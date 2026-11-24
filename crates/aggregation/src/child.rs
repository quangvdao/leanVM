//! The guest's witness for one child proof, and the opening shapes it shares with the guest's constants.

use super::cells::count;
use super::error::AggregateError;
use super::hints::SubHints;
use super::stripe::{stack_hints, stack_log};
use fiat_shamir_prover::merkle::{hash_pair, hash_to_scalars, hash_words};
use lean_vm_core::cpu::from_scalar;
use primitives::field::F128;
use verifier::layout::{SizeSource, TABLE_LOG_FLOORS, stack_blocks};

/// The stacked WHIR opening of a `2^mu` stack at a rate: the verifier's config, then each level's message columns
/// after its folds and its Merkle tree's depth. The hint builder needs it for the real opening and the placeholder map
/// for every candidate size, so it lives here: a candidate whose shape differed from the real one would compile a
/// guest that cannot open the proof it is handed. `None` for a shape no proof has.
pub(super) fn whir_levels(mu: usize, log_inv_rate: usize) -> Option<(verifier::WhirConfig, Vec<usize>, Vec<usize>)> {
    let config = verifier::derive_config(mu, log_inv_rate).ok()?;
    let log_msg_cols: Vec<usize> = config
        .folds
        .iter()
        .scan(mu, |cols, &fold| {
            *cols -= fold;
            Some(*cols)
        })
        .collect();
    let depth = log_msg_cols
        .iter()
        .zip(&config.log_inv_rates)
        .map(|(c, r)| c + r)
        .collect();
    Some((config, log_msg_cols, depth))
}

/// The path bits a query's Merkle walk dispatches on at a time, the guest's `QW` (`walk_path`).
pub(super) const QUERY_WINDOW_BITS: usize = 6;
/// The cells `decode_query_windows` holds a squeezed word's windows in, the guest's `QUERY_WINDOWS_CAP`.
pub(super) const QUERY_WINDOWS_CAP: usize = 42;

/// The depth of the cap a level's paths are walked into, which leaves the walked part whole windows of
/// [`QUERY_WINDOW_BITS`] bits: at least two levels deep, and from `log2(queries) - 5` to five levels below that.
pub(super) fn merkle_cap_depth(queries: usize, depth: usize) -> usize {
    let near = (queries.next_power_of_two().ilog2() as usize)
        .saturating_sub(QUERY_WINDOW_BITS - 1)
        .max(2)
        .min(depth);
    depth - (depth - near) / QUERY_WINDOW_BITS * QUERY_WINDOW_BITS
}

/// The first column of each column group: a maximal run of committed columns, in compact order, sharing one kappa
/// source. The stacked order (descending kappa, then index) keeps a group's columns consecutive, since any other
/// column of equal kappa lies outside it.
pub(super) fn col_groups(sources: &[(usize, usize)]) -> Vec<usize> {
    (0..sources.len())
        .filter(|&c| c == 0 || sources[c] != sources[c - 1])
        .collect()
}

/// The first block of each block group: as [`col_groups`], over the bus blocks, a group never crossing the start of a
/// side (`side_starts`), since a side's blocks tile its own cube.
pub(super) fn block_groups(sources: &[(usize, usize)], side_starts: &[usize]) -> Vec<usize> {
    (0..sources.len())
        .filter(|&b| side_starts.contains(&b) || sources[b] != sources[b - 1])
        .collect()
}

/// The guest's witness for verifying one child (`guests/lean_ethereum.py`): the proof stream, the shape the guest
/// certifies against the announced sizes (each table's log height, the tallest, the pointer width), the stack's
/// placement, the program's value where the link reads it (`bc_value`, the claim the child defers) and the opening's
/// Merkle paths. Nothing here says the child verifies: the guest's run does.
pub(super) fn gen_verify(
    kbc: usize,
    proof: &verifier::Proof,
    summary: &verifier::Summary,
    bc_value: F128,
) -> Result<SubHints, AggregateError> {
    // The guest's buffers have fixed capacities, so its constants do not depend on a child's shape. A child that
    // verifies fits them.
    if proof.stream.len() > STREAM_CAP {
        return Err(AggregateError::InvalidChild);
    }
    let heights: Vec<usize> = summary
        .table_rows
        .iter()
        .zip(TABLE_LOG_FLOORS)
        .map(|(&rows, floor)| (rows.next_power_of_two().trailing_zeros() as usize).max(floor))
        .collect();
    let cycles: u64 = summary.table_rows.iter().sum();
    let bits = (u64::BITS - cycles.leading_zeros()) as usize;
    let width = bits.max(summary.log_memory).max(kbc).max(1);
    let n_rows = *heights.iter().max().unwrap();
    let kappas: Vec<usize> = stack_blocks()
        .iter()
        .map(|block| {
            let base = match block.source {
                SizeSource::Memory => summary.log_memory,
                SizeSource::Table(t) => heights[t],
            };
            base + block.adj
        })
        .collect();
    let mut stream: Vec<F128> = proof.stream.iter().copied().map(from_scalar).collect();
    stream.resize(STREAM_CAP, F128::ZERO);

    let mut hints: SubHints = vec![
        ("stream", stream),
        ("table_logs", heights.iter().map(|&h| count(h)).collect()),
        // The tallest table's log height, the pointer width, and whether the width is the cycles' bit length.
        (
            "link_shape",
            vec![
                count(n_rows),
                count(width),
                if width == bits { F128::ONE } else { F128::ZERO },
            ],
        ),
    ];
    hints.extend(stack_hints(&kappas));
    hints.push(("bc_value", vec![bc_value]));
    let (config, _, depth) =
        whir_levels(stack_log(&kappas), summary.log_inv_rate).ok_or(AggregateError::InvalidChild)?;
    hints.extend(merkle_hints(config.queries, &depth, proof));
    Ok(hints)
}

/// The stacked WHIR opening's Merkle witness, per level: each query's leaf row and the sibling of every node it
/// climbs through below the level's cap, then the cap, every node of the top [`merkle_cap_depth`] levels the queries
/// reach, and which of its nodes they reach. The upper tree is shared by a level's queries; a subtree no query reaches
/// stays zero, unread.
fn merkle_hints(queries: &[usize], depth: &[usize], proof: &verifier::Proof) -> SubHints {
    let mut hints = Vec::new();
    let levels = queries.iter().zip(depth);
    for (level, ((&queries, &depth), phase)) in levels.zip(&proof.merkle_openings).enumerate() {
        let cap_depth = merkle_cap_depth(queries, depth);
        let path_depth = depth - cap_depth;
        let mut nodes = vec![[0u8; 32]; 2 << cap_depth];
        let mut active = vec![F128::ZERO; 1 << cap_depth];
        for opening in phase {
            let leaf: Vec<u64> = opening.leaf.iter().map(|word| word.0).collect();
            // Level 0's rows are K lanes, a cell each; deeper ones E words, two lanes a cell.
            let row = if level == 0 {
                leaf.iter().map(|&x| F128::new(x, 0)).collect()
            } else {
                leaf.as_chunks().0.iter().map(|&[c0, c1]| F128::new(c0, c1)).collect()
            };
            hints.push(("merkle_leaf_rows", row));
            let mut path_siblings = Vec::with_capacity(2 * path_depth);
            let mut node = hash_words(leaf);
            let mut index = (1 << depth) + opening.leaf_index as usize;
            for (height, sibling) in opening.siblings.iter().enumerate() {
                let (left, right) = if index & 1 == 0 {
                    (&node, sibling)
                } else {
                    (sibling, &node)
                };
                if height < path_depth {
                    path_siblings.extend(hash_to_scalars::<F128>(sibling));
                } else {
                    nodes[index] = node;
                    nodes[index ^ 1] = *sibling;
                    active[index >> 1] = F128::ONE;
                }
                node = hash_pair(left, right);
                index >>= 1;
            }
            nodes[1] = node;
            hints.push(("merkle_children", path_siblings));
        }
        hints.push(("merkle_caps", nodes.iter().flat_map(hash_to_scalars).collect()));
        hints.push(("merkle_cap_active", active));
    }
    hints
}

/// The guest's baked buffer caps, which `placeholder_map` compiles in and
/// `gen_verify` admits against: one definition, so a hinted shape can never
/// outgrow the buffer the guest was compiled with.
pub(super) const MU_CAP: usize = 40;
pub(super) const STREAM_CAP: usize = 8192;

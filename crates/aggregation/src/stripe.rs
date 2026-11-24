//! The binary stripe as the guest certifies it (`certify_stack`): the stack's block groups and the witness that
//! places each block.

// Only the snippet test reads the witness until `verify_sub` certifies the stack.
#![cfg_attr(not(test), allow(dead_code))]

use super::cells::count;
use primitives::field::F128;
use verifier::layout::{MIN_STACKED_LOG, SizeSource, StackBlock, stack_blocks};
use verifier::{INITIAL_FOLDING_FACTOR, MAX_STRIPE_WIDTH};

/// The guest's lane-group ids: group `j < 6` is the group of `2^j` lanes when bit `j` of the lane count is set, and
/// groups 6 and 7 are the two groups of 32 lanes of a full stack.
const LANE_GROUPS: usize = 8;
const _: () = assert!(
    MAX_STRIPE_WIDTH == 5 && INITIAL_FOLDING_FACTOR == 6,
    "certify_stack's lane groups"
);

/// A block's kappa source in the guest's `kappa_base`: 1 for the memory, `2 + t` for table `t`.
pub(super) fn source_index(source: SizeSource) -> usize {
    match source {
        SizeSource::Memory => 1,
        SizeSource::Table(table) => 2 + table,
    }
}

/// The first block of each run of consecutive blocks sharing a source, an adjustment and a ring flag. A run's blocks
/// have one size and one ring flag, so they are consecutive in the stack's order.
pub(super) fn block_groups(blocks: &[StackBlock]) -> Vec<usize> {
    let key = |block: &StackBlock| (block.source, block.adj, block.ring);
    (0..blocks.len())
        .filter(|&i| i == 0 || key(&blocks[i]) != key(&blocks[i - 1]))
        .collect()
}

/// The witness streams `certify_stack` reads for blocks of log sizes `kappas` (in [`stack_blocks`] order): the group
/// order, the lane count's bits, and per block its lane group, whether it stays whole, whether its log size is below
/// its group's width, and its in-lane offset over the lanes of its group.
pub(super) fn stack_hints(kappas: &[usize]) -> Vec<(&'static str, Vec<F128>)> {
    let blocks = stack_blocks();
    assert_eq!(kappas.len(), blocks.len(), "one log size per stack block");
    let taken = |b: usize| kappas[b].max(MAX_STRIPE_WIDTH);
    let starts = block_groups(&blocks);
    let mut order: Vec<usize> = (0..starts.len()).collect();
    order.sort_by_key(|&g| (std::cmp::Reverse(taken(starts[g])), !blocks[starts[g]].ring, starts[g]));

    let total: u64 = (0..blocks.len()).map(|b| 1u64 << taken(b)).sum();
    let log = (64 - (total - 1).leading_zeros() as usize).max(MIN_STACKED_LOG);
    let lane = log - INITIAL_FOLDING_FACTOR;
    let lanes = total.div_ceil(1 << lane);
    // Each guest group as (first lane, log width).
    let mut groups = [(0u64, 0usize); LANE_GROUPS];
    if lanes == 64 {
        groups[6] = (0, 5);
        groups[7] = (32, 5);
    } else {
        for (j, group) in groups.iter_mut().enumerate().take(6) {
            *group = ((lanes >> (j + 1)) << (j + 1), j);
        }
    }
    let valid = |j: usize| {
        if j < 6 {
            lanes >> j & 1 == 1 && lanes < 64
        } else {
            lanes == 64
        }
    };

    let bit = |b: bool| if b { F128::ONE } else { F128::ZERO };
    let mut block_hints = vec![[F128::ZERO; 4]; blocks.len()];
    let mut at = 0u64;
    for &g in &order {
        let end = starts.get(g + 1).copied().unwrap_or(blocks.len());
        for b in starts[g]..end {
            let j = (0..LANE_GROUPS)
                .find(|&j| valid(j) && (at >> lane) >= groups[j].0 && (at >> lane) < groups[j].0 + (1 << groups[j].1))
                .expect("every block starts in a lane group");
            let (first, width) = groups[j];
            let whole = taken(b) >= width + lane;
            let w_eff = if whole { 0 } else { width };
            let into = at - (first << lane);
            block_hints[b] = [
                count(j),
                bit(whole),
                bit(kappas[b] < w_eff),
                count((into >> w_eff) as usize),
            ];
            at += 1 << taken(b);
        }
    }
    vec![
        ("stk_group_order", order.iter().map(|&g| count(g)).collect()),
        ("stk_lane_bits", (0..7).map(|k| bit(lanes >> k & 1 == 1)).collect()),
        ("stk_block", block_hints.into_iter().flatten().collect()),
    ]
}

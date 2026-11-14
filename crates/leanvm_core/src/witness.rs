//! `K`-valued columns stacked into one committed witness (§sec:stacking): columns laid
//! end to end, largest first at aligned offsets, into one multilinear `q` over
//! `F64`. An evaluation claim on column `i` at `ζ ∈ E` becomes the claim
//! `q̂(ζ, sel_i) = c` on the stack, where `sel_i` is the high-bit selector of
//! the column's offset.

use primitives::field::F64;

/// Where a column sits in the stacked witness. A [`Placement::VIRTUAL`] column has no
/// placement of its own: either its words are committed inside another column, and its
/// evaluation claims settle there (the KECCAK columns `q_keccak` carries), or they are not
/// committed at all and something else proves them (the KECCAK output lanes).
///
/// A pointer column of the band layout can be split ([`verifier::stack_offsets`]): its
/// rows `j·2^piece_vars..` then sit at `offset + j·2^lane`, a lane down per piece, and
/// only the ring switch reads it, piece by piece.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub n_vars: usize,
    pub offset: usize,
    pub piece_vars: usize,
}

impl Placement {
    pub const VIRTUAL: Placement = Placement {
        n_vars: usize::MAX,
        offset: 0,
        piece_vars: usize::MAX,
    };

    pub(crate) fn is_virtual(&self) -> bool {
        self.n_vars == usize::MAX
    }

    pub(crate) fn is_split(&self) -> bool {
        !self.is_virtual() && self.piece_vars < self.n_vars
    }

    /// The offset of a whole (unsplit) column.
    pub(crate) fn whole(&self) -> usize {
        assert!(!self.is_split(), "a split column has no single window");
        self.offset
    }

    /// Each piece's offset, in row order, in a stack of `2^lane`-word lanes.
    pub(crate) fn piece_offsets(&self, lane: usize) -> impl Iterator<Item = usize> + '_ {
        (0..1usize << (self.n_vars - self.piece_vars)).map(move |j| self.offset + (j << lane))
    }
}

/// The committed stack's shape. `mu` is the log size everything public is derived
/// from (the claims' selector coords, the PCS level ladder), while `n_lanes` counts
/// the `2^(mu - LOG_BATCH)`-word lane blocks that actually carry data: the PCS
/// commits `committed_len()` words and never encodes or hashes the zero tail past
/// them. The level-0 openings leave out the zero words every opened row starts
/// with, the absent lanes among them.
#[derive(Clone, Copy, Debug)]
pub struct StackShape {
    pub mu: usize,
    pub n_lanes: usize,
}

impl StackShape {
    /// Words actually committed: the placed columns rounded up to a whole lane.
    pub fn committed_len(&self) -> usize {
        // Both fields are `pub`, and in release the shift below would mask an
        // underflowed amount into a plausible wrong length rather than panic.
        assert!(self.mu >= crate::pcs::LOG_BATCH, "a stack is at least one lane block");
        self.n_lanes << (self.mu - crate::pcs::LOG_BATCH)
    }
}

/// Lay `2^kappa`-sized items end to end, largest first at aligned offsets and ties
/// broken by index, returning the per-item offset and `Σ 2^kappa`. A `None`
/// kappa takes no space and its offset is meaningless. Depends only on the sizes, so
/// the verifier reconstructs the same tiling.
pub(crate) fn stack_offsets(kappas: &[Option<usize>]) -> (Vec<usize>, usize) {
    let n = kappas.len();
    let mut order: Vec<usize> = (0..n).filter(|&i| kappas[i].is_some()).collect();
    order.sort_by(|&a, &b| kappas[b].unwrap().cmp(&kappas[a].unwrap()).then(a.cmp(&b)));

    let mut offsets = vec![0usize; n];
    let mut off = 0usize;
    for &i in &order {
        let kappa = kappas[i].unwrap();
        assert!(
            kappa < usize::BITS as usize,
            "kappa {kappa} is a log-size, not a length"
        );
        offsets[i] = off;
        off += 1 << kappa;
    }
    (offsets, off)
}

/// Per-column placements and the stack's [`StackShape`] from the columns' log-sizes
/// alone, by [`verifier::stack_offsets`]. A `None` kappa marks a virtual (uncommitted)
/// column. The `pointer` columns, which the opening reads at one point, go in a band at
/// the start of the lanes, split where wider than it. Depends only on lengths and which
/// columns are pointers, and the verifier runs the same function, so it reconstructs
/// the same layout.
pub fn placements_of(kappas: &[Option<usize>], pointer: &[bool]) -> (Vec<Placement>, StackShape) {
    let committed: Vec<usize> = (0..kappas.len()).filter(|&i| kappas[i].is_some()).collect();
    let sizes: Vec<usize> = committed.iter().map(|&i| kappas[i].unwrap()).collect();
    let shared: Vec<bool> = committed.iter().map(|&i| pointer[i]).collect();
    // Floor at the PCS minimum (WHIR's level ladder needs room); tiny
    // witnesses zero-pad up.
    let (spots, mu) = verifier::stack_offsets(&sizes, &shared, crate::pcs::MIN_STACKED_LOG, crate::pcs::LOG_BATCH);
    let mut placements = vec![Placement::VIRTUAL; kappas.len()];
    for ((&i, spot), &n_vars) in committed.iter().zip(&spots).zip(&sizes) {
        placements[i] = Placement {
            n_vars,
            offset: spot.offset as usize,
            piece_vars: spot.piece,
        };
    }
    let placed: usize = sizes.iter().map(|&k| 1usize << k).sum();
    // The columns fit in the fewest whole lanes (the band layout keeps to them too), and
    // the lanes past those are never committed at all.
    let n_lanes = placed.div_ceil(1 << (mu - crate::pcs::LOG_BATCH)).max(1);
    (placements, StackShape { mu, n_lanes })
}

/// Chunk width for the bulk writes below: big enough to amortize the dispatch,
/// small enough to spread one column across cores.
const FILL_CHUNK: usize = 1 << 16;

/// The uninitialized stacked witness: [`StackShape::committed_len`] slots, the
/// placed columns rounded up to a whole lane rather than all the way to `2^mu`.
///
/// # Safety
/// Every slot must be written before it is read. [`split_stack`] hands out one
/// window per whole committed column and zeroes every word no column covers, and
/// [`scatter`] writes a split column's pieces, which together cover the whole
/// allocation, so the obligation reduces to each column's fill writing its own window.
pub unsafe fn alloc_stack(shape: StackShape) -> Vec<F64> {
    // SAFETY: forwarded to the caller by the contract above.
    unsafe { primitives::uninit_vec::<F64>(shape.committed_len()) }
}

/// Carve the stack into one mutable window per whole committed column, in column order,
/// and zero every word no column covers. A virtual column gets an empty window: it is
/// not in the stack, so its values need storage of their own. So does a split column,
/// whose pieces [`scatter`] writes once it is filled.
///
/// Writing each column into its final place is what lets the whole witness be
/// written once. Copying columns in afterwards would move the entire stack a
/// second time for nothing, and allocating the stack zeroed
/// would memset all `2^m` slots only for the columns to overwrite them.
///
/// Safe despite [`alloc_stack`]'s uninitialized allocation: the windows and the split
/// columns' pieces are disjoint (checked here), this zeroes the rest, and a split
/// column's pieces are left to [`scatter`].
pub fn split_stack<'a>(q: &'a mut [F64], placements: &[Placement], lane: usize) -> Vec<&'a mut [F64]> {
    // Every piece of every committed column: (offset, length, column, whole).
    let mut pieces: Vec<(usize, usize, usize, bool)> = Vec::new();
    for (i, p) in placements.iter().enumerate().filter(|(_, p)| !p.is_virtual()) {
        if p.is_split() {
            pieces.extend(p.piece_offsets(lane).map(|offset| (offset, 1 << p.piece_vars, i, false)));
        } else {
            pieces.push((p.offset, 1 << p.n_vars, i, true));
        }
    }
    pieces.sort_unstable();

    let mut windows: Vec<&mut [F64]> = Vec::with_capacity(placements.len());
    windows.resize_with(placements.len(), || &mut []);
    let mut rest = q;
    let mut at = 0usize;
    for (offset, len, i, whole) in pieces {
        assert!(offset >= at, "the stacked columns overlap");
        let (gap, tail) = rest.split_at_mut(offset - at);
        parallel::chunks_mut(gap, FILL_CHUNK, |_, pad| pad.fill(F64::ZERO));
        let (window, tail) = tail.split_at_mut(len);
        if whole {
            windows[i] = window;
        }
        rest = tail;
        at = offset + len;
    }
    parallel::chunks_mut(rest, FILL_CHUNK, |_, pad| pad.fill(F64::ZERO));
    windows
}

/// Write a split column, filled contiguously in `column`, into its pieces of the stack.
pub fn scatter(q: &mut [F64], placement: &Placement, lane: usize, column: &[F64]) {
    assert_eq!(column.len(), 1 << placement.n_vars, "a split column spans its rows");
    for (offset, rows) in placement.piece_offsets(lane).zip(column.chunks_exact(1 << placement.piece_vars)) {
        q[offset..offset + rows.len()].copy_from_slice(rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape has to cover the placed columns with as few lanes as possible: the
    /// prover commits `committed_len()` words, so anything short of `placed` would
    /// leave a claim outside the commitment, and any extra whole lane is work the
    /// change exists to avoid. `mu` stays the announced `2^mu` size both sides
    /// derive their layout from.
    #[test]
    fn placements_cover_the_columns_in_the_fewest_lanes() {
        let lanes = 1usize << crate::pcs::LOG_BATCH;
        // A power-of-two total, one word over one, one just under, a virtual column
        // in the middle, and a stack far below the MIN_STACKED_LOG floor.
        let cases: [Vec<Option<usize>>; 5] = [
            vec![Some(20), Some(20)],
            vec![Some(20), Some(20), Some(0)],
            vec![Some(20), Some(19), Some(18), Some(17)],
            vec![Some(21), None, Some(19), None, Some(15)],
            vec![Some(3), Some(2)],
        ];
        for kappas in cases {
            let placed: usize = kappas.iter().flatten().map(|k| 1usize << k).sum();
            let (placements, shape) = placements_of(&kappas, &vec![false; kappas.len()]);
            let lane_block = 1usize << (shape.mu - crate::pcs::LOG_BATCH);

            assert_eq!(
                shape.mu,
                crate::log2_ceil_usize(placed).max(crate::pcs::MIN_STACKED_LOG)
            );
            assert!((1..=lanes).contains(&shape.n_lanes), "lane count out of range");
            assert!(shape.committed_len() >= placed, "columns must fit the commitment");
            assert!(
                shape.committed_len() - placed < lane_block,
                "a whole lane of the commitment carries no data"
            );
            assert!(shape.committed_len() <= 1usize << shape.mu);
            // And every column really does live inside those lanes.
            for p in placements.iter().filter(|p| !p.is_virtual()) {
                assert!(p.offset + (1usize << p.n_vars) <= shape.committed_len());
            }
        }
    }

    /// The opening reads every pointer column at one point, over the in-lane window its pieces cover, so the
    /// pointers must sit in a narrow band at the start of their lanes, split when a column is wider than the band. The
    /// windows of [`split_stack`] and the pieces of [`scatter`] must then still cover every committed word once, the
    /// pieces holding the column's rows in order.
    #[test]
    fn pointer_columns_sit_in_a_band_and_the_stack_holds_them() {
        // Pointer columns wider than the band (split), at its width, and narrower, among other columns, a virtual
        // one included.
        let mut columns = vec![(Some(15), true); 2];
        columns.extend([(Some(13), true); 2]);
        columns.extend([(Some(12), true); 8].iter().chain(&[(Some(12), false); 16]));
        columns.extend([(Some(10), true); 32].iter().chain(&[(Some(10), false); 192]));
        columns.extend([(Some(14), false), (None, false), (Some(9), false), (Some(8), false)]);
        let (kappas, pointer): (Vec<Option<usize>>, Vec<bool>) = columns.into_iter().unzip();
        let (placements, shape) = placements_of(&kappas, &pointer);
        let lane = shape.mu - crate::pcs::LOG_BATCH;
        assert!(placements.iter().any(|p| p.is_split()), "the case splits a column");
        let band = (0..kappas.len()).filter(|&i| pointer[i] && kappas[i].is_some()).map(|i| placements[i].piece_vars).max().unwrap();
        assert!(band < lane, "the band ({band}) is narrower than a lane ({lane})");
        for i in (0..kappas.len()).filter(|&i| pointer[i] && kappas[i].is_some()) {
            for offset in placements[i].piece_offsets(lane) {
                assert!(offset % (1 << lane) + (1 << placements[i].piece_vars) <= 1 << band, "column {i} is in the band");
            }
        }

        // Write column i's row u as i·2^32 + u + 1 through the windows and the scatter, and read it back.
        let value = |i: usize, u: usize| F64(((i as u64) << 32) + u as u64 + 1);
        let mut q = vec![F64(u64::MAX); shape.committed_len()];
        let windows = split_stack(&mut q, &placements, lane);
        for (i, window) in windows.into_iter().enumerate() {
            for (u, word) in window.iter_mut().enumerate() {
                *word = value(i, u);
            }
        }
        for (i, p) in placements.iter().enumerate().filter(|(_, p)| p.is_split()) {
            let column: Vec<F64> = (0..1 << p.n_vars).map(|u| value(i, u)).collect();
            scatter(&mut q, p, lane, &column);
        }
        let mut covered = 0;
        for (i, p) in placements.iter().enumerate().filter(|(_, p)| !p.is_virtual()) {
            for (j, offset) in p.piece_offsets(lane).enumerate() {
                for r in 0..1usize << p.piece_vars {
                    assert_eq!(q[offset + r], value(i, (j << p.piece_vars) + r), "column {i} row {}", (j << p.piece_vars) + r);
                }
                covered += 1 << p.piece_vars;
            }
        }
        assert_eq!(q.iter().filter(|w| w.0 != 0).count(), covered, "every other word is zero");
    }
}

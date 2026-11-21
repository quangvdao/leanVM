//! `K`-valued columns stacked into one committed witness (§sec:stacking): columns laid
//! end to end, largest first at aligned offsets, into one multilinear `q` over
//! `F64`. An evaluation claim on column `i` at `ζ ∈ E` becomes the claim
//! `q̂(ζ, sel_i) = c` on the stack, where `sel_i` is the high-bit selector of
//! the column's offset.

use ::pcs::stack_open::{StackClaim, SumPart};
use primitives::field::{F64, F128};

/// Where a column sits in the stacked witness. A [`Placement::VIRTUAL`] column has no
/// placement of its own: either its words are committed inside another column, and its
/// evaluation claims settle there (the KECCAK columns `q_keccak` carries), or they are not
/// committed at all and something else proves them (the KECCAK output lanes).
///
/// A column of the stripe or band layout can be split ([`verifier::stack_offsets`]): its
/// rows `j·2^piece_vars..` then sit at `offset + j·2^lane`, a lane down per piece. The ring
/// switch reads it piece by piece, and a point claim at one striped stack point
/// ([`Placement::claim`]).
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

    /// Each piece's offset, in row order, in a stack of `2^lane`-word lanes.
    pub(crate) fn piece_offsets(&self, lane: usize) -> impl Iterator<Item = usize> + '_ {
        (0..1usize << (self.n_vars - self.piece_vars)).map(move |j| self.offset + (j << lane))
    }

    /// The `2^n_vars` rows from `start` (a multiple of `2^n_vars`) as a column of their own:
    /// whole when they fit in one piece, else the run of pieces holding them
    /// (`verifier::multilinear::Placement::rows`).
    pub(crate) fn rows(&self, start: usize, n_vars: usize, lane: usize) -> Placement {
        assert!(start.is_multiple_of(1 << n_vars) && start + (1 << n_vars) <= 1 << self.n_vars);
        let piece = self.piece_vars;
        Placement {
            n_vars,
            offset: self.offset + ((start >> piece) << lane) + (start & ((1 << piece) - 1)),
            piece_vars: piece.min(n_vars),
        }
    }

    /// The claim that the column takes `value` at row `point`, on the stack: a point claim on
    /// a whole column, a striped one on a split column, its pieces at the low end of the point.
    pub(crate) fn claim(&self, point: Vec<F128>, value: F128) -> StackClaim {
        let (low, lanes) = self.split_point(point);
        StackClaim::new(self.offset, low, lanes, value)
    }

    /// One part of a [`StackClaim::Sum`] at row `point`, as [`Placement::claim`] places it.
    pub(crate) fn sum_part(&self, point: Vec<F128>, scale: F128) -> SumPart {
        let (low_point, lane_point) = self.split_point(point);
        SumPart {
            offset: self.offset,
            low_point,
            lane_point,
            scale,
        }
    }

    /// The column's rows in `q`, the stack.
    pub(crate) fn view<'a>(&self, q: &'a [F64], lane: usize) -> Col<'a> {
        assert!(!self.is_virtual(), "a virtual column is not in the stack");
        let len = 1 << self.n_vars;
        if self.is_split() {
            let end = self.offset + (((len >> self.piece_vars) - 1) << lane) + (1 << self.piece_vars);
            Col::split(&q[self.offset..end], self.piece_vars, lane, len)
        } else {
            Col::whole(&q[self.offset..][..len])
        }
    }

    /// [`Placement::view`], to write.
    #[cfg(test)]
    pub(crate) fn view_mut<'a>(&self, q: &'a mut [F64], lane: usize) -> ColMut<'a> {
        let col = self.view(q, lane);
        let (piece, lane, len) = (col.piece, col.lane, col.len);
        ColMut {
            words: parallel::SendPtr(q[self.offset..].as_mut_ptr()),
            piece,
            lane,
            len,
            _q: std::marker::PhantomData,
        }
    }

    fn split_point(&self, mut point: Vec<F128>) -> (Vec<F128>, Vec<F128>) {
        assert!(
            !self.is_virtual() && point.len() == self.n_vars,
            "a committed column's row point"
        );
        let lanes = point.split_off(self.piece_vars);
        (point, lanes)
    }
}

/// A column's rows in order, wherever the stack holds them: one run of words for a whole column, or for a split one
/// ([`Placement`]) `2^piece` rows per piece, piece `j` starting `j·2^lane` words after the first.
#[derive(Debug)]
pub struct Col<'a, T = F64> {
    /// From the column's first word to the end of its last piece.
    words: &'a [T],
    piece: usize,
    lane: usize,
    len: usize,
}

// By hand: the derives would demand `T: Clone` and `T: Copy`.
impl<T> Clone for Col<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Col<'_, T> {}

impl<'a, T: Copy> Col<'a, T> {
    /// A column held as one run of words.
    pub const fn whole(words: &'a [T]) -> Self {
        Self {
            words,
            piece: usize::BITS as usize - 1,
            lane: 0,
            len: words.len(),
        }
    }

    /// `len` rows, `2^piece` per piece, piece `j` at `words[j·2^lane..]`.
    pub fn split(words: &'a [T], piece: usize, lane: usize, len: usize) -> Self {
        assert!(
            piece <= lane && len.is_multiple_of(1 << piece),
            "pieces of whole rows inside their lanes"
        );
        assert!(len == 0 || words.len() >= (((len >> piece) - 1) << lane) + (1 << piece));
        Self {
            words,
            piece,
            lane,
            len,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Where row `u` is in `words`.
    #[inline(always)]
    fn at(&self, u: usize) -> usize {
        ((u >> self.piece) << self.lane) + (u & ((1 << self.piece) - 1))
    }

    /// Row `u`, which the column has.
    #[inline(always)]
    fn read(&self, u: usize) -> &'a T {
        // SAFETY: [`Col::split`] checks that `words` reaches the last row's word, and every row below `len` is at or
        // before it; the callers check `u < len`.
        unsafe { self.words.get_unchecked(self.at(u)) }
    }

    /// Row `u`, if the column has it.
    #[inline(always)]
    pub fn get(&self, u: usize) -> Option<T> {
        (u < self.len).then(|| *self.read(u))
    }

    /// Rows in one piece: the whole column if it is one run.
    pub fn piece_len(&self) -> usize {
        self.len.min(1 << self.piece)
    }

    /// The pieces, in row order: one run of [`Self::piece_len`] rows each.
    pub fn pieces(&self) -> impl ExactSizeIterator<Item = &'a [T]> + use<'a, T> {
        let (words, piece, lane) = (self.words, self.piece_len(), self.lane);
        (0..self.len.div_ceil(piece.max(1))).map(move |j| &words[j << lane..][..piece])
    }

    /// Rows `start..end` as runs of words, each with its first row.
    pub fn runs(&self, start: usize, end: usize) -> impl Iterator<Item = (usize, &'a [T])> + use<'a, T> {
        assert!(start <= end && end <= self.len);
        let (col, piece) = (*self, self.piece_len().max(1));
        let mut u = start;
        std::iter::from_fn(move || {
            (u < end).then(|| {
                let len = (piece - u % piece).min(end - u);
                let run = (u, &col.words[col.at(u)..][..len]);
                u += len;
                run
            })
        })
    }

    /// Rows `start..start + len`, which lie in one piece.
    pub fn run(&self, start: usize, len: usize) -> &'a [T] {
        assert!(
            start + len <= self.len && (len == 0 || start / self.piece_len() == (start + len - 1) / self.piece_len())
        );
        &self.words[self.at(start)..][..len]
    }

    /// Rows `start..end` as one run of words: borrowed when they lie in one piece, copied otherwise.
    pub fn window(&self, start: usize, end: usize) -> std::borrow::Cow<'a, [T]> {
        assert!(start <= end && end <= self.len);
        let piece = self.piece_len().max(1);
        if start == end {
            std::borrow::Cow::Borrowed(&[])
        } else if start / piece == (end - 1) / piece {
            std::borrow::Cow::Borrowed(&self.words[self.at(start)..][..end - start])
        } else {
            std::borrow::Cow::Owned(self.runs(start, end).flat_map(|(_, run)| run.iter().copied()).collect())
        }
    }

    /// The first `len` rows.
    pub fn prefix(&self, len: usize) -> Self {
        assert!(len <= self.len);
        Self { len, ..*self }
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + use<'a, T> {
        self.pieces().flat_map(|piece| piece.iter().copied())
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.iter().collect()
    }

    /// The last row that satisfies `f`.
    pub fn rposition(&self, f: impl Fn(T) -> bool) -> Option<usize> {
        let piece = self.piece_len();
        let pieces: Vec<&[T]> = self.pieces().collect();
        pieces
            .iter()
            .enumerate()
            .rev()
            .find_map(|(j, p)| p.iter().rposition(|&v| f(v)).map(|r| j * piece + r))
    }
}

impl<'a> Col<'a, F64> {
    /// A pointer column's words.
    pub fn words(self) -> Col<'a, u64> {
        Col {
            words: primitives::cast_slice(self.words),
            piece: self.piece,
            lane: self.lane,
            len: self.len,
        }
    }
}

impl<'a, T: Copy> From<&'a [T]> for Col<'a, T> {
    fn from(words: &'a [T]) -> Self {
        Self::whole(words)
    }
}

impl<'a, T: Copy> From<&'a Vec<T>> for Col<'a, T> {
    fn from(words: &'a Vec<T>) -> Self {
        Self::whole(words)
    }
}

impl<T: Copy> std::ops::Index<usize> for Col<'_, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, u: usize) -> &T {
        assert!(u < self.len, "row {u} of a column of {}", self.len);
        self.read(u)
    }
}

/// Several columns' multilinear extensions at one `point`, in one dispatch: a split column is the `eq`-weighted sum of
/// its pieces at the point's low end.
pub fn mle_eval_cols(cols: &[Col], point: &[F128]) -> Vec<F128> {
    let mut values = vec![F128::ZERO; cols.len()];
    let mut logs: Vec<usize> = cols.iter().map(|c| c.piece_len().trailing_zeros() as usize).collect();
    logs.sort_unstable();
    logs.dedup();
    for log in logs {
        let of: Vec<usize> = (0..cols.len()).filter(|&c| cols[c].piece_len() == 1 << log).collect();
        let pieces: Vec<&[F64]> = of.iter().flat_map(|&c| cols[c].pieces()).collect();
        let at = primitives::multilinear::mle_eval_many_par(&pieces, &point[..log]);
        let eq = primitives::multilinear::eq_table_serial(&point[log..]);
        let mut at = at.into_iter();
        for &c in &of {
            values[c] = eq.iter().zip(at.by_ref()).map(|(&e, v)| e * v).sum();
        }
    }
    values
}

/// A column to write, as [`Col`] reads it: the window a fill writes, in the stack or in a buffer of its own. Rows on
/// different threads may be written at once, so each write is `unsafe`, and the caller keeps them disjoint.
pub struct ColMut<'a> {
    words: parallel::SendPtr<F64>,
    piece: usize,
    lane: usize,
    len: usize,
    _q: std::marker::PhantomData<&'a mut [F64]>,
}

// SAFETY: a `ColMut` writes only through `unsafe` calls whose callers keep the rows of concurrent writes disjoint.
unsafe impl Sync for ColMut<'_> {}
unsafe impl Send for ColMut<'_> {}

impl<'a> ColMut<'a> {
    /// No window: a column held elsewhere.
    pub const EMPTY: Self = Self {
        words: parallel::SendPtr(std::ptr::NonNull::dangling().as_ptr()),
        piece: 0,
        lane: 0,
        len: 0,
        _q: std::marker::PhantomData,
    };

    pub fn whole(words: &'a mut [F64]) -> Self {
        Self {
            words: parallel::SendPtr(words.as_mut_ptr()),
            piece: usize::BITS as usize - 1,
            lane: 0,
            len: words.len(),
            _q: std::marker::PhantomData,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn piece_len(&self) -> usize {
        self.len.min(1 << self.piece)
    }

    /// Write `values` to rows `start..`, piece by piece.
    ///
    /// # Safety
    /// The rows are the column's, and no concurrent write or read reaches them.
    #[inline]
    pub unsafe fn write(&self, stream: &primitives::stream::Stream, start: usize, values: &[F64]) {
        debug_assert!(start + values.len() <= self.len);
        let mask = (1usize << self.piece) - 1;
        let mut u = start;
        let mut values = values;
        while !values.is_empty() {
            // Up to the end of row `u`'s piece: shifts and masks only, as fills call this a few rows at a time.
            let len = (mask - (u & mask) + 1).min(values.len());
            let at = ((u >> self.piece) << self.lane) + (u & mask);
            // SAFETY: forwarded to the caller; row `u`'s run lies in its piece.
            unsafe { stream.copy_uninit(self.words.0.add(at), &values[..len]) };
            values = &values[len..];
            u += len;
        }
    }

    /// Rows `start..start + len`, which lie in one piece, as a window of their own.
    pub fn rows(&mut self, start: usize, len: usize) -> ColMut<'_> {
        let piece = self.piece_len();
        assert!(start + len <= self.len && (len == 0 || start / piece == (start + len - 1) / piece));
        let at = ((start >> self.piece) << self.lane) + (start & ((1 << self.piece) - 1));
        // SAFETY: the rows are in one piece of this window, which `self`'s borrow keeps from any other.
        ColMut::whole(unsafe { std::slice::from_raw_parts_mut(self.words.0.add(at), len) })
    }

    /// Disjoint runs of `len` rows each, which split the column, every run in one piece.
    pub fn runs_of(self, len: usize) -> Vec<&'a mut [F64]> {
        assert!(len > 0 && self.len.is_multiple_of(len) && self.piece_len().is_multiple_of(len));
        (0..self.len / len)
            .map(|k| {
                let start = k * len;
                let at = ((start >> self.piece) << self.lane) + (start & ((1 << self.piece) - 1));
                // SAFETY: the runs are disjoint rows of this window, each in one piece, and `self` is consumed.
                unsafe { std::slice::from_raw_parts_mut(self.words.0.add(at), len) }
            })
            .collect()
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
/// column. The `ring` columns (the pointers and q_keccak), which the opening reads
/// through ring weights, are cut across the first lanes with every other column that
/// fits there, or else go in a band at the start of the lanes. Depends only on lengths and which
/// columns are pointers, and the verifier runs the same function, so it reconstructs
/// the same layout.
pub fn placements_of(kappas: &[Option<usize>], ring: &[bool]) -> (Vec<Placement>, StackShape) {
    let committed: Vec<usize> = (0..kappas.len()).filter(|&i| kappas[i].is_some()).collect();
    let sizes: Vec<usize> = committed.iter().map(|&i| kappas[i].unwrap()).collect();
    let ring: Vec<bool> = committed.iter().map(|&i| ring[i]).collect();
    // Floor at the PCS minimum (WHIR's level ladder needs room); tiny
    // witnesses zero-pad up.
    let (spots, mu) = verifier::stack_offsets(&sizes, &ring, crate::pcs::MIN_STACKED_LOG, crate::pcs::LOG_BATCH);
    let mut placements = vec![Placement::VIRTUAL; kappas.len()];
    for ((&i, spot), &n_vars) in committed.iter().zip(&spots).zip(&sizes) {
        placements[i] = Placement {
            n_vars,
            offset: spot.offset as usize,
            piece_vars: spot.piece,
        };
    }
    let placed: usize = sizes.iter().map(|&k| 1usize << k).sum();
    // The columns fit in the fewest whole lanes (the stripe and band layouts keep to them too), and
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
/// window per committed column and zeroes every word no column covers, so the
/// obligation reduces to each column's fill writing its own window.
pub unsafe fn alloc_stack(shape: StackShape) -> Vec<F64> {
    // SAFETY: forwarded to the caller by the contract above.
    unsafe { primitives::uninit_vec::<F64>(shape.committed_len()) }
}

/// Carve the stack into one window per committed column, in column order, and zero every word no column covers. A
/// split column's window is its pieces ([`ColMut`]). A virtual column gets an empty window: it is not in the stack, so
/// its values need storage of their own.
///
/// Writing each column into its final place is what lets the whole witness be
/// written once. Copying columns in afterwards would move the entire stack a
/// second time for nothing, and allocating the stack zeroed
/// would memset all `2^m` slots only for the columns to overwrite them.
///
/// Safe despite [`alloc_stack`]'s uninitialized allocation: the windows are disjoint (checked here), and this zeroes
/// the rest.
pub fn split_stack<'a>(q: &'a mut [F64], placements: &[Placement], lane: usize) -> Vec<ColMut<'a>> {
    // Every piece of every committed column: (offset, length).
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    for p in placements.iter().filter(|p| !p.is_virtual()) {
        let len = 1usize << p.piece_vars.min(p.n_vars);
        pieces.extend(p.piece_offsets(lane).map(|offset| (offset, len)));
    }
    pieces.sort_unstable();
    let mut at = 0usize;
    for (offset, len) in pieces {
        assert!(offset >= at, "the stacked columns overlap");
        parallel::chunks_mut(&mut q[at..offset], FILL_CHUNK, |_, pad| pad.fill(F64::ZERO));
        at = offset + len;
    }
    parallel::chunks_mut(&mut q[at..], FILL_CHUNK, |_, pad| pad.fill(F64::ZERO));

    let words = parallel::SendPtr(q.as_mut_ptr());
    placements
        .iter()
        .map(|p| {
            if p.is_virtual() {
                return ColMut::EMPTY;
            }
            ColMut {
                // SAFETY: the column's pieces lie in `q`, checked disjoint from every other column's above.
                words: parallel::SendPtr(unsafe { words.add(p.offset) }),
                piece: if p.is_split() {
                    p.piece_vars
                } else {
                    usize::BITS as usize - 1
                },
                lane: if p.is_split() { lane } else { 0 },
                len: 1 << p.n_vars,
                _q: std::marker::PhantomData,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A window inside one piece borrows the stack's words; one across pieces copies the same rows in order.
    #[test]
    fn a_window_is_its_rows_in_order() {
        // 4 rows a piece, pieces 8 words apart: row `u` is word `(u / 4)·8 + u % 4`.
        let words: Vec<u64> = (0..28).collect();
        let col = Col::split(&words, 2, 3, 16);
        let rows = |start: usize, end: usize| (start..end).map(|u| col[u]).collect::<Vec<_>>();
        for (start, end) in [(0, 4), (5, 7), (4, 4), (16, 16), (2, 6), (3, 16), (0, 16)] {
            let window = col.window(start, end);
            assert_eq!(*window, rows(start, end));
            let one_piece = end == start || start / 4 == (end - 1) / 4;
            assert_eq!(matches!(window, std::borrow::Cow::Borrowed(_)), one_piece);
        }
    }

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

    /// The opening reads every ring column (here the pointers) at one point, so the stripe layout must cut each into
    /// one piece per lane of the first lanes, all at one in-lane offset. The windows of [`split_stack`] must then still
    /// cover every committed word once, the pieces holding the column's rows in order.
    #[test]
    fn ring_columns_are_striped_and_the_stack_holds_them() {
        // Pointer columns of several sizes among other columns, a virtual one included.
        let mut columns = vec![(Some(15), true); 2];
        columns.extend([(Some(13), true); 2]);
        columns.extend([(Some(12), true); 8].iter().chain(&[(Some(12), false); 16]));
        columns.extend([(Some(10), true); 32].iter().chain(&[(Some(10), false); 192]));
        columns.extend([(Some(14), false), (None, false), (Some(9), false), (Some(8), false)]);
        let (kappas, pointer): (Vec<Option<usize>>, Vec<bool>) = columns.into_iter().unzip();
        let (placements, shape) = placements_of(&kappas, &pointer);
        let lane = shape.mu - crate::pcs::LOG_BATCH;
        let width = (shape.n_lanes.ilog2() as usize).min(verifier::MAX_STRIPE_WIDTH);
        for i in (0..kappas.len()).filter(|&i| pointer[i] && kappas[i].is_some()) {
            let p = placements[i];
            assert_eq!(p.n_vars - p.piece_vars, width, "column {i} has a piece per stripe lane");
            for (j, offset) in p.piece_offsets(lane).enumerate() {
                assert_eq!(offset, (j << lane) + p.offset, "column {i}'s piece {j} is in lane {j}");
            }
            assert!(p.offset < 1 << lane, "column {i} starts in lane 0");
        }

        // Write column i's row u as i·2^32 + u + 1 through the windows, and read it back.
        let value = |i: usize, u: usize| F64(((i as u64) << 32) + u as u64 + 1);
        let mut q = vec![F64(u64::MAX); shape.committed_len()];
        let windows = split_stack(&mut q, &placements, lane);
        let stream = primitives::stream::Stream::new();
        for (i, window) in windows.iter().enumerate() {
            let column: Vec<F64> = (0..window.len()).map(|u| value(i, u)).collect();
            // A write at an odd row and length crosses pieces.
            let cut = column.len().min(3);
            // SAFETY: the rows are the window's, written once.
            unsafe {
                window.write(&stream, 0, &column[..cut]);
                window.write(&stream, cut, &column[cut..]);
            }
        }
        drop(windows);
        for (i, p) in placements.iter().enumerate().filter(|(_, p)| !p.is_virtual()) {
            let col = p.view(&q, lane);
            assert!(
                (0..1 << p.n_vars).all(|u| col[u] == value(i, u)),
                "column {i} reads back in row order"
            );
            let runs: Vec<F64> = col
                .runs(1, col.len())
                .flat_map(|(_, run)| run.iter().copied())
                .collect();
            assert_eq!(runs, col.to_vec()[1..], "column {i}'s runs");
        }
        let mut covered = 0;
        for (i, p) in placements.iter().enumerate().filter(|(_, p)| !p.is_virtual()) {
            for (j, offset) in p.piece_offsets(lane).enumerate() {
                for r in 0..1usize << p.piece_vars {
                    assert_eq!(
                        q[offset + r],
                        value(i, (j << p.piece_vars) + r),
                        "column {i} row {}",
                        (j << p.piece_vars) + r
                    );
                }
                covered += 1 << p.piece_vars;
            }
        }
        assert_eq!(
            q.iter().filter(|w| w.0 != 0).count(),
            covered,
            "every other word is zero"
        );
    }
}

//! Multilinear and stacking helpers.

use crate::field::{E, GEN, ONE, Scaled, ZERO};
use crate::{VerifierError, require};

/// `eq(i, point)` for every index `i`, LSB-first, at one product per entry: `v * (1 + r) = v + v * r`.
pub fn eq_kernel(point: &[E]) -> Vec<E> {
    let mut out = vec![ZERO; 1 << point.len()];
    out[0] = ONE;
    for (k, &r) in point.iter().enumerate() {
        let (low, high) = out[..2 << k].split_at_mut(1 << k);
        let scale = Scaler::new(r, low.len());
        for (v, h) in low.iter_mut().zip(high) {
            *h = scale.mul(*v);
            *v += *h;
        }
    }
    out
}

/// Multiplication by `r`, through [`Scaled`] once it multiplies enough values to repay the table.
enum Scaler {
    Table(Scaled),
    Plain(E),
}

impl Scaler {
    fn new(r: E, uses: usize) -> Self {
        if uses >= 256 {
            Self::Table(Scaled::new(r))
        } else {
            Self::Plain(r)
        }
    }

    #[inline]
    fn mul(&self, value: E) -> E {
        match self {
            Self::Table(table) => table.mul(value),
            Self::Plain(r) => value * *r,
        }
    }
}

pub fn multilinear_eval<T: Copy + Into<E>>(mle: &[T], point: &[E]) -> Result<E, VerifierError> {
    require(
        point.len() < 64 && mle.len() == 1 << point.len(),
        VerifierError::MultilinearSize,
    )?;
    let mut cur: Vec<E> = mle.iter().map(|&value| value.into()).collect();
    for &r in point {
        let scale = Scaler::new(r, cur.len() / 2);
        for i in 0..cur.len() / 2 {
            cur[i] = cur[2 * i] + scale.mul(cur[2 * i] + cur[2 * i + 1]);
        }
        cur.truncate(cur.len() / 2);
    }
    Ok(cur[0])
}

pub const fn log2_ceil(value: u64) -> usize {
    (64 - value.saturating_sub(1).leading_zeros()) as usize
}

pub fn log2_strict(value: u64) -> Result<usize, VerifierError> {
    require(value.is_power_of_two(), VerifierError::NotPowerOfTwo)?;
    Ok(value.trailing_zeros() as usize)
}

pub fn eq_eval(left: &[E], right: &[E]) -> E {
    assert_eq!(left.len(), right.len());
    let mut result = ONE;
    for (&x, &y) in left.iter().zip(right) {
        result *= ONE + x + y;
    }
    result
}

pub fn dot(left: &[E], right: &[E]) -> E {
    assert_eq!(left.len(), right.len());
    let mut result = ZERO;
    for (&x, &y) in left.iter().zip(right) {
        result += x * y;
    }
    result
}

/// `pad_n(point)`, the multilinear extension of `[z >= n]`, `n <= 2^len(point)` (lemma lem:suffix).
pub fn padding_weight(point: &[E], n: u64) -> E {
    let bit = |k: usize| E::new(n >> k & 1, 0);
    let low = point
        .iter()
        .enumerate()
        .fold(ONE, |pad, (k, &r)| (ONE + bit(k)) * r + (ONE + bit(k) + r) * pad);
    (ONE + bit(point.len())) * low
}

/// MLE of `[1, g, g^2, ...]` at an LSB-first point.
pub fn index_mle(point: &[E]) -> E {
    let mut result = ONE;
    let mut generator_power = GEN;
    for &challenge in point {
        result *= ONE + challenge * (ONE + generator_power);
        generator_power = generator_power.square();
    }
    result
}

/// A polynomial at `point`, by Horner over its coefficients, constant first.
pub fn poly_eval(coefficients: &[E], point: E) -> E {
    coefficients.iter().rev().fold(ZERO, |acc, &c| acc * point + c)
}

/// Where something sits in the stacked cube: a claim's point fills the low `variables` coordinates, the bits of `index` fixing the rest.
///
/// A pointer column of the band layout ([`stack_offsets`]) can be split: then `piece_variables < variables`, and piece `j`
/// (its rows `j·2^piece_variables..`) sits at `index + j·2^lane`, one lane down per piece. Only ring-switch claims read
/// a split column, piece by piece ([`Placement::pieces`]).
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub variables: usize,
    pub index: u64,
    pub piece_variables: usize,
}

impl Placement {
    pub fn whole(variables: usize, index: u64) -> Self {
        Self {
            variables,
            index,
            piece_variables: variables,
        }
    }

    pub fn is_split(&self) -> bool {
        self.piece_variables < self.variables
    }

    pub fn stack_point(&self, point: &[E], stack_log: usize) -> Vec<E> {
        assert!(!self.is_split(), "a point claim reads a whole column");
        let above = selector_point(self.index >> self.variables, stack_log - self.variables);
        [point, &above].concat()
    }

    /// eq weight of the coordinates above the window.
    pub fn eq_above(&self, point: &[E]) -> E {
        assert!(!self.is_split(), "a ring weight reads one piece at a time");
        let bits = selector_point(self.index >> self.variables, point.len() - self.variables);
        eq_eval(&bits, &point[self.variables..])
    }

    /// The pieces of a column claimed at row `point`, in a stack of `2^lane`-word lanes: each piece's whole placement
    /// and its row weight `eq(j, point[piece_variables..])`. The column at `point` is the sum of its pieces at
    /// `point[..piece_variables]` under these weights.
    pub fn pieces(&self, lane: usize, point: &[E]) -> Vec<(Placement, E)> {
        assert_eq!(point.len(), self.variables, "a column's row point");
        let above = &point[self.piece_variables..];
        (0..1u64 << above.len())
            .map(|j| {
                let piece = Placement::whole(self.piece_variables, self.index + (j << lane));
                (piece, eq_eval(&selector_point(j, above.len()), above))
            })
            .collect()
    }
}

/// Where [`stack_offsets`] puts a block: `2^(size - piece)` pieces of `2^piece` words, piece `j` (the block's words
/// `j·2^piece..`) at `offset + j·2^lane`. An unsplit block is one piece, `piece == size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spot {
    pub offset: u64,
    pub piece: usize,
}

/// Lay out `2^sizes[i]`-word blocks in a stack of `2^lane_fold` lanes, and return each block's [`Spot`] and the stack's
/// log size, `log2_ceil` of the total and at least `min_log`. Depends only on its arguments, so the verifier rebuilds
/// the prover's layout.
///
/// The `shared` blocks (the pointer columns, which the opening reads through ring weights at one point) go in a band:
/// in-lane words `[0, 2^band)` of adjacent lanes, a block larger than `2^band` split into `2^band`-word pieces, one per
/// lane. The opening projects every lane pair its ring weights touch, over the union of their in-lane windows, so a
/// band that is narrow and contiguous keeps that union small in every fold round. The band is the narrowest one with
/// which every block still fits in the `⌈total / lane⌉` lanes the dense layout uses ([`banded`]), and with no shared
/// block, or none that fits, the layout is the dense one ([`dense`]).
pub fn stack_offsets(sizes: &[usize], shared: &[bool], min_log: usize, lane_fold: usize) -> (Vec<Spot>, usize) {
    let total: u64 = sizes.iter().map(|&size| 1u64 << size).sum();
    let log = log2_ceil(total).max(min_log);
    let lane = log - lane_fold;
    let lanes = total.div_ceil(1 << lane);
    let pointer: u64 = sizes.iter().zip(shared).filter(|(_, s)| **s).map(|(&size, _)| 1u64 << size).sum();
    if pointer > 0 {
        for band in 0..=lane {
            if pointer.div_ceil(1 << band) > lanes {
                continue;
            }
            if let Some(spots) = banded(sizes, shared, lane, log, band, lanes) {
                return (spots, log);
            }
        }
    }
    (dense(sizes, shared, lane), log)
}

/// The band layout of [`stack_offsets`] with band width `2^band`, if it fits in `lanes` lanes. The band starts at lane
/// 0, which every fold round pairs with the lanes after it, so no round pairs a band lane with one outside it that the
/// band could have filled. In it go the split blocks largest first, a lane per piece, and then the smaller shared blocks
/// largest first, packed. Every other block then goes largest first, ties by index, at the lowest free offset aligned
/// to its size, past the band in its lanes or in the lanes after it.
fn banded(sizes: &[usize], shared: &[bool], lane: usize, log: usize, band: usize, lanes: u64) -> Option<Vec<Spot>> {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&index| (std::cmp::Reverse(sizes[index]), index));
    let mut spots = vec![Spot { offset: 0, piece: 0 }; sizes.len()];
    let mut next = 0u64;
    for &index in order.iter().filter(|&&i| shared[i] && sizes[i] > band) {
        spots[index] = Spot {
            offset: next << lane,
            piece: band,
        };
        next += 1 << (sizes[index] - band);
    }
    let mut at = 0u64;
    for &index in order.iter().filter(|&&i| shared[i] && sizes[i] <= band) {
        if at == 1 << band {
            (next, at) = (next + 1, 0);
        }
        spots[index] = Spot {
            offset: (next << lane) + at,
            piece: sizes[index],
        };
        at += 1 << sizes[index];
    }
    let band_end = next + u64::from(at > 0);
    // Free aligned blocks `(offset, log)`: the band lanes past the band, and the stack after them.
    let mut free: Vec<(u64, usize)> = Vec::new();
    for l in 0..band_end {
        let used = if l + 1 == band_end && at > 0 { at } else { 1 << band };
        free.extend(aligned_blocks((l << lane) + used, (l + 1) << lane));
    }
    free.extend(aligned_blocks(band_end << lane, 1 << log));
    let mut end = band_end << lane;
    for &index in order.iter().filter(|&&i| !shared[i]) {
        let size = sizes[index];
        let (pick, _) = free.iter().enumerate().filter(|(_, b)| b.1 >= size).min_by_key(|(_, b)| b.0)?;
        let (offset, log) = free.swap_remove(pick);
        free.extend(aligned_blocks(offset + (1 << size), offset + (1 << log)));
        spots[index] = Spot { offset, piece: size };
        end = end.max(offset + (1 << size));
    }
    (end.div_ceil(1 << lane) <= lanes).then_some(spots)
}

/// `[start, end)` as aligned power-of-two blocks `(offset, log)`, each as large as its offset's alignment allows.
fn aligned_blocks(mut start: u64, end: u64) -> Vec<(u64, usize)> {
    let mut blocks = Vec::new();
    while start < end {
        let mut log = start.trailing_zeros().min(63) as usize;
        while start + (1 << log) > end {
            log -= 1;
        }
        blocks.push((start, log));
        start += 1 << log;
    }
    blocks
}

/// Blocks end to end, largest first at aligned offsets and ties by index. Inside a size class smaller than a lane,
/// the `shared` blocks sit at as few in-lane offsets as [`arrange`] finds.
fn dense(sizes: &[usize], shared: &[bool], lane: usize) -> Vec<Spot> {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&index| (std::cmp::Reverse(sizes[index]), index));
    let mut spots = vec![Spot { offset: 0, piece: 0 }; sizes.len()];
    let mut start = 0;
    for class in order.chunk_by(|&a, &b| sizes[a] == sizes[b]) {
        let size = sizes[class[0]];
        let per_lane = 1usize << lane.saturating_sub(size).min(usize::BITS as usize - 1);
        for (&index, slot) in class.iter().zip(arrange(class, shared, per_lane)) {
            spots[index] = Spot {
                offset: start + ((slot as u64) << size),
                piece: size,
            };
        }
        start += (class.len() as u64) << size;
    }
    spots
}

/// The slot of each block in a size class of `per_lane` slots to a lane. The class starts at a multiple of its block
/// size, so slots `p` and `q` share an in-lane offset exactly when `p ≡ q mod per_lane`. The shared blocks, in index
/// order, fill the residue with the most free slots (the lowest on a tie) and move to the next such one when it is
/// full. The other blocks then take the free slots in index order.
fn arrange(class: &[usize], shared: &[bool], per_lane: usize) -> Vec<usize> {
    let n = class.len();
    if per_lane == 1 || per_lane >= n {
        return (0..n).collect();
    }
    let mut free: Vec<Vec<usize>> = (0..per_lane)
        .map(|residue| (residue..n).step_by(per_lane).rev().collect())
        .collect();
    let mut slots = vec![usize::MAX; n];
    let mut residue: Option<usize> = None;
    for (slot, &index) in slots.iter_mut().zip(class) {
        if shared[index] {
            let at = match residue {
                Some(at) if !free[at].is_empty() => at,
                _ => (0..per_lane)
                    .max_by_key(|&at| (free[at].len(), std::cmp::Reverse(at)))
                    .unwrap_or(0),
            };
            residue = Some(at);
            *slot = free[at].pop().unwrap_or(usize::MAX);
        }
    }
    let mut rest: Vec<usize> = free.into_iter().flatten().collect();
    rest.sort_unstable();
    for (slot, free) in slots.iter_mut().filter(|slot| **slot == usize::MAX).zip(rest) {
        *slot = free;
    }
    slots
}

fn selector_point(selector: u64, length: usize) -> Vec<E> {
    (0..length).map(|bit| E::new(selector >> bit & 1, 0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every piece of every block, as `(offset, length, block)`.
    fn pieces(sizes: &[usize], spots: &[Spot], lane: usize) -> Vec<(u64, u64, usize)> {
        let mut pieces: Vec<(u64, u64, usize)> = Vec::new();
        for (block, (&size, spot)) in sizes.iter().zip(spots).enumerate() {
            for j in 0..1u64 << (size - spot.piece) {
                pieces.push((spot.offset + (j << lane), 1 << spot.piece, block));
            }
        }
        pieces.sort_unstable();
        pieces
    }

    /// The band layout must keep what the dense one guarantees (disjoint, aligned pieces in the fewest whole lanes)
    /// and put every pointer piece at in-lane offsets below the band width, which is what it is for.
    #[test]
    fn the_band_layout_is_disjoint_aligned_and_in_the_fewest_lanes() {
        // The 16-blob and 500-SPHINCS aggregations' columns, `(log size, pointer?) × count`, a stack of only pointer
        // columns, a pointer larger than a lane, and no pointer at all.
        let expand = |counts: &[((usize, bool), usize)]| -> Vec<(usize, bool)> {
            counts.iter().flat_map(|&(column, n)| std::iter::repeat_n(column, n)).collect()
        };
        let blob = expand(&[
            ((22, false), 3),
            ((21, false), 9),
            ((21, true), 5),
            ((20, false), 19),
            ((20, true), 10),
            ((19, false), 10),
            ((19, true), 5),
            ((17, false), 5),
            ((17, true), 3),
            ((16, false), 1),
            ((16, true), 10),
        ]);
        let sphincs = expand(&[
            ((24, false), 1),
            ((23, false), 2),
            ((20, false), 23),
            ((20, true), 13),
            ((18, false), 11),
            ((18, true), 15),
            ((17, false), 10),
            ((17, true), 5),
        ]);
        let cases: [(Vec<(usize, bool)>, Option<usize>); 5] = [
            (blob, Some(20)),
            (sphincs, Some(20)),
            (vec![(18, true); 64], None),
            (vec![(23, true), (20, false), (19, true), (18, false), (18, false)], None),
            (vec![(20, false), (19, false), (18, false)], None),
        ];
        for (case, expected_band) in cases {
            let (sizes, shared): (Vec<usize>, Vec<bool>) = case.iter().copied().unzip();
            let (spots, log) = stack_offsets(&sizes, &shared, 18, 6);
            let lane = log - 6;
            let total: u64 = sizes.iter().map(|&s| 1u64 << s).sum();
            let lanes = total.div_ceil(1 << lane);
            let pieces = pieces(&sizes, &spots, lane);
            for window in pieces.windows(2) {
                assert!(window[0].0 + window[0].1 <= window[1].0, "pieces overlap: {window:?}");
            }
            for &(offset, length, _) in &pieces {
                assert_eq!(offset % length, 0, "a piece is aligned to its size");
                assert!(offset + length <= lanes << lane, "a piece is in the fewest whole lanes");
                assert!(length <= 1 << lane || offset % (1 << lane) == 0);
            }
            let band = (0..sizes.len()).filter(|&i| shared[i]).map(|i| spots[i].piece).max();
            if let Some(expected) = expected_band {
                assert_eq!(band, Some(expected), "the narrowest band that fits");
                let first = pieces.iter().find(|p| shared[p.2]).map(|p| p.0);
                assert_eq!(first, Some(0), "the band starts at lane 0");
            }
            if let Some(band) = band {
                for &(offset, length, block) in pieces.iter().filter(|p| shared[p.2]) {
                    let in_lane = offset % (1 << lane);
                    assert!(in_lane + length <= 1 << band.max(sizes[block].min(lane)), "pointer {block} is in the band");
                }
            }
        }
    }
}

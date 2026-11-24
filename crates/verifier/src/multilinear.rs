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
/// A column smaller than its lane group ([`stack_offsets`]) is split: then `piece_variables < variables`, and piece
/// `j` (its rows `j·2^piece_variables..`) sits at `index + j·2^lane`, one lane down per piece. A ring-switch claim reads
/// a split column piece by piece ([`Placement::pieces`]). A point claim still reads it at one stack point
/// ([`Placement::stack_point`]): its pieces share an in-lane window and fill an aligned run of lanes, so the row
/// coordinates above the pieces are lane coordinates.
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

    /// The stack point of a claim at row `point`, in a stack of `2^stack_log` words in `2^lane`-word lanes. For a split
    /// column the low `piece_variables` coordinates are the in-lane ones, the in-lane offset's bits come next, then
    /// the rest of the row point on the lowest lane coordinates and the first lane's bits above them.
    pub fn stack_point(&self, point: &[E], stack_log: usize, lane: usize) -> Vec<E> {
        assert_eq!(point.len(), self.variables, "a column's row point");
        if !self.is_split() {
            let above = selector_point(self.index >> self.variables, stack_log - self.variables);
            return [point, &above].concat();
        }
        let (piece, spread) = (self.piece_variables, self.variables - self.piece_variables);
        let (first_lane, in_lane) = (self.index >> lane, self.index & ((1 << lane) - 1));
        assert!(
            piece <= lane && in_lane % (1 << piece) == 0,
            "a piece is aligned inside one lane"
        );
        assert!(
            first_lane % (1 << spread) == 0,
            "the pieces fill an aligned run of lanes"
        );
        [
            &point[..piece],
            &selector_point(in_lane >> piece, lane - piece),
            &point[piece..],
            &selector_point(first_lane >> spread, stack_log - lane - spread),
        ]
        .concat()
    }

    /// The `2^variables` rows from `start` (a multiple of `2^variables`) as a column of their own: whole when they
    /// fit in one piece, else the run of pieces holding them.
    pub fn rows(&self, start: u64, variables: usize, lane: usize) -> Placement {
        assert!(start.is_multiple_of(1 << variables) && start + (1 << variables) <= 1 << self.variables);
        let piece = self.piece_variables;
        let index = self.index + ((start >> piece) << lane) + (start & ((1 << piece) - 1));
        Placement {
            variables,
            index,
            piece_variables: piece.min(variables),
        }
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
/// log size, at least `min_log`. Depends only on its arguments, so the verifier rebuilds the prover's layout.
///
/// The layout is a binary stripe. Each block takes `2^max(size, MAX_STRIPE_WIDTH)` words, and the blocks go end to
/// end, sorted by size down, then `ring` blocks first, then by index, over the `⌈total / lane⌉` lanes those words
/// need. The lanes split into groups of `2^w` lanes, the powers of two in the lane count, largest first, with `w` at
/// most [`MAX_STRIPE_WIDTH`]. A block whose words fall in a group is cut into one piece per lane of the group, all at
/// one in-lane offset, so every claim on it sits at one in-lane window. A block at least as large as its group stays
/// whole.
///
/// Each block's place is a function of the sorted sizes and a running sum, so a verifier that checks the sort can
/// derive it. The `ring` blocks are the pointer columns and q_keccak, which the opening reads through ring weights:
/// among blocks of one size they come first, which keeps them in the wider groups.
pub fn stack_offsets(sizes: &[usize], ring: &[bool], min_log: usize, lane_fold: usize) -> (Vec<Spot>, usize) {
    assert_eq!(sizes.len(), ring.len(), "one ring flag per block");
    let taken = |index: usize| sizes[index].max(MAX_STRIPE_WIDTH);
    let total: u64 = (0..sizes.len()).map(|index| 1u64 << taken(index)).sum();
    let log = log2_ceil(total).max(min_log);
    let lane = log - lane_fold;
    let lanes = total.div_ceil(1 << lane);
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&index| (std::cmp::Reverse(taken(index)), !ring[index], index));
    // Groups `(first lane, log lanes)`: the powers of two in `lanes`, largest first, none wider than the cap.
    let mut groups: Vec<(u64, usize)> = Vec::new();
    let mut first = 0u64;
    while first < lanes {
        let width = ((lanes - first).ilog2() as usize).min(MAX_STRIPE_WIDTH);
        groups.push((first, width));
        first += 1 << width;
    }
    let mut spots = vec![Spot { offset: 0, piece: 0 }; sizes.len()];
    let (mut group, mut at) = (0, 0u64);
    for &index in &order {
        while at >= (groups[group].0 + (1 << groups[group].1)) << lane {
            group += 1;
        }
        let (first, width) = groups[group];
        let (size, words, into) = (sizes[index], 1u64 << taken(index), at - (first << lane));
        // A block of whole lanes stays whole, even across groups. Any other block is inside one group: its start is
        // aligned to its size, and so is the group's end, the sizes going down.
        spots[index] = if taken(index) >= width + lane {
            Spot {
                offset: at,
                piece: size,
            }
        } else {
            assert!(
                into + words <= 1 << (width + lane),
                "block {index} is inside one lane group"
            );
            Spot {
                offset: (first << lane) + (into >> width),
                piece: size.saturating_sub(width),
            }
        };
        at += words;
    }
    (spots, log)
}

/// The widest lane group of [`stack_offsets`], log: 32 lanes, which keeps every piece of q_keccak whole Keccak slots
/// (it has at least 32), so the prover reads and writes each slot as one run of words.
pub const MAX_STRIPE_WIDTH: usize = 5;

/// The lanes [`stack_offsets`] fills, the lanes past them all zero.
pub fn stack_lanes(sizes: &[usize], stack_log: usize, lane_fold: usize) -> u64 {
    let total: u64 = sizes.iter().map(|&size| 1u64 << size.max(MAX_STRIPE_WIDTH)).sum();
    total.div_ceil(1 << (stack_log - lane_fold)).max(1)
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

    /// The layout's pieces are disjoint, aligned and in [`stack_lanes`] lanes, and every block smaller than its lane
    /// group is cut into one piece per lane of the group (or per word, when smaller than the group), at one in-lane
    /// offset. A claim at a boolean row reads the word the row is at, through the stack point and through `rows`.
    fn check(sizes: &[usize], min_log: usize) {
        let ring: Vec<bool> = (0..sizes.len()).map(|i| i % 3 == 0).collect();
        let (spots, log) = stack_offsets(sizes, &ring, min_log, 6);
        let lane = log - 6;
        let lanes = stack_lanes(sizes, log, 6);
        assert!(lanes <= 64, "the stack holds the lanes");
        let padded: u64 = sizes.iter().map(|&s| 1u64 << s.max(MAX_STRIPE_WIDTH)).sum();
        assert_eq!(log, log2_ceil(padded).max(min_log));
        let all = pieces(sizes, &spots, lane);
        for window in all.windows(2) {
            assert!(window[0].0 + window[0].1 <= window[1].0, "pieces overlap: {window:?}");
        }
        for &(offset, length, _) in &all {
            assert_eq!(offset % length, 0, "a piece is aligned to its size");
            assert!(offset + length <= lanes << lane, "a piece is in the filled lanes");
            assert!(length <= 1 << lane || offset % (1 << lane) == 0);
        }
        // The groups, as in `stack_offsets`.
        let mut groups: Vec<(u64, usize)> = Vec::new();
        let mut first = 0u64;
        while first < lanes {
            let width = ((lanes - first).ilog2() as usize).min(MAX_STRIPE_WIDTH);
            groups.push((first, width));
            first += 1 << width;
        }
        for (i, (&size, spot)) in sizes.iter().zip(&spots).enumerate() {
            let start = spot.offset >> lane;
            let &(first, width) = groups.iter().rev().find(|g| g.0 <= start).unwrap();
            if size.max(MAX_STRIPE_WIDTH) >= width + lane {
                assert_eq!(spot.piece, size, "block {i} of whole lanes is whole");
                continue;
            }
            assert_eq!(
                size - spot.piece,
                size.min(width),
                "block {i} has a piece per lane of its group"
            );
            assert!(
                start + (1 << (size - spot.piece)) <= first + (1 << width),
                "block {i} is in its group"
            );
        }
        let bits = |value: u64, length: usize| -> Vec<E> { selector_point(value, length) };
        for (i, (&size, spot)) in sizes.iter().zip(&spots).enumerate().filter(|(_, (s, _))| **s <= 20) {
            let placement = Placement {
                variables: size,
                index: spot.offset,
                piece_variables: spot.piece,
            };
            let at = |row: u64| spot.offset + ((row >> spot.piece) << lane) + (row & ((1 << spot.piece) - 1));
            for row in [0, 1, (1 << size) / 3, (1 << size) - 1]
                .into_iter()
                .filter(|&r| r < 1 << size)
            {
                let point = placement.stack_point(&bits(row, size), log, lane);
                assert_eq!(point, bits(at(row), log), "block {i} row {row}");
            }
            for variables in [0, size.min(3), size.saturating_sub(1)] {
                let start = (((1u64 << size) / 3) >> variables) << variables;
                let sub = placement.rows(start, variables, lane);
                for row in [0, (1 << variables) - 1] {
                    let point = sub.stack_point(&bits(row, variables), log, lane);
                    assert_eq!(point, bits(at(start + row), log), "block {i} rows from {start}");
                }
            }
        }
    }

    #[test]
    fn the_layout_is_a_binary_stripe() {
        // The 16-blob and 500-SPHINCS aggregations' columns, `(log size) × count`, a full stack of lanes, one block
        // filling the stack, blocks smaller than a stripe, and a lane count with every power of two in it.
        let expand = |counts: &[(usize, usize)]| -> Vec<usize> {
            counts
                .iter()
                .flat_map(|&(size, n)| std::iter::repeat_n(size, n))
                .collect()
        };
        let cases = [
            expand(&[(22, 3), (21, 14), (20, 29), (19, 15), (17, 8), (16, 11)]),
            expand(&[(24, 1), (23, 2), (20, 36), (18, 26), (17, 15)]),
            vec![18; 64],
            vec![24],
            vec![17, 3, 0, 4, 5, 1, 1],
            expand(&[(18, 63)]),
            expand(&[(18, 33), (12, 5)]),
        ];
        for sizes in cases {
            check(&sizes, 18);
        }
        // Random stacks, by a fixed LCG.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = |bound: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % bound
        };
        for _ in 0..300 {
            let n = 1 + next(80) as usize;
            let top = 8 + next(14) as usize;
            let sizes: Vec<usize> = (0..n).map(|_| next(top as u64 + 1) as usize).collect();
            check(&sizes, 12 + next(8) as usize);
        }
    }
}

//! Ring switching: the 64 slice claims a reduction leaves become one claim on the committed columns.

use crate::field::{E, GEN, ONE};
use crate::layout::K_BITS;
use crate::multilinear::{Placement, poly_eval};
use crate::transcript::Transcript;

// The Frobenius shifts of the six stages composing Phi, one challenge each.
const RING_MAP_SHIFTS: [usize; 6] = [32, 16, 8, 4, 2, 1];

/// The drawn map, stage by stage: `a_p+1 = a_p + f_p a_p^(2^shift)`.
fn ring_map(mut value: E, challenges: &[E]) -> E {
    for (&challenge, shift) in challenges.iter().zip(RING_MAP_SHIFTS) {
        value += challenge * value.pow(1 << shift);
    }
    value
}

/// The Frobenius powers `x^(2^k)` for `k < 64`.
fn frobenius(mut x: E) -> Vec<E> {
    (0..K_BITS)
        .map(|_| {
            let power = x;
            x = x.square();
            power
        })
        .collect()
}

/// The pieces at one point: the point's Frobenius powers, coordinate by coordinate, and each piece's placement and
/// the Frobenius powers of its scale.
struct PointGroup {
    point: Vec<Vec<E>>,
    pieces: Vec<(Option<Placement>, Vec<E>)>,
}

/// The weight `W(u) = sum_p Phi(scale_p eq(r_p, u))`, extended and evaluated by the opening at `x`:
/// `sum_k c_k prod_n (1 + r_n^(2^k) + x_n) sum_p above_p(x) scale_p^(2^k)`, the product shared by the pieces at
/// one point.
fn ring_weight(groups: &[PointGroup], x: &[E], coefficients: &[E]) -> E {
    E::sum(groups.iter().map(|group| {
        let n = group.point.len();
        let aboves: Vec<E> = group
            .pieces
            .iter()
            .map(|(placement, _)| placement.map_or(ONE, |placement| placement.eq_above(x)))
            .collect();
        E::sum(coefficients.iter().enumerate().map(|(k, &c)| {
            let product = group.point.iter().zip(&x[..n]).fold(c, |acc, (powers, &x)| acc * (ONE + powers[k] + x));
            let scales = E::sum(group.pieces.iter().zip(&aboves).map(|((_, powers), &above)| above * powers[k]));
            product * scales
        }))
    }))
}

/// One piece of a ring-switched claim: the slices of the columns at `placement` (the whole stack for `None`) at
/// `point`, scaled by `scale`.
pub struct RingPiece {
    pub placement: Option<Placement>,
    pub point: Vec<E>,
    pub scale: E,
}

/// The 64 claims `s[i] = sum_p scale_p z_p(i, point_p)` on the bit slices of the pieces' columns.
pub struct RingClaim {
    pub pieces: Vec<RingPiece>,
    pub slices: Vec<E>,
}

/// A claim on the stack: its weight at a stack point, and its value.
pub type RingStackClaim = (Box<dyn Fn(&[E]) -> E>, E);

/// Each claim's 64 slices become one dense claim `sum_u W(u) column(u) = target`.
///
/// Draw Phi once they are all fixed, shared by every claim, then take each target `T = sum_i x^i Phi(s_i)` against
/// the MLE-friendly weight `W = sum_p Phi(scale_p eq(point_p, .))` on each piece's placement.
pub fn ring_switch(claims: Vec<RingClaim>, transcript: &mut Transcript) -> Vec<RingStackClaim> {
    let challenges = transcript.samples(RING_MAP_SHIFTS.len());
    // The same map as a Frobenius sum, `Phi(a) = sum_k c_k a^(2^k)` for k < 64.
    let coefficient = |k: usize| {
        let stages = challenges
            .iter()
            .zip(RING_MAP_SHIFTS)
            .filter(|&(_, shift)| k & shift != 0);
        stages.fold(ONE, |acc, (f, shift)| acc * f.pow(1 << (k % shift)))
    };
    let coefficients: Vec<E> = (0..K_BITS).map(coefficient).collect();
    claims
        .into_iter()
        .map(|claim| {
            let images: Vec<E> = claim.slices.iter().map(|&value| ring_map(value, &challenges)).collect();
            let target = poly_eval(&images, GEN);
            let mut groups: Vec<PointGroup> = Vec::new();
            let mut points: Vec<Vec<E>> = Vec::new();
            for piece in claim.pieces {
                let scale = frobenius(piece.scale);
                match points.iter().position(|p| *p == piece.point) {
                    Some(g) => groups[g].pieces.push((piece.placement, scale)),
                    None => {
                        groups.push(PointGroup {
                            point: piece.point.iter().map(|&r| frobenius(r)).collect(),
                            pieces: vec![(piece.placement, scale)],
                        });
                        points.push(piece.point);
                    }
                }
            }
            let coefficients = coefficients.clone();
            let weight = move |x: &[E]| ring_weight(&groups, x, &coefficients);
            (Box::new(weight) as Box<dyn Fn(&[E]) -> E>, target)
        })
        .collect()
}

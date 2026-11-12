//! Ring switching: the 64 slice claims a reduction leaves become one claim on the committed columns.

use crate::field::{E, GEN, ONE, ZERO};
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

/// The weight `W(u) = Phi(a eq(r, u))`, extended and evaluated by the opening at `r_prime`:
/// `sum_k c_k a^(2^k) prod_n (1 + r_n^(2^k) + r'_n)`.
fn ring_weight(r: &[E], scale: E, r_prime: &[E], coefficients: &[E]) -> E {
    assert_eq!(r.len(), r_prime.len());
    let mut total = ZERO;
    let mut frobenius = r.to_vec();
    let mut scale = scale;
    for &c in coefficients {
        let mut product = c * scale;
        for (value, &challenge) in frobenius.iter_mut().zip(r_prime) {
            product *= ONE + *value + challenge;
            *value = value.square();
        }
        total += product;
        scale = scale.square();
    }
    total
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
            let coefficients = coefficients.clone();
            let weight = move |x: &[E]| {
                E::sum(claim.pieces.iter().map(|piece| {
                    let n = piece.point.len();
                    let above = piece.placement.map_or(ONE, |placement| placement.eq_above(x));
                    above * ring_weight(&piece.point, piece.scale, &x[..n], &coefficients)
                }))
            };
            (Box::new(weight) as Box<dyn Fn(&[E]) -> E>, target)
        })
        .collect()
}

//! Hotlink's index side: the rounds over the index `j` of cycles, memory cells and instructions, lowest bit
//! first, each of degree 2.
//!
//! The summand is a sum of terms `coef·A(j)·B(j)` and `coef·A(j)` over arrays that are zero past their length,
//! so a cycle table stops at `T + 1` and the program at its size while memory fills `2^W`. A pair of rows of a
//! term is summed only where both of its arrays have one.

use super::pointer::Chunks;
use primitives::field::{F192, F192Unreduced};
use std::sync::atomic::{AtomicU64, Ordering};

/// Pairs per task of a round.
const TASK_PAIRS: usize = 1 << 12;

/// One term of the summand, over arrays named by position.
#[derive(Clone, Copy, Debug)]
pub enum Term {
    Product { coef: F192, a: usize, b: usize },
    Single { coef: F192, a: usize },
}

/// What the index side ends at: the point `r`, every array's value there, and the claim the last round leaves.
pub struct IndexEnd {
    pub point: Vec<F192>,
    pub values: Vec<F192>,
    pub claim: F192,
}

#[inline(always)]
fn at(v: &[F192], i: usize) -> F192 {
    v.get(i).copied().unwrap_or(F192::ZERO)
}

/// `Σ_k A(2k + X)·B(2k + X)` over the first `pairs` pairs, in coefficients.
fn product_round(a: &[F192], b: &[F192], pairs: usize) -> [F192; 3] {
    let [lo, hi, mixed] = parallel::fold_reduce(
        pairs.div_ceil(TASK_PAIRS),
        || [F192Unreduced::ZERO; 3],
        |acc, task| {
            for k in task * TASK_PAIRS..pairs.min((task + 1) * TASK_PAIRS) {
                let (a0, a1, b0, b1) = (a[2 * k], at(a, 2 * k + 1), b[2 * k], at(b, 2 * k + 1));
                acc[0] ^= a0.mul_unreduced(b0);
                acc[1] ^= a1.mul_unreduced(b1);
                acc[2] ^= (a0 + a1).mul_unreduced(b0 + b1);
            }
        },
        |mut x, y| {
            for (x, y) in x.iter_mut().zip(y) {
                *x ^= y;
            }
            x
        },
    )
    .map(F192Unreduced::reduce);
    [lo, hi + lo + mixed, mixed]
}

/// `Σ_k A(2k + X)` over the array.
fn single_round(a: &[F192]) -> [F192; 2] {
    let pairs = a.len().div_ceil(2);
    let [lo, hi] = parallel::fold_reduce(
        pairs.div_ceil(TASK_PAIRS),
        || [F192::ZERO; 2],
        |acc, task| {
            for k in task * TASK_PAIRS..pairs.min((task + 1) * TASK_PAIRS) {
                acc[0] += a[2 * k];
                acc[1] += at(a, 2 * k + 1);
            }
        },
        |x, y| [x[0] + y[0], x[1] + y[1]],
    );
    [lo, lo + hi]
}

/// Bind the lowest variable at `x`.
fn fold(a: &[F192], x: F192) -> Vec<F192> {
    let mut out = vec![F192::ZERO; a.len().div_ceil(2)];
    parallel::chunks_mut(&mut out, TASK_PAIRS, |task, chunk| {
        for (j, o) in chunk.iter_mut().enumerate() {
            let k = task * TASK_PAIRS + j;
            let (lo, hi) = (a[2 * k], at(a, 2 * k + 1));
            *o = lo + x * (lo + hi);
        }
    });
    out
}

/// The index side over `n_vars` variables. `round` takes each round polynomial's coefficients and returns the
/// challenge.
pub fn prove(
    n_vars: usize,
    mut arrays: Vec<Vec<F192>>,
    terms: &[Term],
    round: &mut dyn FnMut(&[F192]) -> F192,
) -> IndexEnd {
    assert!(
        arrays.iter().all(|a| a.len() <= 1 << n_vars),
        "an array past 2^{n_vars}"
    );
    let mut point = Vec::with_capacity(n_vars);
    let mut claim = None;
    for _ in 0..n_vars {
        let mut poly = [F192::ZERO; 3];
        for &term in terms {
            match term {
                Term::Product { coef, a, b } => {
                    let (a, b) = (&arrays[a], &arrays[b]);
                    let pairs = a.len().div_ceil(2).min(b.len().div_ceil(2));
                    for (p, c) in poly.iter_mut().zip(product_round(a, b, pairs)) {
                        *p += coef * c;
                    }
                }
                Term::Single { coef, a } => {
                    for (p, c) in poly.iter_mut().zip(single_round(&arrays[a])) {
                        *p += coef * c;
                    }
                }
            }
        }
        debug_assert!(claim.is_none_or(|c: F192| c == poly[1] + poly[2]));
        let x = round(&poly);
        claim = Some(primitives::multilinear::poly_eval(&poly, x));
        for a in &mut arrays {
            *a = fold(a, x);
        }
        point.push(x);
    }
    let values: Vec<F192> = arrays.iter().map(|a| at(a, 0)).collect();
    let last = terms
        .iter()
        .map(|&term| match term {
            Term::Product { coef, a, b } => coef * values[a] * values[b],
            Term::Single { coef, a } => coef * values[a],
        })
        .sum();
    debug_assert!(claim.is_none_or(|c| c == last));
    IndexEnd {
        point,
        values,
        claim: last,
    }
}

/// An array the pointers of many columns add into at once: three `u64` limbs per entry, added by XOR.
pub struct Scatter {
    cells: Vec<[AtomicU64; 3]>,
}

impl Scatter {
    pub fn new(len: usize) -> Self {
        Self {
            cells: (0..len).map(|_| Default::default()).collect(),
        }
    }

    #[inline(always)]
    fn add(&self, j: usize, value: F192) {
        let Some(cell) = self.cells.get(j) else {
            panic!("Hotlink: index {j} past the array of {}", self.cells.len());
        };
        for (c, limb) in cell.iter().zip(value.to_limbs()) {
            if limb != 0 {
                c.fetch_xor(limb, Ordering::Relaxed);
            }
        }
    }

    /// Add `weight(u)` at the index of each row's word plus `shift`, dropping indices past `2^W`. A word whose
    /// chunks are not all one-hot adds at every index of it ([`Chunks::indices`]), which is what the row side's
    /// `Φ_s` of that word sums to.
    pub fn add_words(&self, chunks: &Chunks, words: &[u64], shift: usize, weight: impl Fn(usize) -> F192 + Sync) {
        let top = 1usize << chunks.width;
        parallel::for_each_chunk(words.len().div_ceil(TASK_PAIRS), |start, end| {
            for task in start..end {
                for u in task * TASK_PAIRS..words.len().min((task + 1) * TASK_PAIRS) {
                    let w = weight(u);
                    if w.is_zero() {
                        continue;
                    }
                    let mut add = |j: usize| {
                        let j = j + shift;
                        if j < top {
                            self.add(j, w);
                        }
                    };
                    match chunks.decode(words[u]) {
                        Some(j) => add(j),
                        None => chunks.indices(words[u], &mut add),
                    }
                }
            }
        });
    }

    /// The entries, without the trailing zeros.
    pub fn into_vec(self) -> Vec<F192> {
        let load = |c: &[AtomicU64; 3]| F192::from_limbs(c.each_ref().map(|l| l.load(Ordering::Relaxed)));
        let len = self.cells.iter().rposition(|c| !load(c).is_zero()).map_or(0, |j| j + 1);
        let mut out = vec![F192::ZERO; len];
        parallel::chunks_mut(&mut out, TASK_PAIRS, |task, chunk| {
            for (j, o) in chunk.iter_mut().enumerate() {
                *o = load(&self.cells[task * TASK_PAIRS + j]);
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::pointer::{ChunkEq, shifted};
    use super::*;
    use primitives::multilinear::{eq_table, poly_eval};
    use primitives::test_rng::Rng;

    fn mle(a: &[F192], point: &[F192]) -> F192 {
        eq_table(point).iter().zip(a).map(|(&e, &v)| e * v).sum()
    }

    #[test]
    fn index_side_matches_the_naive_sum_and_end() {
        let mut rng = Rng::new(0x1d5);
        let n = 9;
        let lens = [512usize, 300, 1, 0, 37, 512, 2, 257];
        let arrays: Vec<Vec<F192>> = lens.iter().map(|&l| rng.ext_vec(l)).collect();
        let terms = [
            Term::Product {
                coef: rng.ext(),
                a: 0,
                b: 1,
            },
            Term::Product {
                coef: rng.ext(),
                a: 2,
                b: 1,
            },
            Term::Product {
                coef: rng.ext(),
                a: 3,
                b: 0,
            },
            Term::Product {
                coef: rng.ext(),
                a: 4,
                b: 5,
            },
            Term::Product {
                coef: rng.ext(),
                a: 6,
                b: 7,
            },
            Term::Product {
                coef: rng.ext(),
                a: 7,
                b: 7,
            },
            Term::Single { coef: rng.ext(), a: 2 },
            Term::Single { coef: rng.ext(), a: 7 },
        ];
        let value = |values: &dyn Fn(usize) -> F192| -> F192 {
            terms
                .iter()
                .map(|&t| match t {
                    Term::Product { coef, a, b } => coef * values(a) * values(b),
                    Term::Single { coef, a } => coef * values(a),
                })
                .sum()
        };
        let mut claim: F192 = (0..1 << n).map(|j| value(&|a| at(&arrays[a], j))).sum();
        let end = prove(n, arrays.clone(), &terms, &mut |poly| {
            assert_eq!(poly.len(), 3);
            assert_eq!(claim, poly[1] + poly[2], "a round's sum");
            let x = rng.ext();
            claim = poly_eval(poly, x);
            x
        });
        assert_eq!(claim, end.claim);
        for (a, &v) in arrays.iter().zip(&end.values) {
            assert_eq!(mle(a, &end.point), v);
        }
        assert_eq!(end.claim, value(&|a| end.values[a]));
    }

    /// A scattered array at a point is the sum of its rows' `Φ_0` (or `Φ_1`) there, for any words.
    #[test]
    fn scatter_is_the_row_side_pointer() {
        let mut rng = Rng::new(0x5ca);
        for width in [7usize, 10, 13] {
            let chunks = Chunks::new(width);
            let top = 1usize << width;
            let words: Vec<u64> = (0..300)
                .map(|u| {
                    let sigma = if u % 10 == 0 {
                        top - 1
                    } else {
                        rng.next_u64() as usize % top
                    };
                    let word = chunks.encode(sigma);
                    match u % 9 {
                        0 => word ^ 1 << (rng.next_u32() % 64),
                        1 => word | rng.next_u64() & rng.next_u64(),
                        _ => word,
                    }
                })
                .collect();
            let weights: Vec<F192> = (0..words.len())
                .map(|u| if u % 13 == 0 { F192::ZERO } else { rng.ext() })
                .collect();
            let r = rng.ext_vec(width);
            let eq = ChunkEq::at(&chunks, &r);
            let (low, carry) = ChunkEq::shift(&chunks, &r);
            for shift in [false, true] {
                let scatter = Scatter::new(top);
                scatter.add_words(&chunks, &words, usize::from(shift), |u| weights[u]);
                let array = scatter.into_vec();
                let rows: F192 = words
                    .iter()
                    .zip(&weights)
                    .map(|(&w, &weight)| {
                        let part = |e: &ChunkEq| (0..chunks.d()).map(|i| e.eval(&chunks, w, i)).collect::<Vec<_>>();
                        let phi = if shift {
                            shifted(&part(&eq), &part(&low), &part(&carry))
                        } else {
                            part(&eq).iter().product()
                        };
                        weight * phi
                    })
                    .sum();
                assert_eq!(mle(&array, &r), rows, "width {width}, shift {shift}");
            }
        }
    }
}

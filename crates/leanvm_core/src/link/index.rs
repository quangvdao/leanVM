//! Hotwire's index side: the rounds over the index `j` of cycles, memory cells and instructions, lowest bit
//! first, each of degree 2.
//!
//! The summand is a sum of terms `coef·A(j)·B(j)` and `coef·A(j)` over arrays. An array stores its first entries
//! and has a closed form past them ([`Tail`]): zero for a cycle table, which stops at `T + 1`, and the index's power
//! or `eq` for the tables that pair with it. A pair of rows of a term is summed only where both of its arrays can be
//! nonzero, so a table need only be stored as far as its partner reaches.

use super::pointer::{Chunks, NO_INDEX};
use primitives::field::{F64, F192, F192Unreduced};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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

/// What an array holds past its stored entries, at index `j` over the variables still unbound.
#[derive(Clone, Debug)]
pub enum Tail {
    Zero,
    /// `c·ratio^j`.
    Powers {
        c: F192,
        ratio: F64,
    },
    /// `c·eq(point, j)`, the point lowest variable first.
    Eq {
        c: F192,
        point: Vec<F192>,
    },
}

impl Tail {
    #[inline(never)]
    fn at(&self, j: usize) -> F192 {
        match self {
            Self::Zero => F192::ZERO,
            Self::Powers { c, ratio } => c.mul_base(ratio.pow(j as u128)),
            Self::Eq { c, point } => point
                .iter()
                .enumerate()
                .fold(*c, |e, (h, &z)| e * if j >> h & 1 == 1 { z } else { F192::ONE + z }),
        }
    }

    /// The tail once the lowest variable is bound at `x`.
    fn fold(&self, x: F192) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::Powers { c, ratio } => Self::Powers {
                c: *c * (F192::ONE + x.mul_base(F64::ONE + *ratio)),
                ratio: ratio.square(),
            },
            Self::Eq { c, point } => Self::Eq {
                c: *c * (F192::ONE + point[0] + x),
                point: point[1..].to_vec(),
            },
        }
    }
}

/// An array of the summand: its stored entries, then its tail.
pub struct Array {
    pub head: Vec<F192>,
    pub tail: Tail,
}

impl From<Vec<F192>> for Array {
    fn from(head: Vec<F192>) -> Self {
        Self { head, tail: Tail::Zero }
    }
}

impl Array {
    /// The indices, of `size`, where it can be nonzero.
    fn len(&self, size: usize) -> usize {
        match self.tail {
            Tail::Zero => self.head.len(),
            _ => size,
        }
    }

    fn get(&self, j: usize) -> F192 {
        self.head.get(j).copied().unwrap_or_else(|| self.tail.at(j))
    }
}

/// The pairs a round sums, given where each array can be nonzero.
fn task_pairs(lens: &[usize], terms: &[Term]) -> usize {
    (terms.iter())
        .map(|&t| match t {
            Term::Product { a, b, .. } => lens[a].min(lens[b]),
            Term::Single { a, .. } => lens[a],
        })
        .max()
        .unwrap_or(0)
        .div_ceil(2)
}

/// A task's entries of an array, from index `offset`.
#[derive(Clone, Copy)]
struct View<'a> {
    head: &'a [F192],
    offset: usize,
    /// The indices from `offset` where the array can be nonzero.
    len: usize,
    tail: &'a Tail,
}

impl View<'_> {
    #[inline(always)]
    fn at(&self, i: usize) -> F192 {
        match self.head.get(i) {
            Some(&v) => v,
            None => self.tail.at(self.offset + i),
        }
    }

    /// The pairs read from the stored entries alone.
    #[inline(always)]
    fn stored_pairs(&self) -> usize {
        match self.tail {
            Tail::Zero => self.head.len().div_ceil(2),
            _ => self.head.len() / 2,
        }
    }
}

#[inline(always)]
fn at(v: &[F192], i: usize) -> F192 {
    v.get(i).copied().unwrap_or(F192::ZERO)
}

/// Each term's sums over a task's pairs: `A·B` at `X = 0`, `1` and `∞` for a product (unreduced), `A` at `0` and `1`
/// for a single.
struct Sums {
    products: Vec<[F192Unreduced; 3]>,
    singles: Vec<[F192; 2]>,
}

impl Sums {
    fn new(terms: &[Term]) -> Self {
        let n_products = terms.iter().filter(|t| matches!(t, Term::Product { .. })).count();
        Self {
            products: vec![[F192Unreduced::ZERO; 3]; n_products],
            singles: vec![[F192::ZERO; 2]; terms.len() - n_products],
        }
    }

    /// Add the terms over the pairs `k` in `range` of `arrays`.
    #[inline(always)]
    fn add(&mut self, terms: &[Term], arrays: &[View], range: std::ops::Range<usize>) {
        let (mut p, mut q) = (0, 0);
        for &term in terms {
            match term {
                Term::Product { a, b, .. } => {
                    let (a, b) = (arrays[a], arrays[b]);
                    let end = range.end.min(a.len.div_ceil(2)).min(b.len.div_ceil(2)).max(range.start);
                    let stored = end.min(a.stored_pairs()).min(b.stored_pairs()).max(range.start);
                    let acc = &mut self.products[p];
                    let (ah, bh) = (a.head, b.head);
                    for k in range.start..stored {
                        let (a0, a1, b0, b1) = (ah[2 * k], at(ah, 2 * k + 1), bh[2 * k], at(bh, 2 * k + 1));
                        acc[0] ^= a0.mul_unreduced(b0);
                        acc[1] ^= a1.mul_unreduced(b1);
                        acc[2] ^= (a0 + a1).mul_unreduced(b0 + b1);
                    }
                    for k in stored..end {
                        let (a0, a1, b0, b1) = (a.at(2 * k), a.at(2 * k + 1), b.at(2 * k), b.at(2 * k + 1));
                        acc[0] ^= a0.mul_unreduced(b0);
                        acc[1] ^= a1.mul_unreduced(b1);
                        acc[2] ^= (a0 + a1).mul_unreduced(b0 + b1);
                    }
                    p += 1;
                }
                Term::Single { a, .. } => {
                    let end = range.end.min(arrays[a].len.div_ceil(2)).max(range.start);
                    let acc = &mut self.singles[q];
                    for k in range.start..end {
                        acc[0] += arrays[a].at(2 * k);
                        acc[1] += arrays[a].at(2 * k + 1);
                    }
                    q += 1;
                }
            }
        }
    }

    fn combine(mut self, other: Self) -> Self {
        for (x, y) in self.products.iter_mut().zip(other.products) {
            for (x, y) in x.iter_mut().zip(y) {
                *x ^= y;
            }
        }
        for (x, y) in self.singles.iter_mut().zip(other.singles) {
            x[0] += y[0];
            x[1] += y[1];
        }
        self
    }

    /// The round polynomial's coefficients.
    fn poly(self, terms: &[Term]) -> [F192; 3] {
        let mut poly = [F192::ZERO; 3];
        let (mut products, mut singles) = (self.products.into_iter(), self.singles.into_iter());
        for &term in terms {
            match term {
                Term::Product { coef, .. } => {
                    let [lo, hi, mixed] = products.next().expect("a product's sums").map(F192Unreduced::reduce);
                    poly[0] += coef * lo;
                    poly[1] += coef * (hi + lo + mixed);
                    poly[2] += coef * mixed;
                }
                Term::Single { coef, .. } => {
                    let [lo, hi] = singles.next().expect("a single's sums");
                    poly[0] += coef * lo;
                    poly[1] += coef * (lo + hi);
                }
            }
        }
        poly
    }
}

/// The first round's polynomial, over `size` indices.
fn first_round(arrays: &[Array], terms: &[Term], size: usize) -> [F192; 3] {
    let views: Vec<View> = (arrays.iter())
        .map(|a| View {
            head: &a.head,
            offset: 0,
            len: a.len(size),
            tail: &a.tail,
        })
        .collect();
    let lens: Vec<usize> = views.iter().map(|v| v.len).collect();
    parallel::fold_reduce(
        task_pairs(&lens, terms).div_ceil(TASK_PAIRS),
        || Sums::new(terms),
        |acc, task| acc.add(terms, &views, task * TASK_PAIRS..(task + 1) * TASK_PAIRS),
        Sums::combine,
    )
    .poly(terms)
}

/// Bind the lowest variable of every array, of `size` indices, at `x`, and sum the next round's polynomial, in one
/// pass: a task folds its range of every array and sums the terms over it while it is in cache.
fn fold_round(arrays: &[Array], terms: &[Term], x: F192, size: usize) -> (Vec<Array>, [F192; 3]) {
    let half = size / 2;
    let tails: Vec<Tail> = arrays.iter().map(|a| a.tail.fold(x)).collect();
    let lens: Vec<usize> = arrays.iter().map(|a| a.head.len().div_ceil(2)).collect();
    // SAFETY: the tasks below write every entry, each task its own range, before anything reads it.
    let mut out: Vec<Vec<F192>> = lens.iter().map(|&n| unsafe { primitives::uninit_vec(n) }).collect();
    let ptrs: Vec<parallel::SendPtr<F192>> = out.iter_mut().map(|o| parallel::SendPtr(o.as_mut_ptr())).collect();
    let summed: Vec<usize> = (lens.iter().zip(&tails))
        .map(|(&n, tail)| if matches!(tail, Tail::Zero) { n } else { half })
        .collect();
    let pairs = (lens.iter().map(|n| n.div_ceil(2)).max().unwrap_or(0)).max(task_pairs(&summed, terms));
    let sums = parallel::fold_reduce(
        pairs.div_ceil(TASK_PAIRS),
        || Sums::new(terms),
        |acc, task| {
            let range = task * TASK_PAIRS..(task + 1) * TASK_PAIRS;
            let offset = 2 * range.start;
            let views: Vec<View> = (arrays.iter().zip(&tails))
                .zip(ptrs.iter().zip(&lens))
                .map(|((a, tail), (ptr, &n))| {
                    let (lo, hi) = (offset.min(n), (2 * range.end).min(n));
                    // SAFETY: entries `lo..hi` of this array are this task's alone.
                    let dst = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(lo), hi - lo) };
                    for (j, o) in dst.iter_mut().enumerate() {
                        let i = 2 * (lo + j);
                        let (a0, a1) = (
                            a.head[i],
                            a.head.get(i + 1).copied().unwrap_or_else(|| a.tail.at(i + 1)),
                        );
                        *o = a0 + x * (a0 + a1);
                    }
                    let len = match tail {
                        Tail::Zero => dst.len(),
                        _ => half.saturating_sub(offset),
                    };
                    View {
                        head: dst,
                        offset,
                        len,
                        tail,
                    }
                })
                .collect();
            // The views start at the task's first pair.
            acc.add(terms, &views, 0..TASK_PAIRS);
        },
        Sums::combine,
    );
    let arrays = out
        .into_iter()
        .zip(tails)
        .map(|(head, tail)| Array { head, tail })
        .collect();
    (arrays, sums.poly(terms))
}

/// The index side over `n_vars` variables. `round` takes each round polynomial's coefficients and returns the
/// challenge.
pub fn prove(
    n_vars: usize,
    mut arrays: Vec<Array>,
    terms: &[Term],
    round: &mut dyn FnMut(&[F192]) -> F192,
) -> IndexEnd {
    assert!(
        arrays.iter().all(|a| a.head.len() <= 1 << n_vars),
        "an array past 2^{n_vars}"
    );
    let mut size = 1usize << n_vars;
    let mut point = Vec::with_capacity(n_vars);
    let mut claim = None;
    let mut poly = first_round(&arrays, terms, size);
    for _ in 0..n_vars {
        debug_assert!(claim.is_none_or(|c: F192| c == poly[1] + poly[2]));
        let x = round(&poly);
        claim = Some(primitives::multilinear::poly_eval(&poly, x));
        (arrays, poly) = fold_round(&arrays, terms, x, size);
        size /= 2;
        point.push(x);
    }
    let values: Vec<F192> = arrays.iter().map(|a| a.get(0)).collect();
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

/// An array the pointers of many columns add into at once: three `u64` limbs per entry, added by XOR, laid out
/// as [`F192`] is so the array becomes one without a copy. An index past its length sets `overflow` and adds
/// nothing, so a caller that sized it for the honest indices can redo it at full size.
pub struct Scatter {
    limbs: Vec<AtomicU64>,
    overflow: AtomicBool,
}

impl Scatter {
    pub fn new(len: usize) -> Self {
        // SAFETY: every limb is zeroed before it is read.
        let limbs = unsafe {
            primitives::filled_vec::<u64>(3 * len, |limbs| {
                parallel::chunks_mut(limbs, 3 * TASK_PAIRS * 16, |_, chunk| chunk.fill(0));
            })
        };
        let mut limbs = std::mem::ManuallyDrop::new(limbs);
        // SAFETY: `AtomicU64` has the size, alignment and bit validity of `u64`.
        let limbs = unsafe { Vec::from_raw_parts(limbs.as_mut_ptr().cast(), limbs.len(), limbs.capacity()) };
        Self {
            limbs,
            overflow: AtomicBool::new(false),
        }
    }

    /// Whether an index fell past the array.
    pub fn overflowed(&self) -> bool {
        self.overflow.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn add(&self, j: usize, value: F192) {
        let Some(cell) = self.limbs.get(3 * j..3 * j + 3) else {
            self.overflow.store(true, Ordering::Relaxed);
            return;
        };
        for (c, limb) in cell.iter().zip(value.to_limbs()) {
            if limb != 0 {
                c.fetch_xor(limb, Ordering::Relaxed);
            }
        }
    }

    /// Add `weight(u)` at the index of each row's word plus `shift`, dropping indices past `2^W`, with `index` the
    /// words' indices ([`Chunks::decode_all`]). A word whose chunks are not all one-hot adds at every index of it
    /// ([`Chunks::indices`]), which is what the row side's `Φ_s` of that word sums to.
    ///
    /// For words that are mostly distinct, such as cycles; [`Scatter::add_repeated_words`] is for words that repeat.
    pub fn add_words(
        &self,
        chunks: &Chunks,
        words: &[u64],
        index: &[u32],
        shift: usize,
        weight: impl Fn(usize) -> F192 + Sync,
    ) {
        self.add_words_with(chunks, words, index, shift, weight, || Direct(self));
    }

    /// [`Scatter::add_words`] for words that repeat, such as instructions and memory cells: each worker sums the
    /// adds to an index that rows hit again soon after, holding back up to `SLOTS` indices ([`Combine`]).
    pub fn add_repeated_words<const SLOTS: usize>(
        &self,
        chunks: &Chunks,
        words: &[u64],
        index: &[u32],
        shift: usize,
        weight: impl Fn(usize) -> F192 + Sync,
    ) {
        self.add_words_with(chunks, words, index, shift, weight, || Combine::<SLOTS>::new(self));
    }

    fn add_words_with<A: Adder>(
        &self,
        chunks: &Chunks,
        words: &[u64],
        index: &[u32],
        shift: usize,
        weight: impl Fn(usize) -> F192 + Sync,
        adder: impl Fn() -> A + Sync,
    ) {
        assert_eq!(index.len(), words.len(), "an index per word");
        let top = 1usize << chunks.width;
        parallel::for_each_chunk(words.len().div_ceil(TASK_PAIRS), |start, end| {
            let mut combine = adder();
            for task in start..end {
                for u in task * TASK_PAIRS..words.len().min((task + 1) * TASK_PAIRS) {
                    let w = weight(u);
                    if w.is_zero() {
                        continue;
                    }
                    let mut add = |j: usize| {
                        let j = j + shift;
                        if j < top {
                            combine.add(j, w);
                        }
                    };
                    match index[u] {
                        NO_INDEX => chunks.indices(words[u], &mut add),
                        j => add(j as usize),
                    }
                }
            }
            combine.flush();
        });
    }

    /// The entries, without the trailing zeros.
    pub fn into_vec(self) -> Vec<F192> {
        let mut limbs = std::mem::ManuallyDrop::new(self.limbs);
        let (len, cap) = (limbs.len() / 3, limbs.capacity());
        assert_eq!(cap % 3, 0);
        // SAFETY: `F192` is `repr(C)` over three `u64` limbs in this order, and `AtomicU64` is a `u64`; the
        // allocation holds `cap / 3` of them at `u64`'s alignment, which is `F192`'s.
        let mut out: Vec<F192> = unsafe { Vec::from_raw_parts(limbs.as_mut_ptr().cast(), len, cap / 3) };
        let last = out.iter().rposition(|v| !v.is_zero()).map_or(0, |j| j + 1);
        out.truncate(last);
        out
    }
}

/// Where a worker of [`Scatter::add_words_with`] sends its adds.
trait Adder {
    fn add(&mut self, j: usize, value: F192);
    fn flush(self);
}

/// Each add straight into the [`Scatter`].
struct Direct<'a>(&'a Scatter);

impl Adder for Direct<'_> {
    #[inline(always)]
    fn add(&mut self, j: usize, value: F192) {
        self.0.add(j, value);
    }

    fn flush(self) {}
}

/// The adds of one worker into a [`Scatter`], with the latest index at each of `N` slots (by its low bits) held
/// back and summed: an index that rows hit again soon after, such as a loop's instruction or a hot
/// memory cell, takes one atomic add per run of hits instead of one per row, and the workers no longer contend
/// for its cache line.
struct Combine<'a, const N: usize> {
    scatter: &'a Scatter,
    keys: Box<[u32; N]>,
    values: Box<[F192; N]>,
}

impl<'a, const N: usize> Combine<'a, N> {
    fn new(scatter: &'a Scatter) -> Self {
        Self {
            scatter,
            keys: vec![u32::MAX; N].into_boxed_slice().try_into().unwrap(),
            values: vec![F192::ZERO; N].into_boxed_slice().try_into().unwrap(),
        }
    }
}

impl<const N: usize> Adder for Combine<'_, N> {
    #[inline(always)]
    fn add(&mut self, j: usize, value: F192) {
        let slot = j % N;
        let key = u32::try_from(j).expect("an index below 2^32");
        if self.keys[slot] == key {
            self.values[slot] += value;
            return;
        }
        if self.keys[slot] != u32::MAX {
            self.scatter.add(self.keys[slot] as usize, self.values[slot]);
        }
        self.keys[slot] = key;
        self.values[slot] = value;
    }

    /// Add what is held back.
    fn flush(self) {
        for (&key, &value) in self.keys.iter().zip(self.values.iter()) {
            if key != u32::MAX {
                self.scatter.add(key as usize, value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::pointer::{ChunkEq, shifted};
    use super::*;
    use primitives::field::F64;
    use primitives::multilinear::{eq_table, poly_eval};
    use primitives::test_rng::Rng;

    fn mle(a: &[F192], point: &[F192]) -> F192 {
        eq_table(point).iter().zip(a).map(|(&e, &v)| e * v).sum()
    }

    #[test]
    fn a_scatter_flags_an_index_past_its_length() {
        let chunks = Chunks::new(9);
        let words: Vec<u64> = [3, 7, 2].iter().map(|&s| chunks.encode(s)).collect();
        let fits = Scatter::new(8);
        let index = chunks.decode_all(&words);
        fits.add_words(&chunks, &words, &index, 0, |_| F192::ONE);
        assert!(!fits.overflowed());
        let short = Scatter::new(7);
        short.add_words(&chunks, &words, &index, 0, |_| F192::ONE);
        assert!(short.overflowed());
        assert_eq!(short.into_vec(), [0, 0, 1, 1].map(|b| F192::new(b, 0, 0)));
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
        let end = prove(
            n,
            arrays.iter().cloned().map(Array::from).collect(),
            &terms,
            &mut |poly| {
                assert_eq!(poly.len(), 3);
                assert_eq!(claim, poly[1] + poly[2], "a round's sum");
                let x = rng.ext();
                claim = poly_eval(poly, x);
                x
            },
        );
        assert_eq!(claim, end.claim);
        for (a, &v) in arrays.iter().zip(&end.values) {
            assert_eq!(mle(a, &end.point), v);
        }
        assert_eq!(end.claim, value(&|a| end.values[a]));
    }

    /// An array with a tail proves as the array it stands for, stored in full.
    #[test]
    fn a_tail_proves_as_its_stored_array() {
        let mut rng = Rng::new(0x7a1);
        let n = 9;
        let (c, ratio, zeta) = (rng.ext(), F64::G.pow(3), rng.ext_vec(n));
        let tails = [
            Tail::Zero,
            Tail::Powers { c, ratio },
            Tail::Eq { c, point: zeta.clone() },
        ];
        let heads = [(0, 300), (1, 64), (2, 32), (1, 1), (0, 51), (2, 2), (0, 1)].map(|(t, len)| {
            let tail = &tails[t];
            let head: Vec<F192> = (0..len)
                .map(|j| if j % 7 == 3 { rng.ext() } else { tail.at(j) })
                .collect();
            Array {
                head,
                tail: tail.clone(),
            }
        });
        let full: Vec<Vec<F192>> = (heads.iter())
            .map(|a| match a.tail {
                Tail::Zero => a.head.clone(),
                _ => (0..1 << n).map(|j| a.get(j)).collect(),
            })
            .collect();
        let terms = [(0, 1), (4, 2), (6, 3), (4, 5), (0, 0)]
            .map(|(a, b)| Term::Product { coef: rng.ext(), a, b })
            .into_iter()
            .chain([
                Term::Single { coef: rng.ext(), a: 1 },
                Term::Single { coef: rng.ext(), a: 5 },
            ])
            .collect::<Vec<_>>();
        let run = |arrays: Vec<Array>| {
            let (mut rng, mut polys) = (Rng::new(0x7a2), Vec::new());
            let end = prove(n, arrays, &terms, &mut |poly| {
                polys.push(poly.to_vec());
                rng.ext()
            });
            (polys, end.values)
        };
        let stored = run(full.into_iter().map(Array::from).collect());
        assert_eq!(run(heads.into_iter().collect()), stored);
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
            let index = chunks.decode_all(&words);
            for shift in [false, true] {
                let scatter = Scatter::new(top);
                scatter.add_words(&chunks, &words, &index, usize::from(shift), |u| weights[u]);
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

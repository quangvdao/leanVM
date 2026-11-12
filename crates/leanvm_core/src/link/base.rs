//! The row side's base terms: what a table's rows add at the pointer-column index `q = 0`, as forms of degree at most
//! two in the table's columns.
//!
//! A table carries one block per weight and gate: `ω_F(u)·form(u)` for the ungated part of its links, then
//! `ω_F(u)·gate(u)·form(u)` for each gate, and for `JUMP` `eq(τ_F, u)·form(u)` with its two identities. A block
//! is degree 4 in a row variable at most: its weight, its gate and the form's products.
//!
//! A form's products mostly share a column (`fp` times an operand, a layout flag times a lane), so each form is kept
//! as `L(u) + Σ_a col_a(u)·L_a(u)` for linear forms `L`, `L_a`. The prover tabulates every `L` once on the explicit
//! rows from the committed `K` columns, then folds those and the factor columns instead of every column a form reads.

use crate::leaf::BusForm;
use primitives::field::{F64, F192, F192Unreduced};

/// What a block's form is multiplied by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weight {
    /// The table's `ω_F`, which vanishes on its padding rows.
    Omega,
    /// `eq(τ_F, u)` on every row, padding rows included.
    Eq,
}

/// One block: `weight(u)·gate(u)·form(u)`.
#[derive(Clone, Debug)]
pub struct Block {
    pub weight: Weight,
    pub gate: Option<usize>,
    pub form: BusForm,
}

/// A linear form over a table's columns, by local index.
#[derive(Clone, Debug)]
struct Linear {
    constant: F192,
    terms: Vec<(usize, F192)>,
}

impl Linear {
    #[inline(always)]
    fn at(&self, column: impl Fn(usize) -> F64) -> F192 {
        let sum = self
            .terms
            .iter()
            .fold(F192Unreduced::ZERO, |acc, &(c, w)| acc ^ w.mul_base_unreduced(column(c)));
        self.constant + sum.reduce()
    }
}

/// A block over the tabulated values: its weight, its gate's slot, the slot of its linear part and a
/// `(factor, linear form)` pair of slots per product.
#[derive(Clone, Debug)]
struct Factored {
    weight: Weight,
    gate: Option<usize>,
    linear: usize,
    products: Vec<(usize, usize)>,
}

/// The products of a form, grouped by a shared column: `(column, Σ w·other)` per group, greedily by the column the
/// most remaining products read.
fn factor(prods: &[(usize, usize, F192)]) -> Vec<(usize, Linear)> {
    let mut left: Vec<(usize, usize, F192)> = prods.iter().copied().filter(|p| !p.2.is_zero()).collect();
    let mut out = Vec::new();
    while !left.is_empty() {
        let mut counts = std::collections::BTreeMap::<usize, usize>::new();
        for &(a, b, _) in &left {
            *counts.entry(a).or_default() += 1;
            if b != a {
                *counts.entry(b).or_default() += 1;
            }
        }
        let column = counts.iter().max_by_key(|&(c, n)| (*n, std::cmp::Reverse(*c))).map(|(&c, _)| c).unwrap();
        let mut terms: Vec<(usize, F192)> = Vec::new();
        left.retain(|&(a, b, w)| {
            if a != column && b != column {
                return true;
            }
            let other = if a == column { b } else { a };
            match terms.iter_mut().find(|(c, _)| *c == other) {
                Some((_, v)) => *v += w,
                None => terms.push((other, w)),
            }
            false
        });
        out.push((column, Linear { constant: F192::ZERO, terms }));
    }
    out
}

/// The base terms of one table.
pub struct Base<'a> {
    blocks: Vec<Factored>,
    /// The factor and gate columns, by local index, in slot order.
    factors: Vec<usize>,
    linears: Vec<Linear>,
    /// Round 1's columns, every row of each, by local index.
    columns: Vec<&'a [F64]>,
    /// After round 1: the slots' values on the explicit rows, `stride` a row.
    folded: Option<Vec<F192>>,
    /// The slots' values on the padrow.
    tail: Vec<F192>,
    stride: usize,
    /// `eq(τ_F, u)` on every row, when a block is weighted by it.
    eq: Option<Vec<F192>>,
}

/// Rows per task of a fold.
const FOLD_ROWS: usize = 512;

/// The most slots a table's base terms use.
const MAX_SLOTS: usize = 128;

#[inline(always)]
fn at(v: &[F192], i: usize) -> F192 {
    v.get(i).copied().unwrap_or(F192::ZERO)
}

/// `p ← p + w(X)·g(X)·f(X)` for linear `w`, `g` given at 0 and 1, `w` one when absent, and the quadratic `f` in
/// coefficients.
#[inline(always)]
fn add_block(acc: &mut [F192], w: Option<(F192, F192)>, gate: Option<(F192, F192)>, f: [F192; 3]) {
    let mut p = [f[0], f[1], f[2], F192::ZERO, F192::ZERO];
    let mut len = 3;
    for (a0, a1) in gate.into_iter().chain(w) {
        let b = a0 + a1;
        p[len] = p[len - 1] * b;
        for j in (1..len).rev() {
            p[j] = (p[j].mul_unreduced(a0) ^ p[j - 1].mul_unreduced(b)).reduce();
        }
        p[0] *= a0;
        len += 1;
    }
    for (o, v) in acc.iter_mut().zip(&p[..len]) {
        *o += *v;
    }
}

/// A block's gate times its form at a row of slot values, unweighted.
#[inline(always)]
fn form_at(b: &Factored, row: &[F192]) -> F192 {
    let products = b
        .products
        .iter()
        .fold(F192Unreduced::ZERO, |acc, &(f, l)| acc ^ row[f].mul_unreduced(row[l]));
    let form = row[b.linear] + products.reduce();
    b.gate.map_or(form, |g| row[g] * form)
}

impl<'a> Base<'a> {
    /// `columns` holds every local column of the table, every row; `padrow` the table's padding row; `eq` the table
    /// of `eq(τ_F, ·)` when a block is weighted by it.
    pub fn new(blocks: Vec<Block>, columns: Vec<&'a [F64]>, padrow: &[F64], eq: Option<Vec<F192>>) -> Self {
        let mut factors: Vec<usize> = Vec::new();
        let mut linears: Vec<Linear> = Vec::new();
        let slot = |c: usize, factors: &mut Vec<usize>| {
            factors.iter().position(|&f| f == c).unwrap_or_else(|| {
                factors.push(c);
                factors.len() - 1
            })
        };
        let mut shaped = Vec::with_capacity(blocks.len());
        for b in &blocks {
            let gate = b.gate.map(|g| slot(g, &mut factors));
            linears.push(Linear {
                constant: b.form.constant,
                terms: b
                    .form
                    .coeffs
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.is_zero())
                    .map(|(i, &c)| (i, c))
                    .collect(),
            });
            let linear = linears.len() - 1;
            let products: Vec<(usize, usize)> = factor(&b.form.prods)
                .into_iter()
                .map(|(column, form)| {
                    linears.push(form);
                    (slot(column, &mut factors), linears.len() - 1)
                })
                .collect();
            shaped.push((b.weight, gate, linear, products));
        }
        let n = factors.len();
        let blocks: Vec<Factored> = shaped
            .into_iter()
            .map(|(weight, gate, linear, products)| Factored {
                weight,
                gate,
                linear: n + linear,
                products: products.into_iter().map(|(f, l)| (f, n + l)).collect(),
            })
            .collect();
        let stride = n + linears.len();
        assert!(stride <= MAX_SLOTS, "a table's base terms use at most {MAX_SLOTS} slots");
        let mut base = Self {
            blocks,
            factors,
            linears,
            columns,
            folded: None,
            tail: vec![F192::ZERO; stride],
            stride,
            eq,
        };
        let mut tail = vec![F192::ZERO; stride];
        base.slots(|c| padrow[c], &mut tail);
        base.tail = tail;
        assert!(
            base.blocks
                .iter()
                .filter(|b| b.weight == Weight::Eq)
                .all(|b| form_at(b, &base.tail).is_zero()),
            "a block weighted on every row vanishes on the padrow"
        );
        assert_eq!(base.eq.is_some(), base.blocks.iter().any(|b| b.weight == Weight::Eq));
        base
    }

    /// The slots' values at a row of `K` values, `column` reading a local column there.
    #[inline(always)]
    fn slots(&self, column: impl Fn(usize) -> F64 + Copy, out: &mut [F192]) {
        let n = self.factors.len();
        for (o, &c) in out.iter_mut().zip(&self.factors) {
            *o = F192::from(column(c));
        }
        for (o, l) in out[n..].iter_mut().zip(&self.linears) {
            *o = l.at(column);
        }
    }

    /// Call `f` on the slots' values at row `u`.
    #[inline(always)]
    fn with_row<R>(&self, u: usize, f: impl FnOnce(&[F192]) -> R) -> R {
        let s = self.stride;
        match &self.folded {
            Some(rows) => f(rows.get(u * s..(u + 1) * s).unwrap_or(&self.tail)),
            None if self.columns.first().is_some_and(|c| u < c.len()) => {
                let mut row = [F192::ZERO; MAX_SLOTS];
                self.slots(|c| self.columns[c][u], &mut row[..s]);
                f(&row[..s])
            }
            None => f(&self.tail),
        }
    }

    /// `Σ_u ω(u)·gate(u)·form(u)` over the `ω` blocks, `omega` on the executed rows: what the blocks sum to
    /// before any row is bound.
    pub fn omega_sum(&self, omega: &[F192]) -> F192 {
        parallel::fold_reduce(
            omega.len().div_ceil(FOLD_ROWS),
            || F192::ZERO,
            |acc, task| {
                for (u, &w) in omega.iter().enumerate().skip(task * FOLD_ROWS).take(FOLD_ROWS) {
                    if w.is_zero() {
                        continue;
                    }
                    let forms: F192 = self.with_row(u, |row| {
                        self.blocks
                            .iter()
                            .filter(|b| b.weight == Weight::Omega)
                            .map(|b| form_at(b, row))
                            .sum()
                    });
                    *acc += w * forms;
                }
            },
            |a, b| a + b,
        )
    }

    /// Whether a block reaches the padding rows, so the rounds must sum every explicit pair.
    pub fn on_padding(&self) -> bool {
        self.eq.is_some()
    }

    /// Add the pair `(2k, 2k + 1)`'s round polynomial: the `ω` blocks without their weight to `omega` when
    /// `with_omega`, the others to `acc`.
    #[inline(always)]
    pub fn pair(&self, k: usize, with_omega: bool, omega: &mut [F192], acc: &mut [F192]) {
        let eq = self.eq.as_ref().map(|e| (at(e, 2 * k), at(e, 2 * k + 1)));
        self.with_row(2 * k, |lo| {
            self.with_row(2 * k + 1, |hi| {
                for block in &self.blocks {
                    let (out, w) = match block.weight {
                        Weight::Omega if with_omega => (&mut *omega, None),
                        Weight::Omega => continue,
                        Weight::Eq => (&mut *acc, eq),
                    };
                    if w.is_some_and(|w| w.0.is_zero() && w.1.is_zero()) {
                        continue;
                    }
                    let (mut c0, mut c1, mut c2) = (F192Unreduced::ZERO, F192Unreduced::ZERO, F192Unreduced::ZERO);
                    for &(f, l) in &block.products {
                        let (a0, a1, b0, b1) = (lo[f], hi[f], lo[l], hi[l]);
                        c0 ^= a0.mul_unreduced(b0);
                        c1 ^= a1.mul_unreduced(b1);
                        c2 ^= (a0 + a1).mul_unreduced(b0 + b1);
                    }
                    let f0 = lo[block.linear] + c0.reduce();
                    let f1 = hi[block.linear] + c1.reduce();
                    let f2 = c2.reduce();
                    add_block(out, w, block.gate.map(|g| (lo[g], hi[g])), [f0, f0 + f1 + f2, f2]);
                }
            })
        });
    }

    /// Bind the lowest row variable at `x`: `explicit` rows after.
    pub fn fold(&mut self, explicit: usize, x: F192) {
        let s = self.stride;
        // SAFETY: every row writes each of its `s` values.
        let folded = unsafe {
            primitives::filled_vec(explicit * s, |out| {
                if s > 0 {
                    parallel::chunks_mut(out, s * FOLD_ROWS, |task, chunk| {
                        for (j, row) in chunk.chunks_mut(s).enumerate() {
                            let k = task * FOLD_ROWS + j;
                            self.with_row(2 * k, |lo| {
                                self.with_row(2 * k + 1, |hi| {
                                    for ((o, &a), &b) in row.iter_mut().zip(lo).zip(hi) {
                                        *o = if a == b { a } else { a + x * (a + b) };
                                    }
                                })
                            });
                        }
                    });
                }
            })
        };
        self.folded = Some(folded);
        if let Some(eq) = &mut self.eq {
            *eq = (0..eq.len().div_ceil(2))
                .map(|k| {
                    let (a, b) = (eq[2 * k], at(eq, 2 * k + 1));
                    a + x * (a + b)
                })
                .collect();
        }
    }

    /// The base terms at the table's end, its rows bound: `omega` is `ω_F` there.
    pub fn end(&self, omega: F192) -> F192 {
        let eq = self.eq.as_ref().map(|e| e[0]);
        self.with_row(0, |row| {
            self.blocks
                .iter()
                .map(|b| {
                    let w = match b.weight {
                        Weight::Omega => omega,
                        Weight::Eq => eq.expect("an eq table"),
                    };
                    w * form_at(b, row)
                })
                .sum()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::multilinear::{eq_table, poly_eval};
    use primitives::test_rng::Rng;

    /// The rounds of a table's base terms sum to its blocks over the rows, and end on the blocks at the bound point.
    #[test]
    fn base_matches_the_naive_sum_and_end() {
        let mut rng = Rng::new(0xba5e);
        let (m, rows, n_cols) = (4usize, 11usize, 5usize);
        let padrow: Vec<F64> = vec![F64::ZERO; n_cols];
        let columns: Vec<Vec<F64>> = (0..n_cols)
            .map(|_| {
                (0..1 << m)
                    .map(|u| if u < 13 { F64(rng.next_u64()) } else { F64::ZERO })
                    .collect()
            })
            .collect();
        let form = |rng: &mut Rng| {
            let mut f = BusForm::new(n_cols);
            f.constant = rng.ext();
            f.coeffs[1] = rng.ext();
            f.coeffs[3] = rng.ext();
            f.prods.push((0, 2, rng.ext()));
            f.prods.push((4, 4, rng.ext()));
            f.prods.push((1, 2, rng.ext()));
            f.prods.push((2, 2, rng.ext()));
            f
        };
        // The eq block must vanish on the zero padrow: no constant.
        let mut eq_form = form(&mut rng);
        eq_form.constant = F192::ZERO;
        let blocks = vec![
            Block {
                weight: Weight::Omega,
                gate: None,
                form: form(&mut rng),
            },
            Block {
                weight: Weight::Omega,
                gate: Some(3),
                form: form(&mut rng),
            },
            Block {
                weight: Weight::Eq,
                gate: None,
                form: eq_form,
            },
        ];
        let tau = rng.ext_vec(m);
        let eq = eq_table(&tau);
        let omega: Vec<F192> = (0..1 << m)
            .map(|u| if u < rows { rng.ext() } else { F192::ZERO })
            .collect();
        let scales = [F192::ONE; 3];
        let row_value = |scales: &[F192], omega_u: F192, eq_u: F192, col: &dyn Fn(usize) -> F192| -> F192 {
            blocks
                .iter()
                .zip(scales)
                .map(|(b, &scale)| {
                    let w = if b.weight == Weight::Omega { omega_u } else { eq_u };
                    let mut f = b.form.constant;
                    for (i, &c) in b.form.coeffs.iter().enumerate() {
                        f += c * col(i);
                    }
                    for &(x, y, c) in &b.form.prods {
                        f += c * col(x) * col(y);
                    }
                    scale * w * b.gate.map_or(F192::ONE, col) * f
                })
                .sum()
        };
        let mut claim: F192 = (0..1 << m)
            .map(|u| row_value(&scales, omega[u], eq[u], &|c| F192::from(columns[c][u])))
            .sum();
        let refs: Vec<&[F64]> = columns.iter().map(|c| &c[..]).collect();
        let mut base = Base::new(blocks.clone(), refs, &padrow, Some(eq.clone()));
        let omega_only = [F192::ONE, F192::ONE, F192::ZERO];
        let omega_sum: F192 = (0..1 << m)
            .map(|u| row_value(&omega_only, omega[u], eq[u], &|c| F192::from(columns[c][u])))
            .sum();
        assert_eq!(base.omega_sum(&omega[..rows]), omega_sum);
        let mut om = omega.clone();
        let mut point = Vec::new();
        let mut explicit = 1 << m;
        for _ in 0..m {
            let mut poly = [F192::ZERO; 5];
            for k in 0..explicit / 2 {
                let mut under = [F192::ZERO; 4];
                base.pair(k, true, &mut under, &mut poly);
                let (w0, w1) = (om[2 * k], om[2 * k] + om[2 * k + 1]);
                for j in 0..5 {
                    poly[j] += at(&under, j) * w0 + if j > 0 { under[j - 1] * w1 } else { F192::ZERO };
                }
            }
            assert_eq!(claim, poly[1..].iter().copied().sum::<F192>());
            let x = rng.ext();
            claim = poly_eval(&poly, x);
            om = (0..om.len() / 2)
                .map(|k| om[2 * k] + x * (om[2 * k] + om[2 * k + 1]))
                .collect();
            explicit /= 2;
            base.fold(explicit, x);
            point.push(x);
        }
        assert_eq!(claim, base.end(om[0]));
        let eqs = eq_table(&point);
        let mle = |f: &dyn Fn(usize) -> F192| (0..1 << m).map(|u| eqs[u] * f(u)).sum::<F192>();
        let cols: Vec<F192> = (0..n_cols).map(|c| mle(&|u| F192::from(columns[c][u]))).collect();
        let expected = row_value(&scales, mle(&|u| omega[u]), eq_eval_at(&tau, &point), &|c| cols[c]);
        assert_eq!(claim, expected);
    }

    fn eq_eval_at(a: &[F192], b: &[F192]) -> F192 {
        primitives::multilinear::eq_eval(a, b)
    }
}

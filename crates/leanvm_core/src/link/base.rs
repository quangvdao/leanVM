//! The row side's base terms: what a table's rows add at the pointer-column index `q = 0`, as forms of degree at most
//! two in the table's columns.
//!
//! A table carries one block per weight and gate: `ω_F(u)·form(u)` for the ungated part of its links, then
//! `ω_F(u)·gate(u)·form(u)` for each gate, and for `JUMP` `eq(τ_F, u)·form(u)` with its two identities. A block
//! is degree 4 in a row variable at most: its weight, its gate and the form's products.
//!
//! Round 1 reads the committed `K` columns directly. Later rounds read the columns the blocks name, folded into `E`,
//! on the explicit rows, with one tail row for the padrow after them.

use crate::leaf::BusForm;
use primitives::field::{F64, F192};

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

/// A form over the columns a table's blocks read, sparse and renumbered to them.
#[derive(Clone, Debug)]
struct Sparse {
    weight: Weight,
    gate: Option<usize>,
    constant: F192,
    linear: Vec<(usize, F192)>,
    products: Vec<(usize, usize, F192)>,
}

impl Sparse {
    /// The form at one row, from its columns' values.
    fn at(&self, value: impl Fn(usize) -> F192) -> F192 {
        let linear = self.linear.iter().map(|&(c, w)| w * value(c)).sum::<F192>();
        let products = self
            .products
            .iter()
            .map(|&(a, b, w)| w * value(a) * value(b))
            .sum::<F192>();
        self.constant + linear + products
    }
}

/// The base terms of one table.
pub struct Base<'a> {
    blocks: Vec<Sparse>,
    /// The local columns the blocks read, in the numbering the sparse forms use.
    used: Vec<usize>,
    /// Round 1's columns, every row of each, by local index.
    columns: Vec<&'a [F64]>,
    /// `eq(τ_F, u)` on every row, when a block is weighted by it.
    eq: Option<Vec<F192>>,
    /// After round 1: the used columns on the explicit rows, one row of `used.len()` after another, and the padrow.
    folded: Option<Vec<F192>>,
    tail: Vec<F192>,
}

/// Rows per task of a fold.
const FOLD_ROWS: usize = 512;

#[inline(always)]
fn at(v: &[F192], i: usize) -> F192 {
    v.get(i).copied().unwrap_or(F192::ZERO)
}

/// `p ← p + w(X)·g(X)·f(X)` for linear `w`, `g` given at 0 and 1 and the quadratic `f` in coefficients.
#[inline(always)]
fn add_block(acc: &mut [F192], w: (F192, F192), gate: Option<(F192, F192)>, f: [F192; 3]) {
    let mut p = [f[0], f[1], f[2], F192::ZERO, F192::ZERO];
    let mut len = 3;
    for (a0, a1) in gate.into_iter().chain([w]) {
        let b = a0 + a1;
        p[len] = p[len - 1] * b;
        for j in (1..len).rev() {
            p[j] = p[j] * a0 + p[j - 1] * b;
        }
        p[0] *= a0;
        len += 1;
    }
    for (o, v) in acc.iter_mut().zip(&p[..len]) {
        *o += *v;
    }
}

impl<'a> Base<'a> {
    /// `columns` holds every local column of the table, every row; `padrow` the table's padding row; `eq` the table
    /// of `eq(τ_F, ·)` when a block is weighted by it.
    pub fn new(blocks: Vec<Block>, columns: Vec<&'a [F64]>, padrow: &[F64], eq: Option<Vec<F192>>) -> Self {
        let mut used: Vec<usize> = blocks
            .iter()
            .flat_map(|b| {
                let linear = b
                    .form
                    .coeffs
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.is_zero())
                    .map(|(i, _)| i);
                let products = b.form.prods.iter().flat_map(|&(x, y, _)| [x, y]);
                linear.chain(products).chain(b.gate).collect::<Vec<_>>()
            })
            .collect();
        used.sort_unstable();
        used.dedup();
        let index = |c: usize| used.binary_search(&c).expect("a used column");
        let blocks: Vec<Sparse> = blocks
            .into_iter()
            .map(|b| Sparse {
                weight: b.weight,
                gate: b.gate.map(index),
                constant: b.form.constant,
                linear: b
                    .form
                    .coeffs
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.is_zero())
                    .map(|(i, &c)| (index(i), c))
                    .collect(),
                products: b.form.prods.iter().map(|&(x, y, c)| (index(x), index(y), c)).collect(),
            })
            .collect();
        let tail: Vec<F192> = used.iter().map(|&c| F192::from(padrow[c])).collect();
        assert!(
            blocks
                .iter()
                .filter(|b| b.weight == Weight::Eq)
                .all(|b| (b.gate.map_or(F192::ONE, |g| tail[g]) * b.at(|c| tail[c])).is_zero()),
            "a block weighted on every row vanishes on the padrow"
        );
        assert_eq!(eq.is_some(), blocks.iter().any(|b| b.weight == Weight::Eq));
        Self {
            blocks,
            used,
            columns,
            eq,
            folded: None,
            tail,
        }
    }

    /// `Σ_u ω(u)·gate(u)·form(u)` over the `ω` blocks, `omega` on the executed rows: what the blocks sum to before
    /// any row is bound.
    pub fn omega_sum(&self, omega: &[F192]) -> F192 {
        parallel::fold_reduce(
            omega.len().div_ceil(FOLD_ROWS),
            || F192::ZERO,
            |acc, task| {
                for (u, &w) in omega.iter().enumerate().skip(task * FOLD_ROWS).take(FOLD_ROWS) {
                    if w.is_zero() {
                        continue;
                    }
                    for b in self.blocks.iter().filter(|b| b.weight == Weight::Omega) {
                        let gate = b.gate.map_or(F192::ONE, |g| self.value(g, u));
                        *acc += w * gate * b.at(|c| self.value(c, u));
                    }
                }
            },
            |a, b| a + b,
        )
    }

    /// Whether a block reaches the padding rows, so the rounds must sum every explicit pair.
    pub fn on_padding(&self) -> bool {
        self.eq.is_some()
    }

    #[inline(always)]
    fn value(&self, c: usize, u: usize) -> F192 {
        match &self.folded {
            None => self.columns[self.used[c]]
                .get(u)
                .map_or(self.tail[c], |&v| F192::from(v)),
            Some(rows) => {
                let n = self.used.len();
                rows.get(u * n + c).copied().unwrap_or(self.tail[c])
            }
        }
    }

    /// Add the pair `(2k, 2k + 1)`'s round polynomial, `omega` the table's `ω_F` there.
    #[inline(always)]
    pub fn pair(&self, k: usize, omega: (F192, F192), acc: &mut [F192]) {
        let (u0, u1) = (2 * k, 2 * k + 1);
        let pair = |c: usize| (self.value(c, u0), self.value(c, u1));
        let eq = self.eq.as_ref().map(|e| (at(e, u0), at(e, u1)));
        for block in &self.blocks {
            let w = match block.weight {
                Weight::Omega => omega,
                Weight::Eq => eq.expect("an eq table"),
            };
            if w.0.is_zero() && w.1.is_zero() {
                continue;
            }
            let (f0, f1) = (block.at(|c| pair(c).0), block.at(|c| pair(c).1));
            let f2: F192 = block
                .products
                .iter()
                .map(|&(a, b, w)| {
                    let (a, b) = (pair(a), pair(b));
                    w * (a.0 + a.1) * (b.0 + b.1)
                })
                .sum();
            add_block(acc, w, block.gate.map(pair), [f0, f0 + f1 + f2, f2]);
        }
    }

    /// Bind the lowest row variable at `x`: `explicit` rows after.
    pub fn fold(&mut self, explicit: usize, x: F192) {
        let n = self.used.len();
        let mut out = vec![F192::ZERO; explicit * n];
        if n > 0 {
            parallel::chunks_mut(&mut out, n * FOLD_ROWS, |task, chunk| {
                for (j, row) in chunk.chunks_mut(n).enumerate() {
                    let k = task * FOLD_ROWS + j;
                    for (c, o) in row.iter_mut().enumerate() {
                        let (a, b) = (self.value(c, 2 * k), self.value(c, 2 * k + 1));
                        *o = if a == b { a } else { a + x * (a + b) };
                    }
                }
            });
        }
        if let Some(eq) = &mut self.eq {
            *eq = (0..eq.len().div_ceil(2))
                .map(|k| {
                    let (a, b) = (eq[2 * k], at(eq, 2 * k + 1));
                    a + x * (a + b)
                })
                .collect();
        }
        self.folded = Some(out);
    }

    /// The base terms at the table's end, its rows bound: `omega` is `ω_F` there.
    pub fn end(&self, omega: F192) -> F192 {
        let eq = self.eq.as_ref().map(|e| e[0]);
        self.blocks
            .iter()
            .map(|b| {
                let w = match b.weight {
                    Weight::Omega => omega,
                    Weight::Eq => eq.expect("an eq table"),
                };
                let gate = b.gate.map_or(F192::ONE, |g| self.value(g, 0));
                w * gate * b.at(|c| self.value(c, 0))
            })
            .sum()
    }

    /// The used columns at the table's end, by local index.
    pub fn values(&self) -> impl Iterator<Item = (usize, F192)> + '_ {
        self.used
            .iter()
            .enumerate()
            .map(|(c, &local)| (local, self.value(c, 0)))
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
        let row_value = |u: usize, omega_u: F192, eq_u: F192, col: &dyn Fn(usize) -> F192| -> F192 {
            let _ = u;
            blocks
                .iter()
                .map(|b| {
                    let w = if b.weight == Weight::Omega { omega_u } else { eq_u };
                    let mut f = b.form.constant;
                    for (i, &c) in b.form.coeffs.iter().enumerate() {
                        f += c * col(i);
                    }
                    for &(x, y, c) in &b.form.prods {
                        f += c * col(x) * col(y);
                    }
                    w * b.gate.map_or(F192::ONE, col) * f
                })
                .sum()
        };
        let mut claim: F192 = (0..1 << m)
            .map(|u| row_value(u, omega[u], eq[u], &|c| F192::from(columns[c][u])))
            .sum();
        let refs: Vec<&[F64]> = columns.iter().map(|c| &c[..]).collect();
        let mut base = Base::new(blocks.clone(), refs, &padrow, Some(eq.clone()));
        let mut om = omega.clone();
        let mut point = Vec::new();
        let mut explicit = 1 << m;
        for _ in 0..m {
            let mut poly = [F192::ZERO; 5];
            for k in 0..explicit / 2 {
                base.pair(k, (om[2 * k], om[2 * k + 1]), &mut poly);
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
        let expected = row_value(0, mle(&|u| omega[u]), eq_eval_at(&tau, &point), &|c| cols[c]);
        assert_eq!(claim, expected);
    }

    fn eq_eval_at(a: &[F192], b: &[F192]) -> F192 {
        primitives::multilinear::eq_eval(a, b)
    }
}

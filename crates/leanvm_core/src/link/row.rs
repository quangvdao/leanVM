//! Hotlink's row side: one sum-check over every table's rows, then over the pointer columns.
//!
//! The rows are bound lowest variable first, so a round pairs consecutive rows of a table, which are consecutive
//! executions of its opcode. A table of `2^m` rows takes part from round 1 and is finished after `m` rounds; from
//! then on its value is multiplied by the all-zeros corner factor `1 + x_h` of each further row variable, once.
//! The pointer-column variables `q` come last.
//!
//! A pointer column `q` of table `F` contributes, at row `u`,
//!
//! `c_q(u)·Φ_0(u; r) + c⁺_q(u)·Φ_1(u; r) + ω_F(u)·Σ_i ε_{q,i}·(D_i(u; ρ_i)·D_i(u; 1 + ρ_i) + π_{w_i}(ρ_i))`,
//!
//! the column's coefficient `c_q = ω_F·(constant + aux) + act_F·(constant + aux)` times the pointer at the index
//! point `r`, the cycle pointer's `c⁺_q = α_+·ω_F` times the pointer shifted by one (carry matrices, from Binary
//! Jolt's precompile link), plus the mirror certificate (the mirror test of Emile Wambsgans), which vanishes on every executed row whose chunks
//! are one-hot. Two rows whose chunk agrees make that chunk's factor constant in the round that pairs them, and
//! leave its certificate term zero: the prover skips both (equal-value skipping), and a chunk that agrees across a
//! whole folded group keeps a tag saying so.
//!
//! Rows from `explicit` on hold the padrow, whose pointer words are zero, so every folded group of them holds the
//! padrow's values: the prover keeps them as one tail rather than folding them. The coefficients vanish past the
//! executed rows, so only the pairs up to there are summed, unless a table's base terms reach its padding rows.
//!
//! A table's base terms (see [`super::base`]) belong to the pointer-column index `q = 0`: the row rounds add them
//! like any summand, and the column rounds carry their total `B` as `eq(0, q)·B`.

use super::base::Base;
use super::pointer::{ChunkEq, Chunks, mirror};
use primitives::field::F192;

/// Room for the coefficients of one round: degree `max(d + 2, 4) ≤ 10`.
pub const MAX_COEFFS: usize = 12;

/// No one-hot value: a tag for a chunk that is not one value across its group.
const MIXED: u8 = u8::MAX;

/// Pairs per task of a round.
const TASK_PAIRS: usize = 256;

/// A column's coefficient on one of its table's two row weights: `constant`, plus a scale times a linear form of the
/// table's columns when `aux` names one.
#[derive(Clone, Copy, Debug)]
pub struct Coef {
    pub constant: F192,
    pub aux: Option<(usize, F192)>,
}

impl Coef {
    pub const ZERO: Self = Self {
        constant: F192::ZERO,
        aux: None,
    };

    pub fn constant(constant: F192) -> Self {
        Self { constant, aux: None }
    }
}

/// One pointer column, as the row side reads it.
pub struct RowColumn<'a> {
    /// The words, one per row of the table.
    pub words: &'a [u64],
    /// Coefficient on `ω_F`.
    pub omega: Coef,
    /// Coefficient on `act_F`.
    pub act: Coef,
    /// The coefficient of `ω_F·Φ_1`, the shifted pointer, for the cycle pointer.
    pub plus: Option<F192>,
    /// The certificate's coefficient per chunk.
    pub cert: Vec<F192>,
}

/// One table, as the row side reads it.
pub struct RowTable<'a> {
    pub log_rows: usize,
    /// Executed rows, the support of `ω_F` and `act_F`.
    pub rows: usize,
    /// Rows from here on hold the padrow.
    pub explicit: usize,
    /// `ω_F` on the executed rows.
    pub omega: Vec<F192>,
    /// The linear forms the coefficients name, each on the explicit rows, and their values on the padrow.
    pub aux: Vec<Vec<F192>>,
    pub aux_tail: Vec<F192>,
    pub columns: Vec<RowColumn<'a>>,
    /// The table's base terms, at `q = 0`.
    pub base: Option<Base<'a>>,
}

/// The functionals the row side evaluates pointers with: `D_i` at the index point `r`, the shift's `A_i` and
/// `B_i` there, and `D_i` at the certificate point `ρ` and at `1 + ρ`.
pub struct RowPoints {
    pub eq: ChunkEq,
    pub cert: ChunkEq,
    pub cert_bar: ChunkEq,
    pub low: ChunkEq,
    pub carry: ChunkEq,
    /// `π_{w_i}(ρ)` per chunk.
    pub mirrors: Vec<F192>,
}

impl RowPoints {
    pub fn new(chunks: &Chunks, r: &[F192], rho: &[F192]) -> Self {
        let (low, carry) = ChunkEq::shift(chunks, r);
        Self {
            low,
            carry,
            eq: ChunkEq::at(chunks, r),
            cert: ChunkEq::certificate(chunks, rho, false),
            cert_bar: ChunkEq::certificate(chunks, rho, true),
            mirrors: chunks.widths.iter().map(|&w| mirror(&rho[..w])).collect(),
        }
    }
}

/// A column at the end of the row rounds: its 64 bit-slices at `χ_F`, and the values they give,
/// `D_i(χ_F; r)`, `D_i(χ_F; ρ)`, `D_i(χ_F; 1 + ρ)`, `A_i(χ_F)` and `B_i(χ_F)`, `5d` in all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnEnd {
    pub slices: Vec<F192>,
    pub values: Vec<F192>,
}

/// `Σ_u eq(χ, u)·bit_b(word_u)` for each bit `b`.
pub fn bit_slices(words: &[u64], chi: &[F192]) -> Vec<F192> {
    let eq = primitives::multilinear::eq_table(chi);
    let n = words.len().min(eq.len());
    let slices = parallel::fold_reduce(
        n.div_ceil(TASK_PAIRS * 16),
        || [F192::ZERO; 64],
        |acc, task| {
            for u in task * TASK_PAIRS * 16..n.min((task + 1) * TASK_PAIRS * 16) {
                let mut w = words[u];
                while w != 0 {
                    acc[w.trailing_zeros() as usize] += eq[u];
                    w &= w - 1;
                }
            }
        },
        |mut a, b| {
            add_into(&mut a, &b);
            a
        },
    );
    slices.to_vec()
}

impl ColumnEnd {
    pub fn from_slices(chunks: &Chunks, points: &RowPoints, slices: Vec<F192>) -> Self {
        let tables = [&points.eq, &points.cert, &points.cert_bar, &points.low, &points.carry];
        let values = tables
            .iter()
            .flat_map(|e| {
                (0..chunks.d())
                    .map(|i| e.from_slices(chunks, &slices, i))
                    .collect::<Vec<_>>()
            })
            .collect();
        Self { slices, values }
    }
}

/// A table's values at the end of the row rounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableEnd {
    pub omega: F192,
    pub act: F192,
    pub aux: Vec<F192>,
    /// `Π (1 + χ_h)` over the row variables past the table's.
    pub corner: F192,
    pub columns: Vec<ColumnEnd>,
    /// The base terms at `χ_F`, without the corner factor.
    pub base: F192,
}

/// What the row side ends at.
pub struct RowEnd {
    /// `χ`, one coordinate per row round.
    pub point: Vec<F192>,
    /// `q*`, one coordinate per pointer-column round.
    pub column_point: Vec<F192>,
    pub tables: Vec<TableEnd>,
    /// The claim the last round leaves.
    pub claim: F192,
}

// ---- small polynomials in coefficient form -------------------------------------

type Poly = [F192; MAX_COEFFS];

/// `p ← p·(a + b·X)` for `p` of `len` coefficients; the new length.
#[inline(always)]
fn mul_linear(p: &mut Poly, len: usize, a: F192, b: F192) -> usize {
    p[len] = p[len - 1] * b;
    for j in (1..len).rev() {
        p[j] = p[j] * a + p[j - 1] * b;
    }
    p[0] *= a;
    len + 1
}

/// `p ← p·f` for the linear `f` with `f(0) = pair.0`, `f(1) = pair.1`.
#[inline(always)]
fn mul_pair(p: &mut Poly, len: usize, pair: (F192, F192)) -> usize {
    mul_linear(p, len, pair.0, pair.0 + pair.1)
}

/// The product of the linear functions through two pairs.
#[inline(always)]
fn pair_product(a: (F192, F192), b: (F192, F192)) -> [F192; 3] {
    let (c0, c2) = (a.0 * b.0, (a.0 + a.1) * (b.0 + b.1));
    [c0, a.1 * b.1 + c0 + c2, c2]
}

#[inline(always)]
fn add_into(acc: &mut [F192], p: &[F192]) {
    for (o, v) in acc.iter_mut().zip(p) {
        *o += *v;
    }
}

#[inline(always)]
fn at(v: &[F192], i: usize) -> F192 {
    v.get(i).copied().unwrap_or(F192::ZERO)
}

#[inline(always)]
fn is_zero_pair(p: (F192, F192)) -> bool {
    p.0.is_zero() && p.1.is_zero()
}

/// The functionals of a chunk, in a column's values: `D` at `r`, at `ρ`, at `1 + ρ`, then the shift's `A` and `B`.
const AT_R: usize = 0;
const AT_RHO: usize = 1;
const AT_RHO_BAR: usize = 2;
const LOW: usize = 3;
const CARRY: usize = 4;

/// `c(X)·Φ_0(X) + c⁺(X)·Φ_1(X)` for one column at one pair, added to `acc`. `pair(t, i)` is functional `t` of
/// chunk `i` at the two rows; with `skip`, a chunk whose `same(i)` holds is a constant factor `first(i)`.
#[inline(always)]
fn pointer_pair(
    acc: &mut Poly,
    d: usize,
    coef: [F192; 3],
    plus: Option<[F192; 2]>,
    skip: bool,
    same: impl Fn(usize) -> bool,
    first: impl Fn(usize) -> F192,
    pair: impl Fn(usize, usize) -> (F192, F192),
) {
    let mut p = [F192::ZERO; MAX_COEFFS];
    p[..3].copy_from_slice(&coef);
    let mut len = 3;
    let mut start = 0;
    if let Some(plus) = plus {
        let carry = pair(CARRY, 0);
        if !is_zero_pair(carry) {
            // Chunk 0 is all ones in a row: the carry runs on, so the shift takes the full matrix product.
            let mut x1 = [F192::ZERO; MAX_COEFFS];
            x1[..2].copy_from_slice(&plus);
            let mut y = x1;
            let mut x1_len = mul_pair(&mut x1, 2, pair(LOW, 0));
            let mut y_len = mul_pair(&mut y, 2, carry);
            len = mul_pair(&mut p, len, pair(AT_R, 0));
            for i in 1..d {
                let e = pair(AT_R, i);
                len = mul_pair(&mut p, len, e);
                let mut ya = y;
                let ya_len = mul_pair(&mut ya, y_len, pair(LOW, i));
                x1_len = mul_pair(&mut x1, x1_len, e);
                add_into(&mut x1, &ya[..ya_len]);
                x1_len = x1_len.max(ya_len);
                y_len = mul_pair(&mut y, y_len, pair(CARRY, i));
            }
            add_into(acc, &p[..len]);
            add_into(acc, &x1[..x1_len]);
            return;
        }
        // No carry out of chunk 0: `Φ_1 = A_0·Π_{i ≥ 1} D_i`, which shares every factor past chunk 0 with `Φ_0`.
        len = mul_pair(&mut p, len, pair(AT_R, 0));
        let mut x1 = [F192::ZERO; MAX_COEFFS];
        x1[..2].copy_from_slice(&plus);
        let x1_len = mul_pair(&mut x1, 2, pair(LOW, 0));
        add_into(&mut p, &x1[..x1_len]);
        start = 1;
    }
    let mut scalar = F192::ONE;
    let mut linear = [(F192::ZERO, F192::ZERO); 8];
    let mut n_linear = 0;
    for i in start..d {
        if skip && same(i) {
            scalar *= first(i);
        } else {
            linear[n_linear] = pair(AT_R, i);
            n_linear += 1;
        }
    }
    if scalar != F192::ONE {
        for v in &mut p[..len] {
            *v *= scalar;
        }
    }
    for &e in &linear[..n_linear] {
        len = mul_pair(&mut p, len, e);
    }
    add_into(acc, &p[..len]);
}

/// `ω(X)·Σ_i ε_i·(D_i(ρ)·D_i(1 + ρ) + π_i)(X)` for one column at one pair, added to `acc`. With `skip`, a chunk
/// that is one one-hot value at both rows (`clean`) adds zero, one one-hot at both rows (`one_hot`) adds
/// `b·b̄·(X + X²)`, and one whose values agree (`same`) a constant.
#[inline(always)]
fn cert_pair(
    acc: &mut Poly,
    d: usize,
    omega: (F192, F192),
    cert: &[F192],
    mirrors: &[F192],
    skip: bool,
    clean: impl Fn(usize) -> bool,
    one_hot: impl Fn(usize) -> bool,
    same: impl Fn(usize) -> bool,
    pair: impl Fn(usize, usize) -> (F192, F192),
) {
    let mut m = [F192::ZERO; 3];
    let mut any = false;
    for i in 0..d {
        if skip && clean(i) {
            continue;
        }
        any = true;
        let (x, y, eps) = (pair(AT_RHO, i), pair(AT_RHO_BAR, i), cert[i]);
        if skip && one_hot(i) {
            let t = eps * (x.0 + x.1) * (y.0 + y.1);
            m[1] += t;
            m[2] += t;
        } else if skip && same(i) {
            m[0] += eps * (x.0 * y.0 + mirrors[i]);
        } else {
            let q = pair_product(x, y);
            m[0] += eps * (q[0] + mirrors[i]);
            m[1] += eps * q[1];
            m[2] += eps * q[2];
        }
    }
    if any {
        add_weighted(acc, omega, m);
    }
}

/// `ω(X)·m(X)` added to `acc`, `m` of three coefficients.
#[inline(always)]
fn add_weighted(acc: &mut Poly, omega: (F192, F192), m: [F192; 3]) {
    let (w0, w1) = (omega.0, omega.0 + omega.1);
    acc[0] += w0 * m[0];
    acc[1] += w0 * m[1] + w1 * m[0];
    acc[2] += w0 * m[2] + w1 * m[1];
    acc[3] += w1 * m[2];
}

/// The coefficient of a column at a pair of rows: `ω(X)·(constant + aux(X)) + act(X)·(constant + aux(X))`.
#[inline(always)]
fn coef_poly(omega: (F192, F192), act: (F192, F192), aux: &[(F192, F192)], c_omega: Coef, c_act: Coef) -> [F192; 3] {
    let weight = |c: Coef| match c.aux {
        None => (c.constant, c.constant),
        Some((a, s)) => (c.constant + s * aux[a].0, c.constant + s * aux[a].1),
    };
    let mut out = [F192::ZERO; 3];
    for (w, c) in [(omega, c_omega), (act, c_act)] {
        if is_zero_pair(w) || (c.constant.is_zero() && c.aux.is_none()) {
            continue;
        }
        let p = pair_product(w, weight(c));
        for (o, v) in out.iter_mut().zip(p) {
            *o += v;
        }
    }
    out
}

/// `c⁺(X) = plus·ω(X)`.
#[inline(always)]
fn plus_poly(plus: Option<F192>, omega: (F192, F192)) -> Option<[F192; 2]> {
    plus.map(|a| [a * omega.0, a * (omega.0 + omega.1)])
}

// ---- per-table state ----------------------------------------------------------------

/// Values in rows of `stride`, explicit on a prefix and one tail row after it.
struct Strided {
    data: Vec<F192>,
    tail: Vec<F192>,
    stride: usize,
}

impl Strided {
    #[inline(always)]
    fn row(&self, k: usize) -> &[F192] {
        let s = self.stride;
        self.data.get(k * s..(k + 1) * s).unwrap_or(&self.tail)
    }

    /// Bind the lowest variable at `x`: `len` explicit rows after.
    fn fold(&self, len: usize, x: F192) -> Vec<F192> {
        let s = self.stride;
        let mut out = vec![F192::ZERO; len * s];
        if s == 0 {
            return out;
        }
        parallel::chunks_mut(&mut out, s * TASK_PAIRS, |task, chunk| {
            for (j, row) in chunk.chunks_mut(s).enumerate() {
                let k = task * TASK_PAIRS + j;
                let (lo, hi) = (self.row(2 * k), self.row(2 * k + 1));
                for ((o, &a), &b) in row.iter_mut().zip(lo).zip(hi) {
                    *o = if a == b { a } else { a + x * (a + b) };
                }
            }
        });
        out
    }
}

#[inline(always)]
fn tag(tags: &[u8], d: usize, k: usize, i: usize) -> u8 {
    tags.get(k * d + i).copied().unwrap_or(MIXED)
}

/// A pointer column after round 1: per row its chunks' functionals (`3d`, or `5d` for a shifted column), and per
/// chunk a tag, the one-hot value the whole folded group holds or [`MIXED`].
struct Column<'a> {
    words: &'a [u64],
    values: Strided,
    tags: Vec<u8>,
    omega: Coef,
    act: Coef,
    plus: Option<F192>,
    cert: Vec<F192>,
}

struct State<'a> {
    log_rows: usize,
    /// Support of `ω` and `act`.
    support: usize,
    /// Explicit rows.
    explicit: usize,
    omega: Vec<F192>,
    act: Vec<F192>,
    aux: Strided,
    /// Round 1 reads the words; later rounds the folded columns.
    words: Vec<RowColumn<'a>>,
    columns: Vec<Column<'a>>,
    /// The table's row variables.
    m: usize,
    /// The table's value once its rows are bound, and the corner factor since.
    done: Option<(F192, F192)>,
    base: Option<Base<'a>>,
}

impl<'a> State<'a> {
    fn new(table: RowTable<'a>, chunks: &Chunks, points: &RowPoints) -> Self {
        assert!(table.rows <= table.explicit && table.explicit <= 1 << table.log_rows);
        assert_eq!(table.omega.len(), table.rows);
        let aux = Strided {
            data: (0..table.explicit)
                .flat_map(|u| table.aux.iter().map(move |a| a[u]))
                .collect(),
            tail: table.aux_tail,
            stride: table.aux.len(),
        };
        let mut state = Self {
            log_rows: table.log_rows,
            m: table.log_rows,
            support: table.rows,
            explicit: table.explicit,
            omega: table.omega,
            act: vec![F192::ONE; table.rows],
            aux,
            words: table.columns,
            columns: Vec::new(),
            done: None,
            base: table.base,
        };
        if state.log_rows == 0 {
            // One row: its values are the table's end from the start.
            let words = std::mem::take(&mut state.words);
            state.columns = words
                .into_iter()
                .map(|column| materialize(column, 1, F192::ZERO, chunks, points, false))
                .collect();
            if let Some(base) = &mut state.base {
                base.fold(1, F192::ZERO);
            }
            state.finish(chunks, points);
        }
        state
    }

    /// Rows bound: the table's value at its end.
    fn finish(&mut self, chunks: &Chunks, points: &RowPoints) {
        let d = chunks.d();
        let end = self.end(F192::ONE);
        let value: F192 = self
            .columns
            .iter()
            .zip(&end.columns)
            .map(|(column, values)| {
                let (coef, plus, certs) = column_coefs(column, &end, d);
                final_value(chunks, points, coef, plus, &values.values, &certs)
            })
            .sum();
        self.done = Some((value + end.base, F192::ONE));
    }

    #[inline(always)]
    fn aux_pair(&self, k: usize) -> [(F192, F192); 4] {
        let mut out = [(F192::ZERO, F192::ZERO); 4];
        let (lo, hi) = (self.aux.row(2 * k), self.aux.row(2 * k + 1));
        for (o, (&a, &b)) in out.iter_mut().zip(lo.iter().zip(hi)) {
            *o = (a, b);
        }
        out
    }

    /// The round polynomial over this table's pairs.
    fn round(&self, chunks: &Chunks, points: &RowPoints, skip: bool) -> Poly {
        let pairs = if self.base.as_ref().is_some_and(Base::on_padding) {
            self.explicit.div_ceil(2)
        } else {
            self.support.div_ceil(2)
        };
        parallel::fold_reduce(
            pairs.div_ceil(TASK_PAIRS),
            || [F192::ZERO; MAX_COEFFS],
            |acc, task| {
                for k in task * TASK_PAIRS..pairs.min((task + 1) * TASK_PAIRS) {
                    let omega = (at(&self.omega, 2 * k), at(&self.omega, 2 * k + 1));
                    if let Some(base) = &self.base {
                        base.pair(k, omega, acc);
                    }
                    if is_zero_pair(omega) && 2 * k >= self.act.len() {
                        continue;
                    }
                    if self.columns.is_empty() {
                        self.pair_words(k, chunks, points, skip, acc);
                    } else {
                        self.pair_folded(k, chunks, points, skip, acc);
                    }
                }
            },
            |mut a, b| {
                add_into(&mut a, &b);
                a
            },
        )
    }

    /// Round 1: the pair `(2k, 2k + 1)` straight from the words.
    fn pair_words(&self, k: usize, chunks: &Chunks, points: &RowPoints, skip: bool, acc: &mut Poly) {
        let omega = (at(&self.omega, 2 * k), at(&self.omega, 2 * k + 1));
        let act = (at(&self.act, 2 * k), at(&self.act, 2 * k + 1));
        let aux = self.aux_pair(k);
        let d = chunks.d();
        let tables = [&points.eq, &points.cert, &points.cert_bar, &points.low, &points.carry];
        for column in &self.words {
            let word = |u: usize| column.words.get(u).copied().unwrap_or(0);
            let (u0, u1) = (word(2 * k), word(2 * k + 1));
            let same = |i: usize| chunks.bits(u0, i) == chunks.bits(u1, i);
            let one_hot = |i: usize| chunks.bits(u0, i).is_power_of_two() && chunks.bits(u1, i).is_power_of_two();
            let pair = |t: usize, i: usize| (tables[t].eval(chunks, u0, i), tables[t].eval(chunks, u1, i));
            let coef = coef_poly(omega, act, &aux, column.omega, column.act);
            pointer_pair(
                acc,
                d,
                coef,
                plus_poly(column.plus, omega),
                skip,
                same,
                |i| points.eq.eval(chunks, u0, i),
                pair,
            );
            cert_pair(
                acc,
                d,
                omega,
                &column.cert,
                &points.mirrors,
                skip,
                |i| same(i) && chunks.bits(u0, i).is_power_of_two(),
                one_hot,
                same,
                pair,
            );
        }
    }

    /// Later rounds: the pair `(2k, 2k + 1)` of the folded columns.
    fn pair_folded(&self, k: usize, chunks: &Chunks, points: &RowPoints, skip: bool, acc: &mut Poly) {
        let omega = (at(&self.omega, 2 * k), at(&self.omega, 2 * k + 1));
        let act = (at(&self.act, 2 * k), at(&self.act, 2 * k + 1));
        let aux = self.aux_pair(k);
        let d = chunks.d();
        for column in &self.columns {
            let (lo, hi) = (column.values.row(2 * k), column.values.row(2 * k + 1));
            let pair = |t: usize, i: usize| (lo[t * d + i], hi[t * d + i]);
            let coef = coef_poly(omega, act, &aux, column.omega, column.act);
            pointer_pair(
                acc,
                d,
                coef,
                plus_poly(column.plus, omega),
                skip,
                |i| lo[i] == hi[i],
                |i| lo[i],
                pair,
            );
            let clean = |i: usize| {
                let t = tag(&column.tags, d, 2 * k, i);
                t != MIXED && t == tag(&column.tags, d, 2 * k + 1, i)
            };
            let same = |i: usize| lo[d + i] == hi[d + i] && lo[2 * d + i] == hi[2 * d + i];
            cert_pair(
                acc,
                d,
                omega,
                &column.cert,
                &points.mirrors,
                skip,
                clean,
                |_| false,
                same,
                pair,
            );
        }
    }

    /// Bind the lowest row variable at `x`.
    fn fold(&mut self, x: F192, chunks: &Chunks, points: &RowPoints, skip: bool) {
        let support = self.support.div_ceil(2);
        let explicit = self.explicit.div_ceil(2);
        let fold_vec = |v: &[F192], len: usize| -> Vec<F192> {
            (0..len)
                .map(|k| {
                    let (a, b) = (at(v, 2 * k), at(v, 2 * k + 1));
                    a + x * (a + b)
                })
                .collect()
        };
        self.omega = fold_vec(&self.omega, support);
        self.act = fold_vec(&self.act, support);
        self.aux.data = self.aux.fold(explicit, x);
        if let Some(base) = &mut self.base {
            base.fold(explicit, x);
        }
        let d = chunks.d();
        if self.columns.is_empty() {
            let words = std::mem::take(&mut self.words);
            self.columns = words
                .into_iter()
                .map(|column| materialize(column, explicit, x, chunks, points, skip))
                .collect();
        } else {
            for column in &mut self.columns {
                column.values.data = column.values.fold(explicit, x);
                column.tags = (0..explicit * d)
                    .map(|j| {
                        let (k, i) = (j / d, j % d);
                        let t = tag(&column.tags, d, 2 * k, i);
                        if t == tag(&column.tags, d, 2 * k + 1, i) {
                            t
                        } else {
                            MIXED
                        }
                    })
                    .collect();
            }
        }
        self.support = support;
        self.explicit = explicit;
        self.log_rows -= 1;
        if self.log_rows == 0 {
            self.finish(chunks, points);
        }
    }

    fn end(&self, corner: F192) -> TableEnd {
        TableEnd {
            omega: at(&self.omega, 0),
            act: at(&self.act, 0),
            aux: self.aux.row(0).to_vec(),
            corner,
            columns: self
                .columns
                .iter()
                .map(|c| ColumnEnd {
                    slices: Vec::new(),
                    values: c.values.row(0).to_vec(),
                })
                .collect(),
            base: self.base.as_ref().map_or(F192::ZERO, |b| b.end(at(&self.omega, 0))),
        }
    }
}

/// A column's coefficient, shifted coefficient and certificate weights at a table's end, times its corner factor.
fn column_coefs(column: &Column, end: &TableEnd, d: usize) -> (F192, F192, Vec<F192>) {
    let weight = |c: Coef| c.constant + c.aux.map_or(F192::ZERO, |(a, s)| s * end.aux[a]);
    let coef = end.corner * (end.omega * weight(column.omega) + end.act * weight(column.act));
    let plus = end.corner * end.omega * column.plus.unwrap_or(F192::ZERO);
    let certs = (0..d).map(|i| end.corner * end.omega * column.cert[i]).collect();
    (coef, plus, certs)
}

/// A column after round 1, folded at `x` straight from its words.
fn materialize<'a>(
    column: RowColumn<'a>,
    len: usize,
    x: F192,
    chunks: &Chunks,
    points: &RowPoints,
    skip: bool,
) -> Column<'a> {
    let d = chunks.d();
    let n_tables = if column.plus.is_some() { 5 } else { 3 };
    let s = n_tables * d;
    let word = |u: usize| column.words.get(u).copied().unwrap_or(0);
    let mut values = vec![F192::ZERO; len * s];
    let tables = [&points.eq, &points.cert, &points.cert_bar, &points.low, &points.carry];
    parallel::chunks_mut(&mut values, s * TASK_PAIRS, |task, chunk| {
        for (j, row) in chunk.chunks_mut(s).enumerate() {
            let k = task * TASK_PAIRS + j;
            let (u0, u1) = (word(2 * k), word(2 * k + 1));
            for i in 0..d {
                let same = chunks.bits(u0, i) == chunks.bits(u1, i);
                for (t, table) in tables[..n_tables].iter().enumerate() {
                    let e0 = table.eval(chunks, u0, i);
                    row[t * d + i] = if skip && same {
                        e0
                    } else {
                        e0 + x * (e0 + table.eval(chunks, u1, i))
                    };
                }
            }
        }
    });
    let tags = (0..len * d)
        .map(|j| {
            let (k, i) = (j / d, j % d);
            let (b0, b1) = (chunks.bits(word(2 * k), i), chunks.bits(word(2 * k + 1), i));
            if b0 == b1 && b0.is_power_of_two() {
                b0.trailing_zeros() as u8
            } else {
                MIXED
            }
        })
        .collect();
    Column {
        words: column.words,
        values: Strided {
            data: values,
            tail: vec![F192::ZERO; s],
            stride: s,
        },
        tags,
        omega: column.omega,
        act: column.act,
        plus: column.plus,
        cert: column.cert,
    }
}

/// The row side over `tables`. `round` takes each round polynomial's coefficients and returns the challenge;
/// `claim` is what the rows sum to, checked in debug builds. `skip` turns equal-value skipping on, which every
/// proof does; off, it is only there to measure.
pub fn prove(
    chunks: &Chunks,
    points: &RowPoints,
    tables: Vec<RowTable>,
    skip: bool,
    round: &mut dyn FnMut(&[F192]) -> F192,
) -> RowEnd {
    let d = chunks.d();
    let n_coeffs = row_coeffs(d);
    let n_rounds = tables.iter().map(|t| t.log_rows).max().unwrap_or(0);
    let mut states: Vec<State> = tables.into_iter().map(|t| State::new(t, chunks, points)).collect();
    let mut point = Vec::with_capacity(n_rounds);
    let mut claim = None;
    for _ in 0..n_rounds {
        let mut poly = [F192::ZERO; MAX_COEFFS];
        for state in &states {
            match state.done {
                None => add_into(&mut poly, &state.round(chunks, points, skip)),
                Some((value, corner)) => {
                    poly[0] += value * corner;
                    poly[1] += value * corner;
                }
            }
        }
        debug_assert!(claim.is_none_or(|c: F192| c == poly[1..].iter().copied().sum::<F192>()));
        let x = round(&poly[..n_coeffs]);
        claim = Some(primitives::multilinear::poly_eval(&poly[..n_coeffs], x));
        for state in &mut states {
            match &mut state.done {
                None => state.fold(x, chunks, points, skip),
                Some((_, corner)) => *corner *= F192::ONE + x,
            }
        }
        point.push(x);
    }

    // Each column's values at its table's end come from its bit-slices there, as the verifier's do, so the
    // column rounds see `A_i` and `B_i` of every column, shifted or not.
    let ends: Vec<TableEnd> = states
        .iter()
        .map(|s| {
            let mut end = s.end(s.done.map_or(F192::ONE, |(_, corner)| corner));
            for (column, column_end) in s.columns.iter().zip(&mut end.columns) {
                let full = ColumnEnd::from_slices(chunks, points, bit_slices(column.words, &point[..s.m]));
                debug_assert_eq!(column_end.values, full.values[..column_end.values.len()]);
                *column_end = full;
            }
            end
        })
        .collect();
    let mut columns = Vec::new();
    for (state, end) in states.iter().zip(&ends) {
        for (column, values) in state.columns.iter().zip(&end.columns) {
            let (coef, plus, certs) = column_coefs(column, end, d);
            columns.push((coef, plus, values.values.clone(), certs));
        }
    }
    let base = ends.iter().map(|e| e.corner * e.base).sum();
    let (column_point, claim) = prove_columns(chunks, points, columns, base, claim, round);
    RowEnd {
        point,
        column_point,
        tables: ends,
        claim,
    }
}

/// Coefficients of a row round: degree `d + 2` from a gated coefficient times a pointer, 4 from the base terms.
pub fn row_coeffs(d: usize) -> usize {
    (d + 2).max(4) + 1
}

/// Coefficients of a column round: degree `d + 1` from a pointer, 3 from a certificate.
pub fn column_coeffs(d: usize) -> usize {
    (d + 1).max(3) + 1
}

/// The rounds over the pointer-column index `q`, lowest bit first, on vectors padded with zero columns. Each
/// column is its coefficient, shifted coefficient, `5d` chunk values and `d` certificate weights; `base` is the
/// tables' base terms, at `q = 0`.
fn prove_columns(
    chunks: &Chunks,
    points: &RowPoints,
    mut columns: Vec<(F192, F192, Vec<F192>, Vec<F192>)>,
    mut base: F192,
    mut claim: Option<F192>,
    round: &mut dyn FnMut(&[F192]) -> F192,
) -> (Vec<F192>, F192) {
    let d = chunks.d();
    let n_coeffs = column_coeffs(d);
    let n_vars = columns.len().next_power_of_two().trailing_zeros() as usize;
    columns.resize(
        1 << n_vars,
        (F192::ZERO, F192::ZERO, vec![F192::ZERO; 5 * d], vec![F192::ZERO; d]),
    );
    let mut point = Vec::with_capacity(n_vars);
    for _ in 0..n_vars {
        let mut poly = [F192::ZERO; MAX_COEFFS];
        poly[0] += base;
        poly[1] += base;
        for k in 0..columns.len() / 2 {
            let (lo, hi) = (&columns[2 * k], &columns[2 * k + 1]);
            let pair = |t: usize, i: usize| (lo.2[t * d + i], hi.2[t * d + i]);
            let shifted = !(lo.1.is_zero() && hi.1.is_zero());
            pointer_pair(
                &mut poly,
                d,
                [lo.0, lo.0 + hi.0, F192::ZERO],
                shifted.then_some([lo.1, lo.1 + hi.1]),
                false,
                |_| false,
                |_| F192::ZERO,
                pair,
            );
            for i in 0..d {
                let m = pair_product(pair(AT_RHO, i), pair(AT_RHO_BAR, i));
                add_weighted(&mut poly, (lo.3[i], hi.3[i]), [m[0] + points.mirrors[i], m[1], m[2]]);
            }
        }
        debug_assert!(claim.is_none_or(|c: F192| c == poly[1..].iter().copied().sum::<F192>()));
        let x = round(&poly[..n_coeffs]);
        claim = Some(primitives::multilinear::poly_eval(&poly[..n_coeffs], x));
        base *= F192::ONE + x;
        let fold = |a: F192, b: F192| a + x * (a + b);
        let fold_vec = |a: &[F192], b: &[F192]| a.iter().zip(b).map(|(&a, &b)| fold(a, b)).collect::<Vec<_>>();
        columns = (0..columns.len() / 2)
            .map(|k| {
                let (lo, hi) = (&columns[2 * k], &columns[2 * k + 1]);
                (
                    fold(lo.0, hi.0),
                    fold(lo.1, hi.1),
                    fold_vec(&lo.2, &hi.2),
                    fold_vec(&lo.3, &hi.3),
                )
            })
            .collect();
        point.push(x);
    }
    let (coef, plus, values, certs) = &columns[0];
    let last = final_value(chunks, points, *coef, *plus, values, certs) + base;
    debug_assert!(claim.is_none_or(|c| c == last));
    (point, last)
}

/// The summand at one bound point: `c·Φ_0 + c⁺·Φ_1 + Σ_i e_i·(D_i(ρ)·D_i(1 + ρ) + π_i)`, from a column's `3d` or
/// `5d` chunk values.
pub fn final_value(
    chunks: &Chunks,
    points: &RowPoints,
    coef: F192,
    plus: F192,
    values: &[F192],
    certs: &[F192],
) -> F192 {
    let d = chunks.d();
    let pointer: F192 = values[..d].iter().product();
    let shifted = if plus.is_zero() {
        F192::ZERO
    } else {
        plus * super::pointer::shifted(&values[..d], &values[3 * d..4 * d], &values[4 * d..5 * d])
    };
    let cert: F192 = (0..d)
        .map(|i| certs[i] * (values[d + i] * values[2 * d + i] + points.mirrors[i]))
        .sum();
    coef * pointer + shifted + cert
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::base::{Block, Weight};
    use crate::leaf::BusForm;
    use primitives::field::F64;
    use primitives::multilinear::{eq_eval, eq_table, poly_eval};
    use primitives::test_rng::Rng;

    struct Raw {
        log_rows: usize,
        rows: usize,
        explicit: usize,
        omega: Vec<F192>,
        aux: Vec<Vec<F192>>,
        aux_tail: Vec<F192>,
        words: Vec<Vec<u64>>,
        coefs: Vec<(Coef, Coef, Option<F192>, Vec<F192>)>,
        /// Committed columns for the base terms, the padrow past `explicit`.
        k_columns: Vec<Vec<F64>>,
        padrow: Vec<F64>,
        blocks: Vec<Block>,
        tau: Vec<F192>,
    }

    /// Base blocks over three committed columns: one `ω` block, one gated by column 2, one on every row that
    /// vanishes where column 0 does (on the padrow).
    fn random_blocks(rng: &mut Rng) -> Vec<Block> {
        let form = |rng: &mut Rng, constant: bool| {
            let mut f = BusForm::new(3);
            f.constant = if constant { rng.ext() } else { F192::ZERO };
            f.coeffs[0] = rng.ext();
            if constant {
                f.coeffs[1] = rng.ext();
            }
            f.prods.push((0, 1, rng.ext()));
            f
        };
        vec![
            Block { weight: Weight::Omega, gate: None, form: form(rng, true) },
            Block { weight: Weight::Omega, gate: Some(2), form: form(rng, true) },
            Block { weight: Weight::Eq, gate: None, form: form(rng, false) },
        ]
    }

    /// A random table: three columns (the first shifted) whose neighbouring rows share most chunks, some words
    /// not one-hot, padding rows zero past `explicit`.
    fn random_table(rng: &mut Rng, chunks: &Chunks, (log_rows, rows, explicit): (usize, usize, usize)) -> Raw {
        let mask = (1usize << chunks.width) - 1;
        let mut words = Vec::new();
        let mut coefs = Vec::new();
        for c in 0..3 {
            let mut sigma = rng.next_u64() as usize & mask;
            let column = (0..1 << log_rows)
                .map(|u| {
                    if u >= explicit {
                        return 0;
                    }
                    if rng.next_u32() % 3 == 0 {
                        sigma = (sigma + rng.next_u32() as usize % 5) & mask;
                    }
                    if rng.next_u32() % 7 == 0 {
                        // All ones in the low chunks, so the shift carries.
                        sigma |= (1 << (chunks.widths[0] + chunks.widths.get(1).copied().unwrap_or(0))) - 1;
                        sigma &= mask;
                    }
                    let word = chunks.encode(sigma);
                    if rng.next_u32() % 11 == 0 {
                        word ^ 1 << (rng.next_u32() % 64)
                    } else {
                        word
                    }
                })
                .collect();
            words.push(column);
            let coef = |rng: &mut Rng, a: usize| Coef {
                constant: rng.ext(),
                aux: if a < 2 { Some((a, rng.ext())) } else { None },
            };
            let plus = (c == 0).then(|| rng.ext());
            coefs.push((coef(rng, c), coef(rng, c + 1), plus, rng.ext_vec(chunks.d())));
        }
        let padrow = vec![F64::ZERO, F64(rng.next_u64()), F64(rng.next_u64())];
        let k_columns = (0..3)
            .map(|c| {
                (0..1 << log_rows)
                    .map(|u| if u < explicit { F64(rng.next_u64()) } else { padrow[c] })
                    .collect()
            })
            .collect();
        Raw {
            log_rows,
            rows,
            explicit,
            omega: rng.ext_vec(rows),
            aux: (0..2).map(|_| rng.ext_vec(explicit)).collect(),
            aux_tail: rng.ext_vec(2),
            words,
            coefs,
            k_columns,
            padrow,
            blocks: random_blocks(rng),
            tau: rng.ext_vec(log_rows),
        }
    }

    /// The base terms of a table at one row, from its columns' values there.
    fn base_at(t: &Raw, omega: F192, eq: F192, col: &dyn Fn(usize) -> F192) -> F192 {
        t.blocks
            .iter()
            .map(|b| {
                let w = if b.weight == Weight::Omega { omega } else { eq };
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
    }

    #[test]
    fn row_side_matches_the_naive_sum_and_end() {
        let mut rng = Rng::new(0x407);
        for (width, shapes, skip) in [
            (7usize, vec![(3usize, 5usize, 6usize), (2, 4, 4)], true),
            (13, vec![(5, 19, 25), (6, 64, 64), (2, 1, 3), (0, 1, 1)], true),
            (13, vec![(5, 19, 25), (6, 64, 64), (2, 1, 3), (0, 0, 0)], false),
            (22, vec![(4, 16, 16), (7, 100, 100), (3, 7, 8)], true),
            (1, vec![(3, 6, 6), (1, 2, 2)], true),
        ] {
            let chunks = Chunks::new(width);
            let d = chunks.d();
            let points = RowPoints::new(&chunks, &rng.ext_vec(width), &rng.ext_vec(chunks.max_width()));
            let raw: Vec<Raw> = shapes.iter().map(|&s| random_table(&mut rng, &chunks, s)).collect();
            let tables: Vec<RowTable> = raw
                .iter()
                .map(|t| RowTable {
                    log_rows: t.log_rows,
                    rows: t.rows,
                    explicit: t.explicit,
                    omega: t.omega.clone(),
                    aux: t.aux.clone(),
                    aux_tail: t.aux_tail.clone(),
                    columns: t
                        .words
                        .iter()
                        .zip(&t.coefs)
                        .map(|(w, (o, a, p, e))| RowColumn {
                            words: w,
                            omega: *o,
                            act: *a,
                            plus: *p,
                            cert: e.clone(),
                        })
                        .collect(),
                    base: Some(Base::new(
                        t.blocks.clone(),
                        t.k_columns.iter().map(|c| &c[..]).collect(),
                        &t.padrow,
                        Some(eq_table(&t.tau)),
                    )),
                })
                .collect();

            // Per row of a column: its coefficient, `ω`, and the `5d` chunk functionals.
            let chunks_ref = &chunks;
            let row_value = |t: &Raw, u: usize, column: usize| -> (F192, F192, Vec<F192>) {
                let (om, act) = if u < t.rows {
                    (t.omega[u], F192::ONE)
                } else {
                    (F192::ZERO, F192::ZERO)
                };
                let aux = |a: usize| if u < t.explicit { t.aux[a][u] } else { t.aux_tail[a] };
                let w = t.words[column][u];
                let (co, ca, _, _) = &t.coefs[column];
                let weight = |c: &Coef| c.constant + c.aux.map_or(F192::ZERO, |(a, s)| s * aux(a));
                let vals = [&points.eq, &points.cert, &points.cert_bar, &points.low, &points.carry]
                    .iter()
                    .flat_map(|e| (0..d).map(move |i| e.eval(chunks_ref, w, i)))
                    .collect();
                (om * weight(co) + act * weight(ca), om, vals)
            };
            let summand = |t: &Raw, column: usize, coef: F192, om: F192, vals: &[F192]| {
                let (_, _, plus, cert) = &t.coefs[column];
                let certs: Vec<F192> = cert.iter().map(|&e| om * e).collect();
                final_value(&chunks, &points, coef, om * plus.unwrap_or(F192::ZERO), vals, &certs)
            };
            let mut total = F192::ZERO;
            for t in &raw {
                let eq = eq_table(&t.tau);
                for u in 0..1 << t.log_rows {
                    for column in 0..3 {
                        let (coef, om, vals) = row_value(t, u, column);
                        total += summand(t, column, coef, om, &vals);
                    }
                    let om = if u < t.rows { t.omega[u] } else { F192::ZERO };
                    total += base_at(t, om, eq[u], &|c| F192::from(t.k_columns[c][u]));
                }
            }

            let mut claim = total;
            let mut n_coeffs = Vec::new();
            let mut claims = Vec::new();
            let end = prove(&chunks, &points, tables, skip, &mut |poly| {
                assert_eq!(claim, poly[1..].iter().copied().sum::<F192>(), "a round's sum");
                claims.push(claim);
                n_coeffs.push(poly.len());
                let x = rng.ext();
                claim = poly_eval(poly, x);
                x
            });
            claims.push(claim);
            assert_eq!(claim, end.claim);
            let max_log = shapes.iter().map(|s| s.0).max().unwrap();
            assert!(n_coeffs[..max_log].iter().all(|&n| n == row_coeffs(d)));
            assert!(n_coeffs[max_log..].iter().all(|&n| n == column_coeffs(d)));

            // The end, from the rows' multilinear extensions at χ and then the columns' at q*.
            let chi = &end.point;
            let mut columns = Vec::new();
            let mut base = F192::ZERO;
            for (t, table_end) in raw.iter().zip(&end.tables) {
                let eqs = eq_table(&chi[..t.log_rows]);
                let corner: F192 = chi[t.log_rows..].iter().map(|&x| F192::ONE + x).product();
                assert_eq!(table_end.corner, corner);
                // `ω`, `act` and each aux column are separate factors of the summand, so each is its own extension.
                let mle = |f: &dyn Fn(usize) -> F192| eqs.iter().enumerate().map(|(u, &e)| e * f(u)).sum::<F192>();
                let om = mle(&|u| if u < t.rows { t.omega[u] } else { F192::ZERO });
                let act = mle(&|u| if u < t.rows { F192::ONE } else { F192::ZERO });
                let aux: Vec<F192> = (0..2)
                    .map(|a| mle(&|u| if u < t.explicit { t.aux[a][u] } else { t.aux_tail[a] }))
                    .collect();
                assert_eq!(table_end.omega, om);
                let k_values: Vec<F192> = (0..3).map(|c| mle(&|u| F192::from(t.k_columns[c][u]))).collect();
                let table_base = base_at(t, om, eq_eval(&t.tau, &chi[..t.log_rows]), &|c| k_values[c]);
                assert_eq!(table_end.base, table_base);
                base += corner * table_base;
                for column in 0..3 {
                    let mut vals = vec![F192::ZERO; 5 * d];
                    for (u, &e) in eqs.iter().enumerate() {
                        let (_, _, g) = row_value(t, u, column);
                        for (v, g) in vals.iter_mut().zip(g) {
                            *v += e * g;
                        }
                    }
                    let (co, ca, _, _) = &t.coefs[column];
                    let weight = |c: &Coef| c.constant + c.aux.map_or(F192::ZERO, |(a, s)| s * aux[a]);
                    let coef = om * weight(co) + act * weight(ca);
                    assert_eq!(table_end.columns[column].values, vals);
                    let (_, _, plus, cert) = &t.coefs[column];
                    let plus = corner * om * plus.unwrap_or(F192::ZERO);
                    let certs: Vec<F192> = cert.iter().map(|&e| corner * om * e).collect();
                    columns.push((corner * coef, plus, vals, certs));
                }
            }
            let at_chi: F192 = columns
                .iter()
                .map(|(c, p, v, e)| final_value(&chunks, &points, *c, *p, v, e))
                .sum();
            assert_eq!(claims[max_log], at_chi + base, "the row rounds end on the columns' sum");
            let eqq = eq_table(&end.column_point);
            let mle = |f: &dyn Fn(usize) -> F192| (0..columns.len()).map(|q| eqq[q] * f(q)).sum::<F192>();
            let coef = mle(&|q| columns[q].0);
            let plus = mle(&|q| columns[q].1);
            let vals: Vec<F192> = (0..5 * d).map(|j| mle(&|q| columns[q].2[j])).collect();
            let certs: Vec<F192> = (0..d).map(|i| mle(&|q| columns[q].3[i])).collect();
            let base_q: F192 = end.column_point.iter().map(|&x| F192::ONE + x).product();
            assert_eq!(end.claim, final_value(&chunks, &points, coef, plus, &vals, &certs) + base_q * base);
        }
    }
}

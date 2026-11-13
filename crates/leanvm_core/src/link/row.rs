//! Hotwire's row side: one sum-check over every table's rows, then over the pointer columns.
//!
//! The rows are bound lowest variable first, so a round pairs consecutive rows of a table, which are consecutive
//! executions of its opcode. A table of `2^m` rows takes part from round 1 and is finished after `m` rounds; from
//! then on its value is multiplied by the all-zeros corner factor `1 + x_h` of each further row variable, once.
//! The pointer-column variables `q` come last.
//!
//! A pointer column `q` of table `F` contributes, at row `u`,
//!
//! `c_q(u)·Φ_0(u; r) + c⁺_q(u)·Φ_1(u; r) + ω_F(u)·Σ_i ε_{q,i}·(D_i(u; r_i)·D_i(u; 1 + r_i) + π_{w_i}(r_i))`,
//!
//! the column's coefficient `c_q = ω_F·(constant + aux) + act_F·(constant + aux)` times the pointer at the index
//! point `r`, the cycle pointer's `c⁺_q = α_+·ω_F` times the pointer shifted by one (carry matrices, from Binary
//! Jolt's precompile link), plus the mirror certificate (the mirror test of Emile Wambsgans), which vanishes on every executed row whose chunks
//! are one-hot. The certificate reads chunk `i` at the same coordinates `r_i` as the pointer, so the two share
//! `D_i(r_i)`. Two rows whose chunk agrees make that chunk's factor constant in the round that pairs them, and
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
use super::pointer::{ChunkEq, Chunks, NO_INDEX, Shifts, SplitEq, chain, eval_chunk, mirror};
use primitives::field::{F192, F192Unreduced};

/// Room for the coefficients of one round: degree `max(d + 2, 4) ≤ 10`.
pub const MAX_COEFFS: usize = 12;

/// No one-hot value: a tag for a chunk that is not one value across its group.
const MIXED: u8 = u8::MAX;

/// A row past the explicit ones: every chunk [`MIXED`].
const ALL_MIXED: u64 = u64::MAX;

/// Pairs per task of a round.
const TASK_PAIRS: usize = 256;

/// Rounds a table's pointer columns stay words, each folded row read as an `eq`-weighted sum over its group of
/// words, before the prover writes out their values.
const WORD_ROUNDS: usize = 2;

/// Words in a folded row's group before the values are written out.
const MAX_GROUP: usize = 1 << (WORD_ROUNDS - 1);

/// A column's slot tables, and each slot's table times each word's fold weight, while the rounds read words.
type WordTables<'a, 'b> = (&'a [(usize, &'b [F192])], &'a [Vec<F192>]);

/// Chunks of a word, at most.
const MAX_CHUNKS: usize = 8;

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

    fn is_zero(&self) -> bool {
        self.constant.is_zero() && self.aux.is_none()
    }
}

/// Shifted reads of a pointer column that share a coefficient on `ω_F`: `coef·Σ_s μ_s·Φ_s`, the pointer shifted by
/// each `s ≥ 1` of `weights = [(s, μ_s)]`.
#[derive(Clone, Debug)]
pub struct ShiftGroup {
    pub coef: Coef,
    pub weights: Vec<(usize, F192)>,
}

/// One pointer column, as the row side reads it.
pub struct RowColumn<'a> {
    /// The words, one per row of the table.
    pub words: &'a [u64],
    /// Each word's index, or [`NO_INDEX`] ([`Chunks::decode_all`]).
    pub index: Vec<u32>,
    /// Coefficient on `ω_F` of the pointer itself, `Φ_0`.
    pub omega: Coef,
    /// Coefficient on `act_F` of `Φ_0`.
    pub act: Coef,
    /// The shifted reads.
    pub shifts: Vec<ShiftGroup>,
    /// The certificate's coefficient per chunk.
    pub cert: Vec<F192>,
    /// The column's index in the pointer-column rounds.
    pub q: usize,
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
    /// `τ` when `ω_F = θ·eq(τ, ·)` on the executed rows, which lets each round factor `eq(τ_h, X)` out.
    pub tau: Option<Vec<F192>>,
    /// The linear forms the coefficients name, each on the explicit rows, and their values on the padrow.
    pub aux: Vec<Vec<F192>>,
    pub aux_tail: Vec<F192>,
    pub columns: Vec<RowColumn<'a>>,
    /// The table's base terms, at `q = 0`.
    pub base: Option<Base<'a>>,
    /// The first round's polynomial of `base`'s `ω` blocks ([`Base::omega_round`]), which that round then adds in
    /// place of summing them.
    pub base_first: Option<[F192; 5]>,
}

/// The functionals the row side evaluates pointers with: `D_i` at the index point `r`, the shift tables there, and
/// `D_i` at `1 + r` for the certificate.
pub struct RowPoints {
    pub eq: ChunkEq,
    pub bar: ChunkEq,
    pub shifts: Shifts,
    /// `π_{w_i}(r_i)` per chunk.
    pub mirrors: Vec<F192>,
    /// Per chunk, `D_i(r_i)·D_i(1 + r_i)` over a pair of one-hot values `v ≠ v'` in `X + X²`: entry
    /// `v·2^{w_i} + v'` is `e_v·ē_{v'} + e_{v'}·ē_v` for the tables `e` at `r_i` and `ē` at `1 + r_i`.
    pub cert_pairs: Vec<Vec<F192>>,
    /// `eq(r, ·)` over whole pointers.
    pub split: SplitEq,
}

impl RowPoints {
    pub fn new(chunks: &Chunks, r: &[F192], max_shift: usize) -> Self {
        let eq = ChunkEq::at(chunks, r);
        let bar = ChunkEq::mirrored(chunks, r);
        let cert_pairs = eq
            .tables
            .iter()
            .zip(&bar.tables)
            .map(|(e, f)| {
                let n = e.len();
                (0..n * n)
                    .map(|k| {
                        let (v, w) = (k / n, k % n);
                        e[v] * f[w] + e[w] * f[v]
                    })
                    .collect()
            })
            .collect();
        Self {
            shifts: Shifts::new(chunks, &eq, max_shift),
            mirrors: (0..chunks.d()).map(|i| mirror(chunks.part(r, i))).collect(),
            eq,
            bar,
            cert_pairs,
            split: SplitEq::new(r, max_shift),
        }
    }

    /// The layout of a column's values over every shift up to the largest, as the column rounds and the verifier
    /// read them.
    pub fn all_shifts(&self, d: usize) -> Slots {
        Slots::new(d, self.shifts.max_shift, self.shifts.carry)
    }
}

/// Where a column's chunk functionals sit among its values: `D_i` at `r` and at `1 + r` (`d` each), then
/// per shift group chunk 0's table for each carry it passes on, then per later chunk its carry tables.
#[derive(Clone, Copy, Debug)]
pub struct Slots {
    d: usize,
    groups: usize,
    carry: usize,
}

impl Slots {
    pub fn new(d: usize, groups: usize, carry: usize) -> Self {
        Self { d, groups, carry }
    }

    pub fn n_values(&self) -> usize {
        if self.groups == 0 {
            2 * self.d
        } else {
            self.links(1) + 2 * self.carry * (self.d - 1)
        }
    }

    #[inline(always)]
    fn eq(&self, i: usize) -> usize {
        i
    }

    #[inline(always)]
    fn bar(&self, i: usize) -> usize {
        self.d + i
    }

    #[inline(always)]
    fn start(&self, g: usize, c: usize) -> usize {
        2 * self.d + g * (self.carry + 1) + c
    }

    /// The first carry table of chunk `i ≥ 1`; carry `c ≥ 1` in and `out` on is `2·(c − 1) + out` past it.
    #[inline(always)]
    fn links(&self, i: usize) -> usize {
        2 * self.d + self.groups * (self.carry + 1) + (i - 1) * 2 * self.carry
    }

    #[inline(always)]
    fn link(&self, i: usize, c: usize, out: usize) -> usize {
        self.links(i) + 2 * (c - 1) + out
    }

    /// Slot `k`'s chunk and table, the group starts read from `starts`.
    #[inline(always)]
    fn table<'b>(&self, k: usize, points: &'b RowPoints, starts: &'b [Vec<F192>]) -> (usize, &'b [F192]) {
        let d = self.d;
        if k < 2 * d {
            let t = [&points.eq, &points.bar][k / d];
            return (k % d, &t.tables[k % d]);
        }
        let k = k - 2 * d;
        let n_starts = self.groups * (self.carry + 1);
        if k < n_starts {
            return (0, &starts[k]);
        }
        let k = k - n_starts;
        let i = 1 + k / (2 * self.carry);
        (i, &points.shifts.links[i][k % (2 * self.carry)])
    }
}

/// A column at the end of the row rounds: its 64 bit-slices at `χ_F`, and the values they give, laid out by
/// [`RowPoints::all_shifts`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnEnd {
    pub slices: Vec<F192>,
    pub values: Vec<F192>,
}

/// `Σ_u eq[u]·bit_b(word_u)` for each bit `b`.
pub fn bit_slices_eq(words: &[u64], eq: &[F192]) -> Vec<F192> {
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

/// A chunk table at a word's 64 bit-slices: the functional is linear in the bits.
fn from_slices(chunks: &Chunks, slices: &[F192], i: usize, table: &[F192]) -> F192 {
    let start = chunks.word_shift(i);
    table.iter().zip(&slices[start..]).map(|(&e, &s)| e * s).sum()
}

impl ColumnEnd {
    pub fn from_slices(chunks: &Chunks, points: &RowPoints, slices: Vec<F192>) -> Self {
        let slots = points.all_shifts(chunks.d());
        let starts: Vec<Vec<F192>> = (1..=points.shifts.max_shift)
            .flat_map(|s| points.shifts.start[s].clone())
            .collect();
        let values = (0..slots.n_values())
            .map(|k| {
                let (i, table) = slots.table(k, points, &starts);
                from_slices(chunks, &slices, i, table)
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
        p[j] = (p[j].mul_unreduced(a) ^ p[j - 1].mul_unreduced(b)).reduce();
    }
    p[0] *= a;
    len + 1
}

/// `p ← p·f` for the linear `f` with `f(0) = pair.0`, `f(1) = pair.1`.
#[inline(always)]
fn mul_pair(p: &mut Poly, len: usize, pair: (F192, F192)) -> usize {
    mul_linear(p, len, pair.0, pair.0 + pair.1)
}

/// `acc ← acc + p·f` for the linear `f` through `pair`; the length of the sum.
#[inline(always)]
fn add_mul_pair(acc: &mut Poly, acc_len: usize, p: &Poly, len: usize, pair: (F192, F192)) -> usize {
    let (a, b) = (pair.0, pair.0 + pair.1);
    acc[0] += p[0] * a;
    for j in 1..len {
        acc[j] += (p[j].mul_unreduced(a) ^ p[j - 1].mul_unreduced(b)).reduce();
    }
    acc[len] += p[len - 1] * b;
    acc_len.max(len + 1)
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

/// `c(X)·Φ_0(X) + Σ_g c_g(X)·Φ_g(X)` for one column at one pair, added to `acc`: the pointer under its coefficient
/// and each shift group under its own, each coefficient linear and given by its coefficients. `pair(k)` is value slot `k` at the two rows; with `skip`, a chunk whose
/// `same(i)` holds is a constant factor `first(i)`.
///
/// A group's chain starts at chunk 0 with one state per carry passed on. Once no carry is left, every group's value
/// and the pointer's share the remaining factors `Π D_i`, which the skip then shortens.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn pointer_pair(
    acc: &mut Poly,
    slots: &Slots,
    coef: &[F192],
    groups: &[[F192; 2]],
    mut act: Option<(&mut Poly, &[F192])>,
    skip: bool,
    same: impl Fn(usize) -> bool,
    first: impl Fn(usize) -> F192,
    pair: impl Fn(usize) -> (F192, F192),
) {
    let d = slots.d;
    let mut p = [F192::ZERO; MAX_COEFFS];
    p[..coef.len()].copy_from_slice(coef);
    let mut len = coef.len();
    // The pointer's term under the act weight: its coefficient times the same factors, without the shift groups.
    let mut q = [F192::ZERO; MAX_COEFFS];
    let mut q_len = 0;
    if let Some((_, c)) = &act {
        q[..c.len()].copy_from_slice(c);
        q_len = c.len();
    }
    let mut start = 0;
    let live = groups.iter().any(|g| g.iter().any(|c| !c.is_zero()));
    if live {
        let e = pair(slots.eq(0));
        len = mul_pair(&mut p, len, e);
        if q_len > 0 {
            q_len = mul_pair(&mut q, q_len, e);
        }
        // The carry states after chunk `start - 1`, carry `c` at `c - 1`, and whether any is nonzero.
        let mut carries = [[F192::ZERO; MAX_COEFFS]; 4];
        let mut carry_len = [0usize; 4];
        for (g, gc) in groups.iter().enumerate() {
            if gc.iter().all(|c| c.is_zero()) {
                continue;
            }
            let mut gp = [F192::ZERO; MAX_COEFFS];
            gp[..2].copy_from_slice(gc);
            for c in 0..=slots.carry {
                let e = pair(slots.start(g, c));
                if is_zero_pair(e) {
                    continue;
                }
                if c == 0 {
                    len = add_mul_pair(&mut p, len, &gp, 2, e);
                } else {
                    carry_len[c - 1] = add_mul_pair(&mut carries[c - 1], carry_len[c - 1], &gp, 2, e);
                }
            }
        }
        start = 1;
        while start < d && carry_len.iter().any(|&l| l > 0) {
            let i = start;
            let e = pair(slots.eq(i));
            len = mul_pair(&mut p, len, e);
            if q_len > 0 {
                q_len = mul_pair(&mut q, q_len, e);
            }
            let mut out = [F192::ZERO; MAX_COEFFS];
            let mut out_len = 0;
            for c in 1..=slots.carry {
                if carry_len[c - 1] == 0 {
                    continue;
                }
                let (y, y_len) = (carries[c - 1], carry_len[c - 1]);
                let e0 = pair(slots.link(i, c, 0));
                if !is_zero_pair(e0) {
                    len = add_mul_pair(&mut p, len, &y, y_len, e0);
                }
                let e1 = pair(slots.link(i, c, 1));
                if !is_zero_pair(e1) {
                    out_len = add_mul_pair(&mut out, out_len, &y, y_len, e1);
                }
            }
            carry_len = [0; 4];
            carries[0] = out;
            carry_len[0] = out_len;
            start += 1;
        }
    }
    let p_live = !p[..len].iter().all(|c| c.is_zero());
    let q_live = !q[..q_len].iter().all(|c| c.is_zero());
    if !p_live && !q_live {
        return;
    }
    let mut scalar = F192::ONE;
    let mut linear = [(F192::ZERO, F192::ZERO); 8];
    let mut n_linear = 0;
    for i in start..d {
        if skip && same(i) {
            scalar *= first(i);
        } else {
            linear[n_linear] = pair(slots.eq(i));
            n_linear += 1;
        }
    }
    let finish = |acc: &mut Poly, p: &[F192]| finish_pointer(acc, p, scalar, &linear[..n_linear]);
    if p_live {
        finish(acc, &p[..len]);
    }
    if q_live && let Some((act_acc, _)) = &mut act {
        finish(act_acc, &q[..q_len]);
    }
}

/// `acc ← acc + c·f` for `c` of one or two coefficients and the linear `f` through `pair`.
#[inline(always)]
fn add_times_pair(acc: &mut Poly, c: &[F192], pair: (F192, F192)) {
    let (a, b) = (pair.0, pair.0 + pair.1);
    acc[0] += c[0] * a;
    if c.len() == 1 || c[1].is_zero() {
        acc[1] += c[0] * b;
    } else {
        acc[1] += (c[0].mul_unreduced(b) ^ c[1].mul_unreduced(a)).reduce();
        acc[2] += c[1] * b;
    }
}

/// [`pointer_pair`] when at most one chunk differs across the pair's words: the pointer and every shifted chain are
/// then linear in `X`, through their values `phi` and `shifted(g)` at the two rows.
#[inline(always)]
fn linear_pointer(
    acc: &mut Poly,
    coef: &[F192],
    groups: &[[F192; 2]],
    act: Option<(&mut Poly, &[F192])>,
    phi: (F192, F192),
    shifted: impl Fn(usize) -> (F192, F192),
) {
    add_times_pair(acc, coef, phi);
    for (g, gc) in groups.iter().enumerate() {
        if !(gc[0].is_zero() && gc[1].is_zero()) {
            add_times_pair(acc, gc, shifted(g));
        }
    }
    if let Some((act_acc, c)) = act {
        add_times_pair(act_acc, c, phi);
    }
}

/// `at` of each word of a group by its index, folded by the bound challenges `r`, the first for adjacent words.
#[inline(always)]
fn fold_words(index: &[u32], r: &[F192], at: impl Fn(usize) -> F192) -> F192 {
    let mut values = [F192::ZERO; MAX_GROUP];
    for (v, &x) in values.iter_mut().zip(index) {
        *v = at(x as usize);
    }
    let mut len = index.len();
    for &x in r {
        len /= 2;
        for t in 0..len {
            let (a, b) = (values[2 * t], values[2 * t + 1]);
            values[t] = if a == b { a } else { a + x * (a + b) };
        }
    }
    values[0]
}

/// `acc ← acc + p·scalar·Π_j f_j` for the linear `f_j` through `linear[j]`.
#[inline(always)]
fn finish_pointer(acc: &mut Poly, p: &[F192], mut scalar: F192, linear: &[(F192, F192)]) {
    let len = p.len();
    if linear.is_empty() {
        if scalar == F192::ONE {
            add_into(acc, p);
        } else {
            for (a, &v) in acc.iter_mut().zip(p) {
                *a += v * scalar;
            }
        }
        return;
    }
    // The remaining factors' product first, the scalar in its first factor, then one product with `p`: cheaper than
    // multiplying the longer `p` by each factor in turn.
    if len == 1 {
        scalar *= p[0];
    }
    let (a, b) = linear[0];
    let (a, b) = if scalar == F192::ONE {
        (a, b)
    } else {
        (scalar * a, scalar * b)
    };
    let mut r = [F192::ZERO; MAX_COEFFS];
    r[0] = a;
    r[1] = a + b;
    let mut r_len = 2;
    for &e in &linear[1..] {
        r_len = mul_pair(&mut r, r_len, e);
    }
    if len == 1 {
        add_into(acc, &r[..r_len]);
        return;
    }
    for j in 0..len + r_len - 1 {
        let lo = (j + 1).saturating_sub(r_len);
        let sum = (lo..=j.min(len - 1)).fold(F192Unreduced::ZERO, |s, i| s ^ p[i].mul_unreduced(r[j - i]));
        acc[j] += sum.reduce();
    }
}

/// `Σ_i ε_i·(D_i(r_i)·D_i(1 + r_i) + π_i)(X)` for one column at one pair, added to `acc`, from the bar slots, which
/// hold `ε_i·D_i(1 + r_i)` (see [`Column::bars`]), and `ε_i·π_i` in `cert_mirrors`. With `skip`, a chunk that is one
/// one-hot value at both rows (`clean`) adds zero, one one-hot at both rows (`one_hot`) adds `b·b̄·(X + X²)`, and one
/// whose values agree (`same`) a constant.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cert_pair(
    acc: &mut Poly,
    slots: &Slots,
    cert: &[F192],
    cert_mirrors: &[F192],
    skip: bool,
    clean: impl Fn(usize) -> bool,
    one_hot: impl Fn(usize) -> bool,
    same: impl Fn(usize) -> bool,
    pair: impl Fn(usize) -> (F192, F192),
) {
    let mut m = [F192Unreduced::ZERO; 3];
    let mut constant = F192::ZERO;
    for i in 0..slots.d {
        if (skip && clean(i)) || cert[i].is_zero() {
            continue;
        }
        let (x, y) = (pair(slots.eq(i)), pair(slots.bar(i)));
        if skip && one_hot(i) {
            let t = (x.0 + x.1).mul_unreduced(y.0 + y.1);
            m[1] ^= t;
            m[2] ^= t;
        } else if skip && same(i) {
            m[0] ^= x.0.mul_unreduced(y.0);
            constant += cert_mirrors[i];
        } else {
            add_pair_product(&mut m, x, y);
            constant += cert_mirrors[i];
        }
    }
    let m = m.map(F192Unreduced::reduce);
    add_into(acc, &[m[0] + constant, m[1], m[2]]);
}

/// `m ← m + x(X)·y(X)` for the linear `x`, `y` through their pairs, by Karatsuba.
#[inline(always)]
fn add_pair_product(m: &mut [F192Unreduced; 3], x: (F192, F192), y: (F192, F192)) {
    let c0 = x.0.mul_unreduced(y.0);
    let c2 = (x.0 + x.1).mul_unreduced(y.0 + y.1);
    m[0] ^= c0;
    m[1] ^= x.1.mul_unreduced(y.1) ^ c0 ^ c2;
    m[2] ^= c2;
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

/// A coefficient at a pair of rows, `constant + aux(X)`, by its coefficients in `X`: one when it is constant.
#[inline(always)]
fn coef_linear(c: Coef, aux: &[(F192, F192)]) -> ([F192; 2], usize) {
    match c.aux {
        None => ([c.constant, F192::ZERO], 1),
        Some((a, s)) => {
            let (a0, a1) = aux[a];
            let lo = c.constant + s * a0;
            ([lo, s * (a0 + a1)], 2)
        }
    }
}

/// The summand at one bound point from a column's values laid out by `slots`: `c·Φ_0 + Σ_g c_g·Φ_g +
/// Σ_i e_i·(D_i(r_i)·D_i(1 + r_i) + π_i)`.
pub fn summand(slots: &Slots, mirrors: &[F192], coef: F192, groups: &[F192], values: &[F192], certs: &[F192]) -> F192 {
    let d = slots.d;
    let pointer: F192 = (0..d).map(|i| values[slots.eq(i)]).product();
    let shifted: F192 = groups
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(g, &c)| {
            let start: Vec<F192> = (0..=slots.carry).map(|k| values[slots.start(g, k)]).collect();
            c * chain(
                &start,
                d,
                |i| values[slots.eq(i)],
                |i, k, out| values[slots.link(i, k, out)],
            )
        })
        .sum();
    let cert: F192 = (0..d)
        .map(|i| certs[i] * (values[slots.eq(i)] * values[slots.bar(i)] + mirrors[i]))
        .sum();
    coef * pointer + shifted + cert
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
        if s == 0 {
            return Vec::new();
        }
        // SAFETY: every row writes each of its `s` values.
        unsafe {
            primitives::filled_vec(len * s, |out| {
                parallel::chunks_mut(out, s * TASK_PAIRS, |task, chunk| {
                    for (j, row) in chunk.chunks_mut(s).enumerate() {
                        let k = task * TASK_PAIRS + j;
                        let (lo, hi) = (self.row(2 * k), self.row(2 * k + 1));
                        for ((o, &a), &b) in row.iter_mut().zip(lo).zip(hi) {
                            *o = if a == b { a } else { a + x * (a + b) };
                        }
                    }
                });
            })
        }
    }
}

/// Per chunk, its bits when every word of `group` holds the same ones.
#[inline(always)]
fn uniform_chunks(chunks: &Chunks, group: &[u64]) -> [Option<u64>; MAX_CHUNKS] {
    let mut out = [None; MAX_CHUNKS];
    for (i, o) in out.iter_mut().enumerate().take(chunks.d()) {
        let b = chunks.bits(group[0], i);
        if group[1..].iter().all(|&u| chunks.bits(u, i) == b) {
            *o = Some(b);
        }
    }
    out
}

/// Chunk `i`'s `table` at a folded row: its values on the words of `group` folded by the bound challenges `r`, the
/// first for adjacent words. One lookup when the chunk is `uniform` across the group, and no product for a pair of
/// equal values.
#[inline(always)]
fn group_value(chunks: &Chunks, table: &[F192], i: usize, group: &[u64], uniform: Option<u64>, r: &[F192]) -> F192 {
    if uniform.is_some() {
        return eval_chunk(chunks, table, group[0], i);
    }
    let mut values = [F192::ZERO; MAX_GROUP];
    let mut len = group.len() / 2;
    for (v, pair) in values[..len].iter_mut().zip(group.chunks_exact(2)) {
        let a = eval_chunk(chunks, table, pair[0], i);
        *v = if chunks.bits(pair[0], i) == chunks.bits(pair[1], i) {
            a
        } else {
            a + r[0] * (a + eval_chunk(chunks, table, pair[1], i))
        };
    }
    for &x in &r[1..] {
        len /= 2;
        for t in 0..len {
            let (a, b) = (values[2 * t], values[2 * t + 1]);
            values[t] = if a == b { a } else { a + x * (a + b) };
        }
    }
    values[0]
}

/// The weights the bound challenges `r` fold a group of words by, the first for adjacent words: word `t`'s is
/// `Π_j r_j` over the bits `j` set in `t` times `Π_j (1 + r_j)` over the others.
fn fold_weights(r: &[F192]) -> Vec<F192> {
    let mut weights = vec![F192::ONE];
    for &x in r {
        let low: Vec<F192> = weights.iter().map(|&w| w * (F192::ONE + x)).collect();
        weights = low.into_iter().chain(weights.iter().map(|&w| w * x)).collect();
    }
    weights
}

/// Each slot's table times each word's fold weight in turn, `weights.len()` copies of the table.
fn weighted_tables(tables: &[(usize, &[F192])], weights: &[F192]) -> Vec<Vec<F192>> {
    tables
        .iter()
        .map(|&(_, table)| {
            weights
                .iter()
                .flat_map(|&w| table.iter().map(move |&v| w * v))
                .collect()
        })
        .collect()
}

/// [`group_value`] as a sum of lookups: each word's value read from `weighted`, the table times that word's fold
/// weight from [`weighted_tables`], so a group costs no product.
#[inline(always)]
fn weighted_value(
    chunks: &Chunks,
    table: &[F192],
    weighted: &[F192],
    i: usize,
    group: &[u64],
    uniform: Option<u64>,
) -> F192 {
    if uniform.is_some() {
        return eval_chunk(chunks, table, group[0], i);
    }
    let len = table.len();
    group
        .iter()
        .zip(weighted.chunks_exact(len))
        .fold(F192::ZERO, |sum, (&u, w)| sum + eval_chunk(chunks, w, u, i))
}

/// Row `k`'s tags, chunk `i` in byte `i`.
#[inline(always)]
fn row_tags(tags: &[u64], k: usize) -> u64 {
    tags.get(k).copied().unwrap_or(ALL_MIXED)
}

/// The tags of a folded pair: each chunk's tag where the two rows agree, [`MIXED`] where they differ.
#[inline(always)]
fn fold_tags(a: u64, b: u64) -> u64 {
    const LOW: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    let x = a ^ b;
    let differs = (((x & LOW) + LOW) | x) & !LOW;
    a | ((differs >> 7) * 0xff)
}

/// A pointer column: its words, and after round 1 per row its chunk functionals laid out by `slots`, and per chunk a
/// tag, the one-hot value the whole folded group holds or [`MIXED`], packed one byte a chunk.
struct Column<'a> {
    words: &'a [u64],
    slots: Slots,
    /// Chunk 0's table per group and carry passed on, in slot order.
    starts: Vec<Vec<F192>>,
    /// Per group, its low tables in [`SplitEq`].
    split_starts: Vec<Vec<Vec<F192>>>,
    /// Each word's index, or [`NO_INDEX`], while the rounds read words.
    index: Vec<u32>,
    values: Option<Strided>,
    tags: Vec<u64>,
    omega: Coef,
    act: Coef,
    shifts: Vec<ShiftGroup>,
    cert: Vec<F192>,
    /// Per chunk, `D_i` at `1 + r` times the chunk's certificate weight `ε_i` (one when `ε_i` is zero), so a pair's
    /// certificate term is one product of its two slots; [`State::end`] divides the weight back out.
    bars: Vec<Vec<F192>>,
    /// `ε_i·π_i` per chunk.
    cert_mirrors: Vec<F192>,
    q: usize,
}

/// A chunk's certificate weight as [`Column::bars`] scales its table by.
fn bar_scale(eps: F192) -> F192 {
    if eps.is_zero() { F192::ONE } else { eps }
}

impl<'a> Column<'a> {
    fn new(column: RowColumn<'a>, chunks: &Chunks, points: &RowPoints) -> Self {
        let slots = Slots::new(chunks.d(), column.shifts.len(), points.shifts.carry);
        let starts = column
            .shifts
            .iter()
            .flat_map(|g| {
                assert!(
                    g.weights
                        .iter()
                        .all(|&(s, _)| (1..=points.shifts.max_shift).contains(&s))
                );
                points.shifts.combined_start(&g.weights)
            })
            .collect();
        let split_starts = column
            .shifts
            .iter()
            .map(|g| points.split.shifted_tables(&g.weights))
            .collect();
        let words = column.words;
        assert_eq!(column.index.len(), words.len(), "an index per word");
        Self {
            words,
            slots,
            starts,
            split_starts,
            index: column.index,
            values: None,
            tags: Vec::new(),
            omega: column.omega,
            act: column.act,
            shifts: column.shifts,
            bars: (points.bar.tables.iter().zip(&column.cert))
                .map(|(t, &e)| t.iter().map(|&v| bar_scale(e) * v).collect())
                .collect(),
            cert_mirrors: column.cert.iter().zip(&points.mirrors).map(|(&e, &m)| e * m).collect(),
            cert: column.cert,
            q: column.q,
        }
    }

    /// Every slot's table and chunk, in slot order.
    fn tables<'b>(&'b self, points: &'b RowPoints) -> Vec<(usize, &'b [F192])> {
        let d = self.slots.d;
        (0..self.slots.n_values())
            .map(|k| match k.checked_sub(d) {
                Some(i) if i < d => (i, self.bars[i].as_slice()),
                _ => self.slots.table(k, points, &self.starts),
            })
            .collect()
    }

    /// Rows `rows` of the folded values and tags, row `k` its group of `n` words `k·n..(k + 1)·n` folded by the
    /// bound challenges, with `tables` from [`Self::tables`] and `weighted` from [`weighted_tables`].
    #[allow(clippy::too_many_arguments)]
    fn materialize_rows(
        &self,
        tables: &[(usize, &[F192])],
        weighted: &[Vec<F192>],
        n: usize,
        rows: std::ops::Range<usize>,
        values: &mut [F192],
        tags: &mut [u64],
        chunks: &Chunks,
    ) {
        let mut group = [0u64; 2 * MAX_GROUP];
        let mut sigmas = [0u32; 2 * MAX_GROUP];
        for ((row, t), k) in values.chunks_mut(tables.len()).zip(tags).zip(rows) {
            for (o, s) in sigmas[..n].iter_mut().enumerate() {
                *s = self.index.get(k * n + o).copied().unwrap_or(NO_INDEX);
            }
            // Every word one-hot: a chunk is uniform where its digit agrees across the group's indices.
            if !sigmas[..n].contains(&NO_INDEX) {
                let sigmas = &sigmas[..n];
                let diff = sigmas.iter().fold(0, |d, &s| d | (s ^ sigmas[0])) as usize;
                let digit = |sigma: u32, i: usize| chunks.digit(sigma as usize, i);
                for ((o, &(i, table)), weighted) in row.iter_mut().zip(tables).zip(weighted) {
                    *o = if chunks.digit(diff, i) == 0 {
                        table[digit(sigmas[0], i)]
                    } else {
                        let len = table.len();
                        (sigmas.iter().enumerate())
                            .fold(F192::ZERO, |sum, (t, &s)| sum + weighted[t * len + digit(s, i)])
                    };
                }
                *t = ALL_MIXED;
                for i in 0..chunks.d() {
                    if chunks.digit(diff, i) == 0 {
                        *t ^= u64::from(MIXED ^ digit(sigmas[0], i) as u8) << (8 * i);
                    }
                }
                continue;
            }
            for (o, u) in group[..n].iter_mut().enumerate() {
                *u = self.words.get(k * n + o).copied().unwrap_or(0);
            }
            let group = &group[..n];
            let uniform = uniform_chunks(chunks, group);
            for ((o, &(i, table)), weighted) in row.iter_mut().zip(tables).zip(weighted) {
                *o = weighted_value(chunks, table, weighted, i, group, uniform[i]);
            }
            *t = ALL_MIXED;
            for (i, u) in uniform.iter().enumerate().take(chunks.d()) {
                if let Some(b) = u.filter(|b| b.is_power_of_two()) {
                    *t ^= u64::from(MIXED ^ b.trailing_zeros() as u8) << (8 * i);
                }
            }
        }
    }

    /// Rows `rows` of the values and tags with the lowest variable bound at `x`.
    fn fold_rows(&self, rows: std::ops::Range<usize>, values: &mut [F192], tags: &mut [u64], x: F192) {
        let old = self.values.as_ref().expect("the values are written out");
        for ((row, t), k) in values.chunks_mut(old.stride).zip(tags).zip(rows) {
            let (lo, hi) = (old.row(2 * k), old.row(2 * k + 1));
            for ((o, &a), &b) in row.iter_mut().zip(lo).zip(hi) {
                *o = if a == b { a } else { a + x * (a + b) };
            }
            *t = fold_tags(row_tags(&self.tags, 2 * k), row_tags(&self.tags, 2 * k + 1));
        }
    }

    fn set_values(&mut self, data: Vec<F192>, tags: Vec<u64>, stride: usize) {
        self.values = Some(Strided {
            data,
            tail: vec![F192::ZERO; stride],
            stride,
        });
        self.tags = tags;
    }

    /// Add the column's terms at a pair: those under `ω` to `omega`, those under `act` to `act`, each without its
    /// weight. `pointer(acc, coef, groups)` adds a pointer term, `cert(acc)` the certificate's.
    #[inline(always)]
    fn add_terms(
        &self,
        aux: &[(F192, F192)],
        omega: &mut Poly,
        act: &mut Poly,
        pointer: impl FnOnce(&mut Poly, &[F192], &[[F192; 2]], Option<(&mut Poly, &[F192])>),
        cert: impl FnOnce(&mut Poly),
    ) {
        let mut groups = [[F192::ZERO; 2]; 4];
        for (o, g) in groups.iter_mut().zip(&self.shifts) {
            *o = coef_linear(g.coef, aux).0;
        }
        let (coef, len) = coef_linear(self.omega, aux);
        let (act_coef, act_len) = coef_linear(self.act, aux);
        let act = (!self.act.is_zero()).then_some((act, &act_coef[..act_len]));
        pointer(omega, &coef[..len], &groups[..self.shifts.len()], act);
        cert(omega);
    }

    /// The coefficient of `Φ_0`, of each shift `s` (at `s - 1`), and the certificate weights at a table's end, times
    /// its corner factor.
    fn end_coefs(&self, end: &TableEnd, max_shift: usize) -> (F192, Vec<F192>, Vec<F192>) {
        let weight = |c: Coef| c.constant + c.aux.map_or(F192::ZERO, |(a, s)| s * end.aux[a]);
        let coef = end.corner * (end.omega * weight(self.omega) + end.act * weight(self.act));
        let mut shifts = vec![F192::ZERO; max_shift];
        for g in &self.shifts {
            let c = end.corner * end.omega * weight(g.coef);
            for &(s, mu) in &g.weights {
                shifts[s - 1] += c * mu;
            }
        }
        let certs = self.cert.iter().map(|&e| end.corner * end.omega * e).collect();
        (coef, shifts, certs)
    }
}

/// Every column's `len` new rows, in rows of `strides[c]` values and one tag word for column `c`: `fill(c, rows,
/// values, tags)` writes each value and tag of column `c`'s `rows`. One dispatch covers every column.
fn fill_columns(
    strides: &[usize],
    len: usize,
    fill: impl Fn(usize, std::ops::Range<usize>, &mut [F192], &mut [u64]) + Sync,
) -> Vec<(Vec<F192>, Vec<u64>)> {
    // SAFETY: the tasks cover every row of every column once, and `fill` writes each value and tag of its rows.
    let mut out: Vec<(Vec<F192>, Vec<u64>)> = strides
        .iter()
        .map(|&s| unsafe { (primitives::uninit_vec(len * s), primitives::uninit_vec(len)) })
        .collect();
    let ptrs: Vec<_> = out
        .iter_mut()
        .map(|(v, t)| (parallel::SendPtr(v.as_mut_ptr()), parallel::SendPtr(t.as_mut_ptr())))
        .collect();
    let per_column = len.div_ceil(TASK_PAIRS);
    parallel::for_each(strides.len() * per_column, |task| {
        let (c, start) = (task / per_column, task % per_column * TASK_PAIRS);
        let rows = start..len.min(start + TASK_PAIRS);
        let s = strides[c];
        // SAFETY: each task's rows are its own, inside column `c`'s `len` rows.
        let (values, tags) = unsafe {
            (
                std::slice::from_raw_parts_mut(ptrs[c].0.0.add(rows.start * s), rows.len() * s),
                std::slice::from_raw_parts_mut(ptrs[c].1.0.add(rows.start), rows.len()),
            )
        };
        fill(c, rows, values, tags);
    });
    out
}

struct State<'a> {
    log_rows: usize,
    /// Support of `ω` and `act`.
    support: usize,
    /// Explicit rows.
    explicit: usize,
    omega: Vec<F192>,
    act: Vec<F192>,
    /// `ω`'s point, when it is `θ·eq(τ, ·)` on the executed rows.
    tau: Option<Vec<F192>>,
    /// Rows whose whole group of rows is executed: there `act` is one and `ω` is `eq`'s.
    full: usize,
    aux: Strided,
    columns: Vec<Column<'a>>,
    /// The table's row variables.
    m: usize,
    /// The row challenges bound while the pointer columns are still words.
    bound: Vec<F192>,
    /// The table's value once its rows are bound, and the corner factor since.
    done: Option<(F192, F192)>,
    base: Option<Base<'a>>,
    /// [`RowTable::base_first`], until the first round.
    base_first: Option<[F192; 5]>,
    /// Whether a column has a term under `act_F`.
    any_act: bool,
}

impl<'a> State<'a> {
    fn new(table: RowTable<'a>, chunks: &Chunks, points: &RowPoints) -> Self {
        assert!(table.rows <= table.explicit && table.explicit <= 1 << table.log_rows);
        assert_eq!(table.omega.len(), table.rows);
        assert!(
            table.columns.iter().all(|c| c.shifts.len() <= 4),
            "at most four shift groups a column"
        );
        let stride = table.aux.len();
        // SAFETY: every row writes each of its `stride` values.
        let data = unsafe {
            primitives::filled_vec(table.explicit * stride, |data| {
                if stride > 0 {
                    parallel::chunks_mut(data, stride * TASK_PAIRS * 16, |task, chunk| {
                        for (j, row) in chunk.chunks_mut(stride).enumerate() {
                            let u = task * TASK_PAIRS * 16 + j;
                            for (o, a) in row.iter_mut().zip(&table.aux) {
                                *o = a[u];
                            }
                        }
                    });
                }
            })
        };
        let aux = Strided {
            data,
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
            tau: table.tau,
            full: table.rows,
            aux,
            columns: table
                .columns
                .into_iter()
                .map(|c| Column::new(c, chunks, points))
                .collect(),
            bound: Vec::new(),
            done: None,
            base: table.base,
            base_first: table.base_first.filter(|_| table.log_rows > 0),
            any_act: false,
        };
        state.any_act = state.columns.iter().any(|c| !c.act.is_zero());
        if state.log_rows == 0 {
            // One row: its values are the table's end from the start.
            state.materialize(1, chunks, points);
            if let Some(base) = &mut state.base {
                base.fold(1, F192::ZERO);
            }
            state.finish(points);
        }
        state
    }

    /// Write out every column's values and tags at `len` folded rows, from its words and the bound challenges.
    fn materialize(&mut self, len: usize, chunks: &Chunks, points: &RowPoints) {
        let tables: Vec<_> = self.columns.iter().map(|c| c.tables(points)).collect();
        let strides: Vec<usize> = tables.iter().map(Vec::len).collect();
        let weights = fold_weights(&self.bound);
        let weighted: Vec<_> = tables.iter().map(|t| weighted_tables(t, &weights)).collect();
        let (columns, n) = (&self.columns, weights.len());
        let out = fill_columns(&strides, len, |c, rows, values, tags| {
            columns[c].materialize_rows(&tables[c], &weighted[c], n, rows, values, tags, chunks);
        });
        drop(tables);
        for ((column, (values, tags)), s) in self.columns.iter_mut().zip(out).zip(strides) {
            column.set_values(values, tags, s);
            column.index = Vec::new();
        }
    }

    /// Rows bound: the table's value at its end.
    fn finish(&mut self, points: &RowPoints) {
        let end = self.end(F192::ONE);
        let weight = |c: Coef| c.constant + c.aux.map_or(F192::ZERO, |(a, s)| s * end.aux[a]);
        let value: F192 = self
            .columns
            .iter()
            .zip(&end.columns)
            .map(|(column, values)| {
                let coef = end.omega * weight(column.omega) + end.act * weight(column.act);
                let groups: Vec<F192> = column.shifts.iter().map(|g| end.omega * weight(g.coef)).collect();
                let certs: Vec<F192> = column.cert.iter().map(|&e| end.omega * e).collect();
                summand(&column.slots, &points.mirrors, coef, &groups, &values.values, &certs)
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
        let tables: Vec<Vec<(usize, &[F192])>> = self.columns.iter().map(|c| c.tables(points)).collect();
        // Below `whole` pairs, both rows' groups are executed, so `act` is one at both and `ω` is
        // `(ω_0 + ω_1)·eq(τ_h, X)`: the pair's terms under `ω` add to `factored` scaled by `ω_0 + ω_1`, and
        // `eq(τ_h, X)` multiplies their sum once per task.
        let (whole, tau) = match &self.tau {
            Some(tau) => (self.full / 2, tau[self.m - self.log_rows]),
            None => (0, F192::ZERO),
        };
        // While the columns are words, each slot's table times each word's fold weight.
        let weights = fold_weights(&self.bound);
        let weighted: Vec<Vec<Vec<F192>>> = (self.columns.iter().zip(&tables))
            .map(|(c, t)| {
                if c.values.is_none() && weights.len() > 1 {
                    weighted_tables(t, &weights)
                } else {
                    Vec::new()
                }
            })
            .collect();
        let len = (chunks.d() + 2).max(4);
        // A small round still spreads over the workers.
        let per_task = (pairs / 64).clamp(8, TASK_PAIRS);
        let mut poly = parallel::fold_reduce(
            pairs.div_ceil(per_task),
            || [F192::ZERO; MAX_COEFFS],
            |acc, task| {
                let mut factored = [F192Unreduced::ZERO; MAX_COEFFS];
                let mut under_one = [F192::ZERO; MAX_COEFFS];
                let range = task * per_task..pairs.min((task + 1) * per_task);
                for k in range.clone() {
                    let omega = (at(&self.omega, 2 * k), at(&self.omega, 2 * k + 1));
                    // The terms under each weight, summed before the weight multiplies them.
                    let (mut under_omega, mut under_act) = ([F192::ZERO; MAX_COEFFS], [F192::ZERO; MAX_COEFFS]);
                    if let Some(base) = &self.base {
                        let with_omega = !is_zero_pair(omega) && self.base_first.is_none();
                        if with_omega || base.on_padding() {
                            base.pair(k, with_omega, &mut under_omega, acc);
                        }
                    }
                    if is_zero_pair(omega) && 2 * k >= self.act.len() {
                        continue;
                    }
                    let act = (at(&self.act, 2 * k), at(&self.act, 2 * k + 1));
                    let aux = self.aux_pair(k);
                    for ((column, tables), weighted) in self.columns.iter().zip(&tables).zip(&weighted) {
                        let sums = (&mut under_omega, &mut under_act);
                        let words = (tables.as_slice(), weighted.as_slice());
                        match &column.values {
                            None => Self::pair_words(column, words, &self.bound, k, &aux, chunks, points, skip, sums),
                            Some(values) => Self::pair_folded(column, values, k, &aux, skip, sums),
                        }
                    }
                    if k < whole {
                        let scale = omega.0 + omega.1;
                        for (f, &v) in factored.iter_mut().zip(&under_omega[..len]) {
                            *f ^= scale.mul_unreduced(v);
                        }
                        if self.any_act {
                            add_into(&mut under_one, &under_act[..len]);
                        }
                    } else {
                        add_mul_pair(acc, 0, &under_omega, len, omega);
                        if self.any_act {
                            add_mul_pair(acc, 0, &under_act, len, act);
                        }
                    }
                }
                if range.start < whole {
                    let factored = factored.map(F192Unreduced::reduce);
                    add_mul_pair(acc, 0, &factored, len, (F192::ONE + tau, tau));
                    add_into(acc, &under_one[..len]);
                }
            },
            |mut a, b| {
                add_into(&mut a, &b);
                a
            },
        );
        if let Some(first) = &self.base_first {
            add_into(&mut poly, first);
        }
        poly
    }

    /// The first rounds: the pair `(2k, 2k + 1)` straight from the words, each folded row its group of words folded
    /// by the bound challenges `r`.
    #[allow(clippy::too_many_arguments)]
    fn pair_words(
        column: &Column,
        (tables, weighted): WordTables<'_, '_>,
        r: &[F192],
        k: usize,
        aux: &[(F192, F192)],
        chunks: &Chunks,
        points: &RowPoints,
        skip: bool,
        (omega, act): (&mut Poly, &mut Poly),
    ) {
        let n = 1 << r.len();
        if skip {
            let sums = (&mut *omega, &mut *act);
            let done = match n {
                1 => Self::try_indices::<1>(column, (tables, weighted), r, k, aux, chunks, points, sums),
                2 => Self::try_indices::<2>(column, (tables, weighted), r, k, aux, chunks, points, sums),
                _ => false,
            };
            if done {
                return;
            }
        }
        let word = |u: usize| column.words.get(u).copied().unwrap_or(0);
        let (mut lo, mut hi) = ([0u64; MAX_GROUP], [0u64; MAX_GROUP]);
        for t in 0..n {
            lo[t] = word(2 * k * n + t);
            hi[t] = word((2 * k + 1) * n + t);
        }
        let (lo, hi) = (&lo[..n], &hi[..n]);
        let (lo_bits, hi_bits) = (uniform_chunks(chunks, lo), uniform_chunks(chunks, hi));
        let same = |i: usize| lo_bits[i].is_some() && lo_bits[i] == hi_bits[i];
        let one_hot =
            |i: usize| lo_bits[i].is_some_and(u64::is_power_of_two) && hi_bits[i].is_some_and(u64::is_power_of_two);
        let pair = |slot: usize| {
            let (i, table) = tables[slot];
            (
                group_value(chunks, table, i, lo, lo_bits[i], r),
                group_value(chunks, table, i, hi, hi_bits[i], r),
            )
        };
        let slots = &column.slots;
        column.add_terms(
            aux,
            omega,
            act,
            |acc, coef, groups, act| {
                pointer_pair(
                    acc,
                    slots,
                    coef,
                    groups,
                    act,
                    skip,
                    same,
                    |i| points.eq.eval(chunks, lo[0], i),
                    pair,
                )
            },
            |acc| {
                cert_pair(
                    acc,
                    slots,
                    &column.cert,
                    &column.cert_mirrors,
                    skip,
                    |i| same(i) && lo_bits[i].is_some_and(u64::is_power_of_two),
                    one_hot,
                    same,
                    pair,
                )
            },
        );
    }

    /// [`Self::pair_indices`] on the pair `k` of groups of `N` words when every word's chunks are one-hot; whether it
    /// was.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn try_indices<const N: usize>(
        column: &Column,
        tables: WordTables<'_, '_>,
        r: &[F192],
        k: usize,
        aux: &[(F192, F192)],
        chunks: &Chunks,
        points: &RowPoints,
        sums: (&mut Poly, &mut Poly),
    ) -> bool {
        let index = |u: usize| column.index.get(u).copied().unwrap_or(NO_INDEX);
        let sigmas: [[u32; N]; 2] = [0, 1].map(|side| std::array::from_fn(|t| index((2 * k + side) * N + t)));
        if sigmas.iter().any(|s| s.contains(&NO_INDEX)) {
            return false;
        }
        Self::pair_indices::<N>(column, tables, r, sigmas, aux, chunks, points, sums);
        true
    }

    /// [`Self::pair_words`] when every word's chunks are one-hot, from the words' indices, `sigmas[0]` for the
    /// first row's group and `sigmas[1]` for the second's: a chunk
    /// whose digit agrees across them is constant and its certificate term zero, and a pointer with at most one
    /// varying chunk is linear in `X`.
    #[allow(clippy::too_many_arguments)]
    fn pair_indices<const N: usize>(
        column: &Column,
        (tables, weighted): WordTables<'_, '_>,
        r: &[F192],
        sigmas: [[u32; N]; 2],
        aux: &[(F192, F192)],
        chunks: &Chunks,
        points: &RowPoints,
        (omega, act): (&mut Poly, &mut Poly),
    ) {
        let (lo, hi) = (&sigmas[0], &sigmas[1]);
        let (mut lo_diff, mut hi_diff) = (0, 0);
        for (&a, &b) in lo.iter().zip(hi) {
            lo_diff |= a ^ lo[0];
            hi_diff |= b ^ hi[0];
        }
        let diff = lo_diff | hi_diff | (lo[0] ^ hi[0]);
        let digit = |sigma: u32, i: usize| chunks.digit(sigma as usize, i);
        let varying = chunks.varying(diff);
        // Slot `slot`'s value at a folded row of `side`: one lookup when its chunk's digit agrees across the side,
        // else each word's lookup in the weighted tables.
        let value = |slot: usize, side: &[u32], side_diff: u32| {
            let (i, table) = tables[slot];
            if digit(side_diff, i) == 0 {
                return table[digit(side[0], i)];
            }
            let len = table.len();
            (side.iter().enumerate()).fold(F192::ZERO, |sum, (t, &sigma)| {
                sum + weighted[slot][t * len + digit(sigma, i)]
            })
        };
        let pair = |slot: usize| (value(slot, lo, lo_diff), value(slot, hi, hi_diff));
        let slots = &column.slots;
        column.add_terms(
            aux,
            omega,
            act,
            |acc, coef, groups, act| {
                if varying.count_ones() <= 1 {
                    let at = |sigma| points.split.at(sigma);
                    linear_pointer(
                        acc,
                        coef,
                        groups,
                        act,
                        (fold_words(lo, r, at), fold_words(hi, r, at)),
                        |g| {
                            let at = |sigma| points.split.at_shifted(&column.split_starts[g], sigma);
                            (fold_words(lo, r, at), fold_words(hi, r, at))
                        },
                    )
                } else {
                    pointer_pair(
                        acc,
                        slots,
                        coef,
                        groups,
                        act,
                        true,
                        |i| varying >> i & 1 == 0,
                        |i| points.eq.tables[i][digit(lo[0], i)],
                        pair,
                    )
                }
            },
            |acc| {
                let mut m = [F192Unreduced::ZERO; 3];
                let mut constant = F192::ZERO;
                let mut rest = varying;
                while rest != 0 {
                    let i = rest.trailing_zeros() as usize;
                    rest &= rest - 1;
                    let eps = column.cert[i];
                    if eps.is_zero() {
                        continue;
                    }
                    if digit(lo_diff, i) == 0 && digit(hi_diff, i) == 0 {
                        let t = eps.mul_unreduced(
                            points.cert_pairs[i][(digit(lo[0], i) << chunks.widths[i]) + digit(hi[0], i)],
                        );
                        m[1] ^= t;
                        m[2] ^= t;
                    } else {
                        add_pair_product(&mut m, pair(slots.eq(i)), pair(slots.bar(i)));
                        constant += column.cert_mirrors[i];
                    }
                }
                let m = m.map(F192Unreduced::reduce);
                add_into(acc, &[m[0] + constant, m[1], m[2]]);
            },
        );
    }

    /// Later rounds: the pair `(2k, 2k + 1)` of the folded columns.
    #[allow(clippy::too_many_arguments)]
    fn pair_folded(
        column: &Column,
        values: &Strided,
        k: usize,
        aux: &[(F192, F192)],
        skip: bool,
        (omega, act): (&mut Poly, &mut Poly),
    ) {
        let slots = &column.slots;
        let (lo, hi) = (values.row(2 * k), values.row(2 * k + 1));
        let pair = |slot: usize| (lo[slot], hi[slot]);
        let (lo_tags, hi_tags) = (row_tags(&column.tags, 2 * k), row_tags(&column.tags, 2 * k + 1));
        let clean = |i: usize| {
            let t = (lo_tags >> (8 * i)) as u8;
            t != MIXED && t == (hi_tags >> (8 * i)) as u8
        };
        let same = |i: usize| lo[slots.eq(i)] == hi[slots.eq(i)] && lo[slots.bar(i)] == hi[slots.bar(i)];
        column.add_terms(
            aux,
            omega,
            act,
            |acc, coef, groups, act| {
                pointer_pair(acc, slots, coef, groups, act, skip, |i| lo[i] == hi[i], |i| lo[i], pair)
            },
            |acc| {
                cert_pair(
                    acc,
                    slots,
                    &column.cert,
                    &column.cert_mirrors,
                    skip,
                    clean,
                    |_| false,
                    same,
                    pair,
                )
            },
        );
    }

    /// Bind the lowest row variable at `x`.
    fn fold(&mut self, x: F192, chunks: &Chunks, points: &RowPoints) {
        let support = self.support.div_ceil(2);
        let explicit = self.explicit.div_ceil(2);
        let fold_vec = |v: &[F192], len: usize| -> Vec<F192> {
            // SAFETY: every entry is written.
            unsafe {
                primitives::filled_vec(len, |out| {
                    parallel::chunks_mut(out, TASK_PAIRS * 16, |task, chunk| {
                        for (j, o) in chunk.iter_mut().enumerate() {
                            let k = task * TASK_PAIRS * 16 + j;
                            let (a, b) = (at(v, 2 * k), at(v, 2 * k + 1));
                            *o = a + x * (a + b);
                        }
                    });
                })
            }
        };
        self.omega = fold_vec(&self.omega, support);
        self.act = fold_vec(&self.act, support);
        self.aux.data = self.aux.fold(explicit, x);
        if let Some(base) = &mut self.base {
            base.fold(explicit, x);
        }
        self.base_first = None;
        let words = self.columns.first().is_some_and(|c| c.values.is_none());
        if words {
            self.bound.push(x);
        }
        if !words {
            let strides: Vec<usize> = self.columns.iter().map(|c| c.slots.n_values()).collect();
            let columns = &self.columns;
            let out = fill_columns(&strides, explicit, |c, rows, values, tags| {
                columns[c].fold_rows(rows, values, tags, x);
            });
            for ((column, (values, tags)), s) in self.columns.iter_mut().zip(out).zip(strides) {
                column.set_values(values, tags, s);
            }
        } else if self.bound.len() == WORD_ROUNDS || self.log_rows == 1 {
            self.materialize(explicit, chunks, points);
        }
        self.support = support;
        self.explicit = explicit;
        self.full /= 2;
        self.log_rows -= 1;
        if self.log_rows == 0 {
            self.finish(points);
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
                .map(|c| {
                    let mut values = c.values.as_ref().map_or(Vec::new(), |v| v.row(0).to_vec());
                    let d = c.slots.d;
                    for (v, &e) in values.iter_mut().skip(d).take(d).zip(&c.cert) {
                        *v *= bar_scale(e).inv();
                    }
                    ColumnEnd {
                        slices: Vec::new(),
                        values,
                    }
                })
                .collect(),
            base: self.base.as_ref().map_or(F192::ZERO, |b| b.end(at(&self.omega, 0))),
        }
    }
}

/// One column in the pointer-column rounds: its coefficient, its coefficient per shift, its values laid out by
/// [`RowPoints::all_shifts`] and its certificate weights.
type ColumnAt = (F192, Vec<F192>, Vec<F192>, Vec<F192>);

/// The row side over `tables`. `round` takes each round polynomial's coefficients and returns the challenge;
/// `claim` is what the rows sum to, checked in debug builds. `skip` turns equal-value skipping on, which every
/// proof does; off, it is only there to measure. The pointer-column rounds run over `q_vars` variables, each column
/// at its own `q`.
pub fn prove(
    chunks: &Chunks,
    points: &RowPoints,
    tables: Vec<RowTable>,
    q_vars: usize,
    skip: bool,
    round: &mut dyn FnMut(&[F192]) -> F192,
) -> RowEnd {
    let d = chunks.d();
    let n_coeffs = row_coeffs(d);
    let n_rounds = tables.iter().map(|t| t.log_rows).max().unwrap_or(0);
    let mut states: Vec<State> =
        tracing::info_span!("Setup").in_scope(|| tables.into_iter().map(|t| State::new(t, chunks, points)).collect());
    let mut point = Vec::with_capacity(n_rounds);
    let mut claim = None;
    let rounds = tracing::info_span!("Row rounds").entered();
    for h in 0..n_rounds {
        let _round = tracing::info_span!("Row round", round = h).entered();
        let mut poly = [F192::ZERO; MAX_COEFFS];
        tracing::info_span!("Sum").in_scope(|| {
            for state in &states {
                match state.done {
                    None => add_into(&mut poly, &state.round(chunks, points, skip)),
                    Some((value, corner)) => {
                        poly[0] += value * corner;
                        poly[1] += value * corner;
                    }
                }
            }
        });
        debug_assert!(claim.is_none_or(|c: F192| c == poly[1..].iter().copied().sum::<F192>()));
        let x = round(&poly[..n_coeffs]);
        claim = Some(primitives::multilinear::poly_eval(&poly[..n_coeffs], x));
        tracing::info_span!("Fold").in_scope(|| {
            for state in &mut states {
                match &mut state.done {
                    None => state.fold(x, chunks, points),
                    Some((_, corner)) => *corner *= F192::ONE + x,
                }
            }
        });
        point.push(x);
    }

    drop(rounds);

    // Each column's values at its table's end come from its bit-slices there, as the verifier's do, so the
    // column rounds see every shift's tables of every column.
    let _ends = tracing::info_span!("Ends").entered();
    let ends: Vec<TableEnd> = states
        .iter()
        .map(|s| {
            let mut end = s.end(s.done.map_or(F192::ONE, |(_, corner)| corner));
            let eq = primitives::multilinear::eq_table(&point[..s.m]);
            for (column, column_end) in s.columns.iter().zip(&mut end.columns) {
                let full = ColumnEnd::from_slices(chunks, points, bit_slices_eq(column.words, &eq));
                debug_assert_eq!(column_end.values[..2 * d], full.values[..2 * d]);
                *column_end = full;
            }
            end
        })
        .collect();
    let mut columns: Vec<Option<ColumnAt>> = Vec::new();
    columns.resize_with(1 << q_vars, || None);
    for (state, end) in states.iter().zip(&ends) {
        for (column, values) in state.columns.iter().zip(&end.columns) {
            let (coef, shifts, certs) = column.end_coefs(end, points.shifts.max_shift);
            assert!(columns[column.q].is_none(), "two columns at q = {}", column.q);
            columns[column.q] = Some((coef, shifts, values.values.clone(), certs));
        }
    }
    let base = ends.iter().map(|e| e.corner * e.base).sum();
    drop(_ends);
    tracing::info_span!("Free").in_scope(|| drop(states));
    let (column_point, claim) = tracing::info_span!("Pointer-column rounds")
        .in_scope(|| prove_columns(chunks, points, columns, base, claim, round));
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

/// The rounds over the pointer-column index `q`, lowest bit first, an empty `q` a zero column; `base` is the tables'
/// base terms, at `q = 0`.
fn prove_columns(
    chunks: &Chunks,
    points: &RowPoints,
    columns: Vec<Option<ColumnAt>>,
    mut base: F192,
    mut claim: Option<F192>,
    round: &mut dyn FnMut(&[F192]) -> F192,
) -> (Vec<F192>, F192) {
    let d = chunks.d();
    let slots = points.all_shifts(d);
    let n_shifts = points.shifts.max_shift;
    let n_coeffs = column_coeffs(d);
    let n_vars = columns.len().trailing_zeros() as usize;
    let zero = || {
        (
            F192::ZERO,
            vec![F192::ZERO; n_shifts],
            vec![F192::ZERO; slots.n_values()],
            vec![F192::ZERO; d],
        )
    };
    let mut columns: Vec<ColumnAt> = columns.into_iter().map(|c| c.unwrap_or_else(zero)).collect();
    let mut point = Vec::with_capacity(n_vars);
    for _ in 0..n_vars {
        let mut poly = [F192::ZERO; MAX_COEFFS];
        poly[0] += base;
        poly[1] += base;
        for k in 0..columns.len() / 2 {
            let (lo, hi) = (&columns[2 * k], &columns[2 * k + 1]);
            let pair = |slot: usize| (lo.2[slot], hi.2[slot]);
            let groups: Vec<[F192; 2]> = (0..n_shifts).map(|s| [lo.1[s], lo.1[s] + hi.1[s]]).collect();
            pointer_pair(
                &mut poly,
                &slots,
                &[lo.0, lo.0 + hi.0],
                &groups,
                None,
                false,
                |_| false,
                |_| F192::ZERO,
                pair,
            );
            for i in 0..d {
                let m = pair_product(pair(slots.eq(i)), pair(slots.bar(i)));
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
                    fold_vec(&lo.1, &hi.1),
                    fold_vec(&lo.2, &hi.2),
                    fold_vec(&lo.3, &hi.3),
                )
            })
            .collect();
        point.push(x);
    }
    let (coef, shifts, values, certs) = &columns[0];
    let last = summand(&slots, &points.mirrors, *coef, shifts, values, certs) + base;
    debug_assert!(claim.is_none_or(|c| c == last));
    (point, last)
}

#[cfg(test)]
mod tests {
    use super::super::base::{Block, Weight};
    use super::super::pointer::eval_chunk;
    use super::*;
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
        coefs: Vec<(Coef, Coef, Vec<ShiftGroup>, Vec<F192>)>,
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
            Block {
                weight: Weight::Omega,
                gate: None,
                form: form(rng, true),
            },
            Block {
                weight: Weight::Omega,
                gate: Some(2),
                form: form(rng, true),
            },
            Block {
                weight: Weight::Eq,
                gate: None,
                form: form(rng, false),
            },
        ]
    }

    /// A random table: three columns whose neighbouring rows share most chunks, some words not one-hot, padding rows
    /// zero past `explicit`. The first column is read shifted by one, the second by several shifts in two groups.
    fn random_table(
        rng: &mut Rng,
        chunks: &Chunks,
        max_shift: usize,
        (log_rows, rows, explicit): (usize, usize, usize),
    ) -> Raw {
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
                    if rng.next_u32().is_multiple_of(3) {
                        sigma = (sigma + rng.next_u32() as usize % 5) & mask;
                    }
                    if rng.next_u32().is_multiple_of(7) {
                        // All ones in the low chunks, so the shift carries.
                        sigma |= (1 << (chunks.widths[0] + chunks.widths.get(1).copied().unwrap_or(0))) - 1;
                        sigma &= mask;
                    }
                    let word = chunks.encode(sigma);
                    if rng.next_u32().is_multiple_of(11) {
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
            let shifts = match c {
                0 => vec![ShiftGroup {
                    coef: coef(rng, 2),
                    weights: vec![(1, rng.ext())],
                }],
                1 => (0..2)
                    .map(|g| ShiftGroup {
                        coef: coef(rng, 2 * g),
                        weights: (0..3)
                            .map(|_| (1 + rng.next_u32() as usize % max_shift, rng.ext()))
                            .collect(),
                    })
                    .collect(),
                _ => Vec::new(),
            };
            coefs.push((coef(rng, c), coef(rng, c + 1), shifts, rng.ext_vec(chunks.d())));
        }
        let padrow = vec![F64::ZERO, F64(rng.next_u64()), F64(rng.next_u64())];
        let k_columns = (0..3)
            .map(|c| {
                (0..1 << log_rows)
                    .map(|u| if u < explicit { F64(rng.next_u64()) } else { padrow[c] })
                    .collect()
            })
            .collect();
        let tau = rng.ext_vec(log_rows);
        let theta = rng.ext();
        let mut omega = eq_table(&tau);
        omega.truncate(rows);
        omega.iter_mut().for_each(|x| *x *= theta);
        Raw {
            log_rows,
            rows,
            explicit,
            omega,
            aux: (0..2).map(|_| rng.ext_vec(explicit)).collect(),
            aux_tail: rng.ext_vec(2),
            words,
            coefs,
            k_columns,
            padrow,
            blocks: random_blocks(rng),
            tau,
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

    /// A folded pair keeps each chunk's tag where the rows agree and is [`MIXED`] where they differ, byte by byte.
    #[test]
    fn folded_tags_agree_with_the_bytewise_rule() {
        let mut rng = Rng::new(0x7a6);
        for _ in 0..10_000 {
            let pick = |rng: &mut Rng| -> u64 {
                (0..8).fold(0, |t, i| {
                    t | u64::from([0u8, 1, 3, 15, MIXED][rng.next_u64() as usize % 5]) << (8 * i)
                })
            };
            let (a, b) = (pick(&mut rng), pick(&mut rng));
            let want = (0..8).fold(0, |t, i| {
                let (x, y) = ((a >> (8 * i)) as u8, (b >> (8 * i)) as u8);
                t | u64::from(if x == y { x } else { MIXED }) << (8 * i)
            });
            assert_eq!(fold_tags(a, b), want);
        }
    }

    #[test]
    fn row_side_matches_the_naive_sum_and_end() {
        let mut rng = Rng::new(0x407);
        for (width, max_shift, shapes, skip) in [
            (7usize, 12usize, vec![(3usize, 5usize, 6usize), (2, 4, 4)], true),
            (13, 12, vec![(5, 19, 25), (6, 64, 64), (2, 1, 3), (0, 1, 1)], true),
            (13, 3, vec![(5, 19, 25), (6, 64, 64), (2, 1, 3), (0, 0, 0)], false),
            (22, 12, vec![(4, 16, 16), (7, 100, 100), (3, 7, 8)], true),
            (23, 12, vec![(4, 16, 16), (3, 7, 8)], true),
            (1, 1, vec![(3, 6, 6), (1, 2, 2)], true),
        ] {
            let chunks = Chunks::new(width);
            let d = chunks.d();
            let points = RowPoints::new(&chunks, &rng.ext_vec(width), max_shift);
            let raw: Vec<Raw> = shapes
                .iter()
                .map(|&s| random_table(&mut rng, &chunks, max_shift, s))
                .collect();
            let n_columns = 3 * raw.len();
            let q_vars = n_columns.next_power_of_two().trailing_zeros() as usize;
            let mut q = 0;
            let tables: Vec<RowTable> = raw
                .iter()
                .map(|t| RowTable {
                    log_rows: t.log_rows,
                    rows: t.rows,
                    explicit: t.explicit,
                    omega: t.omega.clone(),
                    tau: skip.then(|| t.tau.clone()),
                    aux: t.aux.clone(),
                    aux_tail: t.aux_tail.clone(),
                    columns: t
                        .words
                        .iter()
                        .zip(&t.coefs)
                        .map(|(w, (o, a, g, e))| {
                            q += 1;
                            RowColumn {
                                words: w,
                                index: chunks.decode_all(w),
                                omega: *o,
                                act: *a,
                                shifts: g.clone(),
                                cert: e.clone(),
                                q: q - 1,
                            }
                        })
                        .collect(),
                    base: Some(Base::new(
                        t.blocks.clone(),
                        t.k_columns.iter().map(|c| &c[..]).collect(),
                        &t.padrow,
                        Some(eq_table(&t.tau)),
                    )),
                    base_first: Some(
                        Base::new(
                            t.blocks.clone(),
                            t.k_columns.iter().map(|c| &c[..]).collect(),
                            &t.padrow,
                            Some(eq_table(&t.tau)),
                        )
                        .omega_round(&t.omega, t.tau.first().copied().filter(|_| skip)),
                    ),
                })
                .collect();

            // Per row of a column: its coefficient, its coefficient per shift, `ω`, and the chunk functionals.
            let slots = points.all_shifts(d);
            let starts: Vec<Vec<F192>> = (1..=max_shift).flat_map(|s| points.shifts.start[s].clone()).collect();
            let chunks_ref = &chunks;
            let shift_coefs = |groups: &[ShiftGroup], weight: &dyn Fn(&Coef) -> F192| {
                let mut out = vec![F192::ZERO; max_shift];
                for g in groups {
                    for &(s, mu) in &g.weights {
                        out[s - 1] += weight(&g.coef) * mu;
                    }
                }
                out
            };
            let row_value = |t: &Raw, u: usize, column: usize| -> (F192, Vec<F192>, F192, Vec<F192>) {
                let (om, act) = if u < t.rows {
                    (t.omega[u], F192::ONE)
                } else {
                    (F192::ZERO, F192::ZERO)
                };
                let aux = |a: usize| if u < t.explicit { t.aux[a][u] } else { t.aux_tail[a] };
                let w = t.words[column][u];
                let (co, ca, groups, _) = &t.coefs[column];
                let weight = |c: &Coef| c.constant + c.aux.map_or(F192::ZERO, |(a, s)| s * aux(a));
                let vals = (0..slots.n_values())
                    .map(|k| {
                        let (i, table) = slots.table(k, &points, &starts);
                        eval_chunk(chunks_ref, table, w, i)
                    })
                    .collect();
                let shifts = shift_coefs(groups, &|c| om * weight(c));
                (om * weight(co) + act * weight(ca), shifts, om, vals)
            };
            let row_summand = |t: &Raw, column: usize, coef: F192, shifts: &[F192], om: F192, vals: &[F192]| {
                let certs: Vec<F192> = t.coefs[column].3.iter().map(|&e| om * e).collect();
                summand(&slots, &points.mirrors, coef, shifts, vals, &certs)
            };
            let mut total = F192::ZERO;
            for t in &raw {
                let eq = eq_table(&t.tau);
                for u in 0..1 << t.log_rows {
                    for column in 0..3 {
                        let (coef, shifts, om, vals) = row_value(t, u, column);
                        total += row_summand(t, column, coef, &shifts, om, &vals);
                    }
                    let om = if u < t.rows { t.omega[u] } else { F192::ZERO };
                    total += base_at(t, om, eq[u], &|c| F192::from(t.k_columns[c][u]));
                }
            }

            let mut claim = total;
            let mut n_coeffs = Vec::new();
            let mut claims = Vec::new();
            let end = prove(&chunks, &points, tables, q_vars, skip, &mut |poly| {
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
                    let mut vals = vec![F192::ZERO; slots.n_values()];
                    for (u, &e) in eqs.iter().enumerate() {
                        let (_, _, _, g) = row_value(t, u, column);
                        for (v, g) in vals.iter_mut().zip(g) {
                            *v += e * g;
                        }
                    }
                    let (co, ca, groups, cert) = &t.coefs[column];
                    let weight = |c: &Coef| c.constant + c.aux.map_or(F192::ZERO, |(a, s)| s * aux[a]);
                    let coef = om * weight(co) + act * weight(ca);
                    assert_eq!(table_end.columns[column].values, vals);
                    let shifts = shift_coefs(groups, &|c| corner * om * weight(c));
                    let certs: Vec<F192> = cert.iter().map(|&e| corner * om * e).collect();
                    columns.push((corner * coef, shifts, vals, certs));
                }
            }
            let at_chi: F192 = columns
                .iter()
                .map(|(c, s, v, e)| summand(&slots, &points.mirrors, *c, s, v, e))
                .sum();
            assert_eq!(claims[max_log], at_chi + base, "the row rounds end on the columns' sum");
            let eqq = eq_table(&end.column_point);
            let mle = |f: &dyn Fn(usize) -> F192| (0..columns.len()).map(|q| eqq[q] * f(q)).sum::<F192>();
            let coef = mle(&|q| columns[q].0);
            let shifts: Vec<F192> = (0..max_shift).map(|s| mle(&|q| columns[q].1[s])).collect();
            let vals: Vec<F192> = (0..slots.n_values()).map(|j| mle(&|q| columns[q].2[j])).collect();
            let certs: Vec<F192> = (0..d).map(|i| mle(&|q| columns[q].3[i])).collect();
            let base_q: F192 = end.column_point.iter().map(|&x| F192::ONE + x).product();
            let last = summand(&slots, &points.mirrors, coef, &shifts, &vals, &certs);
            assert_eq!(end.claim, last + base_q * base);
        }
    }
}

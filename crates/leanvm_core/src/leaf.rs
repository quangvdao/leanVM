//! The coordinates a table's links are written in, and the linear forms they accumulate into. The bus that
//! balanced them by a grand product is gone; the recursion guest, frozen at the bus-era layout, still reads the
//! block types.

use primitives::field::{F64, F128, g_pow};
use std::sync::Arc;

/// One tuple coordinate as a function of the block's row `z`.
#[derive(Clone, Debug)]
pub enum Coord {
    /// A public constant (domain separator, opcode, the seed count `1`).
    Const(F64),
    /// A committed column, value `col[z]`.
    Col(usize),
    /// The free increment `g^k · col[z]` (a virtual column, §sec:vm): `k = 1` for the
    /// count/state steps, `k ≤ 12` for a KECCAK state's consecutive cells.
    GCol(usize, u32),
    /// The product `g^k · col_a[z] · col_b[z]` of two committed columns. An address
    /// is `fp·g^o`, so this carries one on the bus without committing it: the
    /// coordinate IS the product, so no column can disagree with it and the binding
    /// constraint that used to say so is unnecessary (§sec:m3).
    Prod(usize, usize, u32),
    /// The index column `g^z` (§sec:idxcol), free via the factored MLE.
    Index,
    /// A public column (the bytecode program, §sec:e2e-bc): not committed; both parties form
    /// its MLE directly, so it raises no claim. Shared rather than owned: push and
    /// pull carry the same thirteen columns.
    Public(Arc<Vec<F64>>),
    /// A sum of `Const`/`Col`/`GCol`/`Prod` terms: any degree-2 form over the
    /// table's columns, which is all §sec:m3 asks of a coordinate. This is what
    /// carries a value a row DERIVES from its columns (an `XOR`/`MUL` result, a
    /// `DEREF` store, a `JUMP` successor) without committing a column for it, and
    /// with it the identity that would have tied the two. Like [`Coord::Prod`],
    /// only a table's blocks may carry one: the table sumcheck settles them,
    /// while a framework block has to split into per-column openings.
    Sum(Vec<Coord>),
}

/// A flushing rule: `2^kappa` rows, each a tuple of coordinates. The first `rows` flush;
/// the rest are a table's padding rows (§sec:e2e-pad), whose leaves are `1`, and
/// `padding` is the tuple at the table's padding row.
#[derive(Clone, Debug)]
pub struct Block {
    pub kappa: usize,
    pub rows: usize,
    pub coords: Vec<Coord>,
    pub padding: Vec<F64>,
}

impl Block {
    /// A block every row of which flushes.
    pub fn full(kappa: usize, coords: Vec<Coord>) -> Self {
        Self {
            kappa,
            rows: 1 << kappa,
            coords,
            padding: Vec::new(),
        }
    }
}

/// A table coordinate's value on a row whose local columns are `row`.
pub(crate) fn coord_at(c: &Coord, row: &[F64]) -> F64 {
    match c {
        Coord::Const(v) => *v,
        Coord::Col(i) => row[*i],
        Coord::GCol(i, k) => g_pow(*k as usize) * row[*i],
        Coord::Prod(i, j, k) => g_pow(*k as usize) * row[*i] * row[*j],
        Coord::Sum(cs) => cs.iter().map(|c| coord_at(c, row)).sum(),
        Coord::Index | Coord::Public(_) => unreachable!("a table's bus block carries no virtual coordinate"),
    }
}

/// Placement of each block in the stacked leaf vector (input order).
#[derive(Clone, Debug)]
pub struct Layout {
    pub mu: usize,
    pub offsets: Vec<usize>,
}

/// An evaluation claim on a committed column, settled against the witness.
/// Reconstructed identically by both sides (its value rides the stream).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnClaim {
    pub col: usize,
    pub point: Vec<F128>,
    pub value: F128,
}

/// Bits indexing a bus tuple's coordinates: `m = 14` coordinates live in the
/// `2^4` slots of the bytecode encoding (§sec:m3, §sec:e2e-bc).
pub const N_TUPLE_BITS: usize = 4;

/// Stack blocks largest-first at aligned offsets; `μ = ⌈log2 Σ 2^{κ_b}⌉`.
pub fn layout(blocks: &[Block]) -> Layout {
    let kappas: Vec<Option<usize>> = blocks.iter().map(|b| Some(b.kappa)).collect();
    let (offsets, placed) = crate::witness::stack_offsets(&kappas);
    Layout {
        mu: crate::log2_ceil_usize(placed.max(1)),
        offsets,
    }
}

/// One table's bus contribution on one side, as a form over that table's committed
/// columns: `Σ_c coeffs[c]·col_c(z) + Σ (a,b,c) c·col_a(z)·col_b(z) + constant`.
/// Every coefficient is a public function of `α`, `β` and the block selectors at
/// `ζ`, because a table's bus blocks carry only `Const`/`Col`/`GCol`/`Prod`
/// coordinates. The table sumcheck sums this against `eq(ζ[..τ], ·)` instead of
/// opening each column at `ζ`, which is why those per-column claims no longer reach
/// the PCS.
///
/// The quadratic part comes from [`Coord::Prod`] and is free: the AIR identities are
/// already degree 2, so a degree-2 form does not raise the round-polynomial degree
/// the batch pays for.
#[derive(Clone, Debug)]
pub struct BusForm {
    pub coeffs: Vec<F128>,
    /// `(col_a, col_b, coeff)` in LOCAL column indices.
    pub(crate) prods: Vec<(usize, usize, F128)>,
    pub constant: F128,
}

impl BusForm {
    /// The zero form over `n_cols` columns.
    pub(crate) fn new(n_cols: usize) -> Self {
        Self {
            coeffs: vec![F128::ZERO; n_cols],
            prods: Vec::new(),
            constant: F128::ZERO,
        }
    }

    /// The same form scaled by `w`. Every coefficient is `E`-valued already, so
    /// folding the side's `ξ`-power in here costs three multiplies once per table
    /// instead of one per row and round of the zerocheck.
    pub fn scaled(&self, w: F128) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|&c| c * w).collect(),
            prods: self.prods.iter().map(|&(a, b, c)| (a, b, c * w)).collect(),
            constant: self.constant * w,
        }
    }

    /// The pointwise sum of several forms over the same columns. Their products are
    /// listed together, a pair named in two forms twice.
    pub fn sum(forms: impl IntoIterator<Item = Self>) -> Self {
        let mut forms = forms.into_iter();
        let mut out = forms.next().expect("a table has at least one bus form");
        for form in forms {
            assert_eq!(out.coeffs.len(), form.coeffs.len(), "forms over the same columns");
            for (slot, c) in out.coeffs.iter_mut().zip(form.coeffs) {
                *slot += c;
            }
            out.constant += form.constant;
            out.prods.extend(form.prods);
        }
        out
    }

    /// The form at one point, `evals` the columns' values there; only its degree-two terms when `quadratic`.
    #[cfg(test)]
    pub(crate) fn eval(&self, evals: &[F128], quadratic: bool) -> F128 {
        let linear = if quadratic { F128::ZERO } else { self.linear(evals) };
        self.prods
            .iter()
            .fold(linear, |acc, &(a, b, c)| acc + evals[a] * evals[b] * c)
    }

    /// `constant + Σ coeffs[i]·evals[i]`.
    #[cfg(test)]
    fn linear(&self, evals: &[F128]) -> F128 {
        self.coeffs
            .iter()
            .zip(evals)
            .fold(self.constant, |acc, (&w, &v)| acc + w * v)
    }
}

/// Accumulate one coordinate of a table's block into that table's form, at
/// coefficient `w`. A [`Coord::Sum`]'s children share `w`, so a derived value
/// lands as the several coefficients and products it is made of.
pub(crate) fn accumulate_form(c: &Coord, w: F128, base: usize, form: &mut BusForm) {
    match c {
        Coord::Const(v) => form.constant += w.mul_base(*v),
        Coord::Col(i) => form.coeffs[*i - base] += w,
        Coord::GCol(i, k) => form.coeffs[*i - base] += w.mul_base(g_pow(*k as usize)),
        Coord::Prod(i, j, k) => form.prods.push((*i - base, *j - base, w.mul_base(g_pow(*k as usize)))),
        Coord::Sum(cs) => {
            for c in cs {
                accumulate_form(c, w, base, form);
            }
        }
        Coord::Index | Coord::Public(_) => {
            unreachable!("a table's bus block carries no virtual coordinate")
        }
    }
}

// One bus GKR point per cache; its prefixes are keyed by length and shared column identity.

/// Selector bits of the stacked bytecode polynomial: the public encoding
/// columns (opcode + twelve operand/immediate slots = thirteen) stack along
/// `2^N_BYTECODE_SELECTORS` slots. A column's slot is its bus tuple coordinate,
/// which is what fixes the width at sixteen rather than at the column count.
pub const N_BYTECODE_SELECTORS: usize = 4;

/// Slot of the first public column: the bytecode block leads with three
/// non-public coordinates (tag, index, read count), and a column's slot IS its
/// bus tuple coordinate.
pub(crate) const BYTECODE_PUBLIC_SLOT: usize = 3;

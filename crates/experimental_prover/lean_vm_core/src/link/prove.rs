//! Hotwire's prover: the challenges, the index side, the row side and the claims they leave on the commitment.
//!
//! After the commitment the verifier draws the link challenges. The index side runs over `j < 2^W` on arrays the
//! pointers scatter, then the prover sends `W⁺(r)`, `S(r)` and the memory term `M_1` at its end `r`, and the verifier
//! draws one fresh coefficient per identity. The row side then continues the same claim over every table's rows and
//! the pointer columns. It ends with each table's non-pointer columns at its row point and the 64 bit-slices of the
//! pointer columns, which become point claims and one ring-switched claim.

use super::base::{Base, Block, Weight};
use super::index::{self, Array, Scatter, SplitEq, Tail, Term};
use super::pointer::Chunks;
use super::row::{self, Coef, RowColumn, RowPoints, RowTable};
use crate::cpu::{KeccakLane, Layout, keccak_value, schema};
use crate::leaf::{BusForm, ColumnClaim, Coord, accumulate_form};
use crate::pcs::{RingSwitchClaim, StackClaim};
use crate::tables::{self, Links, Table};
use crate::transcript::ProverState;
use crate::witness::{Col, mle_eval_cols};
use ::pcs::stack_open::RingPiece;
use primitives::field::{F64, F128};
use primitives::multilinear::eq_table;
use std::borrow::Cow;
use verifier::layout::{BUS_BITS, MEMORY_0, MEMORY_1};

/// Entries per task when filling an array over the index.
const TASK: usize = 1 << 12;

/// Indices each worker holds back while it scatters memory accesses ([`Scatter::add_repeated_words`]).
const MEMORY_SLOTS: usize = 256;

/// Indices each worker holds back while it scatters instructions: more than for memory, since a loop's
/// instructions recur only once per pass through it and every worker runs the same hot loops.
const PROGRAM_SLOTS: usize = 4096;

/// A pointer column's words: the column's own in the stack, or a test's changed copy.
enum Words<'a> {
    Stack(Col<'a, u64>),
    #[cfg(test)]
    Owned(Vec<u64>),
}

impl Words<'_> {
    fn col(&self) -> Col<'_, u64> {
        match self {
            Words::Stack(words) => *words,
            #[cfg(test)]
            Words::Owned(words) => Col::whole(words),
        }
    }
}

/// The words of a table's pointer column that the index side scatters: the column's own.
#[cfg(not(test))]
fn index_words(_: usize, _: usize, words: Col<'_, u64>) -> Words<'_> {
    Words::Stack(words)
}

/// The words of a table's pointer column that the index side scatters: the column's own, unless a test cheats.
#[cfg(test)]
fn index_words(t: usize, column: usize, words: Col<'_, u64>) -> Words<'_> {
    match cheat::INDEX_WORD.get() {
        Some((ct, cc, u, word)) if (ct, cc) == (t, column) && u < words.len() => {
            let mut words = words.to_vec();
            words[u] = word;
            Words::Owned(words)
        }
        _ => Words::Stack(words),
    }
}

/// What a table links, and how Hotwire reads its columns.
pub(crate) struct Plan {
    pub links: Links,
    /// The pointer columns, in Hotwire's order ([`Links::pointers`]).
    pub pointers: Vec<usize>,
    /// The columns that gate an access, in order of first use.
    pub gates: Vec<usize>,
    /// Every other column, in local order: what the prover sends at the row point.
    pub values: Vec<usize>,
}

impl Plan {
    pub(crate) fn of(table: &dyn Table) -> Self {
        let links = tables::links(table);
        let pointers = links.pointers();
        let mut gates = Vec::new();
        for g in links.memory.iter().filter_map(|a| a.gate) {
            if !gates.contains(&g) {
                gates.push(g);
            }
        }
        let values = (0..table.n_committed_columns())
            .filter(|c| !pointers.contains(c))
            .collect();
        Self {
            links,
            pointers,
            gates,
            values,
        }
    }

    /// The aux index of a gate column on the row side: the state form comes first.
    fn gate_aux(&self, gate: usize) -> usize {
        1 + self.gates.iter().position(|&g| g == gate).expect("a gate")
    }

    /// A pointer column's coefficient on `ω_F` from the access it takes unshifted, and its shift groups, one per
    /// gate, from the accesses it takes shifted; `c` is `η_mm·Y(r)`.
    fn access_coefs(&self, pointer: usize, mu: &[F128], c: F128) -> (Coef, Coef, Vec<row::ShiftGroup>) {
        let coef = |gate: Option<usize>| match gate {
            None => Coef::constant(c),
            Some(g) => Coef {
                constant: F128::ZERO,
                aux: Some((self.gate_aux(g), c)),
            },
        };
        let mut omega = None;
        let mut groups: Vec<(Option<usize>, row::ShiftGroup)> = Vec::new();
        for (access, &mu) in self.links.memory.iter().zip(mu) {
            if access.pointer != pointer {
                continue;
            }
            if access.shift == 0 {
                assert!(omega.is_none(), "one unshifted access per pointer");
                let mut base = coef(access.gate);
                base.constant *= mu;
                base.aux = base.aux.map(|(a, s)| (a, s * mu));
                omega = Some(base);
                continue;
            }
            match groups.iter_mut().find(|(g, _)| *g == access.gate) {
                Some((_, group)) => group.weights.push((access.shift, mu)),
                None => groups.push((
                    access.gate,
                    row::ShiftGroup {
                        coef: coef(access.gate),
                        weights: vec![(access.shift, mu)],
                    },
                )),
            }
        }
        (
            omega.unwrap_or(Coef::ZERO),
            Coef::ZERO,
            groups.into_iter().map(|(_, g)| g).collect(),
        )
    }
}

/// The link challenges, drawn after the commitment.
pub(crate) struct Challenges {
    pub theta: Vec<F128>,
    pub tau: Vec<F128>,
    /// On `pc`, `fp`.
    pub gamma_state: [F128; 2],
    /// On the address, the two limbs (`γ_word` and `γ_word·y`) and the presence coordinate.
    pub gamma_mem: [F128; 4],
    /// Per table, per access.
    pub mu: Vec<Vec<F128>>,
    /// `eq(α_bc, s)` over the sixteen slots of the stacked program.
    pub alpha_bc: Vec<F128>,
    pub zeta: Vec<F128>,
    pub eta_start: F128,
    pub eta_cover: F128,
    pub eta_mem: F128,
    pub eta_bc: F128,
    /// Per global pointer column, per chunk: `ε_q·ν_c`.
    pub eps: Vec<Vec<F128>>,
}

impl Challenges {
    /// After the grind their error asks for: a degree-`m + 3 + N_BYTECODE_SELECTORS` polynomial in them on the row
    /// side (`m` the tallest table's row variables), and the `width + 1` of the index side.
    fn sample(ps: &mut ProverState, plans: &[Plan], taus: &[usize], chunks: &Chunks) -> Self {
        let m = taus.iter().copied().max().unwrap_or(0);
        let degree = (m + 3 + crate::leaf::N_BYTECODE_SELECTORS).max(chunks.width + 1);
        ps.grind(fiat_shamir::grinding_bits(degree as u128));
        // The grind's permutation draws `θ_0`; the rest come eight to a permutation, the ones whose count is fixed
        // first.
        let mut theta = vec![ps.sample()];
        let n_accesses: usize = plans.iter().map(|p| p.links.memory.len()).sum();
        let n_pointers: usize = plans.iter().map(|p| p.pointers.len()).sum();
        let fixed = (tables::N_TABLES - 1) + 5 + n_accesses + crate::leaf::N_BYTECODE_SELECTORS + 4 + n_pointers;
        let mut drawn = ps.sample_packed(fixed + m + chunks.width + chunks.d()).into_iter();
        let mut take = |n: usize| -> Vec<F128> { drawn.by_ref().take(n).collect() };
        theta.extend(take(tables::N_TABLES - 1));
        let gamma_state: [F128; 2] = take(2).try_into().unwrap();
        // A word `lo + y·hi` takes `γ_word·y` on its high limb.
        let [g_addr, g_word, g_pres]: [F128; 3] = take(3).try_into().unwrap();
        let gamma_mem = [g_addr, g_word, g_word * F128::Y, g_pres];
        let mu = plans.iter().map(|p| take(p.links.memory.len())).collect();
        let alpha_bc = take(crate::leaf::N_BYTECODE_SELECTORS);
        let [eta_start, eta_cover, eta_mem, eta_bc]: [F128; 4] = take(4).try_into().unwrap();
        let eps = take(n_pointers);
        let tau = take(m);
        let zeta = take(chunks.width);
        let nu = take(chunks.d());
        let eps = eps.iter().map(|&e| nu.iter().map(|&n| e * n).collect()).collect();
        Self {
            theta,
            tau,
            gamma_state,
            gamma_mem,
            mu,
            alpha_bc: eq_table(&alpha_bc),
            zeta,
            eta_start,
            eta_cover,
            eta_mem,
            eta_bc,
            eps,
        }
    }
}

/// The coefficients drawn after the values at `r`, one per identity.
struct Fresh {
    plus: F128,
    spine: F128,
    cert: F128,
    jump: [F128; 2],
}

/// `γ_0·addr + γ_pres + Σ_c γ_{c+1}·v_c` of one access's committed part, at weight `w`, into `form`: its reads are
/// the pointers' coefficients.
fn access_form(access: &tables::Access, gamma: &[F128; 4], w: F128, form: &mut BusForm) {
    accumulate_form(&access.addr, w * gamma[0], 0, form);
    form.constant += w * gamma[3];
    for (c, v) in access.vals.iter().enumerate() {
        accumulate_form(v, w * gamma[1 + c], 0, form);
    }
}

/// A table's payload forms, before the `eq(0, r)` factor: the ungated part `s′_γ + η_mm·Σ μ_p·R_p + η_bc·R^bc`, then
/// one form per gate, `η_mm·Σ μ_p·R_p` over the accesses it gates.
fn payload_forms(table: &dyn Table, plan: &Plan, ch: &Challenges, t: usize) -> (BusForm, Vec<BusForm>) {
    let n = table.n_committed_columns();
    let mut ungated = BusForm::new(n);
    let links = &plan.links;
    for (next, &g) in links.state.next.iter().zip(&ch.gamma_state) {
        accumulate_form(next, g, 0, &mut ungated);
    }
    let a = &ch.alpha_bc;
    let bc = &links.bytecode;
    accumulate_form(&Coord::Col(bc.pc), ch.eta_bc * a[1], 0, &mut ungated);
    ungated.constant += ch.eta_bc * a[2] + (ch.eta_bc * a[3]).mul_base(bc.opcode);
    for (slots, operand) in &bc.operands {
        accumulate_form(operand, ch.eta_bc * a[slots[0]], 0, &mut ungated);
    }
    let mut gated: Vec<BusForm> = plan.gates.iter().map(|_| BusForm::new(n)).collect();
    for (access, &mu) in links.memory.iter().zip(&ch.mu[t]) {
        let form = match access.gate {
            None => &mut ungated,
            Some(g) => &mut gated[plan.gate_aux(g) - 1],
        };
        access_form(access, &ch.gamma_mem, ch.eta_mem * mu, form);
    }
    (ungated, gated)
}

/// A table's base blocks, every form scaled by `scale`, and `JUMP`'s identities on every row.
fn blocks(
    table: &dyn Table,
    plan: &Plan,
    ch: &Challenges,
    t: usize,
    scale: F128,
    jump: Option<[F128; 2]>,
) -> Vec<Block> {
    let (ungated, gated) = payload_forms(table, plan, ch, t);
    let mut out = vec![Block {
        weight: Weight::Omega,
        gate: None,
        form: ungated.scaled(scale),
    }];
    out.extend(plan.gates.iter().zip(gated).map(|(&g, form)| Block {
        weight: Weight::Omega,
        gate: Some(g),
        form: form.scaled(scale),
    }));
    if let Some(alphas) = jump.filter(|_| table.n_constraints() > 0) {
        out.push(Block {
            weight: Weight::Eq,
            gate: None,
            form: table.constraint(&alphas),
        });
    }
    out
}

/// One past the last row that differs from the padrow in any column, at least `rows`, even, at most `2^log_rows`.
fn explicit_rows(columns: &[Col], padrow: &[F64], rows: usize, log_rows: usize) -> usize {
    let last = parallel::map_collect(columns.len(), |c| {
        columns[c].rposition(|v| v != padrow[c]).map_or(0, |u| u + 1)
    });
    let e = last.into_iter().max().unwrap_or(0).max(rows);
    (e + (e & 1)).min(1 << log_rows)
}

/// `f(j, g^j)` for `j < n`, in parallel: each task seeds `g^j` once and steps it by `g`.
fn tabulate_powers(n: usize, f: impl Fn(usize, F64) -> F128 + Sync) -> Vec<F128> {
    tabulate_powers_in(n, |_| (), |(), j, g_j| f(j, g_j))
}

/// [`tabulate_powers`] with `window` made once per task from its entries' range, for `f` to read from.
fn tabulate_powers_in<W>(
    n: usize,
    window: impl Fn(std::ops::Range<usize>) -> W + Sync,
    f: impl Fn(&W, usize, F64) -> F128 + Sync,
) -> Vec<F128> {
    // SAFETY: every entry is written.
    unsafe {
        primitives::filled_vec(n, |out| {
            parallel::chunks_mut(out, TASK, |task, chunk| {
                let mut g = primitives::field::g_pow(task * TASK);
                let start = task * TASK;
                let w = window(start..start + chunk.len());
                for (k, o) in chunk.iter_mut().enumerate() {
                    *o = f(&w, start + k, g);
                    g = primitives::field::mul_by_g(g);
                }
            })
        })
    }
}

/// `f(j)` for `j < n`, in parallel.
fn tabulate(n: usize, f: impl Fn(usize) -> F128 + Sync) -> Vec<F128> {
    // SAFETY: every entry is written.
    unsafe {
        primitives::filled_vec(n, |out| {
            parallel::chunks_mut(out, TASK, |task, chunk| {
                for (k, o) in chunk.iter_mut().enumerate() {
                    *o = f(task * TASK + k);
                }
            })
        })
    }
}

/// `Π_{h ≥ k} (1 + x_h)`: the multilinear extension of `[j < 2^k]`.
fn below_power(point: &[F128], k: usize) -> F128 {
    point[k.min(point.len())..].iter().map(|&x| F128::ONE + x).product()
}

/// What the program side weighs each opcode's slots by, per table and slot (`verifier::link::SlotWeights`): on the
/// rows' weight `ω` (`omega`, and `pc` for a slot read times `pc`), and on `ω·fp` (`frame`).
struct SlotWeights {
    omega: Vec<[F128; 16]>,
    pc: Vec<[F128; 16]>,
    frame: Vec<[F128; 16]>,
}

impl SlotWeights {
    fn new(plans: &[Plan], ch: &Challenges) -> Self {
        let n = plans.len();
        let mut w = Self {
            omega: vec![[F128::ZERO; 16]; n],
            pc: vec![[F128::ZERO; 16]; n],
            frame: vec![[F128::ZERO; 16]; n],
        };
        let (a, gm) = (&ch.alpha_bc, &ch.gamma_mem);
        for (t, plan) in plans.iter().enumerate() {
            w.omega[t][3] = ch.eta_bc * a[3];
            for (slots, _) in &plan.links.bytecode.operands {
                for &s in slots {
                    w.omega[t][s] += ch.eta_bc * a[slots[0]];
                }
            }
            for (access, &mu) in plan.links.memory.iter().zip(&ch.mu[t]) {
                let m = ch.eta_mem * mu;
                if let Some((s, e)) = access.frame {
                    w.frame[t][s] += (m * gm[0]).mul_base(primitives::field::g_pow(e as usize));
                }
                for read in &access.reads {
                    match *read {
                        tables::Read::Word { .. } => {}
                        tables::Read::Slot { slot, scale } => w.omega[t][slot] += m * gm[1] * scale,
                        tables::Read::Frame { slot, scale } => w.frame[t][slot] += m * gm[1] * scale,
                        tables::Read::Pc { slot, scale } => w.pc[t][slot] += m * gm[1] * scale,
                    }
                }
            }
        }
        w
    }
}

/// A table's reads of the word at one pointer: `(constant + Σ c·col)·word(σ)` summed over its accesses under `μ`.
struct WordReads {
    pointer: usize,
    constant: F128,
    cols: Vec<(usize, F128)>,
}

/// Each pointer a table reads words at, with what it reads there under `scale·μ`.
fn word_reads(plan: &Plan, mu: &[F128], scale: F128) -> Vec<WordReads> {
    let mut out: Vec<WordReads> = Vec::new();
    for (access, &mu) in plan.links.memory.iter().zip(mu) {
        for read in &access.reads {
            let tables::Read::Word {
                pointer,
                constant,
                cols,
            } = read
            else {
                continue;
            };
            let at = match out.iter().position(|w| w.pointer == *pointer) {
                Some(at) => at,
                None => {
                    out.push(WordReads {
                        pointer: *pointer,
                        constant: F128::ZERO,
                        cols: Vec::new(),
                    });
                    out.len() - 1
                }
            };
            let w = &mut out[at];
            w.constant += scale * mu * *constant;
            for &(c, coef) in cols {
                match w.cols.iter_mut().find(|(d, _)| *d == c) {
                    Some((_, v)) => *v += scale * mu * coef,
                    None => w.cols.push((c, scale * mu * coef)),
                }
            }
        }
    }
    out
}

/// Where a pointer column of `2^n_vars` rows sits: its own block, or, for one riding a spare slot of `q_keccak`, that
/// slot's rows of it (as `verifier::layout` places it).
fn pointer_placement(layout: &Layout, col: usize, n_vars: usize) -> crate::witness::Placement {
    match keccak_value(col) {
        Some(KeccakLane::Slot(slot)) => {
            let lane = layout.shape.mu - crate::pcs::LOG_BATCH;
            layout.placements[crate::cpu::QKECCAK].rows(slot << n_vars, n_vars, lane)
        }
        _ => layout.placements[col],
    }
}

/// What the link leaves: the memory claim and the table columns' claims at their row points, the pointer columns'
/// ring-switched claim, and the Keccak reduction's claim on the permutation's output.
pub(crate) struct LinkEnd {
    pub memory: StackClaim,
    pub columns: Vec<ColumnClaim>,
    pub ring: RingSwitchClaim,
    pub output: keccak_gkr::OutputClaim,
}

/// Prove the links of an execution whose committed columns are `columns` (every global column, virtual ones
/// included), of `layout`, running `program`.
pub(crate) fn prove(ps: &mut ProverState, layout: &Layout, columns: &[Col], program: &[u64]) -> LinkEnd {
    let chunks = Chunks::new(layout.pointer_width);
    let width = chunks.width;
    let n = 1usize << width;
    let sch = schema();
    let all = tables::tables();
    let plans: Vec<Plan> = all.iter().map(|&t| Plan::of(t)).collect();
    let taus = layout.taus;
    let rows = layout.rows;
    let local = |t: usize| &columns[sch.base[t]..sch.base[t] + all[t].n_committed_columns()];
    let ch = Challenges::sample(ps, &plans, &taus, &chunks);
    let weights = SlotWeights::new(&plans, &ch);
    let cycles: usize = rows.iter().sum();
    let log_mem = layout.placements[MEMORY_0].n_vars;
    // The program's columns, the opcode then the operand slots, from the stacked bytecode, whose slot
    // `BYTECODE_PUBLIC_SLOT + s` holds column `s`.
    let bytecode_size = program.len() >> BUS_BITS;
    let program: Vec<&[F64]> = (program
        .chunks_exact(bytecode_size)
        .skip(crate::leaf::BYTECODE_PUBLIC_SLOT))
    .map(primitives::cast_slice)
    .collect();
    let [g_pc, g_fp] = ch.gamma_state;
    let s_fin = g_pc.mul_base(primitives::field::g_pow(bytecode_size - 1)) + g_fp;

    // ω_F on each table's executed rows, and the state form `s_γ` on its explicit rows.
    let omegas: Vec<Vec<F128>> = (0..tables::N_TABLES)
        .map(|t| {
            // SAFETY: the fill writes every entry.
            let mut e = unsafe {
                primitives::filled_vec(1 << taus[t], |out| {
                    primitives::multilinear::fill_eq_table(&ch.tau[..taus[t]], ch.theta[t], out)
                })
            };
            e.truncate(rows[t]);
            e
        })
        .collect();
    let padrows: Vec<Vec<F64>> = all.iter().map(|t| t.padding()).collect();
    let explicit: Vec<usize> = (0..tables::N_TABLES)
        .map(|t| explicit_rows(local(t), &padrows[t], rows[t], taus[t]))
        .collect();
    let state_at = |t: usize, u: usize| {
        let s = &plans[t].links.state;
        let cols = local(t);
        g_pc.mul_base(cols[s.pc][u]) + g_fp.mul_base(cols[s.fp][u])
    };

    // The base terms' `ω` blocks' first row round, which also gives the index side its payload.
    let mut base_firsts = Vec::new();
    // Each pointer column's indices, in `Plan::pointers` order, decoded once for both sides.
    let pointer_words: Vec<Col<u64>> = (0..tables::N_TABLES)
        .flat_map(|t| plans[t].pointers.iter().map(move |&c| local(t)[c].words()))
        .collect();
    let mut decoded = chunks.decode_many(&pointer_words).into_iter();
    let mut indices: Vec<Vec<Vec<u32>>> = (0..tables::N_TABLES)
        .map(|t| decoded.by_ref().take(plans[t].pointers.len()).collect())
        .collect();
    let index_end = tracing::info_span!("Index side").in_scope(|| {
        let scatter = tracing::info_span!("Scatter").entered();
        let pointer = |t: usize, c: usize| {
            let (plan, cols, n_t) = (&plans[t], local(t), rows[t]);
            let k = plan.pointers.iter().position(|&p| p == c).expect("a pointer column");
            match index_words(t, c, cols[c].words().prefix(n_t)) {
                Words::Stack(w) => (Words::Stack(w), Cow::Borrowed(&indices[t][k][..n_t])),
                #[cfg(test)]
                Words::Owned(w) => {
                    let index = chunks.decode_all(&w);
                    (Words::Owned(w), Cow::Owned(index))
                }
            }
        };
        let cycles_at: Vec<_> = (0..tables::N_TABLES)
            .map(|t| pointer(t, plans[t].links.state.cycle))
            .collect();
        let fill = |lens: [usize; 7]| {
            let arrays = lens.map(Scatter::new);
            let [w_plus, spine, cover, w_mem, w_bc, v_word, w_fp] = &arrays;
            // Every table's cycles in one pass over the cycle-indexed arrays.
            let cycle_words: Vec<(Col<u64>, &[u32])> = cycles_at.iter().map(|(w, at)| (w.col(), &at[..])).collect();
            let targets = [(w_plus, 1), (spine, 0), (cover, 0)];
            Scatter::add_sorted_words_to(&chunks, &cycle_words, targets, |t, u| {
                [omegas[t][u], state_at(t, u), F128::ONE]
            });
            for t in 0..tables::N_TABLES {
                let (plan, cols, omega) = (&plans[t], local(t), &omegas[t]);
                let pointer = |c: usize| pointer(t, c);
                // Every access of the table in one pass over its rows.
                let memory = &plan.links.memory[..plan.links.memory.len().min(ch.mu[t].len())];
                let pointers: Vec<_> = memory.iter().map(|access| pointer(access.pointer)).collect();
                let accesses: Vec<(Col<u64>, &[u32], usize)> = (pointers.iter().zip(memory))
                    .map(|((words, at), access)| (words.col(), &at[..], access.shift))
                    .collect();
                w_mem.add_repeated_accesses::<MEMORY_SLOTS>(&chunks, &accesses, |a, u| {
                    let w = ch.mu[t][a] * omega[u];
                    match memory[a].gate {
                        None => w,
                        Some(g) => w.mul_base(cols[g][u]),
                    }
                });
                // The word reads: `γ_word·μ·scale(u)` at the pointer each reads, under `ω`.
                let reads = word_reads(plan, &ch.mu[t], ch.gamma_mem[1]);
                if !reads.is_empty() {
                    let at: Vec<_> = reads.iter().map(|w| pointer(w.pointer)).collect();
                    let entries: Vec<(Col<u64>, &[u32], usize)> =
                        at.iter().map(|(words, at)| (words.col(), &at[..], 0)).collect();
                    v_word.add_repeated_accesses::<MEMORY_SLOTS>(&chunks, &entries, |a, u| {
                        let w = &reads[a];
                        let scale = (w.cols.iter()).fold(w.constant, |acc, &(c, coef)| acc + coef.mul_base(cols[c][u]));
                        scale * omega[u]
                    });
                }
                let (bc, at) = pointer(plan.links.bytecode.pointer);
                w_bc.add_repeated_words::<PROGRAM_SLOTS>(&chunks, bc.col(), &at, 0, |u| omega[u]);
                let fp = cols[plan.links.state.fp];
                w_fp.add_repeated_words::<PROGRAM_SLOTS>(&chunks, bc.col(), &at, 0, |u| omega[u].mul_base(fp[u]));
            }
            arrays
        };
        // Sized for an honest execution's indices: cycles and their successors, memory cells and bytecode
        // entries. An index past these makes the arrays again at the full `2^W`.
        let lens = [
            cycles + 1,
            cycles + 1,
            cycles + 1,
            1 << log_mem,
            bytecode_size,
            1 << log_mem,
            bytecode_size,
        ];
        let mut arrays = fill(lens.map(|l| l.min(n)));
        if arrays.iter().any(Scatter::overflowed) {
            arrays = fill([n; 7]);
        }
        let [w_plus, spine, cover, w_mem, w_bc, v_word, w_fp] = arrays;
        let mut spine = spine.into_vec();
        if spine.len() <= cycles {
            spine.resize(cycles + 1, F128::ZERO);
        }
        spine[cycles] += s_fin;

        drop(scatter);
        let _tables = tracing::info_span!("Index tables").entered();
        // The memory and program tables are `g^j` terms past their cells and instructions: each is stored only as
        // far as the array it multiplies reaches, rounded to a power of two. `eq(ζ, ·)` is not stored at all.
        let [w_plus, cover, w_mem, w_bc, v_word, w_fp] =
            [w_plus, cover, w_mem, w_bc, v_word, w_fp].map(Scatter::into_vec);
        let reach = |own: usize, partner: usize| own.max(partner).max(1).next_power_of_two().min(n);
        let (mem_len, bc_len) = (reach(1 << log_mem, w_mem.len()), reach(bytecode_size, w_bc.len()));
        let gm = ch.gamma_mem;
        let mem = &columns[MEMORY_0..=MEMORY_1];
        // The word table, `lo + y·hi` per cell: the two memory columns' limbs as they are.
        let words = {
            let (m0, m1) = (mem[0].window(0, 1 << log_mem), mem[1].window(0, 1 << log_mem));
            tabulate(1 << log_mem, |j| F128::new(m0[j].0, m1[j].0))
        };
        // Each task reads its memory cells as one run of each column.
        let cells = |r: std::ops::Range<usize>| {
            let (start, end) = (r.start.min(1 << log_mem), r.end.min(1 << log_mem));
            (start, mem[0].window(start, end), mem[1].window(start, end))
        };
        // Each entry's mixed products are summed unreduced and reduced once.
        let y_mem = tabulate_powers_in(mem_len, cells, |(start, m0, m1), j, g_j| {
            let y = gm[0].mul_base_unreduced(g_j);
            if j < 1 << log_mem {
                let cell = gm[1].mul_base_unreduced(m0[j - start]) ^ gm[2].mul_base_unreduced(m1[j - start]);
                (y ^ cell).reduce() + gm[3]
            } else {
                y.reduce()
            }
        });
        // The program side's targets: `X_ω` under the rows' weight and `P_fp` under `ω·fp`, each weighing an
        // instruction's slots by its opcode.
        let a = &ch.alpha_bc;
        let (x_g, x_one) = (ch.eta_bc * a[1], ch.eta_bc * a[2]);
        let reads_pc: Vec<bool> = (weights.pc.iter()).map(|w| w.iter().any(|c| !c.is_zero())).collect();
        let x_bc = tabulate_powers(bc_len, |j, g_j| {
            let mut x = x_g.mul_base_unreduced(g_j);
            if j < bytecode_size {
                let op = program[0][j].0.trailing_zeros() as usize;
                let w = &weights.omega[op];
                for (s, column) in program.iter().enumerate() {
                    x ^= w[3 + s].mul_base_unreduced(column[j]);
                }
                let mut x = x.reduce() + x_one;
                if reads_pc[op] {
                    let p = &weights.pc[op];
                    for (s, column) in program.iter().enumerate().skip(1) {
                        x += p[3 + s].mul_base(g_j * column[j]);
                    }
                }
                return x;
            }
            x.reduce()
        });
        let p_fp = tabulate(reach(bytecode_size, w_fp.len()).min(bytecode_size), |j| {
            let op = program[0][j].0.trailing_zeros() as usize;
            let w = &weights.frame[op];
            let mut x = F128::ZERO.mul_base_unreduced(F64::ZERO);
            for (s, column) in program.iter().enumerate().skip(1) {
                x ^= w[3 + s].mul_base_unreduced(column[j]);
            }
            x.reduce()
        });
        let firsts: Vec<[F128; 5]> = (0..tables::N_TABLES)
            .map(|t| {
                let base = Base::new(
                    blocks(all[t], &plans[t], &ch, t, F128::ONE, None),
                    local(t).to_vec(),
                    &padrows[t],
                    None,
                );
                base.omega_round(&omegas[t], ch.tau.first().copied())
            })
            .collect();
        let payload: F128 = firsts.iter().flat_map(|f| f[1..].iter().copied()).sum();
        drop(_tables);
        base_firsts = firsts;
        let powers = |c: F128| Tail::Powers { c, ratio: F64::G };
        let arrays = vec![
            w_plus.into(),
            spine.into(),
            cover.into(),
            w_mem.into(),
            Array {
                head: y_mem,
                tail: powers(gm[0]),
            },
            w_bc.into(),
            Array {
                head: x_bc,
                tail: powers(x_g),
            },
            Array {
                head: Vec::new(),
                tail: Tail::Eq(SplitEq::new(F128::ONE, ch.zeta.clone())),
            },
            vec![F128::ONE].into(),
            vec![payload].into(),
            v_word.into(),
            words.into(),
            w_fp.into(),
            p_fp.into(),
        ];
        let terms = [
            Term::Product {
                coef: F128::ONE,
                a: 0,
                b: 1,
            },
            Term::Product {
                coef: ch.eta_mem,
                a: 3,
                b: 4,
            },
            Term::Product {
                coef: F128::ONE,
                a: 5,
                b: 6,
            },
            Term::Product {
                coef: ch.eta_cover,
                a: 7,
                b: 2,
            },
            Term::Product {
                coef: ch.eta_start,
                a: 8,
                b: 1,
            },
            Term::Single { coef: F128::ONE, a: 9 },
            Term::Product {
                coef: ch.eta_mem,
                a: 10,
                b: 11,
            },
            Term::Product {
                coef: F128::ONE,
                a: 12,
                b: 13,
            },
        ];
        let _rounds = tracing::info_span!("Index rounds").entered();
        index::prove(width, arrays, &terms, &mut |poly| {
            ps.add_round_poly(poly, false);
            ps.sample()
        })
    });

    // The values at `r`, then the fresh coefficients.
    let r = &index_end.point;
    let (w_plus_r, spine_r) = (index_end.values[0], index_end.values[1]);
    let r_mem = &r[..log_mem];
    let below_mem = below_power(r, log_mem);
    let mem_values = mle_eval_cols(&columns[MEMORY_0..=MEMORY_1], r_mem);
    // The word table at `r`.
    let m1 = (mem_values[0] + F128::Y * mem_values[1]) * below_mem;
    let y_mem_r = index_end.values[4];
    let x_bc_r = index_end.values[6];
    let p_fp_r = index_end.values[13];
    debug_assert_eq!(index_end.values[11], m1);
    debug_assert_eq!(
        y_mem_r,
        ch.gamma_mem[0] * index_mle(r) + ch.gamma_mem[3] * below_mem + ch.gamma_mem[1] * m1
    );
    #[cfg(test)]
    let m1 = m1 + cheat::MEMORY_TERM.get();
    ps.add_scalars(&[w_plus_r, spine_r, m1]);
    let fresh = Fresh {
        plus: ps.sample(),
        spine: ps.sample(),
        cert: ps.sample(),
        jump: [ps.sample(), ps.sample()],
    };
    let eq0_r = below_power(r, 0);
    let eq_zeta_r = index_end.values[7];
    // The row side's claim, as the verifier forms it from the index side's.
    let eq_cycles_r: F128 = (r.iter().enumerate())
        .map(|(h, &x)| if cycles >> h & 1 == 1 { x } else { F128::ONE + x })
        .product();
    let row_claim = index_end.claim
        + w_plus_r * spine_r
        + ch.eta_start * eq0_r * spine_r
        + fresh.plus * w_plus_r
        + fresh.spine * (spine_r + eq_cycles_r * s_fin);

    let row_end = tracing::info_span!("Row side").in_scope(|| {
        let max_shift = plans.iter().map(|p| p.links.max_shift()).max().unwrap_or(0).max(1);
        let points = RowPoints::new(&chunks, r, max_shift);
        let mut q = 0;
        let mut tables_in = Vec::with_capacity(tables::N_TABLES);
        for t in 0..tables::N_TABLES {
            let (plan, cols) = (&plans[t], local(t));
            let ex = explicit[t];
            let pad = &padrows[t];
            let mut aux = vec![tabulate(ex, |u| state_at(t, u))];
            aux.extend(plan.gates.iter().map(|&g| tabulate(ex, |u| F128::from(cols[g][u]))));
            let s = &plan.links.state;
            let mut aux_tail = vec![g_pc.mul_base(pad[s.pc]) + g_fp.mul_base(pad[s.fp])];
            aux_tail.extend(plan.gates.iter().map(|&g| F128::from(pad[g])));
            // `fp`, the coefficient of the program side's `P_fp`, then each pointer's word reads' column part.
            let fp_aux = aux.len();
            aux.push(tabulate(ex, |u| F128::from(cols[s.fp][u])));
            aux_tail.push(F128::from(pad[s.fp]));
            let reads = word_reads(plan, &ch.mu[t], F128::ONE);
            let mut read_aux = Vec::with_capacity(reads.len());
            for w in &reads {
                if w.cols.is_empty() {
                    read_aux.push(None);
                    continue;
                }
                read_aux.push(Some(aux.len()));
                let at = |row: &dyn Fn(usize) -> F64| {
                    w.cols.iter().fold(F128::ZERO, |acc, &(c, k)| acc + k.mul_base(row(c)))
                };
                aux.push(tabulate(ex, |u| at(&|c| cols[c][u])));
                aux_tail.push(at(&|c| pad[c]));
            }
            let word_coef = ch.eta_mem * ch.gamma_mem[1] * m1;
            let mut row_columns = Vec::with_capacity(plan.pointers.len());
            for (k, &pointer) in plan.pointers.iter().enumerate() {
                let (omega, act, shifts) = match k {
                    0 => (
                        Coef::ZERO,
                        Coef {
                            constant: ch.eta_cover * eq_zeta_r,
                            aux: Some((0, fresh.spine)),
                        },
                        vec![row::ShiftGroup {
                            coef: Coef::constant(fresh.plus),
                            weights: vec![(1, F128::ONE)],
                        }],
                    ),
                    1 => (
                        Coef {
                            constant: x_bc_r,
                            aux: Some((fp_aux, p_fp_r)),
                        },
                        Coef::ZERO,
                        Vec::new(),
                    ),
                    _ => {
                        let (mut omega, act, shifts) = plan.access_coefs(pointer, &ch.mu[t], ch.eta_mem * y_mem_r);
                        if let Some(i) = reads.iter().position(|w| w.pointer == pointer) {
                            omega.constant += word_coef * reads[i].constant;
                            if let Some(a) = read_aux[i] {
                                assert!(omega.aux.is_none(), "a gated pointer reads no word");
                                omega.aux = Some((a, word_coef));
                            }
                        }
                        (omega, act, shifts)
                    }
                };
                row_columns.push(RowColumn {
                    words: cols[pointer].words(),
                    index: std::mem::take(&mut indices[t][k]),
                    omega,
                    act,
                    shifts,
                    cert: ch.eps[q].iter().map(|&e| fresh.cert * e).collect(),
                    q,
                    piece_log: pointer_placement(layout, sch.base[t] + pointer, taus[t]).piece_vars,
                });
                q += 1;
            }
            let jump = (all[t].n_constraints() > 0).then(|| eq_table(&ch.tau[..taus[t]]));
            let base = Base::new(
                blocks(all[t], plan, &ch, t, eq0_r, Some(fresh.jump)),
                cols.to_vec(),
                pad,
                jump,
            );
            tables_in.push(RowTable {
                log_rows: taus[t],
                rows: rows[t],
                explicit: ex,
                omega: omegas[t].clone(),
                tau: Some(ch.tau[..taus[t]].to_vec()),
                aux,
                aux_tail,
                columns: row_columns,
                base: Some(base),
                base_first: Some(base_firsts[t].map(|c| eq0_r * c)),
            });
        }
        let q_vars = q.next_power_of_two().trailing_zeros() as usize;
        row::prove(
            &chunks,
            &points,
            tables_in,
            q_vars,
            Some(row_claim),
            true,
            &mut |poly| {
                ps.add_round_poly(poly, false);
                ps.sample()
            },
        )
    });

    // Each table's other columns at its row point, then the pointer columns' bit-slices.
    let chi = &row_end.point;
    let mut column_claims = Vec::new();
    let mut lanes = [F128::ZERO; primitives::hash::STATE_LANES];
    let column_values = tracing::info_span!("Column values").entered();
    for t in 0..tables::N_TABLES {
        let point = &chi[..taus[t]];
        let cols = local(t);
        let values = mle_eval_cols(&plans[t].values.iter().map(|&c| cols[c]).collect::<Vec<_>>(), point);
        ps.add_scalars(&values);
        for (&c, &value) in plans[t].values.iter().zip(&values) {
            let col = sch.base[t] + c;
            match keccak_value(col) {
                Some(KeccakLane::Output(l)) => lanes[l] = value,
                _ => column_claims.push(ColumnClaim {
                    col,
                    point: point.to_vec(),
                    value,
                }),
            }
        }
    }
    drop(column_values);
    let eq_q = eq_table(&row_end.column_point);
    let mut slices = vec![F128::ZERO; 64];
    let mut pieces = Vec::new();
    let mut q = 0;
    // The row side's ends hold each pointer column's bit-slices at its table's row point.
    for t in 0..tables::N_TABLES {
        let point = &chi[..taus[t]];
        for (&pointer, end) in plans[t].pointers.iter().zip(&row_end.tables[t].columns) {
            for (s, &v) in slices.iter_mut().zip(&end.slices) {
                *s += eq_q[q] * v;
            }
            // A split column is its pieces, each at the low end of the row point. The row side's
            // per-piece bit-slices are the opening's first projection of the piece's lane onto them.
            let placement = &pointer_placement(layout, sch.base[t] + pointer, taus[t]);
            let lane = layout.shape.mu - crate::pcs::LOG_BATCH;
            let weights = primitives::multilinear::eq_table_serial(&point[placement.piece_vars..]);
            for ((offset, &weight), slices) in placement.piece_offsets(lane).zip(&weights).zip(&end.pieces) {
                pieces.push(RingPiece {
                    offset,
                    point: point[..placement.piece_vars].to_vec(),
                    scale: eq_q[q] * weight,
                    slices: (placement.piece_vars <= lane).then(|| slices.clone()),
                });
            }
            q += 1;
        }
    }
    ps.add_scalars(&slices);
    let memory = StackClaim::Sum {
        parts: (MEMORY_0..=MEMORY_1)
            .map(|c| layout.placements[c].sum_part(r_mem.to_vec(), [F128::ONE, F128::Y][c - MEMORY_0] * below_mem))
            .collect(),
        value: m1,
    };
    LinkEnd {
        memory,
        columns: column_claims,
        ring: RingSwitchClaim {
            pieces,
            s_hat_v: Some(slices),
        },
        output: keccak_gkr::OutputClaim {
            instance: chi[..taus[verifier::layout::OP_KECCAK]].to_vec(),
            lanes,
        },
    }
}

/// `Π_h (1 + x_h·(1 + g^{2^h}))`, the multilinear extension of `j ↦ g^j`.
fn index_mle(point: &[F128]) -> F128 {
    let mut g = F64::G;
    let mut out = F128::ONE;
    for &x in point {
        out *= F128::ONE + x.mul_base(F64::ONE + g);
        g *= g;
    }
    out
}

/// Test-only ways for the prover to cheat past its witness, for the negative tests: the word the index side scatters
/// for one row of one pointer column, and a change to the memory term it sends.
#[cfg(test)]
pub(crate) mod cheat {
    use primitives::field::F128;
    use std::cell::Cell;

    thread_local! {
        /// `(table, local column, row, word)`: the index side scatters `word` for that row instead of its own.
        pub(crate) static INDEX_WORD: Cell<Option<(usize, usize, usize, u64)>> = const { Cell::new(None) };
        /// Added to `M_1` before it is sent.
        pub(crate) static MEMORY_TERM: Cell<F128> = const { Cell::new(F128::ZERO) };
    }
}

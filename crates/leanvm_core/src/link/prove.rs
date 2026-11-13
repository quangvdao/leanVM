//! Hotwire's prover: the challenges, the index side, the row side and the claims they leave on the commitment.
//!
//! After the commitment the verifier draws the link challenges. The index side runs over `j < 2^W` on arrays the
//! pointers scatter, then the prover sends `W⁺(r)`, `S(r)` and the memory term `M_1` at its end `r`, and the verifier
//! draws one fresh coefficient per identity. The row side then continues the same claim over every table's rows and
//! the pointer columns. It ends with each table's non-pointer columns at its row point and the 64 bit-slices of the
//! pointer columns, which become point claims and one ring-switched claim.

use super::base::{Base, Block, Weight};
use super::index::{self, Array, Scatter, Tail, Term};
use super::pointer::Chunks;
use super::row::{self, Coef, RowColumn, RowPoints, RowTable};
use crate::cpu::{BytecodeColumns, KeccakLane, Layout, MEM_LO, MEM_TOP, keccak_value, schema};
use crate::leaf::{BusForm, ColumnClaim, Coord, accumulate_form};
use crate::pcs::{RingSwitchClaim, SlotClaim};
use crate::tables::{self, Links, Table};
use crate::transcript::ProverState;
use ::pcs::stack_open::{RingPiece, SumPart};
use primitives::field::{F64, F192};
use primitives::multilinear::{eq_table, mle_eval_par};
use std::borrow::Cow;

/// Entries per task when filling an array over the index.
const TASK: usize = 1 << 12;

/// The words of a pointer column.
fn words(column: &[F64]) -> &[u64] {
    // SAFETY: `F64` is `#[repr(transparent)]` over `u64`.
    unsafe { std::slice::from_raw_parts(column.as_ptr().cast(), column.len()) }
}

/// The words of a table's pointer column that the index side scatters: the column's own.
#[cfg(not(test))]
fn index_words(_: usize, _: usize, words: &[u64]) -> Cow<'_, [u64]> {
    Cow::Borrowed(words)
}

/// The words of a table's pointer column that the index side scatters: the column's own, unless a test cheats.
#[cfg(test)]
fn index_words(t: usize, column: usize, words: &[u64]) -> Cow<'_, [u64]> {
    match cheat::INDEX_WORD.get() {
        Some((ct, cc, u, word)) if (ct, cc) == (t, column) && u < words.len() => {
            let mut words = words.to_vec();
            words[u] = word;
            Cow::Owned(words)
        }
        _ => Cow::Borrowed(words),
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
    fn access_coefs(&self, pointer: usize, mu: &[F192], c: F192) -> (Coef, Coef, Vec<row::ShiftGroup>) {
        let coef = |gate: Option<usize>| match gate {
            None => Coef::constant(c),
            Some(g) => Coef {
                constant: F192::ZERO,
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
    pub theta: Vec<F192>,
    pub tau: Vec<F192>,
    /// On `pc`, `fp`.
    pub gamma_state: [F192; 2],
    /// On the address, the three limbs and the presence coordinate.
    pub gamma_mem: [F192; 5],
    /// Per table, per access.
    pub mu: Vec<Vec<F192>>,
    /// `eq(α_bc, s)` over the sixteen slots of the stacked program.
    pub alpha_bc: Vec<F192>,
    pub zeta: Vec<F192>,
    pub eta_start: F192,
    pub eta_cover: F192,
    pub eta_mem: F192,
    pub eta_bc: F192,
    /// Per global pointer column, per chunk.
    pub eps: Vec<Vec<F192>>,
}

impl Challenges {
    fn sample(ps: &mut ProverState, plans: &[Plan], taus: &[usize], chunks: &Chunks) -> Self {
        let theta = ps.sample_vec(tables::N_TABLES);
        let tau = ps.sample_vec(taus.iter().copied().max().unwrap_or(0));
        let gamma_state = [ps.sample(), ps.sample()];
        let gamma_mem = std::array::from_fn(|_| ps.sample());
        let mu = plans.iter().map(|p| ps.sample_vec(p.links.memory.len())).collect();
        let alpha_bc = ps.sample_vec(crate::leaf::N_BYTECODE_SELECTORS);
        let zeta = ps.sample_vec(chunks.width);
        let [eta_start, eta_cover, eta_mem, eta_bc] = std::array::from_fn(|_| ps.sample());
        let n_pointers: usize = plans.iter().map(|p| p.pointers.len()).sum();
        let eps = (0..n_pointers).map(|_| ps.sample_vec(chunks.d())).collect();
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
    plus: F192,
    spine: F192,
    cert: F192,
    jump: [F192; 2],
}

/// `γ_0·addr + γ_pres + Σ_c γ_{c+1}·v_c` of one access, at weight `w`, into `form`.
fn access_form(access: &tables::Access, gamma: &[F192; 5], w: F192, form: &mut BusForm) {
    accumulate_form(&access.addr, w * gamma[0], 0, form);
    form.constant += w * gamma[4];
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
    for (i, operand) in bc.operands.iter().enumerate() {
        accumulate_form(operand, ch.eta_bc * a[4 + i], 0, &mut ungated);
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
    scale: F192,
    jump: Option<[F192; 2]>,
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
fn explicit_rows(columns: &[&[F64]], padrow: &[F64], rows: usize, log_rows: usize) -> usize {
    let last = parallel::map_collect(columns.len(), |c| {
        columns[c].iter().rposition(|&v| v != padrow[c]).map_or(0, |u| u + 1)
    });
    let e = last.into_iter().max().unwrap_or(0).max(rows);
    (e + (e & 1)).min(1 << log_rows)
}

/// `g^j` for `j < n`.
fn index_powers(n: usize) -> Vec<F64> {
    // SAFETY: every entry is written.
    unsafe {
        primitives::filled_vec(n, |out| {
            parallel::chunks_mut(out, TASK, |task, chunk| {
                let mut g = primitives::field::g_pow(task * TASK);
                for o in chunk {
                    *o = g;
                    g *= F64::G;
                }
            })
        })
    }
}

/// `f(j)` for `j < n`, in parallel.
fn tabulate(n: usize, f: impl Fn(usize) -> F192 + Sync) -> Vec<F192> {
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
fn below_power(point: &[F192], k: usize) -> F192 {
    point[k.min(point.len())..].iter().map(|&x| F192::ONE + x).product()
}

/// What the link leaves: the memory claim and the table columns' claims at their row points, the pointer columns'
/// ring-switched claim, and the Keccak reduction's claim on the permutation's output.
pub(crate) struct LinkEnd {
    pub memory: SlotClaim,
    pub columns: Vec<ColumnClaim>,
    pub ring: RingSwitchClaim,
    pub output: keccak_gkr::OutputClaim,
}

/// Prove the links of an execution whose committed columns are `columns` (every global column, virtual ones
/// included), of `layout`, running `program`.
pub(crate) fn prove(ps: &mut ProverState, layout: &Layout, columns: &[&[F64]], program: &BytecodeColumns) -> LinkEnd {
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
    let cycles: usize = rows.iter().sum();
    let log_mem = layout.placements[MEM_LO].n_vars;
    let bytecode_size = program[0].len();
    let [g_pc, g_fp] = ch.gamma_state;
    let s_fin = g_pc.mul_base(primitives::field::g_pow(bytecode_size - 1)) + g_fp;

    // ω_F on each table's executed rows, and the state form `s_γ` on its explicit rows.
    let omegas: Vec<Vec<F192>> = (0..tables::N_TABLES)
        .map(|t| {
            let mut e = eq_table(&ch.tau[..taus[t]]);
            e.truncate(rows[t]);
            e.iter_mut().for_each(|x| *x *= ch.theta[t]);
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
    let mut indices: Vec<Vec<Vec<u32>>> = (0..tables::N_TABLES)
        .map(|t| {
            plans[t]
                .pointers
                .iter()
                .map(|&c| chunks.decode_all(words(local(t)[c])))
                .collect()
        })
        .collect();
    let index_end = tracing::info_span!("Index side").in_scope(|| {
        let scatter = tracing::info_span!("Scatter").entered();
        let fill = |lens: [usize; 5]| {
            let arrays = lens.map(Scatter::new);
            let [w_plus, spine, cover, w_mem, w_bc] = &arrays;
            for t in 0..tables::N_TABLES {
                let (plan, cols, omega, n_t) = (&plans[t], local(t), &omegas[t], rows[t]);
                let pointer = |c: usize| {
                    let k = plan.pointers.iter().position(|&p| p == c).expect("a pointer column");
                    match index_words(t, c, &words(cols[c])[..n_t]) {
                        Cow::Borrowed(w) => (Cow::Borrowed(w), Cow::Borrowed(&indices[t][k][..n_t])),
                        Cow::Owned(w) => {
                            let index = chunks.decode_all(&w);
                            (Cow::Owned(w), Cow::Owned(index))
                        }
                    }
                };
                let (cycle, at) = pointer(plan.links.state.cycle);
                w_plus.add_words(&chunks, &cycle, &at, 1, |u| omega[u]);
                spine.add_words(&chunks, &cycle, &at, 0, |u| state_at(t, u));
                cover.add_words(&chunks, &cycle, &at, 0, |_| F192::ONE);
                for (access, &mu) in plan.links.memory.iter().zip(&ch.mu[t]) {
                    let ((words, at), s) = (pointer(access.pointer), access.shift);
                    match access.gate {
                        None => w_mem.add_repeated_words(&chunks, &words, &at, s, |u| mu * omega[u]),
                        Some(g) => {
                            w_mem.add_repeated_words(&chunks, &words, &at, s, |u| (mu * omega[u]).mul_base(cols[g][u]))
                        }
                    }
                }
                let (bc, at) = pointer(plan.links.bytecode.pointer);
                w_bc.add_repeated_words(&chunks, &bc, &at, 0, |u| omega[u]);
            }
            arrays
        };
        // Sized for an honest execution's indices: cycles and their successors, memory cells and bytecode
        // entries. An index past these makes the arrays again at the full `2^W`.
        let mut arrays = fill([cycles + 1, cycles + 1, cycles + 1, 1 << log_mem, bytecode_size].map(|l| l.min(n)));
        if arrays.iter().any(Scatter::overflowed) {
            arrays = fill([n; 5]);
        }
        let [w_plus, spine, cover, w_mem, w_bc] = arrays;
        let mut spine = spine.into_vec();
        if spine.len() <= cycles {
            spine.resize(cycles + 1, F192::ZERO);
        }
        spine[cycles] += s_fin;

        drop(scatter);
        let _tables = tracing::info_span!("Index tables").entered();
        // The memory and program tables are `g^j` terms past their cells and instructions, and `eq(ζ, ·)` is a
        // product: each is stored only as far as the array it multiplies reaches, rounded to a power of two.
        let [w_plus, cover, w_mem, w_bc] = [w_plus, cover, w_mem, w_bc].map(Scatter::into_vec);
        let reach = |own: usize, partner: usize| own.max(partner).max(1).next_power_of_two().min(n);
        let (mem_len, bc_len, zeta_len) = (
            reach(1 << log_mem, w_mem.len()),
            reach(bytecode_size, w_bc.len()),
            reach(1, cover.len()),
        );
        let idx = index_powers(mem_len.max(bc_len));
        let gm = ch.gamma_mem;
        let mem = &columns[MEM_LO..=MEM_TOP];
        let y_mem = tabulate(mem_len, |j| {
            let mut y = gm[0].mul_base(idx[j]);
            if j < 1 << log_mem {
                y += gm[4] + gm[1].mul_base(mem[0][j]) + gm[2].mul_base(mem[1][j]) + gm[3].mul_base(mem[2][j]);
            }
            y
        });
        let a = &ch.alpha_bc;
        let x_bc = tabulate(bc_len, |j| {
            let mut x = a[1].mul_base(idx[j]);
            if j < bytecode_size {
                x += a[2];
                for (s, column) in program.iter().enumerate() {
                    x += a[3 + s].mul_base(column[j]);
                }
            }
            x
        });
        drop(idx);
        let firsts: Vec<[F192; 5]> = (0..tables::N_TABLES)
            .map(|t| {
                let base = Base::new(
                    blocks(all[t], &plans[t], &ch, t, F192::ONE, None),
                    local(t).to_vec(),
                    &padrows[t],
                    None,
                );
                base.omega_round(&omegas[t], ch.tau.first().copied())
            })
            .collect();
        let payload: F192 = firsts.iter().flat_map(|f| f[1..].iter().copied()).sum();
        let low = zeta_len.trailing_zeros() as usize;
        let above = below_power(&ch.zeta, low);
        // SAFETY: the fill writes every entry.
        let eq_zeta = unsafe {
            primitives::filled_vec(zeta_len, |out| {
                primitives::multilinear::fill_eq_table(&ch.zeta[..low], above, out)
            })
        };
        drop(_tables);
        base_firsts = firsts;
        let powers = |c: F192| Tail::Powers { c, ratio: F64::G };
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
                tail: powers(a[1]),
            },
            Array {
                head: eq_zeta,
                tail: Tail::Eq {
                    c: F192::ONE,
                    point: ch.zeta.clone(),
                },
            },
            vec![F192::ONE].into(),
            vec![payload].into(),
        ];
        let terms = [
            Term::Product {
                coef: F192::ONE,
                a: 0,
                b: 1,
            },
            Term::Product {
                coef: ch.eta_mem,
                a: 3,
                b: 4,
            },
            Term::Product {
                coef: ch.eta_bc,
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
            Term::Single { coef: F192::ONE, a: 9 },
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
    let mem_values: Vec<F192> = (MEM_LO..=MEM_TOP).map(|c| mle_eval_par(columns[c], r_mem)).collect();
    let m1: F192 = (0..3).map(|c| ch.gamma_mem[1 + c] * mem_values[c]).sum::<F192>() * below_mem;
    let y_mem_r = index_end.values[4];
    let x_bc_r = index_end.values[6];
    debug_assert_eq!(
        y_mem_r,
        ch.gamma_mem[0] * index_mle(r) + ch.gamma_mem[4] * below_mem + m1
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
            aux.extend(plan.gates.iter().map(|&g| tabulate(ex, |u| F192::from(cols[g][u]))));
            let s = &plan.links.state;
            let mut aux_tail = vec![g_pc.mul_base(pad[s.pc]) + g_fp.mul_base(pad[s.fp])];
            aux_tail.extend(plan.gates.iter().map(|&g| F192::from(pad[g])));
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
                            weights: vec![(1, F192::ONE)],
                        }],
                    ),
                    1 => (Coef::constant(ch.eta_bc * x_bc_r), Coef::ZERO, Vec::new()),
                    _ => plan.access_coefs(pointer, &ch.mu[t], ch.eta_mem * y_mem_r),
                };
                row_columns.push(RowColumn {
                    words: words(cols[pointer]),
                    index: std::mem::take(&mut indices[t][k]),
                    omega,
                    act,
                    shifts,
                    cert: ch.eps[q].iter().map(|&e| fresh.cert * e).collect(),
                    q,
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
        row::prove(&chunks, &points, tables_in, q_vars, true, &mut |poly| {
            ps.add_round_poly(poly, false);
            ps.sample()
        })
    });

    // Each table's other columns at its row point, then the pointer columns' bit-slices.
    let chi = &row_end.point;
    let mut column_claims = Vec::new();
    let mut lanes = [F192::ZERO; primitives::hash::STATE_LANES];
    let column_values = tracing::info_span!("Column values").entered();
    for t in 0..tables::N_TABLES {
        let point = &chi[..taus[t]];
        let cols = local(t);
        let values: Vec<F192> = plans[t].values.iter().map(|&c| mle_eval_par(cols[c], point)).collect();
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
    let mut slices = vec![F192::ZERO; 64];
    let mut pieces = Vec::new();
    let mut q = 0;
    // The row side's ends hold each pointer column's bit-slices at its table's row point.
    for t in 0..tables::N_TABLES {
        let point = &chi[..taus[t]];
        for (&pointer, end) in plans[t].pointers.iter().zip(&row_end.tables[t].columns) {
            for (s, &v) in slices.iter_mut().zip(&end.slices) {
                *s += eq_q[q] * v;
            }
            pieces.push(RingPiece {
                offset: layout.placements[sch.base[t] + pointer].offset,
                point: point.to_vec(),
                scale: eq_q[q],
            });
            q += 1;
        }
    }
    ps.add_scalars(&slices);
    let memory = SlotClaim::Sum {
        parts: (MEM_LO..=MEM_TOP)
            .map(|c| SumPart {
                offset: layout.placements[c].offset,
                low_point: r_mem.to_vec(),
                scale: ch.gamma_mem[1 + c - MEM_LO] * below_mem,
            })
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
            instance: chi[..taus[tables::KECCAK_TABLE]].to_vec(),
            lanes,
        },
    }
}

/// `Π_h (1 + x_h·(1 + g^{2^h}))`, the multilinear extension of `j ↦ g^j`.
fn index_mle(point: &[F192]) -> F192 {
    let mut g = F64::G;
    let mut out = F192::ONE;
    for &x in point {
        out *= F192::ONE + x.mul_base(F64::ONE + g);
        g *= g;
    }
    out
}

/// Test-only ways for the prover to cheat past its witness, for the negative tests: the word the index side scatters
/// for one row of one pointer column, and a change to the memory term it sends.
#[cfg(test)]
pub(crate) mod cheat {
    use primitives::field::F192;
    use std::cell::Cell;

    thread_local! {
        /// `(table, local column, row, word)`: the index side scatters `word` for that row instead of its own.
        pub(crate) static INDEX_WORD: Cell<Option<(usize, usize, usize, u64)>> = const { Cell::new(None) };
        /// Added to `M_1` before it is sent.
        pub(crate) static MEMORY_TERM: Cell<F192> = const { Cell::new(F192::ZERO) };
    }
}

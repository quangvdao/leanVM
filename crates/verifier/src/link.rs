//! Hotlink, the verifier's side: one-hot pointer links checked by one sum-check, in place of the bus.
//!
//! Every row of an instruction table names its cycle, the memory cells it touches and its instruction by
//! pointers, each committed one-hot in one `K` word. One sum-check runs over the index `j` (the index side), the
//! rows `u` and the pointer columns `q` (the row side). The index side checks the three links against the scattered
//! pointers: the state chain through the virtual spine `S` and the coverage count `C`, memory and the program by
//! agreement. The row side checks that the scattered arrays are the pointers the rows commit, and that every
//! executed pointer is one-hot. The one-hot check is the mirror test of Emile Wambsgans; one-hot addressing follows
//! Shout (Setty and Thaler); the cycle pointer's shift by one uses the carry matrices of Binary Jolt's precompile
//! link. The prover's side is `leanvm_core::link`.

use crate::field::{E, ONE, ZERO};
use crate::layout::{Access, BUS_BITS, GLOBAL_COLUMN_BASES, K_BITS, Layout, N_TABLES, gpow};
use crate::multilinear::{eq_eval, eq_kernel, index_mle, log2_ceil, padding_weight};
use crate::ring_switch::{RingClaim, RingPiece};
use crate::transcript::{Transcript, sumcheck};
use crate::{VerifierError, require};

/// The widest pointer one word holds.
pub const MAX_WIDTH: usize = 24;

/// How a `width`-bit pointer splits into chunks: the fewest chunks whose one-hot blocks fit 64 bits, widths
/// differing by at most one, the wider first. Chunk `i` covers index bits from `index_shifts[i]` and word bits
/// from `word_shifts[i]`.
#[derive(Clone, Debug)]
pub struct Chunks {
    pub width: usize,
    pub widths: Vec<usize>,
    index_shifts: Vec<usize>,
    word_shifts: Vec<usize>,
}

impl Chunks {
    pub fn new(width: usize) -> Result<Self, VerifierError> {
        require((1..=MAX_WIDTH).contains(&width), VerifierError::AnnouncedTableSizes)?;
        let widths = (1..=width)
            .map(|d| (0..d).map(|i| width / d + usize::from(i < width % d)).collect::<Vec<_>>())
            .find(|widths| widths.iter().map(|&w| 1usize << w).sum::<usize>() <= 64)
            .expect("one chunk per bit fits");
        let prefix = |f: &dyn Fn(usize) -> usize| {
            widths
                .iter()
                .scan(0, |at, &w| {
                    let start = *at;
                    *at += f(w);
                    Some(start)
                })
                .collect()
        };
        Ok(Self {
            width,
            index_shifts: prefix(&|w| w),
            word_shifts: prefix(&|w| 1 << w),
            widths,
        })
    }

    pub fn d(&self) -> usize {
        self.widths.len()
    }

    pub fn max_width(&self) -> usize {
        self.widths[0]
    }

    fn part<'a>(&self, point: &'a [E], i: usize) -> &'a [E] {
        &point[self.index_shifts[i]..self.index_shifts[i] + self.widths[i]]
    }
}

/// The functionals a pointer word is read with, per chunk: `D_i` at the index point `r`, at the certificate point
/// `ρ` and at `1 + ρ`, and the shift's `A_i` and `B_i` at `r`. Each is a table over the chunk's values.
pub struct Functionals {
    tables: [Vec<Vec<E>>; 5],
    /// `π_{w_i}(ρ) = Π_h ρ_h·(1 + ρ_h)` per chunk.
    pub mirrors: Vec<E>,
}

impl Functionals {
    pub fn new(chunks: &Chunks, r: &[E], rho: &[E]) -> Self {
        let eq: Vec<Vec<E>> = (0..chunks.d()).map(|i| eq_kernel(chunks.part(r, i))).collect();
        let bar: Vec<E> = rho.iter().map(|&x| ONE + x).collect();
        let cert = chunks.widths.iter().map(|&w| eq_kernel(&rho[..w])).collect();
        let cert_bar = chunks.widths.iter().map(|&w| eq_kernel(&bar[..w])).collect();
        // A_i[v] = eq(r_i, v + 1) below the top value, B_i[top] = eq(r_i, 0): adding one to a chunk at `v`.
        let low = eq
            .iter()
            .map(|t| (0..t.len()).map(|v| t.get(v + 1).copied().unwrap_or(ZERO)).collect())
            .collect();
        let carry = eq
            .iter()
            .map(|t| (0..t.len()).map(|v| if v + 1 == t.len() { t[0] } else { ZERO }).collect())
            .collect();
        let mirrors = chunks
            .widths
            .iter()
            .map(|&w| rho[..w].iter().map(|&x| x * (ONE + x)).fold(ONE, |a, b| a * b))
            .collect();
        Self {
            tables: [eq, cert, cert_bar, low, carry],
            mirrors,
        }
    }

    /// The `5d` values of a word whose 64 bit-slices are `slices`: each functional is linear in the bits.
    pub fn values(&self, chunks: &Chunks, slices: &[E]) -> Vec<E> {
        self.tables
            .iter()
            .flat_map(|tables| {
                (0..chunks.d()).map(move |i| {
                    let start = chunks.word_shifts[i];
                    E::sum(tables[i].iter().zip(&slices[start..]).map(|(&e, &s)| e * s))
                })
            })
            .collect()
    }

    /// `c·Φ_0 + c⁺·Φ_1 + Σ_i e_i·(D_i(ρ)·D_i(1 + ρ) + π_i)` from a word's `5d` values: the pointer at `r`, the
    /// pointer plus one at `r` (the carry recursion over the chunks), and the mirror certificate.
    pub fn summand(&self, coef: E, plus: E, values: &[E], certs: &[E]) -> E {
        let d = self.mirrors.len();
        let pointer = values[..d].iter().fold(ONE, |a, &b| a * b);
        let (mut x, mut y) = (values[3 * d], values[4 * d]);
        for i in 1..d {
            (x, y) = (x * values[i] + y * values[3 * d + i], y * values[4 * d + i]);
        }
        let cert = E::sum((0..d).map(|i| certs[i] * (values[d + i] * values[2 * d + i] + self.mirrors[i])));
        coef * pointer + plus * x + cert
    }
}

fn product(values: impl IntoIterator<Item = E>) -> E {
    values.into_iter().fold(ONE, |acc, value| acc * value)
}

/// `LT_n(point)`, the multilinear extension of `[j < n]`, for `n ≤ 2^len(point)`.
pub fn less_than(point: &[E], n: u64) -> E {
    ONE + padding_weight(point, n)
}

/// `eq(point, j)` at the integer `j`, LSB-first.
pub fn eq_at(point: &[E], j: u64) -> E {
    point
        .iter()
        .enumerate()
        .fold(ONE, |acc, (h, &x)| acc * if j >> h & 1 == 1 { x } else { ONE + x })
}

/// `Σ_{g < n} Π_h e_h(g_h)` with `e_h(0) = low[h]` and `e_h(1) = high[h]`, for `n ≤ 2^len`: a walk over the
/// binary digits of `n`, from the top.
fn prefix_sum(low: &[E], high: &[E], n: u64) -> E {
    let len = low.len();
    let mut full = Vec::with_capacity(len + 1);
    full.push(ONE);
    for h in 0..len {
        full.push(full[h] * (low[h] + high[h]));
    }
    if n >= 1 << len {
        return full[len];
    }
    let mut total = ZERO;
    let mut prefix = ONE;
    for h in (0..len).rev() {
        if n >> h & 1 == 1 {
            total += prefix * low[h] * full[h];
            prefix *= high[h];
        } else {
            prefix *= low[h];
        }
    }
    total
}

/// `ω_F(χ) = θ_F·Σ_{u < n} eq(τ, u)·eq(χ, u)`, the row weight at the table's row point.
fn omega_at(theta: E, tau: &[E], chi: &[E], n: u64) -> E {
    let low: Vec<E> = tau.iter().zip(chi).map(|(&t, &x)| (ONE + t) * (ONE + x)).collect();
    let high: Vec<E> = tau.iter().zip(chi).map(|(&t, &x)| t * x).collect();
    theta * prefix_sum(&low, &high, n)
}

/// Coefficients of a row round and of a pointer-column round.
fn round_coefficients(d: usize) -> (usize, usize) {
    ((d + 2).max(4) + 1, (d + 1).max(3) + 1)
}

/// What the link leaves on the commitment, and where it read the program.
pub struct LinkEnd {
    /// The memory term at `r`: the memory's low point, each limb's scale, and the value.
    pub memory: (Vec<E>, [E; 3], E),
    /// Per table, its row point and its non-pointer columns' values there, as local column and value.
    pub columns: Vec<(Vec<E>, Vec<(usize, E)>)>,
    /// The pointer columns' 64 bit-slices, one piece per pointer column.
    pub ring: RingClaim,
    pub bytecode_point: Vec<E>,
}

/// The pointer width every execution of these sizes takes: the cycle count's bits, the memory's and the program's.
pub fn pointer_width(log_memory: usize, log_bytecode: usize, cycles: u64) -> usize {
    let t = (u64::BITS - cycles.leading_zeros()) as usize;
    t.max(log_memory).max(log_bytecode).max(1)
}

/// Verify the links of the execution `layout` announces, from right after the commitment.
pub fn verify_link(layout: &Layout, transcript: &mut Transcript) -> Result<LinkEnd, VerifierError> {
    let tables = &layout.tables;
    let heights = &layout.table_log_heights;
    let rows = &layout.table_rows;
    let cycles: u64 = rows.iter().sum();
    let chunks = Chunks::new(pointer_width(layout.log_memory, layout.log_bytecode, cycles))?;
    let (width, d) = (chunks.width, chunks.d());
    let pointers: Vec<Vec<usize>> = tables.iter().map(|t| t.links.pointers()).collect();
    let n_pointers: usize = pointers.iter().map(Vec::len).sum();

    // The challenges.
    let theta = transcript.samples(N_TABLES); let theta = vec![theta[0]; N_TABLES];
    let tau = transcript.samples(heights.iter().copied().max().unwrap_or(0));
    let gamma_state = transcript.samples(2);
    let gamma_mem = transcript.samples(5);
    let mu: Vec<Vec<E>> = tables.iter().map(|t| transcript.samples(t.links.memory.len())).collect();
    let alpha = transcript.samples(BUS_BITS);
    let zeta = transcript.samples(width);
    let [eta_start, eta_cover, eta_mem, eta_bc] = [0; 4].map(|_| transcript.sample());
    let rho = transcript.samples(chunks.max_width());
    let eps: Vec<Vec<E>> = (0..n_pointers).map(|_| transcript.samples(d)).collect();
    let slot = eq_kernel(&alpha);
    let s_init = gamma_state[0] + gamma_state[1];
    let s_fin = gamma_state[0] * gpow((1 << layout.log_bytecode) - 1) + gamma_state[1];

    // The index side, then the values at `r` and the fresh coefficients.
    let claim = eta_cover * less_than(&zeta, cycles) + eta_start * s_init;
    let (r, claim) = sumcheck(transcript, claim, 3, &vec![None; width])?;
    let sent = transcript.next_scalars(3)?;
    let (w_plus, spine, m1) = (sent[0], sent[1], sent[2]);
    let [plus, spine_coef, cert, jump_0, jump_1] = [0; 5].map(|_| transcript.sample());
    let below = |k: usize| product(r[k.min(width)..].iter().map(|&x| ONE + x));
    let y_mem = gamma_mem[0] * index_mle(&r) + gamma_mem[4] * below(layout.log_memory) + m1;
    let bytecode_point = [&r[..layout.log_bytecode], &alpha[..]].concat();
    let program = if transcript.checked() {
        layout.bytecode.evaluate(&bytecode_point)?
    } else {
        ZERO
    };
    let x_bc = slot[1] * index_mle(&r) + below(layout.log_bytecode) * (slot[2] + program);
    let eq0_r = below(0);
    let claim = claim
        + w_plus * spine
        + eta_start * eq0_r * spine
        + plus * w_plus
        + spine_coef * (spine + eq_at(&r, cycles) * s_fin);

    // The row side.
    let (row_count, column_count) = round_coefficients(d);
    let n_rows = heights.iter().copied().max().unwrap_or(0);
    let (chi, claim) = sumcheck(transcript, claim, row_count, &vec![None; n_rows])?;
    let q_vars = log2_ceil(n_pointers as u64);
    let (q_point, claim) = sumcheck(transcript, claim, column_count, &vec![None; q_vars])?;
    let eq_q = eq_kernel(&q_point);

    // The values at the row point, and what they make of the summand there.
    let mut columns = Vec::with_capacity(N_TABLES);
    let (mut coef, mut shifted, mut certs, mut base) = (ZERO, ZERO, vec![ZERO; d], ZERO);
    let mut pieces = Vec::with_capacity(n_pointers);
    let mut q = 0;
    for (t, table) in tables.iter().enumerate() {
        let point = chi[..heights[t]].to_vec();
        let corner = product(chi[heights[t]..].iter().map(|&x| ONE + x));
        let mut values = vec![ZERO; table.width()];
        let mut sent = Vec::new();
        for c in (0..table.width()).filter(|c| !pointers[t].contains(c)) {
            values[c] = transcript.next_scalar()?;
            sent.push((c, values[c]));
        }
        let links = &table.links;
        let omega = omega_at(theta[t], &tau[..heights[t]], &point, rows[t]);
        let act = less_than(&point, rows[t]);
        let state = links.state();
        let s_gamma = gamma_state[0] * values[state.pc] + gamma_state[1] * values[state.fp];

        // The base terms: the payload under ω, and JUMP's identities on every row.
        let access_value =
            |a: &Access| gamma_mem[0] * a.address.evaluate(&values) + gamma_mem[4] + E::sum(
                a.values.iter().zip(&gamma_mem[1..4]).map(|(v, &g)| g * v.evaluate(&values)),
            );
        let bc = links.bytecode_link();
        let r_bc = slot[1] * values[bc.pc]
            + slot[2]
            + slot[3] * gpow(bc.opcode as u64)
            + E::sum(bc.operands.iter().enumerate().map(|(i, f)| slot[4 + i] * f.evaluate(&values)));
        let next = gamma_state[0] * state.next[0].evaluate(&values) + gamma_state[1] * state.next[1].evaluate(&values);
        let memory = E::sum(links.memory.iter().zip(&mu[t]).map(|(a, &m)| {
            let gate = a.gate.map_or(ONE, |g| values[g]);
            m * gate * access_value(a)
        }));
        let mut table_base = omega * eq0_r * (next + eta_mem * memory + eta_bc * r_bc);
        if table.n_constraints() > 0 {
            let identities = (table.constraints)(&values);
            let eq_tau = eq_eval(&tau[..heights[t]], &point);
            table_base += eq_tau * (jump_0 * identities[0] + jump_1 * identities[1]);
        }
        base += corner * table_base;

        // The pointer columns' coefficients.
        for (k, &pointer) in pointers[t].iter().enumerate() {
            let e = eq_q[q] * corner;
            match k {
                0 => {
                    coef += e * act * (eta_cover * eq_eval(&zeta, &r) + spine_coef * s_gamma);
                    shifted += e * omega * plus;
                }
                1 => coef += e * omega * eta_bc * x_bc,
                _ => {
                    let access = &links.memory[k - 2];
                    let gate = access.gate.map_or(ONE, |g| values[g]);
                    coef += e * omega * gate * eta_mem * mu[t][k - 2] * y_mem;
                }
            }
            for (c, &epsilon) in certs.iter_mut().zip(&eps[q]) {
                *c += e * omega * cert * epsilon;
            }
            pieces.push(RingPiece {
                placement: layout.placements[GLOBAL_COLUMN_BASES[t] + pointer],
                point: point.clone(),
                scale: eq_q[q],
            });
            q += 1;
        }
        columns.push((point, sent));
    }
    let slices = transcript.next_scalars(K_BITS)?;
    if transcript.checked() {
        let functionals = Functionals::new(&chunks, &r, &rho);
        let values = functionals.values(&chunks, &slices);
        let eq0_q = product(q_point.iter().map(|&x| ONE + x));
        let last = functionals.summand(coef, shifted, &values, &certs) + eq0_q * base;
        require(last == claim, VerifierError::LinkTerminal)?;
    }
    let below_mem = below(layout.log_memory);
    Ok(LinkEnd {
        memory: (
            r[..layout.log_memory].to_vec(),
            [gamma_mem[1] * below_mem, gamma_mem[2] * below_mem, gamma_mem[3] * below_mem],
            m1,
        ),
        columns,
        ring: RingClaim { pieces, slices },
        bytecode_point,
    })
}

// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// CREDIT: https://github.com/bcc-research/bolt-rs, MIT.
// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Ported from bolt-rs (https://github.com/bcc-research/bolt-rs,
// `whir_recursive.rs`).

//! Field-independent configuration and soundness analysis for WHIR.
//!
//! Source of truth: `doc/leanvm/body/b-polynomial-commitment-scheme.tex`, annex B of
//! the spec ("The polynomial commitment scheme"), Theorem `thm:rbr`. Its
//! per-verifier-message error table maps onto the per-level checks in
//! [`WhirSecurityConfig::validate`]:
//!
//! - batching challenges -> `johnson_algebraic_bits` (one challenge per level,
//!   powers of it over the level's claim list, as in the doc),
//! - fold challenge `s_j` -> `(E + 2 L)/|F|`, checked as one combined bound by
//!   `capacity_mca_fold_bits`,
//! - OOD challenge -> `paper_ood_bits`,
//! - query message -> `(1 - gamma)^t`, plus [`QUERY_GRINDING_BITS`].
//!
//! Round-by-round (RBR) soundness means every entry individually clears
//! [`SECURITY_BITS`]: the Fiat--Shamir error per random-oracle query is the
//! MAX of the entries, not their sum.

// ===================================================================
// Config
// ===================================================================

// The production WHIR configuration: rate-1/2 Johnson list decoding with
// OOD binding and 128-bit round-by-round soundness over F192.

/// Round-by-round soundness target (bits): every verifier-challenge transition
/// must have conditional failure probability at most `2^-SECURITY_BITS`.
pub const SECURITY_BITS: usize = 128;

/// L0 code rate index: `rho_0 = 2^-LOG_INV_RATE_0` (rate 1/2).
pub const LOG_INV_RATE_0: usize = 1;

/// CLI-selectable L0 rates are `2^-r` for `r = 1, 2, 3, 4`.
pub const MIN_LOG_INV_RATE: usize = 1;
pub const MAX_LOG_INV_RATE: usize = 4;

/// Validate a production WHIR inverse-rate logarithm.
pub fn validate_log_inv_rate(log_inv_rate: usize) -> Result<(), String> {
    if !(MIN_LOG_INV_RATE..=MAX_LOG_INV_RATE).contains(&log_inv_rate) {
        return Err(format!(
            "log_inv_rate must be in {MIN_LOG_INV_RATE}..={MAX_LOG_INV_RATE}, got {log_inv_rate}"
        ));
    }
    Ok(())
}

/// Per-level query-phase proof-of-work budget. These bits are ground after the
/// level commitment and before its query positions are sampled, so the query
/// count only needs to close the remaining `SECURITY_BITS - 17` bits.
pub const QUERY_GRINDING_BITS: usize = 17;

pub const INITIAL_FOLDING_FACTOR: usize = 6;
pub const SUBSEQUENT_FOLDING_FACTOR: usize = 4;

/// Logarithmic reduction of the total Reed--Solomon domain after the initial
/// fold. With the production six-variable initial fold, `3` changes the
/// inverse-rate logarithm by `6 - 3 = 3` at the first recursive level.
pub const RS_DOMAIN_INITIAL_REDUCTION_FACTOR: usize = 3;

/// After each subsequent fold, shrink the total Reed--Solomon domain by one
/// bit. This mirrors WHIR's recursive-domain schedule; unlike the initial
/// reduction, it is deliberately fixed rather than a tuning parameter.
const RS_DOMAIN_SUBSEQUENT_REDUCTION_FACTOR: usize = 1;

const _: () = assert!(RS_DOMAIN_INITIAL_REDUCTION_FACTOR <= INITIAL_FOLDING_FACTOR);
const _: () = assert!(RS_DOMAIN_SUBSEQUENT_REDUCTION_FACTOR <= SUBSEQUENT_FOLDING_FACTOR);

/// Folding stops once at most this many variables remain: the residual
/// polynomial (`yr`, at most `2^RESIDUAL_MAX_LOG` coefficients) is sent in
/// clear instead of committed and folded further.
pub const RESIDUAL_MAX_LOG: usize = 5;

// The recursion guest rotates the terminal point left by the lane fold to index it
// by witness coordinate, and the residual segment is what the last lane challenges
// rotate past, so the residual may never be longer than that fold
// (`rec_aggregation`'s placeholder emitter re-checks this per derived candidate).
const _: () = assert!(RESIDUAL_MAX_LOG <= INITIAL_FOLDING_FACTOR);

/// Shape plus per-level soundness parameters for one WHIR opening. Prover
/// and verifier read exactly the same numbers, hence the single struct and the
/// [`VerifierConfig`] alias.
#[derive(Clone, Debug)]
pub struct ProverConfig {
    pub log_inv_rates: Vec<usize>,
    pub level_steps: usize,
    pub initial_k: usize,
    pub level_ks: Vec<usize>,
    /// Per-level query counts (L0, L1, ..., L_r). Length = level_steps + 1.
    /// [`WhirSecurityConfig::derive_config_with_log_inv_rate`] fills these
    /// from the per-level soundness analysis.
    pub queries: Vec<usize>,
    /// Per-level **query-phase** PoW grinding bits (L0, L1, ..., L_r), ground
    /// post-commit/pre-queries. Length = level_steps + 1. Each bit here
    /// substitutes for ~1/log₂(1/(1−γ)) queries at that level.
    pub grinding_bits: Vec<usize>,
    /// Per-commit-level out-of-domain samples (L0, ..., L_r), taken right
    /// after the level's Merkle root enters the transcript. `[0]` must be 0:
    /// L0 is bound by the opening's own (post-commit, random-point)
    /// evaluation claim. Length = level_steps + 1.
    pub ood_samples: Vec<usize>,
}

pub type VerifierConfig = ProverConfig;

/// The per-level shape table a [`VerifierConfig`] implies for a
/// `log_n`-variable opening: the numbers every consumer of the multilevel
/// protocol (the verifier itself, recursion harnesses) otherwise re-derives.
#[derive(Clone, Debug)]
pub struct LevelShapes {
    /// Level count (`level_steps + 1`).
    pub levels: usize,
    /// Fold count per level: `initial_k` then `level_ks`.
    pub ks: Vec<usize>,
    /// Log message columns entering each level's fold (`log_n - initial_k`,
    /// then descending by each level's `k`).
    pub log_msg_cols: Vec<usize>,
    /// Committed block length per level (`msg_cols * inv_rate`).
    pub block_len: Vec<usize>,
    /// The residual cube dimension left after every fold.
    pub yr_log_n: usize,
}

impl ProverConfig {
    /// See [`LevelShapes`].
    pub fn level_shapes(&self, log_n: usize) -> LevelShapes {
        let r = self.level_steps;
        let ks: Vec<usize> = std::iter::once(self.initial_k)
            .chain(self.level_ks.iter().copied())
            .collect();
        let mut log_msg_cols = vec![log_n - self.initial_k];
        for i in 0..r {
            log_msg_cols.push(log_msg_cols[i] - self.level_ks[i]);
        }
        let block_len: Vec<usize> = (0..=r)
            .map(|i| 1usize << (log_msg_cols[i] + self.log_inv_rates[i]))
            .collect();
        LevelShapes {
            levels: r + 1,
            ks,
            yr_log_n: *log_msg_cols.last().unwrap(),
            log_msg_cols,
            block_len,
        }
    }
}

/// Soundness (in bits) the query phase must close on its own at every level
/// (the "100 bits from queries always" policy).
#[cfg(test)]
const UDR_TARGET_BITS: f64 = 100.0;

/// Number of queries for 100-bit soundness in the **unique-decoding regime**
/// at rate `2^(-log_inv_rate)`: `γ = δ/2 = (1−ρ)/2`, per-query soundness
/// `log₂(1/(1−γ))` (see [`udr_per_query_bits`]). Within the unique decoding
/// radius the prover is pinned to a single codeword, so there is no list and
/// no union-bound term: queries close the full target by themselves.
/// Per-query soundness saturates below 1 bit (`γ < 1/2`), so slimmer codes
/// bottom out near `UDR_TARGET_BITS` queries: 243 at rate 1/2, 148 at 1/4,
/// 121 at 1/8, 110 at 1/16, 105 at 1/32.
#[cfg(test)]
pub fn udr_queries(log_inv_rate: usize) -> usize {
    assert!(log_inv_rate > 0, "log_inv_rate=0 (rate 1) has no soundness");
    let per_q = udr_per_query_bits_asymptotic(log_inv_rate);
    (UDR_TARGET_BITS / per_q).ceil() as usize
}

/// Build an ad-hoc WHIR config from the raw PCS shape, WITHOUT the
/// per-level soundness derivation of
/// [`WhirSecurityConfig::derive_config_with_log_inv_rate`].
/// `log_n` is the packed-witness log size (= `m - LOG_PACKING`).
///
/// Strategy: 3-bit recursive folds (`k_i = 3`) with **decreasing rate** (one
/// rate step per level) until the residual is small (`≤ 5` bits), asserting
/// `block_len ≥ udr_queries(rate)` at every level. Returns `Err` when no
/// feasible config exists (e.g. `log_n` too small for the chosen rate).
///
/// Test-support only: the small F64 PCS tests exercise sizes below the
/// production derivation's feasibility floor, where they fall back to this
/// shape. Production callers use the audited, per-level-sound path.
#[cfg(test)]
pub fn default_config(log_n: usize, log_batch_size: usize, log_inv_rate: usize) -> Result<ProverConfig, String> {
    let initial_k = log_batch_size;
    if log_n > initial_k && (1usize << (log_n - initial_k + log_inv_rate)) < udr_queries(log_inv_rate) {
        return Err("L0 block_len < udr_queries: log_n too small for chosen rate".into());
    }
    // Smallest rate strictly above the previous one that still fits the level's
    // query count inside its block length.
    let shape = derive_ladder(log_n, initial_k, log_inv_rate, |rate_running, _fold, cols_next| {
        let mut next_rate = rate_running + 1;
        while (1usize << (cols_next + next_rate)) < udr_queries(next_rate) {
            next_rate += 1;
            if next_rate > 20 {
                return Err("could not find feasible recursive rate (level too deep)".into());
            }
        }
        Ok(next_rate)
    })?;

    let n_levels = shape.log_inv_rates.len();
    Ok(ProverConfig {
        queries: shape.log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
        log_inv_rates: shape.log_inv_rates,
        level_steps: shape.k_levels.len() - 1,
        initial_k,
        level_ks: shape.k_levels[1..].to_vec(),
        grinding_bits: vec![0usize; n_levels],
        ood_samples: vec![0usize; n_levels],
    })
}

/// Shared config for a `2^log_n`-word witness, preferring the production profile at [`LOG_INV_RATE_0`] and falling back to [`default_config`] below its feasibility floor.
#[cfg(test)]
pub(crate) fn test_config_for(log_n: usize) -> ProverConfig {
    if let Ok(config) = crate::whir::config_for_rate(log_n, LOG_INV_RATE_0) {
        return config;
    }
    for log_batch_size in (1..=5).rev() {
        for log_inv_rate in 1..=4 {
            if let Ok(config) = default_config(log_n, log_batch_size, log_inv_rate) {
                return config;
            }
        }
    }
    panic!("no feasible whir config at log_n = {log_n}");
}

/// Level-ladder shape: per-level dims (index 0 = L0) plus the residual.
struct LadderShape {
    log_inv_rates: Vec<usize>,
    log_msg_cols: Vec<usize>,
    k_levels: Vec<usize>,
    yr_log_n: usize,
}

/// Descend the level ladder, folding [`SUBSEQUENT_FOLDING_FACTOR`] variables per
/// level until at most [`RESIDUAL_MAX_LOG`] remain. `next_rate` picks each new
/// level's inverse-rate logarithm from `(previous rate, fold just taken, message
/// dimension the new level carries)`, which is the only thing that separates the
/// production ladder from the test-support one.
fn derive_ladder(
    log_n: usize,
    initial_k: usize,
    log_inv_rate: usize,
    mut next_rate: impl FnMut(usize, usize, usize) -> Result<usize, String>,
) -> Result<LadderShape, String> {
    if log_n <= initial_k {
        return Err("log_n must be > initial_k".into());
    }
    let mut shape = LadderShape {
        log_inv_rates: vec![log_inv_rate],
        log_msg_cols: vec![log_n - initial_k],
        k_levels: vec![initial_k],
        yr_log_n: 0,
    };
    let mut n_running = log_n - initial_k;
    let mut rate_running = log_inv_rate;
    let mut fold_running = initial_k;
    while n_running > RESIDUAL_MAX_LOG {
        let k = SUBSEQUENT_FOLDING_FACTOR.min(n_running);
        let log_msg_cols_next = n_running - k;
        let rate = next_rate(rate_running, fold_running, log_msg_cols_next)?;
        shape.log_inv_rates.push(rate);
        shape.log_msg_cols.push(log_msg_cols_next);
        shape.k_levels.push(k);
        n_running -= k;
        rate_running = rate;
        fold_running = k;
    }
    if shape.k_levels.len() < 2 {
        return Err("log_n too small: needs at least 2 fold levels".into());
    }
    shape.yr_log_n = n_running;
    Ok(shape)
}

/// Production ladder: the total RS domain loses
/// [`RS_DOMAIN_INITIAL_REDUCTION_FACTOR`] bits after the initial fold, then
/// exactly one bit per subsequent fold, so a fold of `k` variables raises the
/// inverse-rate logarithm by `k - reduction`.
fn derive_ladder_shape(log_n: usize, initial_k: usize, log_inv_rate: usize) -> Result<LadderShape, String> {
    let mut domain_reduction = RS_DOMAIN_INITIAL_REDUCTION_FACTOR;
    derive_ladder(log_n, initial_k, log_inv_rate, |rate_running, fold_running, _cols| {
        let rate_increase = fold_running.checked_sub(domain_reduction).ok_or_else(|| {
            format!("folding factor {fold_running} is smaller than RS domain reduction {domain_reduction}")
        })?;
        domain_reduction = RS_DOMAIN_SUBSEQUENT_REDUCTION_FACTOR;
        Ok(rate_running + rate_increase)
    })
}

// ===================================================================
// Security configuration schema
// ===================================================================
//
// Auditable, per-level spec for a WHIR instance: query count, grinding
// bits, slack-from-Johnson, and the proximity-gap analysis the parameters were
// derived under.
//
// That analysis is always the Johnson radius with explicit slack `eta`
// (gamma = (1 - sqrt(rho)) - eta) WITH out-of-domain binding (`doc/leanvm/body/b-polynomial-commitment-scheme.tex`,
// Thm `thm:rbr`). The capacity MCA theorem (`thm:mca-johnson`) gives the
// proximity-gap exceptional set `E = O(n / eta^3)`. The parameter search
// checks the combined fold row `(E + 2L)/|F|` exactly. Binding to a
// single codeword of the (Johnson-bounded) interleaved list is via
// `ood_samples` explicit multilinear OOD evaluations, except at L0, where the
// opening's own post-commit random evaluation claim plays the OOD role (union
// over the list, `L*mu/q`), so `ood_samples = 0`. Plain Johnson without OOD
// binding would be unsound at these parameters: the query phase would pay a
// union bound over the interleaved list (19 to 52 bits here) that the query
// counts do not include.
//
// Grinding always lands after the level's Merkle root is observed and before
// its query positions are sampled, the standard FRI/STARK placement.

/// Parameters for a single level in the multilevel WHIR ladder.
/// L0 = the upstream `pcs::commit` output (reused, not re-committed);
/// L1 .. L_{r−1} are the level commits; the final residual `yr` block
/// is described separately in [`FinalBlockConfig`].
#[derive(Clone, Debug)]
pub struct WhirLevelConfig {
    /// PCS rate at this level: codeword expansion factor = 2^log_inv_rate.
    pub log_inv_rate: usize,
    /// Message dimension at this level (log of the number of field columns in
    /// the codeword). `log_msg_cols + log_inv_rate = log_2(block_len)`.
    pub log_msg_cols: usize,
    /// Log of lane width per Merkle leaf at this level. For L0 = `initial_k`;
    /// for L_i (i ≥ 1) = the previous level's `k`.
    pub log_num_interleaved: usize,
    /// Number of sumcheck folds taken at this level. For L0 = `initial_k`
    /// (the lane fold); for L_i (i ≥ 1) = the level fold k_{i−1}.
    pub k: usize,
    /// Slack from the Johnson radius: γ = (1 − √ρ) − η.
    pub eta: f64,
    /// Integer agreement threshold `A = ceil((sqrt(D/n) + eta) n)` used by
    /// the capacity MCA theorem.
    pub agreement: usize,
    /// Interpolation multiplicity in the capacity MCA certificate.
    pub interpolation_m: usize,
    /// Jet-degree cutoff in the capacity MCA certificate.
    pub jet_degree: usize,
    /// Auxiliary interpolation height in the capacity MCA certificate.
    pub interpolation_height: usize,
    /// Exact Johnson list bound used by fold, OOD, and algebraic checks.
    pub list_bound: usize,
    /// Whether this is the theorem's closed parameter choice or an explicitly
    /// checked finite interpolation certificate.
    pub certificate_variant: JohnsonCertificateVariant,
    /// Number of codeword position queries opened at this level (the FRI
    /// query phase). Bounds the per-query soundness term `(1−γ)^Q`.
    pub queries: usize,
    /// **Query-phase** PoW grinding bits, ground post-commit/pre-queries.
    /// Each bit substitutes for ~1/log₂(1/(1−γ)) queries at this level.
    pub grinding_bits: usize,
    /// Out-of-domain samples taken right after this level's commit enters
    /// the transcript. Each binds the prover to a single codeword of the
    /// interleaved list via a multilinear evaluation claim.
    /// Must be 0 at L0 (bound by the opening's own post-commit evaluation
    /// claim) and ≥ 1 at deeper levels.
    pub ood_samples: usize,
    /// Security target this level guarantees, post-grinding.
    pub target_security_bits: usize,
}

/// Provenance of a capacity MCA interpolation certificate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JohnsonCertificateVariant {
    /// The all-characteristic closed parameter choice from `thm:mca-johnson`.
    Closed,
    /// A finite parameter choice checked directly from the interpolation
    /// surplus inequality.
    Finite,
}

/// Descriptor for the final-residual block (`yr`) sent in the clear at the
/// end of the last fold level. It has no commit and no queries, so the
/// only meaningful parameter is its dimension.
#[derive(Clone, Debug)]
pub struct FinalBlockConfig {
    /// `log_2(|yr|)`: number of extension-field values sent in the clear. The
    /// last fold level's sumcheck stops at this dim instead of folding to 1.
    pub yr_log_n: usize,
}

/// Complete security spec for one WHIR instance, covering a single
/// `(hash, m)` pair.
///
/// **Validation invariants** (checked by [`Self::validate`]):
/// 1. `initial_k + Σ levels[1..].k + final_block.yr_log_n == log_n`.
/// 2. Each level's combined fold-challenge bits reach `target_security_bits`.
/// 3. Each level's query soundness reaches `target_security_bits −
///    grinding_bits` (queries cover what grinding doesn't).
/// 4. `eta` is finite and inside the Johnson range for the level's rate.
/// 5. `log_msg_cols`, `log_num_interleaved`, `k` match the
///    level-shape constraint (each level's input dim equals the
///    previous level's `log_msg_cols`).
#[derive(Clone, Debug)]
pub struct WhirSecurityConfig {
    /// Block-encoder log size: m = log₂(witness bit count).
    pub m: usize,
    /// Committed-witness log dimension.
    pub log_n: usize,
    /// L0 lane fold. Must equal the upstream `PcsParams::log_batch_size` so
    /// the L0 commit can be reused without re-committing.
    pub initial_k: usize,
    /// Round-by-round security target (bits): `validate()` asserts that every
    /// error term associated with a verifier challenge clears at least this
    /// much. This is an RBR target, not a claim that the sum of all interactive
    /// failure probabilities is bounded by `2^-target_security_bits`.
    pub target_security_bits: usize,
    /// Identifier of the proximity-gap analysis used. Self-documents which
    /// theorem the per-level parameters were derived from.
    pub analysis_version: String,
    /// Per-level parameters, in order L0, L1, L2, ....
    pub levels: Vec<WhirLevelConfig>,
    /// Final residual block descriptor.
    pub final_block: FinalBlockConfig,
}

/// Extension-field size used for soundness analysis: `q = 2^192`.
const ANALYSIS_LOG_Q: f64 = 192.0;

/// BCHKS25 parameter `rho = k/n` for an RS code of dimension `k + 1`.
/// Our message has `2^log_msg_cols` coefficients (degree strictly below that
/// value), so `k = 2^log_msg_cols - 1`. This differs perceptibly from the
/// nominal code rate at the small recursive levels.
fn reduced_rate(log_inv_rate: usize, log_msg_cols: usize) -> f64 {
    let dimension = (log_msg_cols as f64).exp2();
    (dimension - 1.0) / ((log_msg_cols + log_inv_rate) as f64).exp2()
}

#[derive(Clone, Copy, Debug)]
struct CapacityMcaCertificate {
    agreement: usize,
    interpolation_m: usize,
    jet_degree: usize,
    interpolation_height: usize,
    list_bound: usize,
    variant: JohnsonCertificateVariant,
}

fn checked_product(values: &[u128], context: &str) -> Result<u128, String> {
    values.iter().try_fold(1u128, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| format!("{context}: integer overflow"))
    })
}

/// Check the finite interpolation surplus certificate behind the capacity MCA
/// theorem. This is run for both the closed recipe and the finite refinement,
/// so the hard-coded refinement is never trusted as a constant.
fn validate_interpolation_support(
    n: usize,
    degree: usize,
    agreement: usize,
    interpolation_m: usize,
    jet_degree: usize,
    height: usize,
) -> Result<(), String> {
    if degree == 0 || agreement <= degree || agreement > n || interpolation_m == 0 || jet_degree == 0 {
        return Err("invalid capacity MCA interpolation parameters".into());
    }

    let n = n as u128;
    let degree = degree as u128;
    let agreement = agreement as u128;
    let interpolation_m = interpolation_m as u128;
    let jet_degree = jet_degree as u128;
    let height = height as u128;
    let max_jet_degree = checked_product(&[interpolation_m, agreement], "interpolation support")?
        .checked_sub(1)
        .ok_or_else(|| "interpolation support: empty numerator".to_string())?
        / degree;
    if jet_degree > max_jet_degree {
        return Err(format!(
            "capacity MCA jet degree {jet_degree} exceeds interpolation limit {max_jet_degree}"
        ));
    }

    let u = jet_degree.min(interpolation_m - 1);
    let jet_triangle = checked_product(&[jet_degree, jet_degree + 1], "jet triangle")? / 2;
    let jet_square_sum = checked_product(&[jet_degree, jet_degree + 1, 2 * jet_degree + 1], "jet square sum")? / 6;
    let u_triangle = checked_product(&[u, u + 1], "equation triangle")? / 2;
    let u_square_sum = checked_product(&[u, u + 1, 2 * u + 1], "equation square sum")? / 6;

    let variables_slope = checked_product(&[jet_degree + 1, interpolation_m, agreement], "variable slope")?
        .checked_sub(checked_product(&[degree, jet_triangle], "variable slope")?)
        .ok_or_else(|| "capacity MCA variable slope is negative".to_string())?;
    let variables_moment = checked_product(&[interpolation_m, agreement, jet_triangle], "variable moment")?
        .checked_sub(checked_product(&[degree, jet_square_sum], "variable moment")?)
        .ok_or_else(|| "capacity MCA variable moment is negative".to_string())?;
    let equations_slope = checked_product(&[u + 1, interpolation_m], "equation slope")?
        .checked_sub(u_triangle)
        .ok_or_else(|| "capacity MCA equation slope is negative".to_string())?;
    let equations_moment = checked_product(&[interpolation_m, u_triangle], "equation moment")?
        .checked_sub(u_square_sum)
        .ok_or_else(|| "capacity MCA equation moment is negative".to_string())?;
    let slope_rhs = checked_product(&[n, equations_slope], "interpolation slope")?;
    let surplus_slope = variables_slope
        .checked_sub(slope_rhs)
        .filter(|surplus| *surplus > 0)
        .ok_or_else(|| "capacity MCA interpolation has no positive slope surplus".to_string())?;
    let moment_rhs = checked_product(&[n, equations_moment], "interpolation moment")?;
    let required_by_moment = variables_moment.saturating_sub(moment_rhs) / surplus_slope;
    let required_height = jet_degree.max(required_by_moment);
    if height < required_height {
        return Err(format!(
            "capacity MCA height {height} is below the certified minimum {required_height}"
        ));
    }
    Ok(())
}

fn johnson_list_bound(n: usize, degree: usize, agreement: usize, jet_degree: usize) -> Result<usize, String> {
    let n = n as u128;
    let degree = degree as u128;
    let agreement = agreement as u128;
    let denominator = checked_product(&[agreement, agreement], "Johnson list denominator")?
        .checked_sub(checked_product(&[n, degree], "Johnson list denominator")?)
        .filter(|denominator| *denominator > 0)
        .ok_or_else(|| "agreement threshold is not beyond the Johnson radius".to_string())?;
    let numerator = checked_product(&[n, agreement - degree], "Johnson list numerator")?;
    let pairwise = numerator / denominator;
    usize::try_from(pairwise.min(jet_degree as u128)).map_err(|_| "Johnson list bound does not fit usize".into())
}

fn closed_capacity_certificate(
    log_inv_rate: usize,
    log_msg_cols: usize,
    agreement: usize,
) -> Result<CapacityMcaCertificate, String> {
    let rho = reduced_rate(log_inv_rate, log_msg_cols);
    let sqrt_rho = rho.sqrt();
    let n = 1usize << (log_msg_cols + log_inv_rate);
    let degree = (1usize << log_msg_cols) - 1;
    let mut interpolation_m = ((sqrt_rho / (2.0 * (agreement as f64 / n as f64 - sqrt_rho))).ceil() as usize).max(3);
    let closed_m_holds = |m: usize| -> Result<bool, String> {
        let lhs = checked_product(
            &[4, m as u128, m as u128, agreement as u128, agreement as u128],
            "closed MCA multiplicity",
        )?;
        let twice_m_plus_one = 2 * m as u128 + 1;
        let rhs = checked_product(
            &[twice_m_plus_one, twice_m_plus_one, degree as u128, n as u128],
            "closed MCA multiplicity",
        )?;
        Ok(lhs >= rhs)
    };
    while interpolation_m > 3 && closed_m_holds(interpolation_m - 1)? {
        interpolation_m -= 1;
    }
    while !closed_m_holds(interpolation_m)? {
        interpolation_m += 1;
    }

    let twice_m_plus_one = 2 * interpolation_m as u128 + 1;
    let jet_rhs = checked_product(
        &[twice_m_plus_one, twice_m_plus_one, n as u128],
        "closed MCA jet degree",
    )?;
    let mut jet_degree = ((interpolation_m as f64 + 0.5) / sqrt_rho).ceil() as usize - 1;
    let jet_holds = |b: usize| -> Result<bool, String> {
        Ok(checked_product(&[4, b as u128, b as u128, degree as u128], "closed MCA jet degree")? < jet_rhs)
    };
    while jet_degree > 0 && !jet_holds(jet_degree)? {
        jet_degree -= 1;
    }
    while jet_holds(jet_degree + 1)? {
        jet_degree += 1;
    }
    let height_numerator = jet_rhs;
    let height_denominator = checked_product(&[12, degree as u128], "closed MCA height")?;
    let interpolation_height = usize::try_from((height_numerator - 1) / height_denominator)
        .map_err(|_| "closed MCA height does not fit usize".to_string())?;
    let list_bound = johnson_list_bound(n, degree, agreement, jet_degree)?;
    validate_interpolation_support(n, degree, agreement, interpolation_m, jet_degree, interpolation_height)?;
    Ok(CapacityMcaCertificate {
        agreement,
        interpolation_m,
        jet_degree,
        interpolation_height,
        list_bound,
        variant: JohnsonCertificateVariant::Closed,
    })
}

/// A finite certificate that reaches the query-only Johnson floor. The
/// optimizer still derives `A` and validates the certificate algebraically.
fn finite_capacity_support(n: usize, degree: usize, agreement: usize, queries: usize) -> Option<(usize, usize, usize)> {
    match (n, degree, agreement, queries) {
        (131_072, 65_535, 92_682, 222) => Some((24_954, 35_290, 606_738_001)),
        _ => None,
    }
}

fn capacity_certificate(
    log_inv_rate: usize,
    log_msg_cols: usize,
    agreement: usize,
    queries: usize,
) -> Result<CapacityMcaCertificate, String> {
    let mut certificate = closed_capacity_certificate(log_inv_rate, log_msg_cols, agreement)?;
    let n = 1usize << (log_msg_cols + log_inv_rate);
    let degree = (1usize << log_msg_cols) - 1;
    if let Some((interpolation_m, jet_degree, interpolation_height)) =
        finite_capacity_support(n, degree, agreement, queries)
    {
        validate_interpolation_support(
            n,
            degree,
            certificate.agreement,
            interpolation_m,
            jet_degree,
            interpolation_height,
        )?;
        certificate.interpolation_m = interpolation_m;
        certificate.jet_degree = jet_degree;
        certificate.interpolation_height = interpolation_height;
        certificate.list_bound = johnson_list_bound(n, degree, certificate.agreement, jet_degree)?;
        certificate.variant = JohnsonCertificateVariant::Finite;
    }
    Ok(certificate)
}

/// Return the exact numerator and denominator of the capacity MCA exceptional
/// bound
///
/// `E = (2B-1)H + (n-D)/(A-D) (B + H Psi) + (n-D-1)B`.
fn capacity_mca_exceptional_fraction(level: &WhirLevelConfig) -> Result<(u128, u128), String> {
    let n = (1usize << (level.log_msg_cols + level.log_inv_rate)) as u128;
    let degree = ((1usize << level.log_msg_cols) - 1) as u128;
    let agreement = level.agreement as u128;
    let jet_degree = level.jet_degree as u128;
    let height = level.interpolation_height as u128;
    let denominator = agreement
        .checked_sub(degree)
        .filter(|value| *value > 0)
        .ok_or_else(|| "capacity MCA agreement must exceed the degree".to_string())?;
    let psi = 1u128
        .checked_add(checked_product(
            &[2 * degree - 1, 2 * jet_degree - 1],
            "capacity MCA Psi",
        )?)
        .and_then(|value| value.checked_add(2 * jet_degree.saturating_sub(2 * degree + 1)))
        .ok_or_else(|| "capacity MCA Psi: integer overflow".to_string())?;
    let integral = checked_product(&[2 * jet_degree - 1, height], "capacity MCA integral term")?
        .checked_add(checked_product(
            &[n - degree - 1, jet_degree],
            "capacity MCA integral term",
        )?)
        .ok_or_else(|| "capacity MCA integral term: integer overflow".to_string())?;
    let fractional = checked_product(
        &[
            n - degree,
            jet_degree
                .checked_add(checked_product(&[height, psi], "capacity MCA fractional term")?)
                .ok_or_else(|| "capacity MCA fractional term: integer overflow".to_string())?,
        ],
        "capacity MCA fractional term",
    )?;
    let numerator = checked_product(&[integral, denominator], "capacity MCA exceptional numerator")?
        .checked_add(fractional)
        .ok_or_else(|| "capacity MCA exceptional numerator: integer overflow".to_string())?;
    Ok((numerator, denominator))
}

fn soundness_budget(target_bits: usize) -> Result<u128, String> {
    let exponent = (ANALYSIS_LOG_Q as usize)
        .checked_sub(target_bits)
        .ok_or_else(|| format!("target {target_bits} exceeds the analysis field size"))?;
    1u128
        .checked_shl(exponent as u32)
        .ok_or_else(|| format!("target {target_bits} is too small for an exact u128 soundness budget"))
}

fn capacity_mca_fold_within_target(level: &WhirLevelConfig, target_bits: usize) -> Result<bool, String> {
    let (exceptional_numerator, denominator) = capacity_mca_exceptional_fraction(level)?;
    let list_numerator = checked_product(&[2, level.list_bound as u128, denominator], "capacity MCA list term")?;
    let fold_numerator = exceptional_numerator
        .checked_add(list_numerator)
        .ok_or_else(|| "capacity MCA fold numerator: integer overflow".to_string())?;
    let target_numerator = checked_product(&[soundness_budget(target_bits)?, denominator], "capacity MCA target")?;
    Ok(fold_numerator <= target_numerator)
}

fn capacity_mca_fold_bits(level: &WhirLevelConfig) -> Result<f64, String> {
    let (exceptional_numerator, denominator) = capacity_mca_exceptional_fraction(level)?;
    let fold_numerator = exceptional_numerator
        .checked_add(checked_product(
            &[2, level.list_bound as u128, denominator],
            "capacity MCA list term",
        )?)
        .ok_or_else(|| "capacity MCA fold numerator: integer overflow".to_string())?;
    Ok(ANALYSIS_LOG_Q - (fold_numerator as f64).log2() + (denominator as f64).log2())
}

/// Exact-threshold query soundness in bits. Outside the threshold-A list, a
/// word agrees with every codeword on at most `A - 1` positions.
fn paper_query_bits(level: &WhirLevelConfig) -> f64 {
    let n = (1usize << (level.log_msg_cols + level.log_inv_rate)) as f64;
    level.queries as f64 * (n / (level.agreement - 1) as f64).log2()
}

/// Unique-decoding-regime per-query soundness at `γ = δ/2` (`δ = 1 − ρ`).
/// Test-support only, backing [`udr_queries`] and the ad-hoc
/// [`default_config`] shape used by small F64 PCS tests.
#[cfg(test)]
fn udr_per_query_bits_asymptotic(log_inv_rate: usize) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let gamma = (1.0 - rho) / 2.0;
    (1.0 / (1.0 - gamma)).log2()
}

/// Johnson-bound list size of the interleaved RS code, in log2. The exact
/// integer bound is stored in each level certificate.
///
/// Interleaving preserves relative distance (`V^{⊙m}` has the base code's
/// distance `δ = 1 − ρ`) and only enlarges the alphabet (to `q^m`). The
/// Johnson bound depends solely on (distance, radius, alphabet size), so the
/// interleaved list size at any radius *below* the Johnson radius `1 − √ρ`
/// is bounded by the very same single-code Johnson list size
///
///   `L_int ≤ L_base ≤ 1/(2·η·√ρ)`,
///
/// with no dependence on `m` and, crucially, no `L_base^r` blow-up.
///
/// The general GGR (Gopalan-Guruswami-Raghavendra, Thm 2.5) interleaved bound
/// `L_int ≤ C(b+r, r)·L_base^r` is only needed to push the list-decoding
/// radius *past* the Johnson bound toward `δ`. WHIR deliberately sits at
/// `θ = 1 − √ρ − η`, strictly below the Johnson radius by slack `η > 0`, so
/// that regime never applies and the plain Johnson bound is both correct and
/// far tighter (it dominates GGR throughout the regime RS can reach).
fn johnson_interleaved_list_log2(list_bound: usize) -> f64 {
    (list_bound as f64).log2()
}

/// Worst algebraic verifier-challenge transition in the production opening:
/// `thm:rbr`'s batch row (`(J−1)·L/|F|` for the powers-of-lambda batching of
/// the PCS annex, Protocol 1 step 1) and the `2L/|F|` part of its fold row.
/// A degree-`d` identity test unioned over a Johnson list of size `L` fails
/// with probability at most `dL/|F|`. The relevant degrees are:
///
/// - the total degree of the GF64-to-GF192 ring-switch batching map (L0 only,
///   but included at every level so the bound also dominates the claim batch
///   entering the next level's list, whatever its query count);
/// - `J − 1 = prev_queries + ood_samples`, the batch polynomial's degree in the
///   level's single lambda. The claims it batches are the ones the PREVIOUS
///   level's query phase raised (`thm:rbr`: `J_i = n_{i-1} + 2`, one per query
///   plus the residual and the OOD claim), so this level's own query count is
///   the wrong quantity: query counts fall with depth, so using it would
///   understate the degree and overstate the bound. At L0 there is no previous
///   level and `J_0` is set by the outer protocol's claim pool rather than by a
///   query count, so 0 is passed; that pool is a few hundred claims, orders below
///   the ring-switch degree the `max` takes anyway; and
/// - 2 for quadratic sumcheck.
fn johnson_algebraic_bits_for(list_bound: usize, prev_queries: usize, ood_samples: usize) -> f64 {
    let log2_l = johnson_interleaved_list_log2(list_bound);
    let degree = crate::ring_switch::RING_SWITCH_SOUNDNESS_DEGREE
        .max(prev_queries + ood_samples)
        .max(2);
    ANALYSIS_LOG_Q - (degree as f64).log2() - log2_l
}

/// `prev_queries` is `levels[i-1].queries`, and 0 for `i = 0`.
fn johnson_algebraic_bits(level: &WhirLevelConfig, prev_queries: usize) -> f64 {
    johnson_algebraic_bits_for(level.list_bound, prev_queries, level.ood_samples)
}

fn johnson_algebraic_within_target(
    level: &WhirLevelConfig,
    prev_queries: usize,
    target_bits: usize,
) -> Result<bool, String> {
    let degree = crate::ring_switch::RING_SWITCH_SOUNDNESS_DEGREE
        .max(prev_queries + level.ood_samples)
        .max(2);
    Ok(
        checked_product(&[degree as u128, level.list_bound as u128], "algebraic soundness")?
            <= soundness_budget(target_bits)?,
    )
}

/// The query count the batch at `levels[i]` carries claims from.
fn prev_queries_at(levels: &[WhirLevelConfig], i: usize) -> usize {
    if i == 0 { 0 } else { levels[i - 1].queries }
}

/// OOD binding bits for a level. `mu_vars` is the level's multilinear
/// variable count (`log_msg_cols + log_num_interleaved`).
///
/// - `ood_samples ≥ 1` (explicit samples): the PCS annex, Lemma `lem:ood` /
///   `thm:rbr`'s OOD row `binom(L,2)·μ/|F|`, generalized to `s` samples: the
///   bad event is two distinct list elements agreeing on all `s` random
///   points of `F^μ` (Schwartz-Zippel, total degree ≤ μ), union over pairs:
///   `bits = s·(192 − log₂ μ) − log₂ binom(L_int, 2)`.
/// - `ood_samples = 0` (L0): the protocol takes no OOD sample at commitment,
///   so the PCS itself is only list binding (the PCS annex, opening paragraph). What this
///   term materializes is the OUTER protocol's binding: the opening's own
///   evaluation claim sits at a post-commit random point, so at most one
///   list member matches it except with `L·μ/|F|` (union over the list, not
///   pairs): `bits = 192 − log₂ L_int − log₂ μ`.
fn paper_ood_bits(list_bound: usize, mu_vars: usize, ood_samples: usize) -> f64 {
    let log2_l = johnson_interleaved_list_log2(list_bound);
    let log2_mu = (mu_vars as f64).log2();
    if ood_samples == 0 {
        ANALYSIS_LOG_Q - log2_l - log2_mu
    } else {
        let pairs = list_bound.saturating_mul(list_bound.saturating_sub(1)) / 2;
        ood_samples as f64 * (ANALYSIS_LOG_Q - log2_mu) - (pairs as f64).log2()
    }
}

fn ood_within_target(level: &WhirLevelConfig, target_bits: usize) -> Result<bool, String> {
    let mu = (level.log_msg_cols + level.log_num_interleaved) as u128;
    let list = level.list_bound as u128;
    let numerator = if level.ood_samples == 0 {
        checked_product(&[list, mu], "OOD soundness")?
    } else {
        // One sample suffices throughout the supported parameter window. This
        // one-sample check is conservative for any larger declared count.
        checked_product(&[list, list.saturating_sub(1), mu], "OOD soundness")? / 2
    };
    Ok(numerator <= soundness_budget(target_bits)?)
}

/// Result of the WHIR-style per-level Johnson-slack search.
struct OptimizedJohnsonLevel {
    eta: f64,
    queries: usize,
    ood_samples: usize,
    certificate: CapacityMcaCertificate,
}

/// Largest integer `a` such that `(a/n)^queries <= 2^-query_bits`, plus one.
/// A word outside the threshold-A list has at most `A - 1 = a` agreements, so
/// this computes the query row without floating-point acceptance decisions.
fn exact_query_agreement(n: usize, queries: usize, query_bits: usize) -> Result<usize, String> {
    let exponent = u32::try_from(queries).map_err(|_| "query count does not fit u32".to_string())?;
    let rhs = num_bigint::BigUint::from(n).pow(exponent);
    let mut safe = 0usize;
    let mut unsafe_bound = n;
    while safe + 1 < unsafe_bound {
        let candidate = safe + (unsafe_bound - safe) / 2;
        let lhs = num_bigint::BigUint::from(candidate).pow(exponent) << query_bits;
        if lhs <= rhs {
            safe = candidate;
        } else {
            unsafe_bound = candidate;
        }
    }
    safe.checked_add(1)
        .ok_or_else(|| "query agreement threshold overflow".into())
}

/// Choose eta independently for one recursive level. Integer query counts are
/// searched from the query-only Johnson floor, and every soundness decision is
/// checked with exact integer arithmetic.
fn optimize_johnson_level(
    level: usize,
    log_inv_rate: usize,
    log_msg_cols: usize,
    log_num_interleaved: usize,
    target_bits: usize,
    query_grinding_bits: usize,
    prev_queries: usize,
) -> Result<OptimizedJohnsonLevel, String> {
    let query_target = target_bits.saturating_sub(query_grinding_bits).max(1);
    let block_len = 1usize << (log_msg_cols + log_inv_rate);
    let degree = (1usize << log_msg_cols) - 1;
    let sqrt_rho = reduced_rate(log_inv_rate, log_msg_cols).sqrt();
    let beyond_johnson = |agreement: usize| -> Result<bool, String> {
        Ok(
            checked_product(&[agreement as u128, agreement as u128], "query Johnson threshold")?
                > checked_product(&[block_len as u128, degree as u128], "query Johnson threshold")?,
        )
    };
    let mut query_floor = (query_target as f64 / -sqrt_rho.log2()).ceil() as usize;
    while query_floor > 1 && beyond_johnson(exact_query_agreement(block_len, query_floor - 1, query_target)?)? {
        query_floor -= 1;
    }
    while !beyond_johnson(exact_query_agreement(block_len, query_floor, query_target)?)? {
        query_floor += 1;
    }

    for queries in query_floor..=block_len {
        let agreement = exact_query_agreement(block_len, queries, query_target)?;
        let eta = agreement as f64 / block_len as f64 - sqrt_rho;
        if !eta.is_finite() || eta <= 0.0 || eta >= 1.0 - sqrt_rho {
            continue;
        }
        let certificate = match capacity_certificate(log_inv_rate, log_msg_cols, agreement, queries) {
            Ok(certificate) => certificate,
            Err(_) => continue,
        };
        let ood_samples = usize::from(level > 0);
        let candidate_level = WhirLevelConfig {
            log_inv_rate,
            log_msg_cols,
            log_num_interleaved,
            k: log_num_interleaved,
            eta,
            agreement: certificate.agreement,
            interpolation_m: certificate.interpolation_m,
            jet_degree: certificate.jet_degree,
            interpolation_height: certificate.interpolation_height,
            list_bound: certificate.list_bound,
            certificate_variant: certificate.variant,
            queries,
            grinding_bits: query_grinding_bits,
            ood_samples,
            target_security_bits: target_bits,
        };
        if !capacity_mca_fold_within_target(&candidate_level, target_bits)?
            || !ood_within_target(&candidate_level, target_bits)?
            || !johnson_algebraic_within_target(&candidate_level, prev_queries, target_bits)?
        {
            continue;
        }
        return Ok(OptimizedJohnsonLevel {
            eta,
            queries,
            ood_samples,
            certificate,
        });
    }

    Err(format!(
        "L{level}: no capacity MCA candidate satisfies {target_bits}-bit Johnson/OOD soundness at rate 1/2^{log_inv_rate}"
    ))
}

impl WhirLevelConfig {
    /// Combined fold-challenge and per-query soundness bits this level delivers.
    fn paper_predicted_bits(&self) -> (f64, f64) {
        (
            capacity_mca_fold_bits(self).expect("validated capacity MCA certificate"),
            paper_query_bits(self),
        )
    }

    /// OOD binding bits this level delivers. See `paper_ood_bits`.
    fn paper_predicted_ood_bits(&self) -> f64 {
        let mu = self.log_msg_cols + self.log_num_interleaved;
        paper_ood_bits(self.list_bound, mu, self.ood_samples)
    }
}

impl WhirSecurityConfig {
    /// Validate that the config is internally consistent and matches the
    /// declared analysis. Returns the first violation found, if any.
    pub fn validate(&self) -> Result<(), String> {
        if self.log_n + crate::LOG_PACKING != self.m {
            return Err(format!(
                "log_n ({}) + LOG_PACKING ({}) != m ({})",
                self.log_n,
                crate::LOG_PACKING,
                self.m
            ));
        }

        // Level shape: initial_k + Σ k (L1+) + yr_log_n = log_n.
        let levels_level_k_sum: usize = self.levels.iter().skip(1).map(|lv| lv.k).sum();
        let yr_log_n = self.final_block.yr_log_n;
        if self.initial_k + levels_level_k_sum + yr_log_n != self.log_n {
            return Err(format!(
                "shape mismatch: initial_k ({}) + Σ k ({}) + yr_log_n ({}) = {} ≠ log_n ({})",
                self.initial_k,
                levels_level_k_sum,
                yr_log_n,
                self.initial_k + levels_level_k_sum + yr_log_n,
                self.log_n,
            ));
        }

        // L0 must have k = initial_k and log_num_interleaved = initial_k.
        let l0 = self.levels.first().ok_or_else(|| "empty levels".to_string())?;
        if l0.k != self.initial_k {
            return Err(format!("L0.k ({}) must equal initial_k ({})", l0.k, self.initial_k));
        }
        if l0.log_num_interleaved != self.initial_k {
            return Err(format!(
                "L0.log_num_interleaved ({}) must equal initial_k ({})",
                l0.log_num_interleaved, self.initial_k
            ));
        }

        // Per-level checks.
        let mut dim_in = self.log_n;
        for (i, lv) in self.levels.iter().enumerate() {
            if lv.log_inv_rate == 0 {
                return Err(format!("L{i}: log_inv_rate=0 gives a rate-one code"));
            }
            if lv.log_msg_cols == 0 {
                return Err(format!("L{i}: log_msg_cols must be positive"));
            }

            // Shape: log_msg_cols + log_num_interleaved = dim_in.
            if lv.log_msg_cols + lv.log_num_interleaved != dim_in {
                return Err(format!(
                    "L{i}: log_msg_cols ({}) + log_num_interleaved ({}) ≠ input dim ({dim_in})",
                    lv.log_msg_cols, lv.log_num_interleaved
                ));
            }

            // Folding `lv.k` variables changes the next level's total RS
            // domain logarithm from `dim_in + rate_i` to
            // `dim_in - lv.k + rate_{i+1}`. Pin that difference to the public
            // initial reduction and to one bit at every later transition.
            if let Some(next) = self.levels.get(i + 1) {
                let domain_reduction = if i == 0 {
                    RS_DOMAIN_INITIAL_REDUCTION_FACTOR
                } else {
                    RS_DOMAIN_SUBSEQUENT_REDUCTION_FACTOR
                };
                let expected_next_rate = lv
                    .log_inv_rate
                    .checked_add(lv.k)
                    .and_then(|r| r.checked_sub(domain_reduction))
                    .ok_or_else(|| format!("L{i}: invalid RS domain reduction {domain_reduction}"))?;
                if next.log_inv_rate != expected_next_rate {
                    return Err(format!(
                        "L{}: log_inv_rate ({}) does not reduce the preceding RS domain by {} bit(s); expected {}",
                        i + 1,
                        next.log_inv_rate,
                        domain_reduction,
                        expected_next_rate,
                    ));
                }
            }

            if lv.queries == 0 {
                return Err(format!("L{i}: query count must be positive"));
            }
            let block_len = 1usize << (lv.log_msg_cols + lv.log_inv_rate);
            let query_target = lv.target_security_bits.saturating_sub(lv.grinding_bits).max(1);
            let expected_agreement = exact_query_agreement(block_len, lv.queries, query_target)?;
            if lv.agreement != expected_agreement {
                return Err(format!(
                    "L{i}: agreement threshold {} is not the exact query threshold {expected_agreement}",
                    lv.agreement
                ));
            }

            // Eta is retained as a readable distance from Johnson, while the
            // security checks use the exact integer threshold above.
            let sqrt_rho = reduced_rate(lv.log_inv_rate, lv.log_msg_cols).sqrt();
            let expected_eta = lv.agreement as f64 / block_len as f64 - sqrt_rho;
            let max_eta = 1.0 - sqrt_rho;
            if !lv.eta.is_finite() || lv.eta <= 0.0 || lv.eta >= max_eta || lv.eta.to_bits() != expected_eta.to_bits() {
                return Err(format!(
                    "L{i}: Johnson eta must equal A/n - sqrt(D/n) in (0, {max_eta}), got {}",
                    lv.eta
                ));
            }

            let expected_certificate =
                capacity_certificate(lv.log_inv_rate, lv.log_msg_cols, lv.agreement, lv.queries)?;
            if (
                lv.interpolation_m,
                lv.jet_degree,
                lv.interpolation_height,
                lv.list_bound,
                lv.certificate_variant,
            ) != (
                expected_certificate.interpolation_m,
                expected_certificate.jet_degree,
                expected_certificate.interpolation_height,
                expected_certificate.list_bound,
                expected_certificate.variant,
            ) {
                return Err(format!(
                    "L{i}: capacity MCA certificate does not match its exact derived parameters"
                ));
            }

            // OOD samples: every level past L0 needs explicit samples, while
            // L0 is bound by the opening's own post-commit evaluation claim.
            if i == 0 && lv.ood_samples != 0 {
                return Err(format!(
                    "L0: ood_samples={} but L0 is bound by the opening's \
                     own evaluation claim (must be 0)",
                    lv.ood_samples
                ));
            }
            if i > 0 && lv.ood_samples == 0 {
                return Err(format!(
                    "L{i}: ood_samples ≥ 1 required past L0 (the query \
                     counts assume single-codeword binding)"
                ));
            }

            // OOD binding clears the target under an exact integer check.
            let ood_pred = lv.paper_predicted_ood_bits();
            if !ood_within_target(lv, lv.target_security_bits)? {
                return Err(format!(
                    "L{i}: OOD binding ({ood_pred:.2} bits) < target ({})",
                    lv.target_security_bits
                ));
            }

            let (fold_pred, q_pred) = lv.paper_predicted_bits();

            // `expected_agreement` was obtained by an exact BigUint comparison,
            // so the query row clears its target without accepting on `q_pred`.
            debug_assert!(q_pred + 1e-10 >= query_target as f64);

            // The fold row is one combined `(E + 2 Lambda)/q` check.
            if !capacity_mca_fold_within_target(lv, lv.target_security_bits)? {
                return Err(format!(
                    "L{i}: fold-challenge soundness ({fold_pred:.2} bits) < target ({})",
                    lv.target_security_bits
                ));
            }

            // The largest list-unioned algebraic identity test (currently the
            // composed ring-switch batching map) is not grindable and must
            // clear the target.
            let algebraic = johnson_algebraic_bits(lv, prev_queries_at(&self.levels, i));
            if !johnson_algebraic_within_target(lv, prev_queries_at(&self.levels, i), lv.target_security_bits)? {
                return Err(format!(
                    "L{i}: list-unioned algebraic soundness ({algebraic:.2} bits) < target ({})",
                    lv.target_security_bits
                ));
            }

            if lv.target_security_bits < self.target_security_bits {
                return Err(format!(
                    "L{i}: target_security_bits ({}) < global target ({})",
                    lv.target_security_bits, self.target_security_bits
                ));
            }

            // Advance dim_in for next level: subtract k (the folds at this level).
            dim_in -= lv.k;
        }

        if dim_in != yr_log_n {
            return Err(format!(
                "after consuming all levels, dim_in ({dim_in}) ≠ yr_log_n ({yr_log_n})"
            ));
        }

        // Round-by-round soundness (doc/leanvm/body/b-polynomial-commitment-scheme.tex, Thm `thm:rbr`): each
        // verifier-challenge transition is checked against
        // `target_security_bits` in the per-level loop above, so the
        // Fiat--Shamir error per random-oracle query is their MAX; ordinary
        // interactive soundness may additionally union-bound over transitions.
        Ok(())
    }

    /// Derive the production security config at witness size `m` for an
    /// explicit L0 rate `2^-log_inv_rate`: Johnson list decoding with OOD
    /// binding and [`SECURITY_BITS`] bits per round under **round-by-round
    /// soundness**, i.e. every verifier-challenge error term (fold, query plus
    /// query grinding, OOD, and algebraic checks) clears the
    /// target individually.
    pub fn derive_config_with_log_inv_rate(m: usize, log_inv_rate: usize) -> Result<Self, String> {
        validate_log_inv_rate(log_inv_rate)?;
        let target_bits = SECURITY_BITS;
        let query_grind: usize = QUERY_GRINDING_BITS;
        let log_n = m
            .checked_sub(crate::LOG_PACKING)
            .ok_or_else(|| format!("m ({m}) < LOG_PACKING ({})", crate::LOG_PACKING))?;
        let initial_k = INITIAL_FOLDING_FACTOR;

        // The ladder geometry is independent of eta. Exact block-length
        // feasibility is checked below by the same per-level optimizer that
        // supplies the production eta and query count.
        let shape = derive_ladder_shape(log_n, initial_k, log_inv_rate)?;
        let n_levels = shape.log_inv_rates.len();

        // Round-by-round target: every verifier-challenge error term (fold,
        // query, OOD, and algebraic checks) must individually clear
        // `target_bits`. We do not add a whole-transcript union-bound margin:
        // this configuration targets 128-bit RBR soundness, as required by the
        // Fiat--Shamir analysis, rather than 128-bit interactive soundness after
        // summing every transition probability.
        let mut levels = Vec::with_capacity(n_levels);
        for i in 0..n_levels {
            let rate = shape.log_inv_rates[i];
            let cols = shape.log_msg_cols[i];
            let ilv = shape.k_levels[i];
            let prev_queries = prev_queries_at(&levels, i);
            let optimized = optimize_johnson_level(i, rate, cols, ilv, target_bits, query_grind, prev_queries)?;

            levels.push(WhirLevelConfig {
                log_inv_rate: rate,
                log_msg_cols: cols,
                log_num_interleaved: ilv,
                k: shape.k_levels[i],
                eta: optimized.eta,
                agreement: optimized.certificate.agreement,
                interpolation_m: optimized.certificate.interpolation_m,
                jet_degree: optimized.certificate.jet_degree,
                interpolation_height: optimized.certificate.interpolation_height,
                list_bound: optimized.certificate.list_bound,
                certificate_variant: optimized.certificate.variant,
                queries: optimized.queries,
                grinding_bits: query_grind,
                ood_samples: optimized.ood_samples,
                target_security_bits: target_bits,
            });
        }

        let analysis_version = "rs_capacity_mca_exact_threshold_width_free_rbr";
        let cfg = Self {
            m,
            log_n,
            initial_k,
            target_security_bits: target_bits,
            analysis_version: analysis_version.into(),
            levels,
            final_block: FinalBlockConfig {
                yr_log_n: shape.yr_log_n,
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Build the shared prover/verifier config, retaining the level shape and dropping security-analysis fields.
    pub fn to_config(&self) -> Result<ProverConfig, String> {
        self.validate()?;
        Ok(ProverConfig {
            log_inv_rates: self.levels.iter().map(|lv| lv.log_inv_rate).collect(),
            level_steps: self.levels.len() - 1,
            initial_k: self.initial_k,
            level_ks: self.levels.iter().skip(1).map(|lv| lv.k).collect(),
            queries: self.levels.iter().map(|lv| lv.queries).collect(),
            grinding_bits: self.levels.iter().map(|lv| lv.grinding_bits).collect(),
            ood_samples: self.levels.iter().map(|lv| lv.ood_samples).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::pretty_integer;

    #[test]
    fn johnson_bound_uses_exact_threshold_and_reduced_rate() {
        // A message of dimension 16 has maximum degree 15, so the theorem's
        // reduced rate at block length 512 is 15/512, not the nominal 1/32.
        assert_eq!(reduced_rate(5, 4), 15.0 / 512.0);

        // At exact dyadic query boundaries the largest safe outside-list
        // agreement is retained, rather than lost to floating-point ceil.
        assert_eq!(exact_query_agreement(131_072, 111, 111).unwrap(), 65_537);
        assert_eq!(exact_query_agreement(524_288, 37, 111).unwrap(), 65_537);

        let closed = capacity_certificate(2, 15, 65_537, 111).unwrap();
        assert_eq!(closed.variant, JohnsonCertificateVariant::Closed);
        assert_eq!(
            (closed.interpolation_m, closed.jet_degree, closed.interpolation_height),
            (16_384, 32_769, 357_946_710)
        );

        let finite = capacity_certificate(1, 16, 92_682, 222).unwrap();
        assert_eq!(finite.variant, JohnsonCertificateVariant::Finite);
        assert_eq!(
            (
                finite.interpolation_m,
                finite.jet_degree,
                finite.interpolation_height,
                finite.list_bound
            ),
            (24_954, 35_290, 606_738_001, 23_784)
        );
    }

    #[test]
    fn production_profile_is_128_bit_johnson_with_query_grinding() {
        let expected = [
            [
                "222,55",
                "222,56,30",
                "222,56,31",
                "222,56,32",
                "222,56,32",
                "222,56,32,22",
                "222,56,32,22",
                "222,56,32,22",
                "223,56,32,23",
                "223,56,32,23,17",
                "223,56,32,23,17",
                "223,56,32,23,17",
                "223,56,32,23,18",
                "223,56,32,23,18,14",
            ],
            [
                "111,44",
                "111,45,27",
                "111,45,28",
                "111,45,28",
                "111,45,28",
                "111,45,28,20",
                "111,45,28,20",
                "112,45,28,21",
                "112,45,28,21",
                "112,45,28,21,16",
                "112,45,28,21,16",
                "112,45,28,21,16",
                "112,45,28,21,16",
                "112,45,28,21,16,13",
            ],
            [
                "74,37",
                "74,37,24",
                "74,37,25",
                "74,37,25",
                "74,37,25",
                "74,37,25,18",
                "75,37,25,19",
                "75,37,25,19",
                "75,37,25,19",
                "75,38,25,19,15",
                "75,38,25,19,15",
                "75,38,25,19,15",
                "75,38,25,19,15",
                "75,38,25,19,15,13",
            ],
            [
                "56,32",
                "56,32,22",
                "56,32,22",
                "56,32,22",
                "56,32,23",
                "56,32,23,17",
                "56,32,23,17",
                "56,32,23,17",
                "56,32,23,18",
                "56,32,23,18,14",
                "56,32,23,18,14",
                "56,32,23,18,14",
                "56,32,23,18,14",
                "56,32,23,18,14,12",
            ],
        ];
        let mut min_fold_bits = f64::INFINITY;
        for log_inv_rate in MIN_LOG_INV_RATE..=MAX_LOG_INV_RATE {
            for log_n in 15..=28 {
                let cfg = WhirSecurityConfig::derive_config_with_log_inv_rate(log_n + crate::LOG_PACKING, log_inv_rate)
                    .unwrap();
                assert_eq!(cfg.target_security_bits, 128);
                assert_eq!(cfg.levels[0].log_inv_rate, log_inv_rate);
                assert_eq!(cfg.levels[0].ood_samples, 0);
                let queries = cfg
                    .levels
                    .iter()
                    .map(|level| level.queries.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                assert_eq!(queries, expected[log_inv_rate - 1][log_n - 15]);
                for (i, level) in cfg.levels.iter().enumerate() {
                    let (fold_bits, query_bits) = level.paper_predicted_bits();
                    let ood_bits = level.paper_predicted_ood_bits();
                    let algebraic_bits = johnson_algebraic_bits(level, prev_queries_at(&cfg.levels, i));
                    min_fold_bits = min_fold_bits.min(fold_bits);
                    assert_eq!(level.grinding_bits, QUERY_GRINDING_BITS);
                    assert!(query_bits + level.grinding_bits as f64 + 1e-10 >= 128.0);
                    assert!(fold_bits >= 128.0);
                    assert!(ood_bits >= 128.0);
                    assert!(algebraic_bits >= 128.0);
                    if i > 0 {
                        assert_eq!(level.ood_samples, 1);
                    }
                }
            }
        }
        assert!(
            (128.0..129.0).contains(&min_fold_bits),
            "query optimization should use the first bit of fold margin: {min_fold_bits}"
        );
    }

    /// Parameter-report helper:
    /// `WHIR_LOG_INV_RATE=2 WHIR_NUM_VARS=22 cargo test --release -p pcs print_whir_query_counts -- --ignored --nocapture`
    #[test]
    #[ignore = "manual parameter report; configure it through environment variables"]
    fn print_whir_query_counts() {
        let env_usize = |name: &str| {
            std::env::var(name)
                .unwrap_or_else(|_| panic!("missing {name}"))
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be a non-negative integer"))
        };
        let log_inv_rate = env_usize("WHIR_LOG_INV_RATE");
        let num_vars = env_usize("WHIR_NUM_VARS");
        let cfg =
            WhirSecurityConfig::derive_config_with_log_inv_rate(num_vars + crate::LOG_PACKING, log_inv_rate).unwrap();

        println!(
            "num_vars={}, rate=1/{}",
            pretty_integer(num_vars),
            pretty_integer(1usize << log_inv_rate)
        );
        for (level, params) in cfg.levels.iter().enumerate() {
            let eta = params.eta;
            println!(
                "L{}: rate=1/{}, queries={}, eta={eta:.12e}, A={}, m={}, B={}, H={}, list={}",
                pretty_integer(level),
                pretty_integer(1usize << params.log_inv_rate),
                pretty_integer(params.queries),
                pretty_integer(params.agreement),
                pretty_integer(params.interpolation_m),
                pretty_integer(params.jet_degree),
                pretty_integer(params.interpolation_height),
                pretty_integer(params.list_bound),
            );
        }
    }
}

//! A self-contained verifier for leanVM execution proofs: single threaded, std only, plain integer arithmetic.
//!
//! Kept close enough to the Lean transcription (`formal/Statement/LeanVmVerifier.lean`) to read side by side: the same functions, in the same order, with the same constants.
//!
//! What the transcription has no counterpart for sits in modules of its own: the prepared [`Bytecode`] and what the statement of an aggregate adds, [`da`].
//!
//! One walk of the protocol serves two entry points. [`verify_execution`] checks the proof. [`replay_unchecked`] reads the same [`Summary`] out of it and checks nothing: every check it leaves out is behind `Transcript::checked`.

#![forbid(unsafe_code)]

mod bytecode;
pub mod da;
mod field;
mod hash;
mod keccak;
mod layout;
mod link;
mod merkle;
mod multilinear;
mod ring_switch;
mod transcript;
mod whir;

pub use bytecode::Bytecode;
pub use field::{E, K};
pub use layout::{
    MAX_LOG_BYTECODE, MAX_LOG_MEMORY, MAX_STACKED_LOG, MAX_TABLE_LOG, MIN_LOG_MEMORY, MIN_STACKED_LOG, TAG,
    min_execution_stacked_log,
};
pub use merkle::MerklePath;
pub use multilinear::stack_offsets;
pub use transcript::Proof;
pub use whir::whir_queries;

use field::{ONE, ZERO, powers};
use hash::{Digest, ROUNDS, STATE_LANES};
use keccak::{KeccakShape, verify_keccak};
use layout::{
    GLOBAL_COLUMN_BASES, KECCAK_OUT, Layout, MEMORY_0, MEMORY_1, MEMORY_2, N_TABLES, OP_KECCAK, QKECCAK, build_layout,
};
use link::verify_link;
use multilinear::{dot, eq_eval, multilinear_eval};
use ring_switch::{RingClaim, RingPiece, ring_switch};
use std::ops::Range;
use transcript::Transcript;
use whir::verify_whir;

/// Why a proof was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierError {
    DigestHalf,
    MultilinearSize,
    NotPowerOfTwo,
    StreamExhausted,
    GrindingNonce,
    MerkleOpeningMissing,
    MerkleOpeningShape,
    MerkleOpeningPosition,
    MerkleRootMismatch,
    StreamNotConsumed,
    MerkleNotConsumed,
    /// The link sum-check's last round does not match the summand at its point.
    LinkTerminal,
    AnnouncedTableSizes,
    WhirShape,
    QueryCountLadder,
    InducedBasisDimensions,
    WhirTerminal,
    WhirNoTerminalLevel,
    KeccakProduct,
    KeccakLayer,
    KeccakOutputOpening,
    KeccakInputOpening,
    AnnouncedHighLimb,
    PcsInverseRate,
    CommittedSize,
}

impl std::fmt::Display for VerifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::DigestHalf => "a digest half is 128-bit",
            Self::MultilinearSize => "multilinear table has the wrong size",
            Self::NotPowerOfTwo => "expected a power of two",
            Self::StreamExhausted => "proof stream exhausted",
            Self::GrindingNonce => "invalid grinding nonce",
            Self::MerkleOpeningMissing => "Merkle opening missing",
            Self::MerkleOpeningShape => "Merkle opening has the wrong shape",
            Self::MerkleOpeningPosition => "Merkle opening at the wrong position",
            Self::MerkleRootMismatch => "Merkle root mismatch",
            Self::StreamNotConsumed => "proof stream not fully consumed",
            Self::MerkleNotConsumed => "Merkle openings not fully consumed",
            Self::LinkTerminal => "link sum-check terminal mismatch",
            Self::AnnouncedTableSizes => "invalid announced table sizes",
            Self::WhirShape => "invalid WHIR shape",
            Self::QueryCountLadder => "tabulated query count does not match the ladder",
            Self::InducedBasisDimensions => "bad induced-basis dimensions",
            Self::WhirTerminal => "WHIR terminal check failed",
            Self::WhirNoTerminalLevel => "WHIR verification ended without a terminal level",
            Self::KeccakProduct => "Keccak product polynomial does not carry the claim",
            Self::KeccakLayer => "Keccak layer evaluations do not match the sumcheck",
            Self::KeccakOutputOpening => "Keccak output opening ends on the wrong value",
            Self::KeccakInputOpening => "Keccak input opening ends on the wrong value",
            Self::AnnouncedHighLimb => "announced size has a nonzero high limb",
            Self::PcsInverseRate => "invalid PCS inverse rate",
            Self::CommittedSize => "committed size outside the PCS window",
        };
        f.write_str(message)
    }
}

impl std::error::Error for VerifierError {}

fn require(condition: bool, error: VerifierError) -> Result<(), VerifierError> {
    if condition { Ok(()) } else { Err(error) }
}

/// The weight a claim puts on the stack, and the value it claims for it.
type StackClaim<'a> = (Box<dyn Fn(&[E]) -> E + 'a>, E);

/// Discharge every claim on the committed stack in one opening: the same powers of one challenge
/// batch the values into the target, and the weights into the basis WHIR evaluates at its terminal point.
fn verify_stacked_opening(
    transcript: &mut Transcript,
    root: Digest,
    stack_log: usize,
    log_inv_rate: usize,
    claims: &[StackClaim],
) -> Result<(), VerifierError> {
    let values: Vec<E> = claims.iter().map(|(_, value)| *value).collect();
    let scales = powers(transcript.sample(), claims.len());
    let evaluate_basis = |point: &[E]| {
        let weights: Vec<E> = claims.iter().map(|(weight, _)| weight(point)).collect();
        dot(&scales, &weights)
    };
    let target = dot(&scales, &values);
    verify_whir(transcript, stack_log, log_inv_rate, target, root, &evaluate_basis)
}

/// A claim that a column's multilinear takes `value` at `point`.
struct ColumnClaim {
    column: usize,
    point: Vec<E>,
    value: E,
}

impl ColumnClaim {
    /// `None` for a column nothing commits, whose claim never reaches the stack.
    fn on_stack(&self, layout: &Layout) -> Option<StackClaim<'static>> {
        let point = layout.placements[self.column]?.stack_point(&self.point, layout.stack_log);
        Some((Box::new(move |x| eq_eval(&point, x)), self.value))
    }
}

/// What a walk of the proof leaves behind, for a prover that proves this verification in turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    /// The announced sizes: the memory's log size and each table's row count.
    pub log_memory: usize,
    pub table_rows: Vec<u64>,
    pub log_inv_rate: usize,
    /// Where the link reads the stacked bytecode. Its value there is the claim an aggregate defers.
    pub bytecode_point: Vec<E>,
    /// The stream positions the Keccak reduction reads.
    pub keccak_scalars: Range<usize>,
    /// The stream positions of the link's index-side values at `r` (`w⁺`, the spine and the memory term) and of the
    /// pointer columns' bit-slices.
    pub values_at_r: Range<usize>,
    pub bit_slices: Range<usize>,
}

/// Verify that `proof` shows an execution of `bytecode` (the stacked bytecode multilinear) on the 256-bit `public_input`.
pub fn verify_execution(bytecode: &Bytecode, public_input: &[u8; 32], proof: &Proof) -> Result<Summary, VerifierError> {
    walk(bytecode, public_input, proof, true)
}

/// NOT a verification: the [`Summary`] of `proof`, read out without checking it. An invalid proof is accepted.
///
/// For a prover that reads its guest's witness out of a proof the guest then verifies in full. It walks the same transcript as [`verify_execution`] and leaves out every check: the sumcheck, link, Keccak and WHIR terminal identities, the proof of work, the bytecode evaluation and the Merkle roots. It still fails on a proof it cannot walk: announced sizes out of range, a stream or openings of the wrong length or shape, a Merkle path at a position the protocol did not open.
pub fn replay_unchecked(bytecode: &Bytecode, public_input: &[u8; 32], proof: &Proof) -> Result<Summary, VerifierError> {
    walk(bytecode, public_input, proof, false)
}

/// Exposed for the Keccak reduction's benchmark and tests: a batch of `2^log_steps` permutations proved alone, a state
/// committed every `depth` rounds, the committed states the whole stack.
///
/// The proof is the commitment root, the output lanes claimed at a challenge point, the reduction, then the ring
/// switch and its opening, as in an execution proof. Its public input is the shape, the words `log_steps`, `depth`,
/// `log_inv_rate` and zero, so a proof of one shape verifies for no other.
pub fn verify_keccak_batch(
    fiat_shamir_iv: &[u8; 32],
    log_steps: usize,
    depth: usize,
    log_inv_rate: usize,
    proof: &Proof,
) -> Result<(), VerifierError> {
    require(
        log_steps >= layout::KECCAK_MIN_STEPS_LOG && depth > 0 && ROUNDS.is_multiple_of(depth),
        VerifierError::CommittedSize,
    )?;
    let shape = KeccakShape { log_steps, depth };
    let stack_log = log_steps + shape.slot_bits();
    require(
        (MIN_STACKED_LOG..=MAX_STACKED_LOG).contains(&stack_log),
        VerifierError::CommittedSize,
    )?;
    let mut statement = [0u8; 32];
    for (bytes, word) in statement
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip([log_steps, depth, log_inv_rate])
    {
        *bytes = (word as u64).to_le_bytes();
    }
    let mut transcript = Transcript::new(proof, Digest(*fiat_shamir_iv), Digest(statement), true);
    let halves = transcript.next_scalars(2)?;
    let root = Digest::from_halves(halves[0], halves[1])?;
    let instance = transcript.samples(log_steps);
    let output = transcript.next_scalars(STATE_LANES)?;
    let openings = verify_keccak(&mut transcript, shape, &instance, &output)?;
    let (point, slices) = openings.inputs;
    let piece = RingPiece {
        placement: None,
        point,
        scale: ONE,
    };
    let mut claims: Vec<StackClaim> = ring_switch(
        vec![RingClaim {
            pieces: vec![piece],
            slices,
        }],
        &mut transcript,
    );
    if let Some((point, word)) = openings.outputs {
        claims.push((Box::new(move |x| eq_eval(&point, x)), word));
    }
    verify_stacked_opening(&mut transcript, root, stack_log, log_inv_rate, &claims)?;
    transcript.finish()?;
    Ok(())
}

fn walk(bytecode: &Bytecode, public_input: &[u8; 32], proof: &Proof, checked: bool) -> Result<Summary, VerifierError> {
    let public_input = Digest(*public_input);
    let mut transcript = Transcript::new(proof, bytecode.fiat_shamir_iv, public_input, checked);

    // 1] memory log-size, table row counts, and log-inv-rate in WHIR
    let announced = transcript.next_scalars(2 + N_TABLES)?;
    require(
        announced.iter().all(|value| value.c1 == 0 && value.c2 == 0),
        VerifierError::AnnouncedHighLimb,
    )?;
    let log_memory = announced[0].c0;
    let table_rows: Vec<u64> = announced[1..1 + N_TABLES].iter().map(|value| value.c0).collect();
    let log_inverse_rate = announced[1 + N_TABLES].c0;
    require((1..=4).contains(&log_inverse_rate), VerifierError::PcsInverseRate)?;
    let layout = build_layout(bytecode, log_memory, &table_rows)?;
    require(
        (MIN_STACKED_LOG..=MAX_STACKED_LOG).contains(&layout.stack_log),
        VerifierError::CommittedSize,
    )?;

    // 2] parse WHIR commitment: one Merkle root (No OOD, our PCS is only List-binding).
    let halves = transcript.next_scalars(2)?;
    let root = Digest::from_halves(halves[0], halves[1])?;

    // 3] The links: one sum-check over the index, the rows and the pointer columns, in place of a bus. It leaves each
    // table's other columns at its row point, a claim on the memory and the pointer columns' bit-slices.
    let link = verify_link(&layout, &mut transcript)?;
    let mut claims: Vec<ColumnClaim> = Vec::new();
    for (base, (point, values)) in GLOBAL_COLUMN_BASES.into_iter().zip(&link.columns) {
        claims.extend(values.iter().map(|&(local, value)| ColumnClaim {
            column: base + local,
            point: point.clone(),
            value,
        }));
    }

    // 4] binding the public input: each memory limb's claim is that limb's public line at the challenge, which the verifier
    // evaluates itself, so the prover sends nothing. Both public words are 128-bit, so the top limb's claim is zero.
    let public_challenge = transcript.sample();
    let (first, second) = public_input.halves();
    let mut public_point = vec![ZERO; layout.placements[MEMORY_0].expect("memory is committed").variables];
    public_point[0] = public_challenge;
    let public_pairs = [(first.c0, second.c0), (first.c1, second.c1), (first.c2, second.c2)];
    for (column, (low, high)) in [MEMORY_0, MEMORY_1, MEMORY_2].into_iter().zip(public_pairs) {
        claims.push(ColumnClaim {
            column,
            point: public_point.clone(),
            value: multilinear_eval(&[K(low), K(high)], &[public_challenge])?,
        });
    }

    // 5] KECCAK validity, by GKR over the rounds. The table's values on its output lanes are the reduction's claim on
    // the permutation's output: they leave the batch, their columns having no placement.
    let (point, values) = &link.columns[OP_KECCAK];
    let lanes: Vec<E> = (KECCAK_OUT..KECCAK_OUT + STATE_LANES)
        .map(|lane| {
            values
                .iter()
                .find(|&&(local, _)| local == lane)
                .map(|&(_, value)| value)
        })
        .collect::<Option<_>>()
        .expect("the output lanes are no pointers");
    let keccak_start = transcript.stream_offset();
    let openings = verify_keccak(&mut transcript, layout.keccak, point, &lanes)?;
    let keccak_scalars = keccak_start..transcript.stream_offset();
    if let Some((point, value)) = openings.outputs {
        claims.push(ColumnClaim {
            column: QKECCAK,
            point,
            value,
        });
    }

    // 6] Ring switching: the Keccak reduction's input slices on q_keccak, then the pointer columns' slices. Both lead
    // the batch, then the memory term and the point claims.
    let qkeccak = layout.placements[QKECCAK].expect("q_keccak is committed");
    let (point, slices) = openings.inputs;
    let keccak_ring = RingClaim {
        pieces: vec![RingPiece {
            placement: Some(qkeccak),
            point,
            scale: ONE,
        }],
        slices,
    };
    let mut stack_claims: Vec<StackClaim<'static>> = ring_switch(vec![keccak_ring, link.ring], &mut transcript);
    let (memory_point, memory_scales, memory_value) = link.memory;
    let memory_points: Vec<(Vec<E>, E)> = [MEMORY_0, MEMORY_1, MEMORY_2]
        .into_iter()
        .zip(memory_scales)
        .map(|(column, scale)| {
            let placement = layout.placements[column].expect("memory is committed");
            (placement.stack_point(&memory_point, layout.stack_log), scale)
        })
        .collect();
    stack_claims.push((
        Box::new(move |x| E::sum(memory_points.iter().map(|(point, scale)| *scale * eq_eval(point, x)))),
        memory_value,
    ));
    stack_claims.extend(claims.iter().filter_map(|claim| claim.on_stack(&layout)));
    verify_stacked_opening(
        &mut transcript,
        root,
        layout.stack_log,
        log_inverse_rate as usize,
        &stack_claims,
    )?;
    transcript.finish()?;
    Ok(Summary {
        log_memory: layout.log_memory,
        table_rows,
        log_inv_rate: log_inverse_rate as usize,
        bytecode_point: link.bytecode_point,
        keccak_scalars,
        values_at_r: link.values_at_r,
        bit_slices: link.bit_slices,
    })
}

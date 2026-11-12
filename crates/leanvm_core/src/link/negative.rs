//! Negative tests: the real prover on a tampered witness, which the verifier must reject, and the two tampers it must
//! accept. Each tamper runs after the witness is built and before it is committed ([`crate::cpu::prove_tampered`]).

use super::pointer::Chunks;
use super::prove::{Plan, cheat};
use crate::cpu::layout::{Witness, schema};
use crate::cpu::{CpuError, DerefMode, Op, OutCells, Program, prove_tampered, verify};
use crate::leaf::Coord;
use crate::tables::{self, KECCAK_TABLE};
use primitives::field::{F64, F192, g_pow};
use verifier::VerifierError;

const XOR: usize = 0;
const MUL: usize = 1;
const SET: usize = 2;
const DEREF: usize = 3;

/// Loop iterations, and `XOR` asserts per iteration: enough cycles that the pointer width (the cycle count's bits)
/// exceeds both the memory's and the program's, so pointers past memory and past the program exist.
const ITERATIONS: usize = 14;
const ASSERTS: usize = 12;
/// Where the loop's code starts, and its first frame.
const LOOP_PC: usize = 30;
const FIRST_FRAME: usize = 48;
const PROGRAM_LEN: usize = 64;

fn w(x: u64) -> F192 {
    F192::new(x, 0, 0)
}

fn gw(k: usize) -> F192 {
    F192::from(g_pow(k))
}

/// A program with a row in every table. Main sets up a Keccak permutation, a `DEREF`, a `MUL` and an `XOR` of zeros
/// (rows 0 of their tables), then runs a loop of `ITERATIONS + 1` iterations, one frame of eight cells each: frame
/// cells 0 to 5 hold `g^fp`, `g^8`, the condition `c_k = g^k + g^n`, the loop's pc, `K = g^n + g^{n+1}` and `g`,
/// and each iteration writes the next frame's from its own, then asserts `ASSERTS` times. The condition reaches zero
/// at `k = n`, where the loop falls through to a `JUMP` to the final state.
fn program() -> Program {
    let n = ITERATIONS;
    let k_const = gw(n) + gw(n + 1);
    let mut prog = vec![Op::Set { o: 2, k: w(3) }, Op::Set { o: 3, k: w(4) }];
    for c in 4..13u32 {
        let k = if c < 9 { w(u64::from(c)) } else { F192::ZERO };
        prog.push(Op::Set { o: c, k });
    }
    prog.push(Op::Keccak {
        m: [2, 3, 2, 3],
        tail: 4,
        b8: 8,
        cap: 9,
        out: 13,
        in_odd: false,
        out_odd: false,
        out_cells: OutCells::One,
    });
    prog.extend([
        Op::Set { o: 25, k: F192::ZERO },
        Op::Mul { a: 25, b: 25, c: 26 },
        Op::Xor { a: 25, b: 25, c: 27 },
        Op::Set { o: 14, k: gw(20) },
        Op::Set { o: 21, k: w(9) },
        Op::Deref {
            o1: 14,
            o2: 1,
            o3: 15,
            mode: DerefMode::Cell,
        },
        Op::Mul { a: 2, b: 3, c: 16 },
        Op::Xor { a: 13, b: 16, c: 17 },
    ]);
    let frame = FIRST_FRAME as u32;
    for (k, o) in [gw(FIRST_FRAME), gw(8), gw(0) + gw(n), gw(LOOP_PC), k_const, gw(1)].into_iter().zip(0..) {
        prog.push(Op::Set { o: frame + o, k });
    }
    prog.extend([
        Op::Set { o: 22, k: F192::ONE },
        Op::Set { o: 23, k: gw(LOOP_PC) },
        Op::Set { o: 24, k: gw(FIRST_FRAME) },
        Op::Jump { oc: 22, od: 23, of: 24 },
    ]);
    assert_eq!(prog.len(), LOOP_PC);
    prog.extend([
        Op::Mul { a: 0, b: 1, c: 8 },
        Op::Set { o: 9, k: gw(8) },
        Op::Set { o: 11, k: gw(LOOP_PC) },
        Op::Set { o: 12, k: k_const },
        Op::Set { o: 13, k: gw(1) },
        Op::Mul { a: 2, b: 5, c: 6 },
        Op::Xor { a: 6, b: 4, c: 10 },
        Op::Xor { a: 1, b: 1, c: 7 },
    ]);
    prog.extend((0..ASSERTS).map(|_| Op::Xor { a: 1, b: 1, c: 7 }));
    prog.push(Op::Jump { oc: 2, od: 3, of: 8 });
    prog.extend([
        Op::Set { o: 16, k: F192::ONE },
        Op::Set { o: 17, k: gw(PROGRAM_LEN - 1) },
        Op::Jump { oc: 16, od: 17, of: 16 },
    ]);
    prog.resize(PROGRAM_LEN, Op::Xor { a: 0, b: 0, c: 0 });
    Program::from_bytecode(prog, 32)
}

const PI: [F192; 2] = [F192::new(7, 0, 0), F192::new(11, 0, 0)];

/// A pointer column's words, or any column's values, in the witness.
fn column(w: &mut Witness, t: usize, c: usize) -> &mut [F64] {
    let p = w.layout.placements[schema().base[t] + c];
    &mut w.q[p.offset..p.offset + (1 << p.n_vars)]
}

fn plan(t: usize) -> Plan {
    Plan::of(tables::tables()[t])
}

fn chunks(w: &Witness) -> Chunks {
    Chunks::new(w.layout.pointer_width)
}

/// Prove with `tamper` applied, and verify.
fn outcome(tamper: impl FnOnce(&mut Witness)) -> Result<(), CpuError> {
    let program = program();
    let (proof, _) = prove_tampered(&program, PI, crate::pcs::TEST_LOG_INV_RATE, tamper).expect("the prover runs");
    cheat::INDEX_WORD.set(None);
    cheat::MEMORY_TERM.set(F192::ZERO);
    verify(&program, &PI, &proof).map(|_| ())
}

fn rejected(tamper: impl FnOnce(&mut Witness)) {
    assert_eq!(outcome(tamper), Err(CpuError::Verifier(VerifierError::LinkTerminal)));
}

/// The bit of another value of chunk `i` than the one `word` holds there.
fn other_bit(chunks: &Chunks, word: u64, i: usize) -> u64 {
    let shift: usize = (0..i).map(|i| 1 << chunks.widths[i]).sum();
    let current = chunks.bits(word, i).trailing_zeros() as usize;
    1 << (shift + (current + 1) % (1 << chunks.widths[i]))
}

/// The honest run's shape: the pointer width exceeds the memory's and the program's, and every table has a row.
#[test]
fn honest_proof_verifies() {
    let exec = program().execute(PI).expect("the program runs");
    assert!(exec.rows().iter().all(|&n| n > 0), "{:?}", exec.rows());
    assert_eq!(outcome(|w| {
        let width = w.layout.pointer_width;
        assert!(width > w.log_mem && 1 << width > PROGRAM_LEN, "width {width}, memory 2^{}", w.log_mem);
        assert!(chunks(w).d() > 1);
    }), Ok(()));
}

#[test]
fn a_chunk_with_two_ones_is_rejected() {
    rejected(|w| {
        let (chunks, cycle) = (chunks(w), plan(SET).links.state.cycle);
        let word = &mut column(w, SET, cycle)[0];
        *word = F64(word.0 | other_bit(&chunks, word.0, 0));
    });
}

#[test]
fn a_chunk_with_no_ones_is_rejected() {
    rejected(|w| {
        let (chunks, pointer) = (chunks(w), plan(SET).links.memory[0].pointer);
        let word = &mut column(w, SET, pointer)[0];
        *word = F64(word.0 & !(chunks.bits(word.0, 1) << (1 << chunks.widths[0])));
    });
}

#[test]
fn swapped_cycle_pointers_are_rejected() {
    rejected(|w| column(w, SET, plan(SET).links.state.cycle).swap(0, 1));
}

#[test]
fn a_duplicate_cycle_pointer_is_rejected() {
    rejected(|w| {
        let cycle = column(w, SET, plan(SET).links.state.cycle);
        cycle[1] = cycle[0];
    });
}

/// A `DEREF` row's value column changed, with the memory as it was.
#[test]
fn a_changed_value_is_rejected() {
    rejected(|w| {
        let value = plan(DEREF)
            .links
            .memory
            .iter()
            .find_map(|a| match a.vals[0] {
                Coord::Col(c) => Some(c),
                _ => None,
            })
            .expect("a DEREF access reads a value column");
        let cell = &mut column(w, DEREF, value)[0];
        *cell += F64::ONE;
    });
}

/// The `SET` that writes cap cell 10 pointed at cell 11, which holds the same zero.
#[test]
fn a_read_of_another_cell_with_the_same_value_is_rejected() {
    rejected(|w| {
        let (chunks, pointer) = (chunks(w), plan(SET).links.memory[0].pointer);
        let words = column(w, SET, pointer);
        let row = words.iter().position(|v| chunks.decode(v.0) == Some(10)).expect("a SET writes cell 10");
        words[row] = F64(chunks.encode(11));
    });
}

/// The mirror test binds executed rows only: two ones in an executed row's chunk are rejected, in a padding row's
/// accepted, since a padding row takes part in no link.
#[test]
fn a_non_one_hot_pointer_counts_on_executed_rows_only() {
    let tamper = |row: fn(&Witness) -> usize| {
        move |w: &mut Witness| {
            let (chunks, row, pointer) = (chunks(w), row(w), plan(SET).links.memory[0].pointer);
            let word = &mut column(w, SET, pointer)[row];
            *word = F64(word.0 | other_bit(&chunks, word.0, 0));
        }
    };
    let executed = |w: &Witness| w.layout.rows[SET] - 1;
    let padding = |w: &Witness| {
        assert!(w.layout.rows[SET] < 1 << w.layout.taus[SET], "SET has padding rows");
        w.layout.rows[SET]
    };
    assert_eq!(outcome(tamper(executed)), Err(CpuError::Verifier(VerifierError::LinkTerminal)));
    assert_eq!(outcome(tamper(padding)), Ok(()));
}

/// A padding row keeps pointer 0 and carries any payload.
#[test]
fn a_padding_row_with_any_payload_is_accepted() {
    assert_eq!(
        outcome(|w| {
            let row = w.layout.rows[SET];
            let plan = plan(SET);
            for &c in &plan.values {
                column(w, SET, c)[row] += F64(5);
            }
        }),
        Ok(())
    );
}

/// Rows 0 of `XOR` and `MUL` both combine two zeros, so swapped whole they still agree with memory; only the opcode
/// each table's program read carries differs, and with one weight shared by both tables those two errors would cancel.
#[test]
fn an_xor_and_mul_row_swapped_are_rejected() {
    rejected(|w| {
        for c in 0..tables::tables()[XOR].n_committed_columns() {
            let (x, m) = (column(w, XOR, c)[0], column(w, MUL, c)[0]);
            column(w, XOR, c)[0] = m;
            column(w, MUL, c)[0] = x;
        }
    });
}

/// Row 0 of `XOR` reads cell 25 twice: both pointers take the same extra one.
#[test]
fn two_pointers_of_a_row_with_the_same_non_one_hot_chunk_are_rejected() {
    rejected(|w| {
        let chunks = chunks(w);
        let accesses = plan(XOR).links.memory;
        let extra = other_bit(&chunks, column(w, XOR, accesses[0].pointer)[0].0, 0);
        for access in &accesses[..2] {
            let word = &mut column(w, XOR, access.pointer)[0];
            assert_eq!(chunks.decode(word.0), Some(25));
            *word = F64(word.0 | extra);
        }
    });
}

/// An executed pointer with an extra one, while the index side scatters the honest word, as a prover would that
/// kept the index side's arrays honest and hoped the row side would not notice.
#[test]
fn a_non_one_hot_pointer_with_honest_index_arrays_is_rejected() {
    rejected(|w| {
        let (chunks, pointer) = (chunks(w), plan(SET).links.memory[0].pointer);
        let word = &mut column(w, SET, pointer)[0];
        cheat::INDEX_WORD.set(Some((SET, pointer, 0, word.0)));
        *word = F64(word.0 | other_bit(&chunks, word.0, 0));
    });
}

/// A row's bytecode pointer moved past the program, into the zero padding below `2^W`.
#[test]
fn a_bytecode_pointer_past_the_program_is_rejected() {
    rejected(|w| {
        let (chunks, pointer) = (chunks(w), plan(SET).links.bytecode.pointer);
        let word = &mut column(w, SET, pointer)[0];
        let pc = chunks.decode(word.0).expect("one-hot");
        *word = F64(chunks.encode(pc + PROGRAM_LEN));
    });
}

/// The `KECCAK` row's write of output cell 13 moved past memory, into the zero padding below `2^W`.
#[test]
fn a_keccak_pointer_past_memory_is_rejected() {
    rejected(|w| {
        let (chunks, log_mem) = (chunks(w), w.log_mem);
        let pointer = plan(KECCAK_TABLE)
            .pointers
            .into_iter()
            .find(|&c| chunks.decode(column(w, KECCAK_TABLE, c)[0].0) == Some(13))
            .expect("KECCAK writes cell 13");
        let word = &mut column(w, KECCAK_TABLE, pointer)[0];
        *word = F64(chunks.encode(13 + (1 << log_mem)));
    });
}

#[test]
fn a_wrong_memory_term_is_rejected() {
    rejected(|_| cheat::MEMORY_TERM.set(F192::ONE));
}

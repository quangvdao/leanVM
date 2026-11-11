//! Stage 1's measurement of the row side's cost on a real trace, run by `LEANVM_HOTLINK_R=1` in place of the
//! proof. Temporary: it goes once the row side runs inside the proof, where the tracing spans time it.

use super::pointer::Chunks;
use super::row::{self, Coef, RowColumn, RowPoints, RowTable};
use crate::cpu::{Op, Trace};
use primitives::field::F192;
use std::time::Instant;

/// Every table's pointers as indices: per table the cycle, memory and bytecode pointer columns.
pub(crate) struct TablePointers {
    pub(crate) name: &'static str,
    pub(crate) cycle: Vec<u32>,
    pub(crate) memory: Vec<Vec<u32>>,
    pub(crate) bytecode: Vec<u32>,
}

/// The pointers of a run, Keccak's 26 cells as plain pointers.
pub(crate) fn pointers(trace: &Trace, prog: &[Op]) -> Vec<TablePointers> {
    let ternary = |pc: u32| match prog[pc as usize] {
        Op::Xor { a, b, c } | Op::Mul { a, b, c } => [a, b, c],
        op => unreachable!("{op:?}"),
    };
    let arith = |name, rows: &crate::cpu::Rows<crate::cpu::Xrow>| TablePointers {
        name,
        cycle: rows.iter().map(|r| r.step).collect(),
        memory: (0..3)
            .map(|k| rows.iter().map(|r| r.fp + ternary(r.pc)[k]).collect())
            .collect(),
        bytecode: rows.iter().map(|r| r.pc).collect(),
    };
    let set = &trace.set;
    let deref = &trace.deref;
    let jump = &trace.jump;
    let keccak = &trace.keccak;
    vec![
        arith("XOR", &trace.xor),
        arith("MUL", &trace.mul),
        TablePointers {
            name: "SET",
            cycle: set.iter().map(|r| r.step).collect(),
            memory: vec![
                set.iter()
                    .map(|r| match prog[r.pc as usize] {
                        Op::Set { o, .. } => r.fp + o,
                        op => unreachable!("{op:?}"),
                    })
                    .collect(),
            ],
            bytecode: set.iter().map(|r| r.pc).collect(),
        },
        TablePointers {
            name: "DEREF",
            cycle: deref.iter().map(|r| r.step).collect(),
            memory: (0..3)
                .map(|k| {
                    deref
                        .iter()
                        .map(|r| match prog[r.pc as usize] {
                            Op::Deref { o1, o3, .. } => [r.fp + o1, r.store, r.fp + o3][k],
                            op => unreachable!("{op:?}"),
                        })
                        .collect()
                })
                .collect(),
            bytecode: deref.iter().map(|r| r.pc).collect(),
        },
        TablePointers {
            name: "JUMP",
            cycle: jump.iter().map(|r| r.step).collect(),
            memory: (0..3)
                .map(|k| {
                    jump.iter()
                        .map(|r| match prog[r.pc as usize] {
                            Op::Jump { oc, od, of } => r.fp + [oc, od, of][k],
                            op => unreachable!("{op:?}"),
                        })
                        .collect()
                })
                .collect(),
            bytecode: jump.iter().map(|r| r.pc).collect(),
        },
        TablePointers {
            name: "KECCAK",
            cycle: keccak.iter().map(|r| r.step).collect(),
            memory: (0..crate::keccak::ROW_CELLS)
                .map(|c| {
                    keccak
                        .iter()
                        .map(|r| crate::tables::KeccakOp::of(&prog[r.pc as usize]).cell(r.fp, c))
                        .collect()
                })
                .collect(),
            bytecode: keccak.iter().map(|r| r.pc).collect(),
        },
    ]
}

/// SplitMix64, for challenges that only have to be generic.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn ext(&mut self) -> F192 {
        F192::new(self.next(), self.next(), self.next())
    }

    fn ext_vec(&mut self, n: usize) -> Vec<F192> {
        (0..n).map(|_| self.ext()).collect()
    }
}

fn ceil_log2(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

/// Time the row side over the run's pointers and print the cost per pointer per row.
pub(crate) fn run(trace: &Trace, prog: &[Op], log_mem: usize, cycles: usize) {
    let width = ceil_log2(cycles + 1).max(log_mem).max(ceil_log2(prog.len()));
    let chunks = Chunks::new(width);
    let d = chunks.d();
    let tables = pointers(trace, prog);
    let words: Vec<(Vec<u64>, Vec<Vec<u64>>, Vec<u64>)> = tables
        .iter()
        .map(|t| {
            let encode = |v: &[u32]| v.iter().map(|&s| chunks.encode(s as usize)).collect::<Vec<u64>>();
            (
                encode(&t.cycle),
                t.memory.iter().map(|m| encode(m)).collect(),
                encode(&t.bytecode),
            )
        })
        .collect();
    eprintln!(
        "Hotlink R: W = {width}, widths {:?}, d = {d}, cycles {cycles}, threads {}",
        chunks.widths,
        parallel::num_threads()
    );
    for (t, (cycle, memory, _)) in tables.iter().zip(&words) {
        let same = |w: &[u64]| {
            let pairs = w.len() / 2;
            let chunk_same: usize = (0..pairs)
                .map(|k| {
                    (0..d)
                        .filter(|&i| chunks.bits(w[2 * k], i) == chunks.bits(w[2 * k + 1], i))
                        .count()
                })
                .sum();
            chunk_same as f64 / (pairs * d).max(1) as f64
        };
        let mem_same: f64 = memory.iter().map(|m| same(m)).sum::<f64>() / memory.len() as f64;
        eprintln!(
            "  {:6} rows {:7}  chunks equal in round 1: cycle {:.3}, memory {:.3}",
            t.name,
            cycle.len(),
            same(cycle),
            mem_same
        );
    }
    let mut mix = Mix(0x401);
    for memory_only in [true, false] {
        for skip in [true, false] {
            let mut pointer_rows = 0usize;
            let row_tables: Vec<RowTable> = words
                .iter()
                .map(|(cycle, memory, bytecode)| {
                    let rows = cycle.len();
                    let mut columns = Vec::new();
                    let cert = |mix: &mut Mix| mix.ext_vec(d);
                    if !memory_only {
                        columns.push(RowColumn {
                            words: cycle,
                            omega: Coef::ZERO,
                            act: Coef {
                                constant: mix.ext(),
                                aux: Some(0),
                            },
                            plus: Some(mix.ext()),
                            cert: cert(&mut mix),
                        });
                    }
                    for m in memory {
                        columns.push(RowColumn {
                            words: m,
                            omega: Coef::constant(mix.ext()),
                            act: Coef::ZERO,
                            plus: None,
                            cert: cert(&mut mix),
                        });
                    }
                    if !memory_only {
                        columns.push(RowColumn {
                            words: bytecode,
                            omega: Coef::constant(mix.ext()),
                            act: Coef::ZERO,
                            plus: None,
                            cert: cert(&mut mix),
                        });
                    }
                    pointer_rows += rows * columns.len();
                    RowTable {
                        log_rows: ceil_log2(rows).max(1),
                        rows,
                        explicit: rows,
                        omega: mix.ext_vec(rows),
                        aux: vec![mix.ext_vec(rows)],
                        aux_tail: vec![F192::ZERO],
                        columns,
                    }
                })
                .collect();
            let points = RowPoints::new(&chunks, &mix.ext_vec(width), &mix.ext_vec(chunks.max_width()));
            let start = Instant::now();
            let mut marks = Vec::new();
            let mut challenges = Mix(7);
            let end = row::prove(&chunks, &points, row_tables, skip, &mut |poly| {
                marks.push((Instant::now(), poly.len()));
                challenges.ext()
            });
            let total = start.elapsed();
            let round1 = marks[0].0 - start;
            let rows_done = marks.iter().filter(|m| m.1 == d + 3).count();
            let row_rounds = marks[rows_done - 1].0 - start;
            eprintln!(
                "  {} pointers, skipping {}: {} pointer-rows, row side {:.1} ms (round 1 {:.1} ms, rounds 2..{} {:.1} ms, \
                 columns {:.1} ms): {:.1} ns per pointer per row (claim {})",
                if memory_only { "memory" } else { "all" },
                if skip { "on" } else { "off" },
                pointer_rows,
                total.as_secs_f64() * 1e3,
                round1.as_secs_f64() * 1e3,
                rows_done,
                (row_rounds - round1).as_secs_f64() * 1e3,
                (total - row_rounds).as_secs_f64() * 1e3,
                total.as_nanos() as f64 / pointer_rows as f64,
                !end.claim.is_zero()
            );
        }
    }
}

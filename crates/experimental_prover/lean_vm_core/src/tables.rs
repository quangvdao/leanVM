//! Per-instruction tables (`doc/leanvm/body/07-instruction-tables.tex`). Each opcode is one [`Table`] impl that declares,
//! in one place, its committed columns, how to fill them from the trace, its links (Hotwire, [`crate::link`]): the
//! state it steps, the instruction it executes and the memory cells it touches, each named by a one-hot pointer
//! column, and its degree-2 constraint. Column indices here are *local* (`0..n_committed_columns`);
//! `cpu`'s schema offsets them to global witness columns.
//!
//! Columns are `K`-valued (`F64`); a **machine word** (memory value) is 128-bit (`E = F128`), two `K` lanes.
//! Nothing a row DERIVES is a column, and neither is anything it only READS where one of its pointers points: an
//! operand the instruction holds, a word in a cell a pointer names. A value linear in such reads, the `fp`-relative
//! addresses, the `XOR`/`MUL` results and the `DEREF` store, is a sum of them ([`Read`]), weighed on the side of the
//! table they read; the rest of a value is a degree-2 form over the row's columns. A table commits what enters a
//! product with another read or must be checked lane by lane: `MUL`'s first word, `DEREF`'s pointer, `JUMP`'s
//! `K`-valued words. `JUMP`'s is-nonzero indicator is the one identity any table has, handed to the table sumcheck
//! as a degree-2 form ([`Table::constraint`]).

use crate::cpu::{Jrow, Memory, Op, Rows, Trace};
use crate::leaf::BusForm;
use crate::leaf::Coord::{self, Col, Const, GCol, Prod};
use crate::link::pointer::Chunks;
use primitives::field::{F64, F128, mul_by_g};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

// ---- shared bus vocabulary ---------------------------------------------------

/// `g^k` at compile time (`g = x`, so repeated `mul_by_g` from `g^0 = 1`).
const fn g_pow(k: usize) -> F64 {
    let mut acc = F64::ONE;
    let mut i = 0;
    while i < k {
        acc = mul_by_g(acc);
        i += 1;
    }
    acc
}

// Domain separators (coordinate 0 of every bus tuple): the g-powers g^0, g^1, g^2.
pub(crate) const SEP_STATE: F64 = g_pow(0);
pub(crate) const SEP_MEM: F64 = g_pow(1);
pub(crate) const SEP_BYTECODE: F64 = g_pow(2);

// Opcodes (coordinate 3 of a bytecode tuple).
pub(crate) const OP_XOR: F64 = g_pow(0);
pub(crate) const OP_MUL: F64 = g_pow(1);
pub(crate) const OP_SET: F64 = g_pow(2);
pub(crate) const OP_DEREF: F64 = g_pow(3);
pub(crate) const OP_JUMP: F64 = g_pow(4);
pub(crate) const OP_KECCAK: F64 = g_pow(5);

// ---- links -------------------------------------------------------------------

/// A row's step of the state chain: its cycle pointer, its `(pc, fp)`, and the successor state it derives.
#[derive(Clone, Debug)]
pub struct StateLink {
    pub cycle: usize,
    pub pc: usize,
    pub fp: usize,
    pub next: [Coord; 2],
}

/// A row's read of the program: its pointer, its `pc`, its opcode, and each operand equation `Σ_{s ∈ slots} BC_s =
/// form` it checks against its instruction, weighed by the first slot's challenge (`verifier::layout::BytecodeLink`).
#[derive(Clone, Debug)]
pub struct BytecodeLink {
    pub pointer: usize,
    pub pc: usize,
    pub opcode: F64,
    pub operands: Vec<(Vec<usize>, Coord)>,
}

/// A term of an access's word that the row reads where one of its pointers points, rather than from its columns
/// (`verifier::layout::Read`).
#[derive(Clone, Debug)]
pub enum Read {
    /// `(constant + Σ c·col)·word(σ)`, the word in the cell the pointer column `pointer` names.
    Word {
        pointer: usize,
        constant: F128,
        cols: Vec<(usize, F128)>,
    },
    /// `scale·BC_slot(σ_bc)`, a slot of the row's instruction.
    Slot { slot: usize, scale: F128 },
    /// `scale·fp·BC_slot(σ_bc)`.
    Frame { slot: usize, scale: F128 },
    /// `scale·pc·BC_slot(σ_bc)`.
    Pc { slot: usize, scale: F128 },
}

/// A row's memory access: its pointer and how far past the pointer's cell it reads, the column gating it if any, the
/// address `g^σ` it names, and the word. The address is `addr + g^e·fp·BC_slot(σ_bc)` for `frame = Some((slot, e))`,
/// and the word is `vals[0] + y·vals[1] + Σ reads`.
#[derive(Clone, Debug)]
pub struct Access {
    pub pointer: usize,
    pub shift: usize,
    pub gate: Option<usize>,
    pub addr: Coord,
    pub frame: Option<(usize, u32)>,
    pub vals: [Coord; 2],
    pub reads: Vec<Read>,
}

/// The read of the word at the cell `pointer` names, once.
fn word(pointer: usize) -> Read {
    Read::Word {
        pointer,
        constant: F128::ONE,
        cols: vec![],
    }
}

/// Everything a table links, in local column indices.
#[derive(Clone, Debug)]
pub struct Links {
    pub state: StateLink,
    pub bytecode: BytecodeLink,
    pub memory: Vec<Access>,
}

impl Links {
    /// The pointer columns, in the order Hotwire numbers them: the cycle, the bytecode, then each access's pointer
    /// in order of first use.
    pub fn pointers(&self) -> Vec<usize> {
        let mut out = vec![self.state.cycle, self.bytecode.pointer];
        for a in &self.memory {
            if !out[2..].contains(&a.pointer) {
                out.push(a.pointer);
            }
        }
        out
    }

    /// The largest shift any access takes.
    pub fn max_shift(&self) -> usize {
        self.memory.iter().map(|a| a.shift).max().unwrap_or(0)
    }
}

/// A table's links.
pub fn links(table: &dyn Table) -> Links {
    let mut f = FlushBuilder::new();
    table.flushes(&mut f);
    Links {
        state: f.state.expect("every table steps the state"),
        bytecode: f.bytecode.expect("every table reads the program"),
        memory: f.memory,
    }
}

// ---- flush builder -----------------------------------------------------------

/// Collects a table's links in *local* column indices, and with them the push/pull tuples the bus
/// carried, which the recursion guest's layout still reads. The push/pull of a memory-checked entry
/// differ only by one coordinate carrying the post-increment `g·count` (`GCol`) instead of the
/// pre-increment (`Col`), the count column now being the access's pointer.
pub struct FlushBuilder {
    pub(crate) push: Vec<Vec<Coord>>,
    pub(crate) pull: Vec<Vec<Coord>>,
    state: Option<StateLink>,
    bytecode: Option<BytecodeLink>,
    memory: Vec<Access>,
}

impl FlushBuilder {
    pub(crate) fn new() -> Self {
        Self {
            push: Vec::new(),
            pull: Vec::new(),
            state: None,
            bytecode: None,
            memory: Vec::new(),
        }
    }

    fn pair(&mut self, push: Vec<Coord>, pull: Vec<Coord>) {
        self.push.push(push);
        self.pull.push(pull);
    }

    /// Fall-through state step: the next pc is `g·pc`, fp unchanged.
    pub(crate) fn state_step(&mut self, cycle: usize, pc: usize, fp: usize) {
        self.state_derived(cycle, pc, fp, GCol(pc, 1), Col(fp));
    }

    /// Explicit state transition (JUMP): the next state, which the row DERIVES from its columns rather than
    /// committing, after `(pc, fp)`.
    pub(crate) fn state_derived(&mut self, cycle: usize, pc: usize, fp: usize, npc: Coord, nfp: Coord) {
        self.pair(
            vec![Const(SEP_STATE), npc.clone(), nfp.clone()],
            vec![Const(SEP_STATE), Col(pc), Col(fp)],
        );
        self.state = Some(StateLink {
            cycle,
            pc,
            fp,
            next: [npc, nfp],
        });
    }

    /// Bytecode read at `pc`, by the pointer column `pointer`: the opcode and the operand equations, each at its
    /// slots. The tuple the bus carried holds each equation's form at its first slot, `width` operand slots in all.
    pub(crate) fn bytecode(
        &mut self,
        pc: usize,
        pointer: usize,
        opcode: F64,
        operands: Vec<(Vec<usize>, Coord)>,
        width: usize,
    ) {
        let mut tuple = vec![Const(F64::ZERO); width];
        for (slots, form) in &operands {
            tuple[slots[0] - 4] = form.clone();
        }
        let mut push = vec![Const(SEP_BYTECODE), Col(pc), GCol(pointer, 1), Const(opcode)];
        let mut pull = vec![Const(SEP_BYTECODE), Col(pc), Col(pointer), Const(opcode)];
        push.extend_from_slice(&tuple);
        pull.extend(tuple);
        self.pair(push, pull);
        self.bytecode = Some(BytecodeLink {
            pointer,
            pc,
            opcode,
            operands,
        });
    }

    /// The access to the cell `shift` past the one the pointer column `pointer` names, when the column `gate` is 1
    /// if there is one; at 0 the access takes part in no link. A gated access reads nothing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn access(
        &mut self,
        pointer: usize,
        shift: usize,
        gate: Option<usize>,
        addr: Coord,
        frame: Option<(usize, u32)>,
        vals: [Coord; 2],
        reads: Vec<Read>,
    ) {
        assert!(gate.is_none() || (frame.is_none() && reads.is_empty()));
        let counter = match gate {
            None => GCol(pointer, 1),
            Some(flag) => Coord::Sum(vec![Col(pointer), Prod(flag, pointer, 1), Prod(flag, pointer, 0)]),
        };
        let mut push = vec![Const(SEP_MEM), addr.clone(), counter];
        let mut pull = vec![Const(SEP_MEM), addr.clone(), Col(pointer)];
        push.extend_from_slice(&vals);
        pull.extend(vals.clone());
        self.pair(push, pull);
        self.memory.push(Access {
            pointer,
            shift,
            gate,
            addr,
            frame,
            vals,
            reads,
        });
    }

    /// An ungated access to the cell at `fp·BC_slot(σ_bc)`.
    pub(crate) fn framed(&mut self, pointer: usize, slot: usize, vals: [Coord; 2], reads: Vec<Read>) {
        self.access(pointer, 0, None, Const(F64::ZERO), Some((slot, 0)), vals, reads);
    }
}

const NONE: [Coord; 2] = [Const(F64::ZERO), Const(F64::ZERO)];

// ---- fill context ------------------------------------------------------------

/// Inputs a table needs to fill its columns: the trace rows, the final memory
/// image (for read values), and `g^0..` for O(1) address/operand lookups.
pub struct FillCtx<'a> {
    pub(crate) trace: &'a Trace,
    pub(crate) mem: &'a Memory,
    pub(crate) gpow: &'a [F64],
    pub(crate) prog: &'a [Op],
    /// The pointer layout every pointer column is written in.
    chunks: &'a Chunks,
    /// This table's height `2^tau`, the length of every window in `out`. Rows past the
    /// trace's are padding rows, each holding `padding` ([`Table::padding`]).
    height: usize,
    padding: Vec<F64>,
    /// How many columns [`Self::runs`] wrote: a fill that skipped some would leave the stacked witness holding
    /// uninitialized slots, so [`fill_table`] checks it.
    filled: AtomicUsize,
}

/// Where one column's values go: its window in the stacked witness, piece by piece for a split column, or a
/// private buffer if the column is virtual.
pub(crate) type ColumnOut<'a> = crate::witness::ColMut<'a>;

/// Rows a fill gathers before streaming them out: whole cache lines of each column.
const RUN: usize = 32;

impl<'a> FillCtx<'a> {
    pub(crate) fn new(
        trace: &'a Trace,
        mem: &'a Memory,
        gpow: &'a [F64],
        prog: &'a [Op],
        chunks: &'a Chunks,
        height: usize,
        padding: Vec<F64>,
    ) -> Self {
        Self {
            trace,
            mem,
            gpow,
            prog,
            chunks,
            height,
            padding,
            filled: AtomicUsize::new(0),
        }
    }

    fn g_at(&self, i: u32) -> F64 {
        self.gpow[i as usize]
    }

    /// The one-hot word of pointer `sigma`.
    fn ptr(&self, sigma: u32) -> F64 {
        F64(self.chunks.encode(sigma as usize))
    }

    /// The three frame offsets of an `XOR`/`MUL` row. A row records
    /// only its `(pc, fp)`; the operands are the instruction's, so they are read
    /// back from the bytecode rather than copied into every row (§the trace rows
    /// in `cpu::trace`).
    fn ternary_operands(&self, pc: u32) -> (u32, u32, u32) {
        match self.prog[pc as usize] {
            Op::Xor { a, b, c } | Op::Mul { a, b, c } => (a, b, c),
            op => unreachable!("a three-operand row's pc {pc} holds {op:?}"),
        }
    }

    /// [`Self::runs`] from one closure per row, which returns the row's local columns.
    fn cols<const N: usize, R: Sync>(&self, out: &mut [ColumnOut], rows: &Rows<R>, f: impl Fn(&R) -> [F64; N] + Sync) {
        self.runs(out, rows.len(), |start, len, buf: &mut [_; N]| {
            for j in 0..len {
                for (column, v) in buf.iter_mut().zip(f(rows.get(start + j))) {
                    column[j] = v;
                }
            }
        });
    }

    /// Write all `N` local columns in one pass, so a row, its instruction and the words it names are read once:
    /// `f(start, len, buf)` writes trace rows `start..start + len` of column `c` to `buf[c][..len]`, and each column
    /// holds its padding value to the end of its window. A column with no window is held elsewhere (a KECCAK input
    /// lane, by `q_keccak`) and is not written.
    fn runs<const N: usize>(
        &self,
        out: &mut [ColumnOut],
        n_rows: usize,
        f: impl Fn(usize, usize, &mut [[F64; RUN]; N]) + Sync,
    ) {
        let n = self.height;
        assert_eq!(out.len(), N, "a fill writes every column of its table");
        assert!(n_rows <= n, "a table's rows must fit its cube");
        let dst: [Option<&ColumnOut>; N] = std::array::from_fn(|c| {
            (!out[c].is_empty()).then(|| {
                assert_eq!(out[c].len(), n, "column {c} has the wrong window length");
                &out[c]
            })
        });
        self.filled.store(N, Relaxed);
        let padding: [[F64; RUN]; N] = std::array::from_fn(|c| [self.padding[c]; RUN]);
        // A run of rows is gathered column by column, then streamed out: the
        // stack is read by a later pass, so its lines need not be fetched.
        parallel::for_each_chunk(n.div_ceil(RUN), |first, end| {
            let stream = primitives::stream::Stream::new();
            for b in first..end {
                let start = b * RUN;
                let len = RUN.min(n - start);
                let mut buf = padding;
                if start < n_rows {
                    f(start, len.min(n_rows - start), &mut buf);
                }
                for (window, column) in dst.iter().zip(&buf) {
                    if let Some(window) = window {
                        // SAFETY: block `b` writes rows `start..start + len` of each window, in bounds and disjoint
                        // from every other block's, and the dispatch blocks until every write is finished.
                        unsafe { window.write(&stream, start, &column[..len]) };
                    }
                }
            }
        });
    }

    /// The two `K`-lanes of the 128-bit word in memory cell `addr`.
    fn limbs(&self, addr: u32) -> [F64; 2] {
        self.mem.word(addr as usize).to_limbs().map(F64)
    }
}

/// Fill one table's columns and check that every window was written. The stack is
/// allocated uninitialized, so a table that wrote nothing would be read as
/// indeterminate bytes rather than caught by a length mismatch.
pub(crate) fn fill_table(table: &dyn Table, ctx: &FillCtx, out: &mut [ColumnOut]) {
    table.fill(ctx, out);
    assert_eq!(ctx.filled.load(Relaxed), out.len(), "a table left columns unwritten");
}

// ---- the trait ---------------------------------------------------------------

/// One instruction table. Indices in [`flushes`](Table::flushes) and
/// [`count_columns`](Table::count_columns) are local to this table.
///
/// Every table's last column is its cycle pointer, `CYC`.
pub trait Table: Sync {
    /// Number of committed columns (local indices `0..n_committed_columns`).
    fn n_committed_columns(&self) -> usize;
    /// Local indices of this table's memory and bytecode pointer columns, which hold the
    /// count columns' places. The recursion guest's layout still gives each a count block.
    fn count_columns(&self) -> &'static [usize];
    /// The row a padding row holds, past the rows the program executed
    /// (§sec:e2e-pad). Public, and it has to satisfy the table's constraint. All
    /// zeros by default, which satisfies `JUMP`'s; its pointer words are zero.
    fn padding(&self) -> Vec<F64> {
        vec![F64::ZERO; self.n_committed_columns()]
    }
    /// How many identities [`constraint`](Table::constraint) folds.
    /// Sizes this table's slice of the batch's disjoint `xi`-range (§constraints).
    /// Defaults to none, which is every table but `JUMP`: a relation whose value
    /// rides the bus as a coordinate needs no identity to tie it (§sec:m3).
    fn n_constraints(&self) -> usize {
        0
    }
    /// The table's identities as one degree-2 form over its columns, in local order
    /// (e.g. `jump::V_COND`), identity `i` weighted by `pows[i]`, this table's slice of
    /// the batch's `xi`-powers. The form vanishes on every valid row (§sec:air). The
    /// slice is exactly [`n_constraints`](Table::n_constraints) long: an identity
    /// indexed past its end panics rather than reaching into the next table's range.
    /// The default is the constraint-free case; a table that declares constraints and
    /// forgets to state them trips the assert instead of dropping them.
    fn constraint(&self, pows: &[F128]) -> BusForm {
        assert!(pows.is_empty(), "a table with constraints must state them");
        BusForm::new(self.n_committed_columns())
    }
    /// Declare the table's links ([`links`]).
    fn flushes(&self, f: &mut FlushBuilder);
    /// Fill this table's columns from the trace: `out[i]` is local column `i`'s
    /// window, already at its padded length. Every window must be written in full, by
    /// one `FillCtx::runs` or `FillCtx::cols` over all the columns.
    fn fill(&self, ctx: &FillCtx, out: &mut [ColumnOut]);
}

/// The tables in fixed order `[XOR, MUL, SET, DEREF, JUMP, KECCAK]`, the
/// order of `row_counts` / `taus` throughout `cpu`.
pub const N_TABLES: usize = 6;

pub fn tables() -> [&'static dyn Table; N_TABLES] {
    [
        &Arith { is_xor: true },
        &Arith { is_xor: false },
        &SetTable,
        &DerefTable,
        &JumpTable,
        &KeccakTable,
    ]
}

/// Each table's least log height: 0, but the Keccak reduction's block of steps for `KECCAK`.
pub const TABLE_LOG_FLOORS: [usize; N_TABLES] = [0, 0, 0, 0, 0, crate::keccak::MIN_STEPS.ilog2() as usize];

/// What each cell a `KECCAK` row touches is addressed by, in flush order (the four `m` cells, the
/// four `tail` cells, `b8`, the four `cap` cells, then the thirteen output cells): the offset
/// operand it follows, in the order of their bytecode slots ([`KeccakOp`]), and how far.
const KECCAK_CELLS: [(usize, u32); crate::keccak::ROW_CELLS] = {
    let mut t = [(0, 0); crate::keccak::ROW_CELLS];
    let mut c = 0;
    while c < t.len() {
        t[c] = match c {
            0..4 => (c, 0),
            4..8 => (4, c as u32 - 4),
            8 => (5, 0),
            9..13 => (6, c as u32 - 9),
            _ => (7, c as u32 - 13),
        };
        c += 1;
    }
    t
};

/// A `KECCAK` instruction: its offset operands `m`, `tail`, `b8`, `cap` and `out` in the order of
/// their bytecode slots, its two layout flags, and the output cells it writes.
#[derive(Clone, Copy)]
pub(crate) struct KeccakOp {
    offsets: [u32; 8],
    pub(crate) in_odd: bool,
    pub(crate) out_odd: bool,
    pub(crate) out_cells: crate::cpu::OutCells,
}

impl KeccakOp {
    pub(crate) fn of(op: &Op) -> Self {
        match *op {
            Op::Keccak {
                m,
                tail,
                b8,
                cap,
                out,
                in_odd,
                out_odd,
                out_cells,
            } => Self {
                offsets: [m[0], m[1], m[2], m[3], tail, b8, cap, out],
                in_odd,
                out_odd,
                out_cells,
            },
            op => unreachable!("not a KECCAK: {op:?}"),
        }
    }

    /// Cell `c` (in flush order) of a row at frame `fp`.
    pub(crate) fn cell(&self, fp: u32, c: usize) -> u32 {
        let (offset, step) = self.cell_parts(c);
        fp + offset + step
    }

    /// Cell `c` of a row at frame `fp` is `fp + offset + step`.
    pub(crate) fn cell_parts(&self, c: usize) -> (u32, u32) {
        let (operand, step) = KECCAK_CELLS[c];
        (self.offsets[operand], step)
    }
}

/// The one limb of a row's read cells that holds no lane, `(cell, high)`, in the even layout and
/// in the odd one.
const FREE_LIMB: [(usize, bool); 2] = [free_limb(false), free_limb(true)];

const fn free_limb(odd: bool) -> (usize, bool) {
    let lanes = verifier::layout::keccak_cell_lanes(odd);
    let mut c = 0;
    loop {
        match lanes[c] {
            (None, _) => return (c, false),
            (_, None) => return (c, true),
            _ => c += 1,
        }
    }
}

// ---- XOR / MUL ---------------------------------------------------------------

/// `XOR` and `MUL_NATIVE`. Neither commits its result: the destination's word is linear in the words its pointers
/// read, `v_A + v_B` for `XOR` and `v_A·v_B` for `MUL`, so it is a sum of reads (§sec:m3). `XOR` commits no word at
/// all; `MUL` commits `v_A`, the coefficient of its read of `v_B`. The three addresses are `fp` times the
/// instruction's offsets, read from its slots 4, 5, 6.
struct Arith {
    is_xor: bool,
}

mod arith {
    pub const PC: usize = 0;
    pub const FP: usize = 1;
    // MUL's first word, two K-limbs.
    pub const A_LO: usize = 2;
    pub const A_HI: usize = 3;
    /// The pointers: the three cells, the instruction and the cycle, from `first(is_xor)` on.
    pub const fn first(is_xor: bool) -> usize {
        if is_xor { 2 } else { 4 }
    }
}

impl Table for Arith {
    fn n_committed_columns(&self) -> usize {
        arith::first(self.is_xor) + 5
    }
    fn count_columns(&self) -> &'static [usize] {
        if self.is_xor { &[2, 3, 4, 5] } else { &[4, 5, 6, 7] }
    }
    fn flushes(&self, f: &mut FlushBuilder) {
        use arith::*;
        let p = first(self.is_xor);
        let (ra, rb, rc, rbc, cyc) = (p, p + 1, p + 2, p + 3, p + 4);
        f.state_step(cyc, PC, FP);
        f.bytecode(PC, rbc, if self.is_xor { OP_XOR } else { OP_MUL }, vec![], 5);
        if self.is_xor {
            f.framed(ra, 4, NONE, vec![word(ra)]);
            f.framed(rb, 5, NONE, vec![word(rb)]);
            f.framed(rc, 6, NONE, vec![word(ra), word(rb)]);
        } else {
            f.framed(ra, 4, [Col(A_LO), Col(A_HI)], vec![]);
            f.framed(rb, 5, NONE, vec![word(rb)]);
            let product = Read::Word {
                pointer: rb,
                constant: F128::ZERO,
                cols: vec![(A_LO, F128::ONE), (A_HI, F128::Y)],
            };
            f.framed(rc, 6, NONE, vec![product]);
        }
    }
    fn fill(&self, ctx: &FillCtx, out: &mut [ColumnOut]) {
        let rows = if self.is_xor { &ctx.trace.xor } else { &ctx.trace.mul };
        if self.is_xor {
            ctx.cols(out, rows, |r| {
                let (a, b, c) = ctx.ternary_operands(r.pc);
                [
                    ctx.g_at(r.pc),
                    ctx.g_at(r.fp),
                    ctx.ptr(r.fp + a),
                    ctx.ptr(r.fp + b),
                    ctx.ptr(r.fp + c),
                    ctx.ptr(r.pc),
                    ctx.ptr(r.step),
                ]
            });
        } else {
            ctx.cols(out, rows, |r| {
                let (a, b, c) = ctx.ternary_operands(r.pc);
                let va = ctx.limbs(r.fp + a);
                [
                    ctx.g_at(r.pc),
                    ctx.g_at(r.fp),
                    va[0],
                    va[1],
                    ctx.ptr(r.fp + a),
                    ctx.ptr(r.fp + b),
                    ctx.ptr(r.fp + c),
                    ctx.ptr(r.pc),
                    ctx.ptr(r.step),
                ]
            });
        }
    }
}

// ---- SET ---------------------------------------------------------------------

struct SetTable;

mod set {
    pub const PC: usize = 0;
    pub const FP: usize = 1;
    pub const R: usize = 2;
    pub const RBC: usize = 3;
    pub const CYC: usize = 4;
    pub const N: usize = 5;
}

impl Table for SetTable {
    fn n_committed_columns(&self) -> usize {
        set::N
    }
    fn count_columns(&self) -> &'static [usize] {
        use set::*;
        &[R, RBC]
    }
    fn flushes(&self, f: &mut FlushBuilder) {
        use set::*;
        f.state_step(CYC, PC, FP);
        f.bytecode(PC, RBC, OP_SET, vec![], 5);
        // The cell at `fp·o` (slot 4) holds the immediate, whose two limbs are slots 5 and 6.
        let immediate = vec![
            Read::Slot {
                slot: 5,
                scale: F128::ONE,
            },
            Read::Slot {
                slot: 6,
                scale: F128::Y,
            },
        ];
        f.framed(R, 4, NONE, immediate);
    }
    fn fill(&self, ctx: &FillCtx, out: &mut [ColumnOut]) {
        let rows = &ctx.trace.set;
        ctx.cols(out, rows, |r| {
            let o = match ctx.prog[r.pc as usize] {
                Op::Set { o, .. } => o,
                op => unreachable!("a SET row's pc {} holds {op:?}", r.pc),
            };
            [
                ctx.g_at(r.pc),
                ctx.g_at(r.fp),
                ctx.ptr(r.fp + o),
                ctx.ptr(r.pc),
                ctx.ptr(r.step),
            ]
        });
    }
}

// ---- DEREF -------------------------------------------------------------------

struct DerefTable;

mod deref {
    pub const PC: usize = 0;
    pub const FP: usize = 1;
    // The pointer word is a SINGLE K-lane, so its extension limb is provably zero: the memory read carries a literal
    // zero there. It addresses the store target, `p·o2`, with the offset `o2`.
    pub const P: usize = 2;
    pub const O2: usize = 3;
    // `f_pc + f_fp`, which is 1 exactly when the store takes a K-valued source rather than the local word.
    pub const S: usize = 4;
    pub const R1: usize = 5;
    pub const R2: usize = 6;
    pub const R3: usize = 7;
    pub const RBC: usize = 8;
    pub const CYC: usize = 9;
    pub const N: usize = 10;
}

impl Table for DerefTable {
    fn n_committed_columns(&self) -> usize {
        deref::N
    }
    fn count_columns(&self) -> &'static [usize] {
        use deref::*;
        &[R1, R2, R3, RBC]
    }
    fn flushes(&self, f: &mut FlushBuilder) {
        use deref::*;
        f.state_step(CYC, PC, FP);
        f.bytecode(PC, RBC, OP_DEREF, vec![(vec![5], Col(O2)), (vec![7, 8], Col(S))], 5);
        f.framed(R1, 4, [Col(P), Const(F64::ZERO)], vec![]);
        // The stored word `(1 + s)·v_3 + f_pc·(g²·pc) + f_fp·fp` (§sec:tab-deref): the local word read again, the
        // virtual return target and the frame, the flags being slots 7 and 8.
        let store = vec![
            Read::Word {
                pointer: R3,
                constant: F128::ONE,
                cols: vec![(S, F128::ONE)],
            },
            Read::Pc {
                slot: 7,
                scale: F128::from(g_pow(2)),
            },
            Read::Frame {
                slot: 8,
                scale: F128::ONE,
            },
        ];
        f.access(R2, 0, None, Prod(P, O2, 0), None, NONE, store);
        f.framed(R3, 6, NONE, vec![word(R3)]);
    }
    fn fill(&self, ctx: &FillCtx, out: &mut [ColumnOut]) {
        let rows = &ctx.trace.deref;
        ctx.cols(out, rows, |r| {
            let (o1, o2, o3, mode) = match ctx.prog[r.pc as usize] {
                Op::Deref { o1, o2, o3, mode } => (o1, o2, o3, mode),
                op => unreachable!("a DEREF row's pc {} holds {op:?}", r.pc),
            };
            let p = ctx.mem.word((r.fp + o1) as usize);
            debug_assert!(p.as_base().is_some(), "deref pointer must be K-valued");
            [
                ctx.g_at(r.pc),
                ctx.g_at(r.fp),
                F64(p.c0),
                ctx.g_at(o2),
                F64(u64::from(mode.f_pc() || mode.f_fp())),
                ctx.ptr(r.fp + o1),
                ctx.ptr(r.store),
                ctx.ptr(r.fp + o3),
                ctx.ptr(r.pc),
                ctx.ptr(r.step),
            ]
        });
    }
}

// ---- JUMP --------------------------------------------------------------------

struct JumpTable;

mod jump {
    pub const PC: usize = 0;
    pub const FP: usize = 1;
    // The condition, destination and frame words are all K-valued, so each is a SINGLE lane whose memory read
    // carries a literal zero above it: the read balances only if the stored word is in K, exactly as for the DEREF
    // pointer. A guest branches on g-powers, never on an arbitrary word: `assert a != b` takes an inverse hint
    // instead of a branch (§sec:prog-div-ne). The three cells sit at `fp` times the instruction's slots 4, 5, 6.
    pub const V_COND: usize = 2;
    pub const V_PC: usize = 3;
    pub const V_FP: usize = 4;
    // Local witness columns (committed, never flushed): the inverse hint `w = c⁻¹` and the taken indicator
    // `b = [c ≠ 0]` it certifies (the `JUMP` table in `doc/leanvm/body/07-instruction-tables.tex`). Both are single
    // K lanes.
    pub const W: usize = 5;
    pub const B: usize = 6;
    pub const RC: usize = 7;
    pub const RD: usize = 8;
    pub const RF: usize = 9;
    pub const RBC: usize = 10;
    pub const CYC: usize = 11;
    pub const N: usize = 12;
}

impl Table for JumpTable {
    fn n_committed_columns(&self) -> usize {
        jump::N
    }
    fn count_columns(&self) -> &'static [usize] {
        use jump::*;
        &[RC, RD, RF, RBC]
    }
    fn n_constraints(&self) -> usize {
        2 // the two indicator identities; the selections ride the state push
    }
    /// `b = cond·w` and `cond·(b+1) = 0` (§sec:tab-jump), weighted `pows[0]` and `pows[1]`.
    ///
    /// The two relations together force `b = [cond ≠ 0]`: when `cond ≠ 0` the second
    /// gives `b = 1` (and the first `w = cond⁻¹`); when `cond = 0` the first gives
    /// `b = 0`. The two selections need no identity: the state push carries each as its
    /// own degree-2 coordinate (§sec:m3).
    ///
    /// The condition is `K`-valued, so both identities are single-lane. Its memory
    /// read carries a literal zero above the low limb, so a word outside
    /// `K` cannot balance the bus; the interpreter rejects one outright.
    fn constraint(&self, pows: &[F128]) -> BusForm {
        use jump::*;
        let mut form = BusForm::new(N);
        form.coeffs[B] = pows[0];
        form.coeffs[V_COND] = pows[1];
        form.prods = vec![(V_COND, W, pows[0]), (V_COND, B, pows[1])];
        form
    }
    fn flushes(&self, f: &mut FlushBuilder) {
        use jump::*;
        // The successor state is DERIVED: `b·d + (b+1)·g·pc` and `b·f + (b+1)·fp`,
        // each degree 2 in K columns, so neither successor is committed. Written
        // out in characteristic 2 as `b·d + b·(g·pc) + g·pc`.
        f.state_derived(
            CYC,
            PC,
            FP,
            Coord::Sum(vec![Prod(B, V_PC, 0), Prod(B, PC, 1), GCol(PC, 1)]),
            Coord::Sum(vec![Prod(B, V_FP, 0), Prod(B, FP, 0), Col(FP)]),
        );
        f.bytecode(PC, RBC, OP_JUMP, vec![], 5);
        f.framed(RC, 4, [Col(V_COND), Const(F64::ZERO)], vec![]);
        f.framed(RD, 5, [Col(V_PC), Const(F64::ZERO)], vec![]);
        f.framed(RF, 6, [Col(V_FP), Const(F64::ZERO)], vec![]);
    }
    fn fill(&self, ctx: &FillCtx, out: &mut [ColumnOut]) {
        use jump::*;
        let rows = &ctx.trace.jump;
        let ins = |r: &Jrow| match ctx.prog[r.pc as usize] {
            Op::Jump { oc, od, of } => (oc, od, of),
            op => unreachable!("a JUMP row's pc {} holds {op:?}", r.pc),
        };
        let cell = |r: &Jrow, o: u32| ctx.mem.word((r.fp + o) as usize);
        ctx.runs(out, rows.len(), |start, len, buf: &mut [_; N]| {
            // The is-nonzero witness `w = c⁻¹` (0 where c = 0) by one batched Montgomery inversion per run: column
            // `W` first holds the running product `acc` of the nonzero conditions before its row, and `acc` ends as
            // their full product (nonzero, hence invertible). The conditions lie in K, a subfield, so the inversion
            // runs in K.
            let mut acc = F64::ONE;
            for j in 0..len {
                let r = rows.get(start + j);
                let (oc, od, of) = ins(r);
                // The three cells are K-valued on every row, taken or not (`cpu::execute` rejects anything else),
                // so each is one lane and the memory flush carries literal zeros above it.
                let c = cell(r, oc);
                debug_assert!(c.as_base().is_some(), "a JUMP condition outside K at pc {}", r.pc);
                let c = F64(c.c0);
                let row = [
                    ctx.g_at(r.pc),
                    ctx.g_at(r.fp),
                    c,
                    F64(cell(r, od).c0),
                    F64(cell(r, of).c0),
                    acc,
                    F64(u64::from(!c.is_zero())),
                    ctx.ptr(r.fp + oc),
                    ctx.ptr(r.fp + od),
                    ctx.ptr(r.fp + of),
                    ctx.ptr(r.pc),
                    ctx.ptr(r.step),
                ];
                for (column, v) in buf.iter_mut().zip(row) {
                    column[j] = v;
                }
                if !c.is_zero() {
                    acc *= c;
                }
            }
            let mut inv = acc.inv();
            for j in (0..len).rev() {
                let c = buf[V_COND][j];
                if c.is_zero() {
                    buf[W][j] = F64::ZERO;
                } else {
                    buf[W][j] *= inv;
                    inv *= c;
                }
            }
        });
    }
}

// ---- KECCAK ---------------------------------------------------------------------

/// `KECCAK` (“KECCAK” in `doc/leanvm/body/07-instruction-tables.tex`): the Keccak-f\[1600\]
/// permutation of a state in memory. The four `m` cells and `b8` are addressed *independently*
/// at `fp·o_i` (`o_i = g^{m[i]}`), so a caller hashing e.g. `(tweak, pp)` need not copy them into
/// adjacent cells; the four `tail` cells, the four `cap` cells and the thirteen output cells are
/// consecutive runs based at `fp·o_T`, `fp·o_C` and `fp·o_O`. No address is committed: each rides
/// the bus as the product `fp·o_X·g^k` (§sec:m3). The permutation relating output lanes to input
/// lanes carries no table constraint either: it is proven by the Keccak reduction over
/// `q_keccak` (§keccak), which leaves this table with no identity of its own.
///
/// The cells read and written hold the lanes in the layout their flag names
/// ([`verifier::layout::keccak_cell_lanes`]), the flags bound by the bytecode read, so a cell's limb is the
/// form `a + flag·(a + b)` of its lane in either layout, of degree two. The one limb of a read
/// cell that holds no lane is the column `FREE`; written, it is zero. The fifty lanes are
/// value columns, listed in `n_committed_columns` (they need a local index for the flushes), but
/// `cpu` treats them as VIRTUAL ([`crate::cpu::keccak_value`]): the input lanes are `q_keccak`'s
/// slots, which hold those words, so their claims route there and the fill leaves them alone,
/// and the output lanes' claims are the Keccak reduction's claim on the output.
struct KeccakTable;

pub(crate) mod keccakt {
    use primitives::hash::STATE_LANES;

    pub const PC: usize = 0;
    pub const FP: usize = 1;
    pub const O_OUT: usize = 2; // the g-power of the output run's offset; the seven read offsets are the instruction's
    pub const IN_ODD: usize = O_OUT + 1; // the layout read …
    pub const OUT_ODD: usize = IN_ODD + 1; // … and the layout written
    pub const OUT_TWO: usize = OUT_ODD + 1; // whether output cell 1 is written …
    pub const OUT_ALL: usize = OUT_TWO + 1; // … and cells 2..13
    pub const FREE: usize = OUT_ALL + 1; // the read limb that holds no lane
    pub const V_IN: usize = FREE + 1; // the 25 input lanes, in lane order …
    pub const V_OUT: usize = V_IN + STATE_LANES; // … and the 25 output lanes
    pub const R0: usize = V_OUT + STATE_LANES; // one pointer per offset operand, at the cell it names
    pub const RBC: usize = R0 + 8;
    pub const CYC: usize = RBC + 1;
    pub const N: usize = CYC + 1;
}

/// A cell limb holding lane `even` in the even layout and `odd` in the odd one, as `flag`
/// selects: `a + flag·(a + b)`, a missing lane being `missing`.
fn layout_limb(even: Option<usize>, odd: Option<usize>, values: usize, flag: usize, missing: Option<usize>) -> Coord {
    let col = |lane: Option<usize>| lane.map(|l| values + l).or(missing);
    let (a, b) = (col(even), col(odd));
    if a == b {
        return a.map_or(Const(F64::ZERO), Col);
    }
    Coord::Sum(
        [a.map(Col), a.map(|a| Prod(flag, a, 0)), b.map(|b| Prod(flag, b, 0))]
            .into_iter()
            .flatten()
            .collect(),
    )
}

impl Table for KeccakTable {
    fn n_committed_columns(&self) -> usize {
        keccakt::N
    }
    fn count_columns(&self) -> &'static [usize] {
        const COUNTS: [usize; 9] = {
            let mut c = [0usize; 9];
            let mut i = 0;
            while i < 8 {
                c[i] = keccakt::R0 + i;
                i += 1;
            }
            c[8] = keccakt::RBC;
            c
        };
        &COUNTS
    }
    /// Zeros, but for the output lanes: a padding row, like the Keccak reduction's padding
    /// steps, permutes zeros, in the even layout.
    fn padding(&self) -> Vec<F64> {
        let mut row = vec![F64::ZERO; keccakt::N];
        let mut out = [0; primitives::hash::STATE_LANES];
        primitives::hash::keccak_f(&mut out);
        for (lane, w) in out.into_iter().enumerate() {
            row[keccakt::V_OUT + lane] = F64(w);
        }
        row
    }
    fn flushes(&self, f: &mut FlushBuilder) {
        use crate::keccak::STATE_CELLS;
        use keccakt::*;
        use verifier::layout::keccak_cell_lanes;
        f.state_step(CYC, PC, FP);
        // `out` and the four flags are slots 11..16 of the instruction; the read offsets, slots 4..11, are read where
        // the row's instruction pointer points.
        let operands = (O_OUT..=OUT_ALL).map(|c| (vec![11 + c - O_OUT], Col(c))).collect();
        f.bytecode(PC, RBC, OP_KECCAK, operands, 12);
        // Thirteen cells read, then up to thirteen written: output cell 0 always, cell 1 as `OUT_TWO` says, the
        // rest as `OUT_ALL` says. A consecutive cell is a free ×g on the product's g-power, and the cell `step` past
        // its operand's pointer.
        let (even, odd) = (keccak_cell_lanes(false), keccak_cell_lanes(true));
        for (c, &(operand, step)) in KECCAK_CELLS.iter().enumerate() {
            let (s, values, flag, free) = if c < STATE_CELLS {
                (c, V_IN, IN_ODD, Some(FREE))
            } else {
                (c - STATE_CELLS, V_OUT, OUT_ODD, None)
            };
            let lo = layout_limb(even[s].0, odd[s].0, values, flag, free);
            let hi = layout_limb(even[s].1, odd[s].1, values, flag, free);
            match c.checked_sub(STATE_CELLS) {
                None => {
                    let frame = Some((4 + operand, step));
                    f.access(
                        R0 + operand,
                        step as usize,
                        None,
                        Const(F64::ZERO),
                        frame,
                        [lo, hi],
                        vec![],
                    );
                }
                Some(out) => {
                    let gate = match out {
                        0 => None,
                        1 => Some(OUT_TWO),
                        _ => Some(OUT_ALL),
                    };
                    let addr = Prod(FP, O_OUT, step);
                    f.access(R0 + operand, step as usize, gate, addr, None, [lo, hi], vec![]);
                }
            }
        }
    }
    /// Every column but the input lanes, which are `q_keccak`'s and have no window here. The output lanes are the
    /// input permuted again, since a row writes only the output cells its bytecode requests.
    fn fill(&self, ctx: &FillCtx, out: &mut [ColumnOut]) {
        use crate::keccak::state_of_cells;
        use keccakt::*;
        let rows = &ctx.trace.keccak;
        let flag = |b: bool| F64(u64::from(b));
        ctx.cols(out, rows, |r| {
            let k = KeccakOp::of(&ctx.prog[r.pc as usize]);
            let (two, all) = k.out_cells.flags();
            let (cell, high) = FREE_LIMB[usize::from(k.in_odd)];
            let free = ctx.mem.word(k.cell(r.fp, cell) as usize);
            let mut lanes = state_of_cells(k.in_odd, |c| ctx.mem.word(k.cell(r.fp, c) as usize));
            primitives::hash::keccak_f(&mut lanes);
            let mut v = [F64::ZERO; N];
            v[PC] = ctx.g_at(r.pc);
            v[FP] = ctx.g_at(r.fp);
            v[O_OUT] = ctx.g_at(k.offsets[7]);
            [v[IN_ODD], v[OUT_ODD], v[OUT_TWO], v[OUT_ALL]] = [k.in_odd, k.out_odd, two, all].map(flag);
            v[FREE] = F64(if high { free.c1 } else { free.c0 });
            v[V_OUT..R0].copy_from_slice(&lanes.map(F64));
            v[R0..RBC].copy_from_slice(&k.offsets.map(|o| ctx.ptr(r.fp + o)));
            v[RBC] = ctx.ptr(r.pc);
            v[CYC] = ctx.ptr(r.step);
            v
        });
    }
}

/// The fewest committed columns of a table's row: a run of `n` steps commits at least `n` times
/// it, which bounds the steps of a provable run (the interpreter's step limit).
pub(crate) const MIN_COMMITTED_COLUMNS: usize = {
    let widths = [arith::first(true) + 5, set::N, deref::N, jump::N, keccakt::N];
    let mut min = widths[0];
    let mut i = 1;
    while i < widths.len() {
        if widths[i] < min {
            min = widths[i];
        }
        i += 1;
    }
    min
};

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::field::powers;

    /// `JUMP`'s two identities vanish on an honest row, taken or not.
    #[test]
    fn jump_identities_vanish_on_an_honest_row() {
        let pows = powers(F128::new(0x9e37_79b9_7f4a_7c15, 0x1234_5678_9abc_def0), 2);
        // The condition is K-valued (a single lane on its read, §sec:tab-jump), so the
        // pair is single-lane: `w = c⁻¹` in K too.
        for cond in [F64::ZERO, F64(0x9e37_79b9_7f4a_7c15)] {
            let mut row = vec![F64::ZERO; jump::N];
            let w = cond.inv();
            row[jump::V_COND] = cond;
            row[jump::W] = w;
            row[jump::B] = if cond.is_zero() { F64::ZERO } else { F64::ONE };
            let row: Vec<F128> = row.into_iter().map(F128::from).collect();
            assert_eq!(
                JumpTable.constraint(&pows).eval(&row, false),
                F128::ZERO,
                "cond = {cond:?}"
            );
        }
    }
}

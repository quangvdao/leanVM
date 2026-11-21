//! Per-opcode trace rows, emitted during execution and assembled into a [`Trace`].
//!
//! A row carries only what the witness fill cannot recover: the step's
//! `(pc, fp)`, its step, and `DEREF`'s resolved store index. Everything else is a function of those:
//! operands and immediates come from `prog[pc]`, addresses from `fp` plus those
//! operands, and values from the final memory image, which is write-once and so
//! still holds what each accessed cell held at the time it was accessed. The
//! rows are the interpreter's largest write stream, so what they do not carry
//! they do not pay for, twice: once writing them and once reading them back.

/// `XOR` or `MUL` row: the three cells are `fp·g^{a,b,c}`.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Xrow {
    pub(crate) pc: u32,
    pub(crate) fp: u32, // frame base: address = fp + offset, operand = g^offset
    /// The step that executed the row, its Hotwire cycle pointer.
    pub(crate) step: u32,
}
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Srow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) step: u32,
}
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Drow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) step: u32,
    /// The store target's cell, `log_g(m[fp + o1]) + o2`.
    pub(crate) store: u32,
}
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Jrow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) step: u32,
}

/// `KECCAK` row. The input lanes are read back from the final memory image, the output lanes their permutation.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Krow {
    pub(crate) pc: u32,
    pub(crate) fp: u32,
    pub(crate) step: u32,
}

/// A row's step, which a segment counts from its own start and the merge makes serial.
pub(crate) trait Row: Copy + Send + Sync {
    fn step(&mut self) -> &mut u32;
}

macro_rules! row {
    ($($t:ty),*) => {
        $(impl Row for $t {
            fn step(&mut self) -> &mut u32 {
                &mut self.step
            }
        })*
    };
}
row!(Xrow, Srow, Drow, Jrow, Krow);

/// Rows in fixed-size chunks, so the trace grows without ever moving what it holds:
/// a `Vec` would copy every row written so far each time it doubled.
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Rows<T> {
    chunks: Vec<Vec<T>>,
    len: usize,
}

const CHUNK_LOG: usize = 14;
const CHUNK: usize = 1 << CHUNK_LOG;

impl<T> Rows<T> {
    pub(crate) fn new() -> Self {
        Self {
            chunks: Vec::new(),
            len: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn push(&mut self, row: T) {
        if self.len & (CHUNK - 1) == 0 {
            self.chunks.push(Vec::with_capacity(CHUNK));
        }
        // SAFETY: a chunk was pushed whenever the last one filled, so there is one.
        unsafe { self.chunks.last_mut().unwrap_unchecked() }.push(row);
        self.len += 1;
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn get(&self, i: usize) -> &T {
        &self.chunks[i >> CHUNK_LOG][i & (CHUNK - 1)]
    }
}

impl<T: Row> Rows<T> {
    /// Append the rows `rows` picks from each part, in order and on the pool, each row's step advanced by its part's
    /// first step.
    fn append(&mut self, parts: &[(&RowSet, u32)], rows: impl Fn(&RowSet) -> &Rows<T>) {
        // Each source chunk, its part's first step, and the row it goes to.
        let mut tasks = Vec::new();
        let mut end = self.len;
        for &(part, first) in parts {
            for chunk in &rows(part).chunks {
                tasks.push((chunk, first, end));
                end += chunk.len();
            }
        }
        self.chunks
            .resize_with(end.div_ceil(CHUNK), || Vec::with_capacity(CHUNK));
        let dst: Vec<_> = self
            .chunks
            .iter_mut()
            .map(|c| parallel::SendPtr(c.as_mut_ptr()))
            .collect();
        parallel::for_each(tasks.len(), |t| {
            let (chunk, first, at) = tasks[t];
            for (i, row) in (at..).zip(chunk) {
                let mut row = *row;
                *row.step() += first;
                // SAFETY: row `i < end` lies in its chunk's capacity, and one task writes it.
                unsafe { dst[i >> CHUNK_LOG].add(i & (CHUNK - 1)).write(row) };
            }
        });
        for (c, chunk) in self.chunks.iter_mut().enumerate().skip(self.len >> CHUNK_LOG) {
            // SAFETY: every row of the chunk below `end` was written, above or before.
            unsafe { chunk.set_len((end - c * CHUNK).min(CHUNK)) };
        }
        self.len = end;
    }
}

/// The rows a walk emits, one [`Rows`] per instruction table.
pub(crate) struct RowSet {
    pub(crate) xor: Rows<Xrow>,
    pub(crate) mul: Rows<Xrow>,
    pub(crate) set: Rows<Srow>,
    pub(crate) deref: Rows<Drow>,
    pub(crate) jump: Rows<Jrow>,
    pub(crate) keccak: Rows<Krow>,
}

impl RowSet {
    pub(crate) fn new() -> Self {
        Self {
            xor: Rows::new(),
            mul: Rows::new(),
            set: Rows::new(),
            deref: Rows::new(),
            jump: Rows::new(),
            keccak: Rows::new(),
        }
    }

    /// Append `parts` in order, on the pool, each row's step advanced by its part's first step.
    pub(crate) fn append(&mut self, parts: &[(&RowSet, u32)]) {
        self.xor.append(parts, |p| &p.xor);
        self.mul.append(parts, |p| &p.mul);
        self.set.append(parts, |p| &p.set);
        self.deref.append(parts, |p| &p.deref);
        self.jump.append(parts, |p| &p.jump);
        self.keccak.append(parts, |p| &p.keccak);
    }
}

#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Trace {
    pub(crate) xor: Rows<Xrow>,
    pub(crate) mul: Rows<Xrow>,
    pub(crate) set: Rows<Srow>,
    pub(crate) deref: Rows<Drow>,
    pub(crate) jump: Rows<Jrow>,
    pub(crate) keccak: Rows<Krow>,
}

impl Trace {
    pub(crate) fn new(rows: RowSet) -> Self {
        let RowSet {
            xor,
            mul,
            set,
            deref,
            jump,
            keccak,
        } = rows;
        Self {
            xor,
            mul,
            set,
            deref,
            jump,
            keccak,
        }
    }

    /// Rows per instruction table, in [`crate::cpu::Stats::TABLES`] order.
    pub(crate) fn row_counts(&self) -> [usize; crate::tables::N_TABLES] {
        [
            self.xor.len(),
            self.mul.len(),
            self.set.len(),
            self.deref.len(),
            self.jump.len(),
            self.keccak.len(),
        ]
    }
}

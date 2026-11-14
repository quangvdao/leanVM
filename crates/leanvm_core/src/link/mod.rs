//! Hotwire: one-hot pointer links checked by sum-checks, in place of the bus.
//!
//! Every row of an instruction table names what it touches by pointers: its cycle, the memory cells it reads
//! and writes, and its instruction in the program. Each pointer is committed one-hot ([`pointer`]), so a
//! sum-check over the index `j` ([`index`]) and one over the rows and pointer columns ([`row`]) together show
//! that every row reads the memory, program and neighbouring state its pointers name, and that the cycle
//! pointers enumerate `0..T` once each. The mirror test of Emile Wambsgans shows each executed pointer is
//! one-hot; one-hot addressing follows Shout (Setty and Thaler); the shift by one of the cycle pointer uses the
//! carry matrices of Binary Jolt's precompile link.

pub mod base;
pub mod index;
pub mod pointer;
pub(crate) mod prove;
pub mod row;

#[cfg(test)]
mod negative;

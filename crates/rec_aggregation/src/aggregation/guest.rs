//! Compiling the guest to a fixed point on its own size.

use super::deferred::{scalar_bytecode, stacked_bytecode};
use super::placeholders::placeholder_map;
use lean_compiler::{compile, parse_with_replacements};
use leanvm_core::cpu::Program;

/// Build the guest bytecode and the stacked table its claims are about. Both are
/// cached, so this only moves the cost out of the first prove or verify.
pub fn warm_up() {
    unified_guest();
    stacked_bytecode();
    scalar_bytecode();
}

/// The aggregation bytecode.
///
/// Its recursion verifier still checks the bus proof: Hotlink has not been ported to the guest. So the guest is
/// compiled from the constants the bus layout gave it ([`frozen_placeholder_map`]), and stays the same program,
/// with the same digest, as before Hotlink. A leaf aggregate proves this program with the Hotlink prover; a node
/// aggregate, whose guest would verify Hotlink child proofs, does not work until the guest is ported.
pub fn unified_guest() -> &'static Program {
    static GUEST: std::sync::OnceLock<Program> = std::sync::OnceLock::new();
    GUEST.get_or_init(|| {
        let guest = compile_with(&frozen_placeholder_map());
        assert_eq!(
            guest.prog().len().trailing_zeros() as usize,
            FROZEN_KBC,
            "the frozen guest constants are for a 2^{FROZEN_KBC} program"
        );
        guest
    })
}

/// The log size of the program the frozen constants were computed for: the bus-era guest's fixed point.
const FROZEN_KBC: usize = 19;

/// The guest constants `placeholder_map(19)` gave on the bus layout (commit e4dcb8d54), one `KEY = value` per line.
fn frozen_placeholder_map() -> std::collections::BTreeMap<String, String> {
    include_str!("bus_guest_constants.txt")
        .lines()
        .map(|line| {
            let (key, value) = line.split_once(" = ").expect("a `KEY = value` line");
            (key.to_string(), value.to_string())
        })
        .collect()
}

/// The guest compiled from the current layout's constants: the self-referential compile the port to Hotlink will
/// restore in [`unified_guest`].
#[allow(dead_code, reason = "unused until the guest verifies Hotlink proofs")]
pub(super) fn compile_guest(kbc: usize) -> Program {
    compile_with(&placeholder_map(kbc))
}

fn compile_with(replacements: &std::collections::BTreeMap<String, String>) -> Program {
    // `DBG_PLACEHOLDERS=path`: dump the baked guest constants, to read alongside
    // a `DBG_PROF_DUMP` profile (the guest's shape is entirely in these).
    if let Ok(path) = std::env::var("DBG_PLACEHOLDERS") {
        let dump: String = replacements.iter().map(|(k, v)| format!("{k} = {v}\n")).collect();
        std::fs::write(&path, dump).expect("write DBG_PLACEHOLDERS");
    }
    let guest = compile(
        &parse_with_replacements(include_str!("../../guests/lean_ethereum.py"), replacements)
            .expect("the repository aggregation guest must parse"),
    );
    // `DBG_DISASM=path`: dump the guest's disassembly, to read alongside a
    // `DBG_PROF_DUMP` per-pc profile. Function boundaries lead the dump, so the
    // pc a failed guest check reports can be resolved to a source function
    // without re-deriving the layout by hand.
    if let Ok(path) = std::env::var("DBG_DISASM") {
        let mut ranges: Vec<_> = guest.fn_ranges.iter().collect();
        ranges.sort_by_key(|(_, entry, _)| *entry);
        let mut dump = String::new();
        for (name, entry, len) in ranges {
            dump += &format!("# fn {entry:>7}..{:<7} {name}\n", entry + len);
        }
        dump += &lean_compiler::disassemble(guest.prog());
        std::fs::write(&path, dump).expect("write DBG_DISASM");
    }
    guest
}

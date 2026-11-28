import LeanVM.Wire.Spec
import Statement.Fast.Keccak

/-!
# The wire encoding on the Rust test's vector

The vector of `proof_round_trips_through_its_one_encoding` (`crates/verifier/src/wire.rs`,
leanVM): a tree of four rows `[0, 0, 1]`, `[0, 2, 3]`, `[0, 0, 4]`,
`[0, 5, 0]`, opened at the queries `3, 1, 3` with leaves of `3` words (built by the test: the sibling at level `l` of query `q` is the node
`(q >> l) ^ 1`), and the stream `[E::new(1, 2)]`. As the Rust asserts: the encoding is
`(4 + 1 + 16) + 4 + (4 + 3·4) + 3·4 + 2·2·8 + 2·32 = 149` bytes (the stream: its length, a
one-byte bitmask and one whole element; the phase count; the queries; three sizes; two rows of two
words, the common zero word dropped; two siblings); it decodes, with Keccak-256 over
Keccak-f[1600], to the proof, whose encoding is the bytes; and one more byte is refused.

The vector of `compact_stream_is_canonical`: eleven stream elements, seven of them compact (LEB128
of one, one, one, two, two, three and eight bytes) and four whole, so `4 + 2 + 18 + 64 + 4` bytes;
they round-trip; the empty proof is eight zero bytes; and decoding refuses a bitmask bit past the
stream, an overlong LEB128 of zero, a LEB128 of nine bytes, and a compact element sent whole.

Each check is run compiled (`#guard`, the hash compiled by `Statement.Fast.Keccak`). The kernel
does not reduce the encoding or the decoding themselves: both sort the positions with core's
`List.mergeSort`, defined by well-founded recursion.

The pruning idea and its formalization on leanerVM's main branch are scaraven's
(https://github.com/scaraven).
-/

namespace LeanVMTests.Wire.RustVector

open LeanVM.Parameters LeanVM.Semantics LeanVM.Protocol LeanVM.Protocol.KeccakHash
open LeanVM.Protocol.Compile

/-- The rows, `F64` words as their bits. -/
def rows : List (List K) := [[0, 0, 1], [0, 2, 3], [0, 0, 4], [0, 5, 0]].map (·.map K.ofBits)

/-- `hash_leaf` (`crates/verifier/src/merkle.rs`): Keccak-256 of the words' bytes. -/
def hashWords (words : List K) : Digest := keccak256 (words.flatMap K.toBytes)

/-- `hash_pair`: Keccak-256 of the two digests' bytes. -/
def hashPair (left right : Digest) : Digest := keccak256 (digestBytes left ++ digestBytes right)

/-- The test's `node level i`: the leaf's digest at level `0`, the parent of two leaves above. -/
def node : ℕ → ℕ → Digest
  | 0, i => hashWords (rows.getD i [])
  | _, i => hashPair (hashWords (rows.getD (2 * i) [])) (hashWords (rows.getD (2 * i + 1) []))

/-- `open(4, …, 3, node, row)` at the query `q`: height `2`. -/
def opening (q : ℕ) : Opening :=
  ⟨q, rows.getD q [], (List.range 2).map fun level ↦ node level ((q >>> level) ^^^ 1)⟩

/-- The test's proof. -/
def proof : Proof :=
  ⟨[E.ofLimbs (K.ofBits 1) (K.ofBits 2)], [[3, 1, 3].map opening]⟩

/-- A proof's fields, comparable. -/
def fields (p : Proof) : List (List UInt8) × List (List (ℕ × List K × List Digest)) :=
  (p.stream.map E.toBytes, p.openings.map (·.map fun o ↦ (o.leafIndex, o.leaf, o.path)))

/-- Decoding with Keccak-256 over Keccak-f[1600], as the deployed verifier does. -/
def decode (bytes : Bytes) : Option Proof :=
  Proof.fromBytes (m := Id) (fun s ↦ pure (keccakF1600 s)) bytes

#guard proof.toBytes.length = (4 + 1 + 16) + 4 + (4 + 3 * 4) + 3 * 4 + 2 * 2 * 8 + 2 * 32

#guard (decode proof.toBytes).map fields == some (fields proof)

#guard (decode (proof.toBytes ++ [0])).isNone

/- One encoding: the decoded proof's encoding is the bytes. -/
#guard (decode proof.toBytes).map Proof.toBytes == some proof.toBytes

/-- An element from its two limbs' lanes. -/
def element (c0 c1 : UInt64) : E := E.ofLimbs (K.ofLane c0) (K.ofLane c1)

/-- The stream of `compact_stream_is_canonical`, around both ends of each LEB128 length and of
`COMPACT_LIMIT`. -/
def compactCases : List E :=
  [element 0 0, element 1 0, element 127 0, element 128 0, element (2 ^ 14 - 1) 0,
    element (2 ^ 14) 0, element (2 ^ 56 - 1) 0, element (2 ^ 56) 0,
    element 0xFFFFFFFFFFFFFFFF 0, element 0 1, element 42 99]

/-- The proof of `compact_stream_is_canonical`: the stream, no phase. -/
def compactProof : Proof := ⟨compactCases, []⟩

/-- The compact proof's bytes. -/
def compactBytes : Bytes := compactProof.toBytes

/-- Its stream elements, comparable. -/
def streamFields (p : Proof) : List (List UInt8) := p.stream.map E.toBytes

#guard compactBytes.length = 4 + 2 + 18 + 64 + 4

#guard (decode compactBytes).map streamFields == some (streamFields compactProof)

#guard (decode compactBytes).map Proof.toBytes == some compactBytes

#guard (⟨[], []⟩ : Proof).toBytes == List.replicate 8 0

#guard ((decode (List.replicate 8 0)).map streamFields) == some []

/- A bitmask bit past the stream (the high bit of the second bitmask byte). -/
#guard (decode (compactBytes.set 5 (compactBytes[5]! ||| 0x80))).isNone

/- An overlong LEB128 of zero. -/
#guard (decode (LeanVM.Prover.u32Bytes 1 ++ [0x01, 0x80, 0x00] ++
  LeanVM.Prover.u32Bytes 0)).isNone

/- A LEB128 of nine bytes. -/
#guard (decode (LeanVM.Prover.u32Bytes 1 ++ [0x01] ++ List.replicate 8 0x80 ++ [0x01] ++
  LeanVM.Prover.u32Bytes 0)).isNone

/- A compact element sent whole. -/
#guard (decode (LeanVM.Prover.u32Bytes 1 ++ [0x00] ++ K.toBytes (K.ofLane 7) ++ K.toBytes 0 ++
  LeanVM.Prover.u32Bytes 0)).isNone

end LeanVMTests.Wire.RustVector

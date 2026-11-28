/-
  LeanVM.Wire.Encoding

  The proof's wire encoding, transcribed from leanVM's `crates/verifier/src/wire.rs`:
  the decoding `Proof.fromBytes` (`Proof::from_bytes`, `read_phase`) and the
  encoding `Proof.toBytes` (`Proof::to_bytes`, `write_phase`).
-/

module

public import Statement.Verifier.Lib

/-!
# The proof's wire encoding

The end statements (`Statement/`) are about the structured `Proof`; the wire encoding is a
compression layer on it. Its transcription is here, and its properties are proved in the other
`LeanVM.Wire` modules: the round trip (`LeanVM.Wire.roundTrip`), the one encoding
(`LeanVM.Wire.oneEncoding`), and that the honest prover's proofs are within the round trip's
hypotheses (`LeanVM.Wire.proveWithHash_wireDecodable`). That this transcription is the Rust's is
test evidence: the wire differential (`scripts/wire-diff.py`).
The pruning idea, and its formalization on leanerVM's main branch, are scaraven's
(https://github.com/scaraven).
-/

@[expose] public section

namespace LeanVM.Protocol.Compile

open LeanVM.Parameters LeanVM.Semantics LeanVM.Protocol.KeccakHash

/-! ## The proof's wire encoding (`crates/verifier/src/wire.rs`)

A proof travels as one byte string, `Proof::to_bytes` (`Proof.toBytes`, below), and the verifier
decodes it, `Proof::from_bytes` (`Proof.fromBytes`, below), then verifies the proof. Every count and
size is a little-endian 32-bit word, every `K` word eight little-endian bytes, every digest its 32
bytes. The stream travels as its length, a bitmask of one bit per element marking the compact ones
(`compactScalar`: a `y` limb of zero and a `y^0` limb below `2 ^ 56`), then each element: a compact
one as its `y^0` limb in canonical unsigned LEB128 (one to eight bytes), any other as its two limbs'
16 bytes (`readStream`). A phase travels pruned: the queries' positions, the height, the leaf width
and the row width, the leaf at each distinct position as its last row-width words (the words before
are zero in every leaf of the phase), and the siblings no other path of the phase gives, bottom
level first and left to right; decoding recomputes the others by hashing, Keccak-256 over `π` as the
verifier hashes (`hash_leaf`, `hash_pair`, `crates/verifier/src/merkle.rs`). Decoding verifies
nothing: it refuses everything but the one encoding of a proof within the bounds below.

The Rust hashes a phase's shared nodes while it reads the phase's siblings; which siblings it reads
depends on the positions only, never on a hash, so here the bytes are read first (`readProof`) and
the shared nodes hashed after (`unprunePhase`), in the Rust's order. On bytes the Rust refuses it
has hashed some nodes, which changes nothing it returns. Sizes are natural numbers here, `usize` on
a 64-bit host in the Rust, where no product of the decoding overflows.
-/

/-- `COMPACT_LIMIT`: the bound below which a stream scalar's `y^0` limb travels compact
(`wire.rs`). -/
def wireCompactLimit : ℕ := 2 ^ 56

/-- `is_compact`: a stream element travels compact when its `y` limb is zero and its `y^0` limb is
below `wireCompactLimit` (`wire.rs`). -/
def compactScalar (x : E) : Bool :=
  decide (x.limb 1 = 0) && decide ((x.limb 0).lane.toNat < wireCompactLimit)

/-- `MAX_PHASES`: the most phases a decoded proof has (`wire.rs`). -/
def wireMaxPhases : ℕ := 8

/-- `MAX_QUERIES`: the most queries a decoded phase has (`wire.rs`). -/
def wireMaxQueries : ℕ := 256

/-- `MAX_LEAF_WORDS`: the most words a decoded leaf has (`wire.rs`). -/
def wireMaxLeafWords : ℕ := 64

/-- `MAX_HEIGHT`: the greatest height of a decoded phase's tree (`wire.rs`). -/
def wireMaxHeight : ℕ := 32

/-- `dedup_by_key`: keep the first of each run of consecutive elements with the same key. -/
def dedupByKey {α β : Type} [DecidableEq β] (key : α → β) : List α → List α
  | [] => []
  | a :: rest => match dedupByKey key rest with
    | [] => [a]
    | b :: kept => if key a = key b then a :: kept else a :: b :: kept

/-- The distinct positions of a phase's queries, ascending (`sort_unstable`, `dedup`). -/
def distinctPositions (queries : List ℕ) : List ℕ :=
  dedupByKey id (queries.mergeSort fun a b ↦ a ≤ b)

/-- The nodes of level `level` a phase's paths climb through: their positions shifted right by
`level`, distinct and ascending, for positions distinct and ascending. -/
def levelNodes (positions : List ℕ) (level : ℕ) : List ℕ :=
  dedupByKey id (positions.map fun position ↦ position >>> level)

/-- The nodes of a level whose sibling a phase carries: those whose sibling `node ^^^ 1` is not a
node of the level, in order (`wire.rs`, `wire.rs`). -/
def carriedSiblings (nodes : List ℕ) : List ℕ :=
  nodes.filter fun node ↦ node ^^^ 1 ∉ nodes

/-- The leaf of `leafWords` words whose words after the leading zeros are `row` (`decode`,
`crates/verifier/src/wire.rs`). -/
def paddedLeaf (row : List K) (leafWords : ℕ) : List K :=
  List.replicate (leafWords - row.length) 0 ++ row

/-! ### Reading (`Reader`, `wire.rs`): each read gives the bytes left or `none` -/

/-- `Reader::bytes`: the first `length` bytes and the rest. -/
def readBytes (length : ℕ) (input : Bytes) : Option (Bytes × Bytes) :=
  if length ≤ input.length then some (input.take length, input.drop length) else none

/-- `Reader::u32`: a little-endian 32-bit word. -/
def readU32 (input : Bytes) : Option (ℕ × Bytes) :=
  (readBytes 4 input).map fun (head, rest) ↦ ((KeccakHash.littleEndianLane head 0).toNat, rest)

/-- `count` little-endian 32-bit words, in order. -/
def readU32s : (count : ℕ) → Bytes → Option (List ℕ × Bytes)
  | 0, input => some ([], input)
  | count + 1, input => do
    let (value, input) ← readU32 input
    let (values, input) ← readU32s count input
    some (value :: values, input)

/-- `Reader::words`: `count` words of `K`, each eight little-endian bytes. -/
def readWords (count : ℕ) (input : Bytes) : Option (List K × Bytes) :=
  (readBytes (8 * count) input).map fun (head, rest) ↦
    ((List.range count).map fun i ↦ K.ofLane (KeccakHash.littleEndianLane head i), rest)

/-- `Reader::hash`: a digest, its 32 bytes. -/
def readDigest (input : Bytes) : Option (Digest × Bytes) :=
  (readBytes 32 input).map fun (head, rest) ↦
    (#v[KeccakHash.littleEndianLane head 0, KeccakHash.littleEndianLane head 1,
      KeccakHash.littleEndianLane head 2, KeccakHash.littleEndianLane head 3], rest)

/-- `Reader::varint`, with `fuel` bytes left to read: canonical unsigned LEB128, seven bits a
byte, least significant first, the high bit set on every byte but the last. Refused past `fuel`
bytes, and refused when the last byte is zero unless it is the first (`first`), the only zero last
byte of a canonical encoding. The Rust reads at most eight bytes (`readVarint 8 true`). -/
def readVarint : (fuel : ℕ) → (first : Bool) → Bytes → Option (ℕ × Bytes)
  | 0, _, _ => none
  | _ + 1, _, [] => none
  | fuel + 1, first, byte :: input =>
    if byte.toNat < 128 then
      if first = true ∨ byte.toNat ≠ 0 then some (byte.toNat, input) else none
    else
      (readVarint fuel false input).map fun (high, input) ↦ (byte.toNat - 128 + 128 * high, input)

/-- Bytes as the little-endian number `∑ b_k 256^k`. -/
def leValue : List UInt8 → ℕ
  | [] => 0
  | byte :: bytes => byte.toNat + 256 * leValue bytes

/-- The low `count` bits of a number, least significant first. -/
def bitsOf : (count : ℕ) → ℕ → List Bool
  | 0, _ => []
  | count + 1, value => decide (value % 2 = 1) :: bitsOf count (value / 2)

/-- One stream element, compact or not as its bitmask bit says: a compact one as its `y^0` limb's
LEB128 (`readVarint 8 true`), its `y` limb zero; any other as its two limbs' words, refused when the
element is compact, which has the shorter encoding. -/
def readElement (compact : Bool) (input : Bytes) : Option (E × Bytes) :=
  if compact then
    (readVarint 8 true input).map fun (value, input) ↦
      (E.ofLimbs (K.ofLane (UInt64.ofNat value)) 0, input)
  else do
    let (words, input) ← readWords 2 input
    let element := E.ofLimbs (words.getD 0 0) (words.getD 1 0)
    guard (compactScalar element = false)
    some (element, input)

/-- The stream elements, one per bitmask bit, in order. -/
def readElements : List Bool → Bytes → Option (List E × Bytes)
  | [], input => some ([], input)
  | compact :: flags, input => do
    let (element, input) ← readElement compact input
    let (elements, input) ← readElements flags input
    some (element :: elements, input)

/-- The stream (`Proof::from_bytes`, `wire.rs`): its length `n`; the bitmask, `⌈n / 8⌉` bytes,
bit `i` (bit `i % 8` of byte `i / 8`) marking element `i` compact, refused unless its bits from `n`
on are zero, here as the bytes' little-endian number being below `2 ^ n`; then the elements
(`readElements`). The Rust first refuses bytes too few for the bitmask and one byte per element,
which `readElements` refuses too. -/
def readStream (input : Bytes) : Option (List E × Bytes) := do
  let (length, input) ← readU32 input
  let (mask, input) ← readBytes ((length + 7) / 8) input
  guard (leValue mask < 2 ^ length)
  readElements (bitsOf length (leValue mask)) input

/-- `count` digests, in order. -/
def readDigests : (count : ℕ) → Bytes → Option (List Digest × Bytes)
  | 0, input => some ([], input)
  | count + 1, input => do
    let (digest, input) ← readDigest input
    let (digests, input) ← readDigests count input
    some (digest :: digests, input)

/-- `count` rows of `rowWords` words each, in order. -/
def readRows (rowWords : ℕ) : (count : ℕ) → Bytes → Option (List (List K) × Bytes)
  | 0, input => some ([], input)
  | count + 1, input => do
    let (row, input) ← readWords rowWords input
    let (rows, input) ← readRows rowWords count input
    some (row :: rows, input)

/-- The siblings a phase carries, from level `level` up for `remaining` levels: at each level, the
sibling of each node of `carriedSiblings` in order, keyed by its node (`wire.rs`). -/
def readCarriedSiblings (positions : List ℕ) :
    (level remaining : ℕ) → Bytes → Option (List (List (ℕ × Digest)) × Bytes)
  | _, 0, input => some ([], input)
  | level, remaining + 1, input => do
    let carried := carriedSiblings (levelNodes positions level)
    let (digests, input) ← readDigests carried.length input
    let (above, input) ← readCarriedSiblings positions (level + 1) remaining input
    some (carried.zip digests :: above, input)

/-- A phase as its encoding gives it, before the shared nodes are hashed: the queries' positions in
sampling order, the tree's height and leaf width, the distinct positions ascending with their
leaves, and, level by level, the siblings it carries keyed by their node. -/
structure PrunedPhase where
  /-- The queries' positions, in the order the verifier samples them. -/
  queries : List ℕ
  /-- The tree's height. -/
  height : ℕ
  /-- The leaf width, in words. -/
  leafWords : ℕ
  /-- The distinct positions, ascending. -/
  positions : List ℕ
  /-- The leaf at each distinct position, whole. -/
  leaves : List (List K)
  /-- Level by level from the leaves, the siblings the phase carries, keyed by their node. -/
  carried : List (List (ℕ × Digest))

/-- `read_phase` up to the hashing (`wire.rs`): the queries' count, at most
`wireMaxQueries`, and positions; the height, the leaf width and the row width; refused unless the
height is at most `wireMaxHeight`, the leaf width at most `wireMaxLeafWords`, the row width at most
the leaf width, and some query's position is below `2 ^ height`, every one being; the rows at
the distinct positions, ascending, refused unless the row width is the least (some leaf's first row
word is nonzero, or the rows are empty); then the carried siblings. -/
def readPhase (input : Bytes) : Option (PrunedPhase × Bytes) := do
  let (count, input) ← readU32 input
  guard (count ≤ wireMaxQueries)
  let (queries, input) ← readU32s count input
  let (height, input) ← readU32 input
  let (leafWords, input) ← readU32 input
  let (rowWords, input) ← readU32 input
  let positions := distinctPositions queries
  guard (height ≤ wireMaxHeight ∧ leafWords ≤ wireMaxLeafWords ∧ rowWords ≤ leafWords ∧
    ∃ last ∈ positions.getLast?, last >>> height = 0)
  let (rows, input) ← readRows rowWords positions.length input
  let leaves := rows.map fun row ↦ paddedLeaf row leafWords
  guard ¬ (0 < rowWords ∧ ∀ leaf ∈ leaves, leaf.getD (leafWords - rowWords) 0 = 0)
  let (carried, input) ← readCarriedSiblings positions 0 height input
  some (⟨queries, height, leafWords, positions, leaves, carried⟩, input)

/-- `count` phases, in order. -/
def readPhases : (count : ℕ) → Bytes → Option (List PrunedPhase × Bytes)
  | 0, input => some ([], input)
  | count + 1, input => do
    let (phase, input) ← readPhase input
    let (phases, input) ← readPhases count input
    some (phase :: phases, input)

/-- `Proof::from_bytes` up to the hashing (`wire.rs`): the stream (`readStream`); the number of
phases, at most `wireMaxPhases`, and the phases; refused unless no byte is left. -/
def readProof (bytes : Bytes) : Option (List E × List PrunedPhase) := do
  let (stream, input) ← readStream bytes
  let (phaseCount, input) ← readU32 input
  guard (phaseCount ≤ wireMaxPhases)
  let (phases, input) ← readPhases phaseCount input
  guard input.isEmpty
  some (stream, phases)

section Unprune

variable {m : Type → Type} [Monad m] (π : KeccakState → m KeccakState)

/-- One level of the climb (`wire.rs`), from the level's nodes ascending with their
digests and the siblings the phase carries at the level: each node's sibling, the level's node at
`node ^^^ 1` if there is one, else the carried one; and the parents ascending, each hashed once, at
its first child, the left child's digest first (`hash_pair`). -/
def climbLevel (nodes carried : List (ℕ × Digest)) :
    m (List (ℕ × Digest) × List (ℕ × Digest)) := do
  let siblingOf (node : ℕ) : Digest :=
    ((nodes.lookup (node ^^^ 1)).orElse fun _ ↦ carried.lookup node).getD default
  let firstChildren := nodes.filter fun (node, _) ↦ node % 2 = 0 ∨ (node ^^^ 1) ∉ nodes.map (·.1)
  let parents ← firstChildren.mapM fun (node, digest) ↦ do
    let (left, right) :=
      if node % 2 = 0 then (digest, siblingOf node) else (siblingOf node, digest)
    let parent ← keccak256With π (digestBytes left ++ digestBytes right)
    pure (node >>> 1, parent)
  pure (nodes.map fun (node, _) ↦ (node, siblingOf node), parents)

/-- Climb the levels from the nodes of the lowest, the carried siblings level by level: each
level's siblings keyed by their node. -/
def climbLevels : List (List (ℕ × Digest)) → List (ℕ × Digest) → m (List (List (ℕ × Digest)))
  | [], _ => pure []
  | carried :: above, nodes => do
    let (siblings, parents) ← climbLevel π nodes carried
    let rest ← climbLevels above parents
    pure (siblings :: rest)

/-- Unprune a phase back into its paths (`wire.rs`): hash the leaves (`hash_leaf`), in
order, climb the levels, and give each query the opening at its position: the leaf there and, at
each level, the sibling of the node its position is under. -/
def unprunePhase (phase : PrunedPhase) : m (List Opening) := do
  let leafDigests ← phase.leaves.mapM fun leaf ↦ keccak256With π (leaf.flatMap K.toBytes)
  let siblings ← climbLevels π phase.carried (phase.positions.zip leafDigests)
  pure (phase.queries.map fun query ↦
    ⟨query, ((phase.positions.zip phase.leaves).lookup query).getD [],
      (List.range phase.height).map fun level ↦
        ((siblings.getD level []).lookup (query >>> level)).getD default⟩)

/-- **The proof the bytes encode** (`Proof::from_bytes`, `wire.rs`), over the permutation `π`:
the bytes read (`readProof`), then each phase unpruned, in order; `none` unless the bytes are
exactly one proof's encoding within the bounds. Nothing is verified. -/
def Proof.fromBytes (bytes : Bytes) : m (Option Proof) :=
  match readProof bytes with
  | none => pure none
  | some (stream, phases) => do
    let openings ← phases.mapM (unprunePhase π)
    pure (some ⟨stream, openings⟩)

end Unprune

end LeanVM.Protocol.Compile

namespace LeanVM.Prover

open LeanVM.Parameters LeanVM.Semantics LeanVM.Protocol LeanVM.Protocol.Compile

/-! ## The proof's wire encoding (`Proof::to_bytes`, `crates/verifier/src/wire.rs`)

The one encoding of a proof, which `Proof.fromBytes` (above) decodes. It hashes
nothing: a phase is pruned into the fewest siblings that authenticate all its paths together, a
sibling left out where another path's node gives it. The Rust asserts that the proof is within the
decoding's bounds and every count fits in 32 bits; here a count is written modulo `2^32`, and the
first path of a phase gives its height and leaf width (none for no path, where the Rust panics).
-/

open LeanVM.Protocol.KeccakHash in
/-- `write_u32`: the four little-endian bytes of a count, modulo `2^32` (`wire.rs`). -/
def u32Bytes (value : ℕ) : List UInt8 :=
  (List.range 4).map fun k ↦ UInt8.ofNat (value / 256 ^ k)

/-- `write_varint`: canonical unsigned LEB128, seven bits a byte, least significant first, the
high bit set on every byte but the last (`wire.rs`). -/
def varintBytes (value : ℕ) : List UInt8 :=
  if value < 128 then [UInt8.ofNat value]
  else UInt8.ofNat (value % 128 + 128) :: varintBytes (value / 128)
termination_by value
decreasing_by omega

/-- The `count` little-endian bytes of a number, modulo `256 ^ count`. -/
def leBytes : (count : ℕ) → ℕ → List UInt8
  | 0, _ => []
  | count + 1, value => UInt8.ofNat (value % 256) :: leBytes count (value / 256)

/-- Bits as the number they are the low bits of, least significant first. -/
def bitsValue : List Bool → ℕ
  | [] => 0
  | bit :: bits => bit.toNat + 2 * bitsValue bits

open LeanVM.Protocol.KeccakHash in
/-- One stream element's bytes: a compact one as its `y^0` limb's LEB128, any other as its two
limbs' 16 bytes (`wire.rs`). -/
def elementBytes (x : E) : List UInt8 :=
  if compactScalar x then varintBytes (x.limb 0).lane.toNat else x.toBytes

/-- The stream's bytes (`Proof::to_bytes`, `wire.rs`): its length `n`; the bitmask, `⌈n / 8⌉`
bytes, bit `i` set when element `i` is compact; then each element (`elementBytes`). -/
def streamToBytes (stream : List E) : List UInt8 :=
  u32Bytes stream.length ++
    leBytes ((stream.length + 7) / 8) (bitsValue (stream.map compactScalar)) ++
    stream.flatMap elementBytes

open LeanVM.Protocol.KeccakHash in
/-- `write_phase` (`wire.rs`): the queries' count and positions, in sampling order; the
height, the leaf width and the row width, the leaf width less the leading zero words every distinct
leaf has; the leaf at each distinct position, ascending (the first path at it), as its last
row-width words; then, level by level from the leaves, the sibling of each node the distinct paths
climb through (the first path's) whose sibling node is not one of them, left to right. -/
def phaseToBytes (paths : List Opening) : List UInt8 :=
  let height := (paths.head?.map fun path ↦ path.path.length).getD 0
  let leafWords := (paths.head?.map fun path ↦ path.leaf.length).getD 0
  let distinct := dedupByKey Opening.leafIndex
    (paths.mergeSort fun a b ↦ a.leafIndex ≤ b.leafIndex)
  let zeros := ((distinct.map fun path ↦
    (path.leaf.takeWhile fun word ↦ decide (word = 0)).length).min?).getD 0
  u32Bytes paths.length ++ paths.flatMap (fun path ↦ u32Bytes path.leafIndex) ++
    u32Bytes height ++ u32Bytes leafWords ++ u32Bytes (leafWords - zeros) ++
    distinct.flatMap (fun path ↦ (path.leaf.drop zeros).flatMap K.toBytes) ++
    (List.range height).flatMap fun level ↦
      let nodes := dedupByKey Prod.fst (distinct.map fun path ↦
        (path.leafIndex >>> level, path.path.getD level default))
      (nodes.filter fun (node, _) ↦ node ^^^ 1 ∉ nodes.map Prod.fst).flatMap
        fun (_, sibling) ↦ digestBytes sibling

/-- **The proof's one encoding** (`Proof::to_bytes`, `wire.rs`): the stream (`streamToBytes`);
the number of phases, then each phase pruned (`phaseToBytes`). -/
def _root_.LeanVM.Protocol.Compile.Proof.toBytes (proof : Proof) : List UInt8 :=
  streamToBytes proof.stream ++
    u32Bytes proof.openings.length ++ proof.openings.flatMap phaseToBytes

end LeanVM.Prover

end

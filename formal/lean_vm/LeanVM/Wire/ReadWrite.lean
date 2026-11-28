/-
  LeanVM.Wire.ReadWrite

  The wire encoding's readers and writers are inverse: each reader (`readU32`, `readWords`, …,
  `LeanVM.Wire.Encoding`) reads back what its writer (`Prover.u32Bytes`, `K.toBytes`, …)
  wrote, and reads nothing else.
-/

module

public import LeanVM.Wire.Encoding
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Readers and writers of the wire encoding

Each reader of `Proof.fromBytes` reads back its writer's bytes and the rest untouched
(`readU32_u32Bytes`, `readWords_toBytes`, `readRows_toBytes`, …), and,
conversely, what a reader accepts is its writer's bytes of what it returns
(`readU32_eq_some_iff`, `readWords_eq_some_iff`, …: the one encoding of each field). The stream
reads back (`readStream_streamToBytes`) and is read only from its one encoding
(`readStream_eq_some_iff`): the LEB128 of a compact element is canonical (`readVarint_eq_some_iff`),
the bitmask's bits past the stream are zero, and an element sent whole is not compact.

The base facts are about one 64-bit lane: `littleEndianLane` of a word's bytes is the word
(`littleEndianLane_wordBytes_append`), and the bytes of the lane of eight bytes are those bytes
(`wordBytes_littleEndianLane`).

The pruning idea and its formalization on leanerVM's main branch are scaraven's
(https://github.com/scaraven).
-/

namespace LeanVM.Wire

open LeanVM.Parameters LeanVM.Protocol LeanVM.Protocol.KeccakHash LeanVM.Protocol.Compile

public section

/-! ## One lane -/

/-- The bytes `b_k` from `8 i` on, as the number `∑ b_k 256^k` below `2 ^ (8 m)`. -/
private theorem sum_bytes_lt (f : ℕ → UInt8) :
    ∀ m, ∑ k ∈ Finset.range m, (f k).toNat * 2 ^ (8 * k) < 2 ^ (8 * m)
  | 0 => by simp
  | m + 1 => by
    rw [Finset.sum_range_succ]
    have ih := sum_bytes_lt f m
    have hf := (f m).toNat_lt
    calc _ < 2 ^ (8 * m) + (f m).toNat * 2 ^ (8 * m) := by omega
      _ = (1 + (f m).toNat) * 2 ^ (8 * m) := by ring
      _ ≤ 2 ^ 8 * 2 ^ (8 * m) := Nat.mul_le_mul_right _ (by omega)
      _ = 2 ^ (8 * (m + 1)) := by rw [← pow_add]; ring_nf

private theorem toNat_foldl_lane (bs : List UInt8) (i : ℕ) :
    ∀ m, m ≤ 8 → ((List.range m).foldl (fun acc k ↦ acc ||| ((bs.getD (8 * i + k) 0).toUInt64 <<<
      (8 * k).toUInt64)) 0).toNat =
        ∑ k ∈ Finset.range m, (bs.getD (8 * i + k) 0).toNat * 2 ^ (8 * k)
  | 0, _ => rfl
  | m + 1, hm => by
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Finset.sum_range_succ,
      UInt64.toNat_or, toNat_foldl_lane bs i m (by omega), UInt64.toNat_shiftLeft]
    have h8 : (8 * m).toUInt64.toNat % 64 = 8 * m := by simp; omega
    have hb := (bs.getD (8 * i + m) 0).toNat_lt
    have hlt : (bs.getD (8 * i + m) 0).toNat * 2 ^ (8 * m) < 2 ^ 64 :=
      calc _ < 2 ^ 8 * 2 ^ (8 * m) := Nat.mul_lt_mul_of_pos_right (by simpa using hb)
            (Nat.two_pow_pos _)
        _ = 2 ^ (8 * m + 8) := by rw [← pow_add]; ring_nf
        _ ≤ 2 ^ 64 := Nat.pow_le_pow_right (by norm_num) (by omega)
    rw [h8, UInt8.toNat_toUInt64, Nat.shiftLeft_eq, Nat.mod_eq_of_lt hlt, Nat.lor_comm,
      ← Nat.shiftLeft_eq, ← Nat.shiftLeft_add_eq_or_of_lt
        (sum_bytes_lt (fun k ↦ bs.getD (8 * i + k) 0) m), Nat.shiftLeft_eq, Nat.add_comm]

/-- **Lane `i` of bytes, as a number:** `∑ b_(8i+k) 256^k`, the bytes past the end zero. -/
theorem toNat_littleEndianLane (bs : List UInt8) (i : ℕ) :
    (littleEndianLane bs i).toNat =
      ∑ k ∈ Finset.range 8, (bs.getD (8 * i + k) 0).toNat * 2 ^ (8 * k) :=
  toNat_foldl_lane bs i 8 le_rfl

/-- Lane `i` of bytes is lane `0` of the bytes from `8 i`. -/
theorem littleEndianLane_eq_drop (bs : List UInt8) (i : ℕ) :
    littleEndianLane bs i = littleEndianLane (bs.drop (8 * i)) 0 := by
  apply UInt64.toNat_inj.mp
  simp only [toNat_littleEndianLane, List.getD_eq_getElem?_getD, List.getElem?_drop,
    Nat.mul_zero, Nat.zero_add]

/-- **The lane of a word's bytes is the word.** -/
theorem littleEndianLane_wordBytes_append (w : UInt64) (rest : List UInt8) :
    littleEndianLane (wordBytes w ++ rest) 0 = w := by
  apply UInt64.toNat_inj.mp
  have hw := w.toNat_lt
  simp only [toNat_littleEndianLane, Finset.sum_range_succ, Finset.sum_range_zero, wordBytes,
    List.range_succ, List.range_zero, List.nil_append, List.map_cons, List.map_nil,
    List.cons_append, List.getD_eq_getElem?_getD, List.getElem?_cons_zero,
    List.getElem?_cons_succ, Option.getD_some, UInt64.toNat_toUInt8, UInt64.toNat_shiftRight,
    Nat.shiftRight_eq_div_pow]
  norm_num
  omega

/-- **The bytes of the lane of eight bytes are those bytes.** -/
theorem wordBytes_littleEndianLane {bs : List UInt8} (h : 8 ≤ bs.length) :
    wordBytes (littleEndianLane bs 0) = bs.take 8 := by
  match bs, h with
  | b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest, _ =>
    have hl := toNat_littleEndianLane (b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest) 0
    simp only [Finset.sum_range_succ, Finset.sum_range_zero, List.getD_eq_getElem?_getD,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, Nat.mul_zero,
      Nat.zero_add] at hl
    have h0 := b0.toNat_lt
    have h1 := b1.toNat_lt
    have h2 := b2.toNat_lt
    have h3 := b3.toNat_lt
    have h4 := b4.toNat_lt
    have h5 := b5.toNat_lt
    have h6 := b6.toNat_lt
    have h7 := b7.toNat_lt
    generalize littleEndianLane _ 0 = w at hl ⊢
    simp only [wordBytes, List.range_succ, List.range_zero, List.nil_append, List.map_cons,
      List.map_nil, List.cons_append, List.take_succ_cons, List.take_zero, List.cons.injEq,
      and_true]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> apply UInt8.toNat_inj.mp <;>
      simp only [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, Nat.shiftRight_eq_div_pow] <;>
      norm_num at hl h0 h1 h2 h3 h4 h5 h6 h7 ⊢ <;> omega

/-- The lanes of words' bytes are the words, whatever follows. -/
theorem lanes_flatMap_wordBytes (rest : List UInt8) :
    ∀ ws : List UInt64,
      (List.range ws.length).map (littleEndianLane (ws.flatMap wordBytes ++ rest)) = ws
  | [] => rfl
  | w :: ws => by
    rw [List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map,
      List.flatMap_cons, List.append_assoc, littleEndianLane_wordBytes_append]
    congr 1
    conv_rhs => rw [← lanes_flatMap_wordBytes rest ws]
    refine List.map_congr_left fun i _ ↦ ?_
    rw [Function.comp_apply, littleEndianLane_eq_drop, littleEndianLane_eq_drop _ i,
      show 8 * i.succ = 8 + 8 * i by rw [Nat.succ_eq_add_one]; ring, ← List.drop_drop,
      List.drop_left' (by simp [wordBytes])]

/-- Bytes of length `8 c` are the bytes of their `c` lanes. -/
theorem flatMap_wordBytes_lanes :
    ∀ (c : ℕ) {bs : List UInt8}, bs.length = 8 * c →
      ((List.range c).map (littleEndianLane bs)).flatMap wordBytes = bs
  | 0, bs, h => by simp_all
  | c + 1, bs, h => by
    rw [List.range_succ_eq_map, List.map_cons, List.map_map, List.flatMap_cons,
      wordBytes_littleEndianLane (by omega)]
    have hmap : (List.range c).map (littleEndianLane bs ∘ Nat.succ) =
        (List.range c).map (littleEndianLane (bs.drop 8)) := by
      refine List.map_congr_left fun i _ ↦ ?_
      rw [Function.comp_apply, littleEndianLane_eq_drop, littleEndianLane_eq_drop _ i,
        List.drop_drop, Nat.add_comm, show 8 * i.succ = 8 * i + 8 by rw [Nat.succ_eq_add_one]; ring]
    rw [hmap, flatMap_wordBytes_lanes c (by simp; omega), List.take_append_drop]

/-! ## `readBytes` -/

/-- **What `readBytes` accepts:** its length's bytes, then the rest. -/
theorem readBytes_eq_some_iff {length : ℕ} {input head rest : List UInt8} :
    readBytes length input = some (head, rest) ↔ head.length = length ∧ input = head ++ rest := by
  unfold readBytes
  split
  · simp only [Option.some.injEq, Prod.mk.injEq]
    constructor
    · rintro ⟨rfl, rfl⟩
      exact ⟨by simp; omega, (List.take_append_drop _ _).symm⟩
    · rintro ⟨rfl, rfl⟩
      simp
  · simp only [reduceCtorEq, false_iff, not_and]
    rintro rfl rfl
    simp_all

/-- `readBytes` reads back the bytes in front. -/
theorem readBytes_append (l rest : List UInt8) : readBytes l.length (l ++ rest) = some (l, rest) :=
  readBytes_eq_some_iff.mpr ⟨rfl, rfl⟩

/-! ## `readU32` -/

@[simp] theorem length_u32Bytes (n : ℕ) : (Prover.u32Bytes n).length = 4 := by
  simp [Prover.u32Bytes]

private theorem toNat_littleEndianLane_four (b0 b1 b2 b3 : UInt8) :
    (littleEndianLane [b0, b1, b2, b3] 0).toNat =
      b0.toNat + b1.toNat * 2 ^ 8 + b2.toNat * 2 ^ 16 + b3.toNat * 2 ^ 24 := by
  rw [toNat_littleEndianLane]
  simp [Finset.sum_range_succ, List.getD_eq_getElem?_getD]

/-- **What `readU32` accepts:** the four bytes of a number below `2 ^ 32`, then the rest. -/
theorem readU32_eq_some_iff {input rest : List UInt8} {n : ℕ} :
    readU32 input = some (n, rest) ↔ n < 2 ^ 32 ∧ input = Prover.u32Bytes n ++ rest := by
  unfold readU32
  rw [Option.map_eq_some_iff]
  constructor
  · rintro ⟨⟨head, rest'⟩, hread, heq⟩
    simp only [Prod.mk.injEq] at heq
    obtain ⟨rfl, rfl⟩ := heq
    obtain ⟨hlen, rfl⟩ := readBytes_eq_some_iff.mp hread
    match head, hlen with
    | [b0, b1, b2, b3], _ =>
      rw [toNat_littleEndianLane_four]
      have h0 := b0.toNat_lt
      have h1 := b1.toNat_lt
      have h2 := b2.toNat_lt
      have h3 := b3.toNat_lt
      refine ⟨by omega, ?_⟩
      simp only [Prover.u32Bytes, List.range_succ, List.range_zero, List.nil_append,
        List.map_cons, List.map_nil, List.cons_append, List.nil_append, List.cons.injEq,
        and_true]
      refine ⟨?_, ?_, ?_, ?_⟩ <;> apply UInt8.toNat_inj.mp <;>
        simp only [UInt8.toNat_ofNat'] <;> norm_num <;> omega
  · rintro ⟨hn, rfl⟩
    refine ⟨(Prover.u32Bytes n, rest), by rw [← length_u32Bytes n, readBytes_append], ?_⟩
    simp only [Prod.mk.injEq, and_true]
    simp only [Prover.u32Bytes, List.range_succ, List.range_zero, List.nil_append,
      List.map_cons, List.map_nil, List.cons_append, toNat_littleEndianLane_four,
      UInt8.toNat_ofNat']
    norm_num
    omega

/-- `readU32` reads back a number below `2 ^ 32`. -/
theorem readU32_u32Bytes {n : ℕ} (hn : n < 2 ^ 32) (rest : List UInt8) :
    readU32 (Prover.u32Bytes n ++ rest) = some (n, rest) :=
  readU32_eq_some_iff.mpr ⟨hn, rfl⟩

/-! ## Lists of fields -/

/-- `count` fields in order, each read by `read`: the shape of `readU32s`, `readDigests` and
`readRows`. -/
def readList {α : Type} (read : List UInt8 → Option (α × List UInt8)) :
    ℕ → List UInt8 → Option (List α × List UInt8)
  | 0, input => some ([], input)
  | count + 1, input => do
    let (value, input) ← read input
    let (values, input) ← readList read count input
    some (value :: values, input)

/-- **What `readList` accepts**, from what its reader accepts: `count` fields' encodings in order,
then the rest. -/
theorem readList_eq_some_iff {α : Type} {read : List UInt8 → Option (α × List UInt8)}
    (encode : α → List UInt8) (valid : α → Prop)
    (hread : ∀ {input value rest}, read input = some (value, rest) ↔
      valid value ∧ input = encode value ++ rest) :
    ∀ {count : ℕ} {input : List UInt8} {values : List α} {rest : List UInt8},
      readList read count input = some (values, rest) ↔
        values.length = count ∧ (∀ v ∈ values, valid v) ∧ input = values.flatMap encode ++ rest
  | 0, input, values, rest => by
    simp only [readList, Option.some.injEq, Prod.mk.injEq, List.length_eq_zero_iff]
    constructor
    · rintro ⟨rfl, rfl⟩
      simp
    · rintro ⟨rfl, -, rfl⟩
      simp
  | count + 1, input, values, rest => by
    simp only [readList, Option.bind_eq_bind, Option.bind_eq_some_iff, Prod.exists]
    constructor
    · rintro ⟨value, input', hv, values', rest', hvs, heq⟩
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl⟩ := heq
      obtain ⟨hvalid, rfl⟩ := hread.mp hv
      obtain ⟨hlen, hall, rfl⟩ := (readList_eq_some_iff encode valid hread).mp hvs
      refine ⟨by simp [hlen], ?_, by simp⟩
      intro v hv
      rcases List.mem_cons.mp hv with rfl | hv
      · exact hvalid
      · exact hall v hv
    · rintro ⟨hlen, hall, rfl⟩
      match values, hlen with
      | value :: values', hlen =>
        refine ⟨value, values'.flatMap encode ++ rest,
          hread.mpr ⟨hall value List.mem_cons_self, by simp⟩, values', rest,
          (readList_eq_some_iff encode valid hread).mpr
            ⟨by simpa using hlen, fun v hv ↦ hall v (List.mem_cons_of_mem _ hv), rfl⟩, rfl⟩

theorem readU32s_eq_readList : readU32s = readList readU32 := by
  funext count input
  induction count generalizing input with
  | zero => rfl
  | succ count ih => simp only [readU32s, readList, ih]

/-- **What `readU32s` accepts:** `count` numbers below `2 ^ 32`, then the rest. -/
theorem readU32s_eq_some_iff {count : ℕ} {input rest : List UInt8} {values : List ℕ} :
    readU32s count input = some (values, rest) ↔ values.length = count ∧
      (∀ n ∈ values, n < 2 ^ 32) ∧ input = values.flatMap Prover.u32Bytes ++ rest := by
  rw [readU32s_eq_readList]
  exact readList_eq_some_iff Prover.u32Bytes (· < 2 ^ 32) readU32_eq_some_iff

/-- `readU32s` reads back numbers below `2 ^ 32`. -/
theorem readU32s_u32Bytes {ns : List ℕ} (hns : ∀ n ∈ ns, n < 2 ^ 32) (rest : List UInt8) :
    readU32s ns.length (ns.flatMap Prover.u32Bytes ++ rest) = some (ns, rest) :=
  readU32s_eq_some_iff.mpr ⟨rfl, hns, rfl⟩

/-! ## `readWords` -/

theorem flatMap_kToBytes (ws : List K) :
    ws.flatMap K.toBytes = (ws.map K.lane).flatMap wordBytes := by
  induction ws with
  | nil => rfl
  | cons w ws ih => simp only [List.flatMap_cons, List.map_cons, ih]; rfl

@[simp] theorem length_flatMap_wordBytes (ws : List UInt64) :
    (ws.flatMap wordBytes).length = 8 * ws.length := by
  induction ws with
  | nil => rfl
  | cons w ws ih => simp only [List.flatMap_cons, List.length_append, ih, List.length_cons,
      wordBytes, List.length_map, List.length_range]; ring

/-- **What `readWords` accepts:** `count` words' bytes, then the rest. -/
theorem readWords_eq_some_iff {count : ℕ} {input rest : List UInt8} {ws : List K} :
    readWords count input = some (ws, rest) ↔
      ws.length = count ∧ input = ws.flatMap K.toBytes ++ rest := by
  unfold readWords
  rw [Option.map_eq_some_iff]
  constructor
  · rintro ⟨⟨head, rest'⟩, hread, heq⟩
    simp only [Prod.mk.injEq] at heq
    obtain ⟨rfl, rfl⟩ := heq
    obtain ⟨hlen, rfl⟩ := readBytes_eq_some_iff.mp hread
    refine ⟨by simp, ?_⟩
    rw [flatMap_kToBytes, List.map_map]
    congr 1
    exact (flatMap_wordBytes_lanes count hlen).symm
  · rintro ⟨rfl, rfl⟩
    refine ⟨(ws.flatMap K.toBytes, rest), ?_, ?_⟩
    · rw [show 8 * ws.length = (ws.flatMap K.toBytes).length by
        rw [flatMap_kToBytes, length_flatMap_wordBytes, List.length_map], readBytes_append]
    · simp only [Prod.mk.injEq, and_true]
      have h := lanes_flatMap_wordBytes [] (ws.map K.lane)
      rw [List.append_nil, ← flatMap_kToBytes, List.length_map] at h
      calc _ = ((List.range ws.length).map
            (littleEndianLane (ws.flatMap K.toBytes))).map K.ofLane := by
            rw [List.map_map]; rfl
        _ = ws := by rw [h, List.map_map]; exact List.map_id' _ |>.trans rfl

/-- `readWords` reads back words. -/
theorem readWords_toBytes (ws : List K) (rest : List UInt8) :
    readWords ws.length (ws.flatMap K.toBytes ++ rest) = some (ws, rest) :=
  readWords_eq_some_iff.mpr ⟨rfl, rfl⟩

/-! ## Digests -/

theorem digestBytes_eq (d : Digest) : digestBytes d = d.toList.flatMap wordBytes := rfl

/-- **What `readDigest` accepts:** a digest's 32 bytes, then the rest. -/
theorem readDigest_eq_some_iff {input rest : List UInt8} {d : Digest} :
    readDigest input = some (d, rest) ↔ input = digestBytes d ++ rest := by
  unfold readDigest
  rw [Option.map_eq_some_iff]
  constructor
  · rintro ⟨⟨head, rest'⟩, hread, heq⟩
    simp only [Prod.mk.injEq] at heq
    obtain ⟨rfl, rfl⟩ := heq
    obtain ⟨hlen, rfl⟩ := readBytes_eq_some_iff.mp hread
    congr 1
    exact (flatMap_wordBytes_lanes 4 hlen).symm
  · rintro rfl
    refine ⟨(digestBytes d, rest), ?_, ?_⟩
    · rw [show 32 = (digestBytes d).length by
        rw [digestBytes_eq, length_flatMap_wordBytes, Vector.length_toList], readBytes_append]
    · simp only [Prod.mk.injEq, and_true]
      have h := lanes_flatMap_wordBytes [] d.toList
      rw [List.append_nil, ← digestBytes_eq, Vector.length_toList] at h
      apply Vector.toList_inj.mp
      rw [← h]
      rfl

theorem readDigests_eq_readList : readDigests = readList readDigest := by
  funext count input
  induction count generalizing input with
  | zero => rfl
  | succ count ih => simp only [readDigests, readList, ih]

/-- **What `readDigests` accepts:** `count` digests' bytes, then the rest. -/
theorem readDigests_eq_some_iff {count : ℕ} {input rest : List UInt8} {ds : List Digest} :
    readDigests count input = some (ds, rest) ↔
      ds.length = count ∧ input = ds.flatMap digestBytes ++ rest := by
  rw [readDigests_eq_readList,
    readList_eq_some_iff digestBytes (fun _ ↦ True) (by simp [readDigest_eq_some_iff])]
  simp

/-- `readDigests` reads back digests. -/
theorem readDigests_digestBytes (ds : List Digest) (rest : List UInt8) :
    readDigests ds.length (ds.flatMap digestBytes ++ rest) = some (ds, rest) :=
  readDigests_eq_some_iff.mpr ⟨rfl, rfl⟩

/-! ## Rows -/

theorem readRows_eq_readList (rowWords : ℕ) :
    readRows rowWords = readList (readWords rowWords) := by
  funext count input
  induction count generalizing input with
  | zero => rfl
  | succ count ih => simp only [readRows, readList, ih]

/-- **What `readRows` accepts:** `count` rows of `rowWords` words, then the rest. -/
theorem readRows_eq_some_iff {rowWords count : ℕ} {input rest : List UInt8}
    {rows : List (List K)} :
    readRows rowWords count input = some (rows, rest) ↔ rows.length = count ∧
      (∀ row ∈ rows, row.length = rowWords) ∧
        input = rows.flatMap (·.flatMap K.toBytes) ++ rest := by
  rw [readRows_eq_readList]
  exact readList_eq_some_iff (fun row : List K ↦ row.flatMap K.toBytes)
    (fun row : List K ↦ row.length = rowWords) readWords_eq_some_iff

/-- `readRows` reads back rows of one width. -/
theorem readRows_toBytes {w : ℕ} {rows : List (List K)} (hrows : ∀ r ∈ rows, r.length = w)
    (rest : List UInt8) :
    readRows w rows.length (rows.flatMap (·.flatMap K.toBytes) ++ rest) = some (rows, rest) :=
  readRows_eq_some_iff.mpr ⟨rfl, hrows, rfl⟩

/-! ## The stream -/

/-! ### LEB128 -/

theorem varintBytes_of_lt {value : ℕ} (h : value < 128) :
    Prover.varintBytes value = [UInt8.ofNat value] := by
  rw [Prover.varintBytes]; simp [h]

theorem varintBytes_of_le {value : ℕ} (h : 128 ≤ value) :
    Prover.varintBytes value =
      UInt8.ofNat (value % 128 + 128) :: Prover.varintBytes (value / 128) := by
  rw [Prover.varintBytes]; simp [show ¬ value < 128 by omega]

private theorem toNat_ofNat_of_lt {n : ℕ} (h : n < 256) : (UInt8.ofNat n).toNat = n := by
  simp only [UInt8.toNat_ofNat']; omega

/-- **What `readVarint` accepts:** with `fuel` bytes, the LEB128 of a number below `128 ^ fuel`,
nonzero unless `first`, then the rest. -/
theorem readVarint_eq_some_iff :
    ∀ {fuel : ℕ} {first : Bool} {input rest : List UInt8} {value : ℕ},
      readVarint fuel first input = some (value, rest) ↔
        0 < fuel ∧ value < 128 ^ fuel ∧ (first = true ∨ value ≠ 0) ∧
          input = Prover.varintBytes value ++ rest
  | 0, first, input, rest, value => by simp [readVarint]
  | fuel + 1, first, [], rest, value => by
    simp only [readVarint, reduceCtorEq, false_iff, not_and]
    intro _ _ _ h
    by_cases hv : value < 128
    · rw [varintBytes_of_lt hv] at h; simp at h
    · rw [varintBytes_of_le (by omega)] at h; simp at h
  | fuel + 1, first, byte :: input, rest, value => by
    have hb := byte.toNat_lt
    have hpow : 128 ≤ 128 ^ (fuel + 1) := Nat.le_self_pow (by omega) _
    by_cases hbyte : byte.toNat < 128
    · simp only [readVarint, hbyte, ↓reduceIte]
      constructor
      · intro h
        split_ifs at h with hc
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨by omega, by omega, hc, ?_⟩
        rw [varintBytes_of_lt hbyte, UInt8.ofNat_toNat]; rfl
      · rintro ⟨-, -, hc, h⟩
        by_cases hv : value < 128
        · rw [varintBytes_of_lt hv] at h
          simp only [List.cons_append, List.nil_append, List.cons.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          rw [toNat_ofNat_of_lt (by omega)]
          simp [hc]
        · rw [varintBytes_of_le (by omega)] at h
          simp only [List.cons_append, List.cons.injEq] at h
          obtain ⟨rfl, -⟩ := h
          rw [toNat_ofNat_of_lt (by omega)] at hbyte
          omega
    · simp only [readVarint, hbyte, ↓reduceIte, Option.map_eq_some_iff, Prod.exists,
        Prod.mk.injEq]
      constructor
      · rintro ⟨high, input', hread, rfl, rfl⟩
        obtain ⟨hfuel, hlt, hne, rfl⟩ := readVarint_eq_some_iff.1 hread
        have hne : high ≠ 0 := by simpa using hne
        refine ⟨by omega, ?_, Or.inr (by omega), ?_⟩
        · rw [pow_succ]; omega
        · rw [varintBytes_of_le (value := byte.toNat - 128 + 128 * high) (by omega),
            show (byte.toNat - 128 + 128 * high) % 128 + 128 = byte.toNat by omega,
            show (byte.toNat - 128 + 128 * high) / 128 = high by omega, UInt8.ofNat_toNat]
          rfl
      · rintro ⟨-, hlt, -, h⟩
        by_cases hv : value < 128
        · rw [varintBytes_of_lt hv] at h
          simp only [List.cons_append, List.nil_append, List.cons.injEq] at h
          obtain ⟨rfl, -⟩ := h
          rw [toNat_ofNat_of_lt (by omega)] at hbyte
          omega
        · rw [varintBytes_of_le (by omega)] at h
          simp only [List.cons_append, List.cons.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          rw [pow_succ] at hlt
          have hfuel : 0 < fuel := by
            rcases Nat.eq_zero_or_pos fuel with rfl | h
            · simp at hlt; omega
            · exact h
          refine ⟨value / 128, rest,
            readVarint_eq_some_iff.2 ⟨hfuel, by omega, Or.inr (by omega), rfl⟩, ?_, rfl⟩
          rw [toNat_ofNat_of_lt (by omega)]
          omega

/-! ### The bitmask -/

theorem leValue_lt : ∀ bytes : List UInt8, leValue bytes < 256 ^ bytes.length
  | [] => by simp [leValue]
  | byte :: bytes => by
    have ih := leValue_lt bytes
    have hb := byte.toNat_lt
    rw [leValue, List.length_cons, pow_succ]
    omega

@[simp] theorem length_leBytes : ∀ count value : ℕ, (Prover.leBytes count value).length = count
  | 0, _ => rfl
  | count + 1, value => by simp [Prover.leBytes, length_leBytes count]

theorem leBytes_leValue :
    ∀ bytes : List UInt8, Prover.leBytes bytes.length (leValue bytes) = bytes
  | [] => rfl
  | byte :: bytes => by
    have hb := byte.toNat_lt
    rw [List.length_cons, leValue, Prover.leBytes,
      show (byte.toNat + 256 * leValue bytes) % 256 = byte.toNat by omega,
      show (byte.toNat + 256 * leValue bytes) / 256 = leValue bytes by omega,
      leBytes_leValue bytes, UInt8.ofNat_toNat]

theorem leValue_leBytes :
    ∀ count value : ℕ, value < 256 ^ count → leValue (Prover.leBytes count value) = value
  | 0, value, h => by simp at h; simp [h, leValue, Prover.leBytes]
  | count + 1, value, h => by
    rw [pow_succ] at h
    rw [Prover.leBytes, leValue, leValue_leBytes count (value / 256) (by omega),
      toNat_ofNat_of_lt (by omega)]
    omega

@[simp] theorem length_bitsOf : ∀ count value : ℕ, (bitsOf count value).length = count
  | 0, _ => rfl
  | count + 1, value => by simp [bitsOf, length_bitsOf count]

theorem bitsValue_lt : ∀ bits : List Bool, Prover.bitsValue bits < 2 ^ bits.length
  | [] => by simp [Prover.bitsValue]
  | bit :: bits => by
    have ih := bitsValue_lt bits
    have hbit : bit.toNat ≤ 1 := by cases bit <;> simp
    rw [Prover.bitsValue, List.length_cons, pow_succ]
    omega

theorem bitsOf_bitsValue : ∀ bits : List Bool, bitsOf bits.length (Prover.bitsValue bits) = bits
  | [] => rfl
  | bit :: bits => by
    have h1 : (bit.toNat + 2 * Prover.bitsValue bits) / 2 = Prover.bitsValue bits := by
      cases bit <;> simp only [Bool.toNat_false, Bool.toNat_true] <;> omega
    have h2 : decide ((bit.toNat + 2 * Prover.bitsValue bits) % 2 = 1) = bit := by
      cases bit <;> simp only [Bool.toNat_false, Bool.toNat_true, decide_eq_true_eq,
        decide_eq_false_iff_not] <;> omega
    rw [List.length_cons, bitsOf, Prover.bitsValue, h1, h2, bitsOf_bitsValue bits]

theorem bitsValue_bitsOf :
    ∀ count value : ℕ, value < 2 ^ count → Prover.bitsValue (bitsOf count value) = value
  | 0, value, h => by simp at h; simp [h, bitsOf, Prover.bitsValue]
  | count + 1, value, h => by
    rw [pow_succ] at h
    rw [bitsOf, Prover.bitsValue, bitsValue_bitsOf count (value / 2) (by omega)]
    by_cases hv : value % 2 = 1 <;> simp [hv] <;> omega

/-- The bitmask's bytes hold the bits of every element. -/
private theorem two_pow_le_mask (length : ℕ) : 2 ^ length ≤ 256 ^ ((length + 7) / 8) :=
  calc 2 ^ length ≤ 2 ^ (8 * ((length + 7) / 8)) := Nat.pow_le_pow_right (by norm_num) (by omega)
    _ = 256 ^ ((length + 7) / 8) := by rw [pow_mul]; norm_num

/-! ### The elements -/

private theorem ofLimbs_limbs (x : E) : E.ofLimbs (x.limb 0) (x.limb 1) = x :=
  CompPoly.Extension.Ext.ext fun i ↦ by fin_cases i <;> rfl

@[simp] theorem limb_ofLimbs_zero (a b : K) : (E.ofLimbs a b).limb 0 = a := rfl

@[simp] theorem limb_ofLimbs_one (a b : K) : (E.ofLimbs a b).limb 1 = b := rfl

/-- **What `readElement` accepts:** an element's bytes, compact as the bit says, then the rest. -/
theorem readElement_eq_some_iff {compact : Bool} {input rest : List UInt8} {x : E} :
    readElement compact input = some (x, rest) ↔
      compactScalar x = compact ∧ input = Prover.elementBytes x ++ rest := by
  cases compact with
  | true =>
    simp only [readElement, ↓reduceIte, Option.map_eq_some_iff, Prod.exists, Prod.mk.injEq]
    constructor
    · rintro ⟨value, input', hread, rfl, rfl⟩
      obtain ⟨-, hlt, -, rfl⟩ := readVarint_eq_some_iff.1 hread
      have hlane : (K.ofLane (UInt64.ofNat value)).lane.toNat = value := by
        change (UInt64.ofNat value).toNat = value
        simp only [UInt64.toNat_ofNat']; norm_num at hlt ⊢; omega
      have hc : compactScalar (E.ofLimbs (K.ofLane (UInt64.ofNat value)) 0) = true := by
        simp only [compactScalar, limb_ofLimbs_zero, limb_ofLimbs_one, hlane, wireCompactLimit,
          decide_true, Bool.true_and, decide_eq_true_eq]
        norm_num at hlt ⊢; omega
      refine ⟨hc, ?_⟩
      simp only [Prover.elementBytes, hc, ↓reduceIte, limb_ofLimbs_zero, hlane]
    · rintro ⟨hc, rfl⟩
      have hc' := hc
      simp only [compactScalar, Bool.and_eq_true, decide_eq_true_eq, wireCompactLimit] at hc'
      obtain ⟨h1, h0⟩ := hc'
      refine ⟨(x.limb 0).lane.toNat, rest, readVarint_eq_some_iff.2
        ⟨by norm_num, lt_of_lt_of_eq (of_decide_eq_true h0) (by norm_num), Or.inl rfl,
          by simp [Prover.elementBytes, hc]⟩, ?_, rfl⟩
      rw [UInt64.ofNat_toNat]
      change E.ofLimbs (x.limb 0) 0 = x
      rw [← h1, ofLimbs_limbs]
  | false =>
    simp only [readElement, Bool.false_eq_true, ↓reduceIte, Option.bind_eq_bind,
      Option.bind_eq_some_iff, Prod.exists, Option.guard_eq_some', Option.some.injEq,
      Prod.mk.injEq]
    constructor
    · rintro ⟨words, input', hwords, -, hc, rfl, rfl⟩
      obtain ⟨hlength, rfl⟩ := readWords_eq_some_iff.1 hwords
      match words, hlength with
      | [a, b], _ =>
        simp only [List.getD_cons_zero, List.getD_cons_succ] at hc ⊢
        refine ⟨hc, ?_⟩
        simp [Prover.elementBytes, hc, E.toBytes]
    · rintro ⟨hc, rfl⟩
      refine ⟨[x.limb 0, x.limb 1], rest, readWords_eq_some_iff.2 ⟨rfl, ?_⟩, ?_⟩
      · simp [Prover.elementBytes, hc, E.toBytes]
      · simp only [List.getD_cons_zero, List.getD_cons_succ, ofLimbs_limbs]
        refine ⟨(), hc, ?_⟩
        simp

/-- **What `readElements` accepts:** the elements' bytes, compact as the bits say, then the
rest. -/
theorem readElements_eq_some_iff :
    ∀ {flags : List Bool} {input rest : List UInt8} {xs : List E},
      readElements flags input = some (xs, rest) ↔
        xs.map compactScalar = flags ∧ input = xs.flatMap Prover.elementBytes ++ rest
  | [], input, rest, xs => by
    simp only [readElements, Option.some.injEq, Prod.mk.injEq, List.map_eq_nil_iff]
    constructor
    · rintro ⟨rfl, rfl⟩; simp
    · rintro ⟨rfl, rfl⟩; simp
  | compact :: flags, input, rest, xs => by
    simp only [readElements, Option.bind_eq_bind, Option.bind_eq_some_iff, Prod.exists,
      Option.some.injEq, Prod.mk.injEq]
    constructor
    · rintro ⟨x, input', hx, xs', rest', hxs, rfl, rfl⟩
      obtain ⟨hc, rfl⟩ := readElement_eq_some_iff.1 hx
      obtain ⟨hcs, rfl⟩ := readElements_eq_some_iff.1 hxs
      simp [hc, hcs]
    · rintro ⟨hmap, rfl⟩
      match xs, hmap with
      | x :: xs', hmap =>
        simp only [List.map_cons, List.cons.injEq] at hmap
        exact ⟨x, xs'.flatMap Prover.elementBytes ++ rest,
          readElement_eq_some_iff.2 ⟨hmap.1, by simp⟩, xs', rest,
          readElements_eq_some_iff.2 ⟨hmap.2, rfl⟩, rfl, rfl⟩

/-! ### The stream -/

private theorem guard_unit {p : Prop} [Decidable p] (hp : p) :
    (guard p : Option Unit) = some () := by
  simp [hp]

/-- `readStream` reads back a stream of fewer than `2 ^ 32` elements. -/
theorem readStream_streamToBytes {stream : List E} (hlength : stream.length < 2 ^ 32)
    (rest : List UInt8) :
    readStream (Prover.streamToBytes stream ++ rest) = some (stream, rest) := by
  have hvalue := bitsValue_lt (stream.map compactScalar)
  rw [List.length_map] at hvalue
  have hmask := leValue_leBytes ((stream.length + 7) / 8) _
    (lt_of_lt_of_le hvalue (two_pow_le_mask stream.length))
  have hbits := bitsOf_bitsValue (stream.map compactScalar)
  rw [List.length_map] at hbits
  simp only [readStream, Prover.streamToBytes, List.append_assoc, readU32_u32Bytes hlength,
    Option.bind_eq_bind, Option.bind_some]
  rw [readBytes_eq_some_iff.2 ⟨length_leBytes _ _, rfl⟩]
  simp only [Option.bind_some, hmask, hbits, guard_unit hvalue,
    readElements_eq_some_iff.2 ⟨rfl, rfl⟩]

/-- **What `readStream` accepts:** the bytes of a stream of fewer than `2 ^ 32` elements, then
the rest. -/
theorem readStream_eq_some_iff {input rest : List UInt8} {stream : List E} :
    readStream input = some (stream, rest) ↔
      stream.length < 2 ^ 32 ∧ input = Prover.streamToBytes stream ++ rest := by
  constructor
  · intro h
    simp only [readStream, Option.bind_eq_bind, Option.bind_eq_some_iff, Prod.exists,
      Option.guard_eq_some'] at h
    obtain ⟨length, input₁, hlength, mask, input₂, hmask, -, hguard, hread⟩ := h
    obtain ⟨hlt, rfl⟩ := readU32_eq_some_iff.1 hlength
    obtain ⟨hmaskLength, rfl⟩ := readBytes_eq_some_iff.1 hmask
    obtain ⟨hflags, rfl⟩ := readElements_eq_some_iff.1 hread
    have hstreamLength : stream.length = length := by
      simpa using congrArg List.length hflags
    subst hstreamLength
    refine ⟨hlt, ?_⟩
    simp only [Prover.streamToBytes, List.append_assoc, hflags,
      bitsValue_bitsOf _ _ hguard, ← hmaskLength, leBytes_leValue]
  · rintro ⟨hlength, rfl⟩
    exact readStream_streamToBytes hlength rest

end

end LeanVM.Wire

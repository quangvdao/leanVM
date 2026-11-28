//! The proof's one encoding, [`Proof::to_bytes`], and its decoding, which unprunes each Merkle phase.

use std::collections::BTreeMap;

use crate::field::{E, K};
use crate::merkle::{MerklePath, hash_leaf, hash_pair};
use crate::transcript::Proof;

/// The most phases a proof, queries a phase, leaf words and tree height may have: at least the protocol's own, and few
/// enough that hostile bytes unprune to a bounded size. Encoding asserts them, so an honest proof always decodes.
pub const MAX_PHASES: usize = 8;
pub const MAX_QUERIES: usize = 256;
pub const MAX_LEAF_WORDS: usize = 64;
pub const MAX_HEIGHT: usize = 32;

/// Exclusive upper bound on compact stream scalars (`c1 == 0 && c0 < COMPACT_LIMIT`), which travel as canonical
/// unsigned LEB128 (1 to 8 bytes) rather than 16 raw bytes.
const COMPACT_LIMIT: u64 = 1u64 << 56;

fn is_compact(element: E) -> bool {
    element.c1 == 0 && element.c0 < COMPACT_LIMIT
}

impl Proof {
    /// Every count and size is a little-endian `u32`, every word a little-endian `u64`.
    ///
    /// - The stream: its length `n`, a `ceil(n / 8)`-byte bitmask (LSB-first, trailing bits zero) marking compact
    ///   scalars (`c1 == 0 && c0 < 2^56`), then each element in order: canonical unsigned LEB128 (1 to 8 bytes) when
    ///   marked compact, or its two words `(c0, c1)` (16 bytes) otherwise.
    /// - The number of phases, then each phase pruned: its queries' count and positions, in the order the verifier
    ///   samples them; the tree's height, the leaf width, and the row width; the leaf at each distinct position,
    ///   ascending, as its last row-width words, the ones before being zero in every leaf of the phase; then the
    ///   siblings no other path's node gives, bottom level first and left to right within a level.
    ///
    /// The row width is the least that drops only zeros, and the sibling count follows from the positions.
    ///
    /// # Panics
    ///
    /// On a proof no prover makes, such as one with an empty phase, or with more phases, queries, leaf words or tree
    /// levels than decoding accepts.
    pub fn to_bytes(&self) -> Vec<u8> {
        assert!(self.merkle_openings.len() <= MAX_PHASES);
        let mut out = Vec::new();
        write_u32(&mut out, self.stream.len());
        for chunk in self.stream.chunks(8) {
            let mut mask = 0u8;
            for (bit, &element) in chunk.iter().enumerate() {
                if is_compact(element) {
                    mask |= 1u8 << bit;
                }
            }
            out.push(mask);
        }
        for &element in &self.stream {
            if is_compact(element) {
                write_varint(&mut out, element.c0);
            } else {
                out.extend(element.c0.to_le_bytes());
                out.extend(element.c1.to_le_bytes());
            }
        }
        write_u32(&mut out, self.merkle_openings.len());
        for phase in &self.merkle_openings {
            write_phase(&mut out, phase);
        }
        out
    }

    /// The proof `bytes` encode, or `None` unless they are exactly one proof's encoding. Nothing is verified.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut input = Reader(bytes);
        let length = input.u32()? as usize;
        let mask_len = length.div_ceil(8);
        // Every stream element takes at least one byte after the bitmask; reject hostile lengths before allocating.
        if input.0.len() < mask_len.checked_add(length)? {
            return None;
        }
        let mask = input.bytes(mask_len)?;
        let rem = length % 8;
        if rem != 0 && (*mask.last()? >> rem) != 0 {
            return None;
        }
        let mut stream = Vec::with_capacity(length);
        for i in 0..length {
            let compact = (mask[i / 8] >> (i % 8)) & 1 != 0;
            let element = if compact {
                E::new(input.varint()?, 0)
            } else {
                let raw = input.bytes(16)?;
                let c0 = u64::from_le_bytes(*raw[..8].as_array()?);
                let c1 = u64::from_le_bytes(*raw[8..].as_array()?);
                let element = E::new(c0, c1);
                if is_compact(element) {
                    return None;
                }
                element
            };
            stream.push(element);
        }
        let phases = input.u32().filter(|&phases| phases as usize <= MAX_PHASES)?;
        let merkle_openings = (0..phases).map(|_| read_phase(&mut input)).collect::<Option<_>>()?;
        input.0.is_empty().then_some(Self {
            stream,
            merkle_openings,
        })
    }
}

/// Prune a phase into its octopus: the fewest siblings that authenticate all its paths together, a sibling being left
/// out where another path's node gives it. No hashing.
fn write_phase(out: &mut Vec<u8>, paths: &[MerklePath]) {
    assert!(!paths.is_empty(), "a Merkle phase opens at least one path");
    let (height, leaf_words) = (paths[0].siblings.len(), paths[0].leaf.len());
    assert!(paths.len() <= MAX_QUERIES && leaf_words <= MAX_LEAF_WORDS && height <= MAX_HEIGHT);
    let mut distinct: Vec<&MerklePath> = paths.iter().collect();
    distinct.sort_by_key(|path| path.leaf_index);
    distinct.dedup_by_key(|path| path.leaf_index);
    let zeros = distinct
        .iter()
        .map(|path| path.leaf.iter().take_while(|word| word.0 == 0).count())
        .min()
        .unwrap_or(0);

    write_u32(out, paths.len());
    for path in paths {
        write_u32(out, usize::try_from(path.leaf_index).expect("a position fits in usize"));
    }
    for size in [height, leaf_words, leaf_words - zeros] {
        write_u32(out, size);
    }
    for path in &distinct {
        out.extend(path.leaf[zeros..].iter().flat_map(|word| word.to_bytes()));
    }
    for level in 0..height {
        let mut nodes: Vec<(u64, [u8; 32])> = distinct
            .iter()
            .map(|path| (path.leaf_index >> level, path.siblings[level]))
            .collect();
        nodes.dedup_by_key(|&mut (node, _)| node);
        for &(node, sibling) in &nodes {
            if nodes.binary_search_by_key(&(node ^ 1), |&(node, _)| node).is_err() {
                out.extend(sibling);
            }
        }
    }
}

/// Unprune a phase's octopus back into its paths, hashing the nodes they share. Nothing is authenticated here, the
/// verifier checks every path; what is refused is anything but the one encoding of a phase within the bounds.
fn read_phase(input: &mut Reader) -> Option<Vec<MerklePath>> {
    let count = input.u32().filter(|&count| count as usize <= MAX_QUERIES)?;
    let queries: Vec<u64> = (0..count).map(|_| input.u32().map(u64::from)).collect::<Option<_>>()?;
    let (height, leaf_words, row_words) = (input.u32()? as usize, input.u32()? as usize, input.u32()? as usize);
    let mut positions = queries.clone();
    positions.sort_unstable();
    positions.dedup();
    if !(height <= MAX_HEIGHT
        && leaf_words <= MAX_LEAF_WORDS
        && row_words <= leaf_words
        && positions.last().is_some_and(|&last| last >> height == 0))
    {
        return None;
    }
    let mut distinct = Vec::with_capacity(positions.len());
    for &leaf_index in &positions {
        let mut leaf = vec![K(0); leaf_words - row_words];
        leaf.extend(input.words(row_words)?);
        distinct.push(MerklePath {
            leaf_index,
            leaf,
            siblings: Vec::new(),
        });
    }
    // The rows start where the zero words every leaf starts with end.
    if row_words > 0 && distinct.iter().all(|path| path.leaf[leaf_words - row_words].0 == 0) {
        return None;
    }

    // Level by level, the nodes the paths climb through.
    let mut nodes: BTreeMap<u64, [u8; 32]> = distinct
        .iter()
        .map(|path| (path.leaf_index, hash_leaf(&path.leaf).0))
        .collect();
    for level in 0..height {
        let mut siblings = BTreeMap::new();
        let mut parents = BTreeMap::new();
        for (&p, node) in &nodes {
            let sibling = match nodes.get(&(p ^ 1)) {
                Some(&sibling) => sibling,
                None => input.hash()?,
            };
            siblings.insert(p, sibling);
            let (left, right) = if p & 1 == 0 { (node, &sibling) } else { (&sibling, node) };
            parents.entry(p >> 1).or_insert_with(|| hash_pair(left, right).0);
        }
        for path in &mut distinct {
            path.siblings.push(siblings[&(path.leaf_index >> level)]);
        }
        nodes = parents;
    }
    let path = |query: &u64| distinct[positions.binary_search(query).expect("a query is a position")].clone();
    Some(queries.iter().map(path).collect())
}

fn write_u32(out: &mut Vec<u8>, value: usize) {
    out.extend(u32::try_from(value).expect("fits in u32").to_le_bytes());
}

/// Canonical unsigned LEB128 for `value < COMPACT_LIMIT` (1 to 8 bytes).
fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    debug_assert!(value < COMPACT_LIMIT);
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Bytes being decoded, each read refused past their end.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn bytes(&mut self, length: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(length)?;
        self.0 = rest;
        Some(head)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(*self.bytes(4)?.as_array()?))
    }

    /// Canonical unsigned LEB128 below `COMPACT_LIMIT` (1 to 8 bytes, no redundant trailing zero byte).
    fn varint(&mut self) -> Option<u64> {
        let mut value = 0u64;
        for shift in (0..56).step_by(7) {
            let byte = *self.bytes(1)?.first()?;
            let low = u64::from(byte & 0x7f);
            if byte & 0x80 == 0 {
                if shift > 0 && low == 0 {
                    return None;
                }
                return Some(value | (low << shift));
            }
            value |= low << shift;
        }
        None
    }

    fn hash(&mut self) -> Option<[u8; 32]> {
        Some(*self.bytes(32)?.as_array()?)
    }

    fn words(&mut self, count: usize) -> Option<Vec<K>> {
        let bytes = self.bytes(count.checked_mul(8)?)?;
        Some(
            bytes
                .as_chunks()
                .0
                .iter()
                .map(|&word| K(u64::from_le_bytes(word)))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A proof survives its encoding, which carries each leaf without the zero words every opened leaf starts with,
    /// and nothing else decodes to it.
    #[test]
    fn proof_round_trips_through_its_one_encoding() {
        let rows = [[0, 0, 1], [0, 2, 3], [0, 0, 4], [0, 5, 0]].map(|row| row.map(K));
        let leaves = rows.map(|row| hash_leaf(&row).0);
        let parents = [hash_pair(&leaves[0], &leaves[1]).0, hash_pair(&leaves[2], &leaves[3]).0];
        let path = |q: usize| MerklePath {
            leaf_index: q as u64,
            leaf: rows[q].to_vec(),
            siblings: vec![leaves[q ^ 1], parents[(q / 2) ^ 1]],
        };
        let proof = Proof {
            stream: vec![E::new(1, 2)],
            merkle_openings: vec![vec![path(3), path(1), path(3)]],
        };
        let bytes = proof.to_bytes();
        // The stream (length, 1-byte bitmask, 16-byte element), then the phase: queries, sizes, rows, siblings.
        assert_eq!(bytes.len(), (4 + 1 + 16) + 4 + (4 + 3 * 4) + 3 * 4 + 2 * 2 * 8 + 2 * 32);
        assert_eq!(Proof::from_bytes(&bytes), Some(proof));
        assert_eq!(Proof::from_bytes(&[&bytes[..], &[0]].concat()), None);
    }

    /// Compact scalars round-trip across boundary sizes, and non-canonical encodings are refused.
    #[test]
    fn compact_stream_is_canonical() {
        let cases = vec![
            E::new(0, 0),
            E::new(1, 0),
            E::new(127, 0),
            E::new(128, 0),
            E::new((1 << 14) - 1, 0),
            E::new(1 << 14, 0),
            E::new(COMPACT_LIMIT - 1, 0),
            E::new(COMPACT_LIMIT, 0),
            E::new(u64::MAX, 0),
            E::new(0, 1),
            E::new(42, 99),
        ];
        let proof = Proof {
            stream: cases,
            merkle_openings: Vec::new(),
        };
        let bytes = proof.to_bytes();
        // 4 (len) + 2 (mask for 11 elements) + (1+1+1+2+2+3+8 compact) + (4*16 full) + 4 (phases=0).
        assert_eq!(bytes.len(), 4 + 2 + 18 + 64 + 4);
        assert_eq!(Proof::from_bytes(&bytes), Some(proof.clone()));

        // Empty proof still encodes as eight zero bytes.
        let empty = Proof {
            stream: Vec::new(),
            merkle_openings: Vec::new(),
        };
        assert_eq!(empty.to_bytes(), [0u8; 8]);
        assert_eq!(Proof::from_bytes(&[0u8; 8]), Some(empty));

        // Non-zero trailing bit in the bitmask is refused.
        let mut bad_mask = bytes.clone();
        bad_mask[5] |= 0x80;
        assert_eq!(Proof::from_bytes(&bad_mask), None);

        // Overlong LEB128 for 0 ([0x80, 0x00]) is refused.
        let bad_varint = [1u32.to_le_bytes().as_slice(), &[0x01, 0x80, 0x00], &0u32.to_le_bytes()].concat();
        assert_eq!(Proof::from_bytes(&bad_varint), None);

        // 9-byte LEB128 (>= 2^56) is refused.
        let too_wide = [
            1u32.to_le_bytes().as_slice(),
            &[0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01],
            &0u32.to_le_bytes(),
        ]
        .concat();
        assert_eq!(Proof::from_bytes(&too_wide), None);

        // Full 16-byte encoding of a compact scalar is refused.
        let non_compact = [
            1u32.to_le_bytes().as_slice(),
            &[0x00],
            &7u64.to_le_bytes(),
            &0u64.to_le_bytes(),
            &0u32.to_le_bytes(),
        ]
        .concat();
        assert_eq!(Proof::from_bytes(&non_compact), None);
    }
}

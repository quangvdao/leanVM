//! One-hot pointers: the chunk layout every Hotlink pointer of a proof shares.
//!
//! A pointer is an index `σ < 2^W`. Its `W` bits split, least significant first, into `d` chunks of widths
//! `w_i`, and chunk `i` is committed as the `2^{w_i}` indicator bits `[σ_i = v]`, starting at bit
//! `Σ_{i' < i} 2^{w_{i'}}` of one `K` word. The layout is the fewest chunks whose indicator bits fit 64, wider
//! chunks first: `W = 22` gives `(4, 3, 3, 3, 3, 3, 3)`. One-hot addressing follows Shout (Setty and Thaler).
//!
//! A chunk at a point `r_i ∈ E^{w_i}` is the linear functional `D_i(r_i) = Σ_v eq(r_i, v)·e_v` of its bits, so a
//! pointer at `r = (r_1, …, r_d)` is `Φ(r) = Π_i D_i(r_i) = eq(r, σ)` when every chunk is one-hot.

use primitives::field::F192;
use primitives::multilinear::eq_table;

/// The widest pointer one `K` word holds: eight chunks of three bits.
pub const MAX_WIDTH: usize = 24;

/// The chunk layout of `W`-bit pointers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunks {
    pub width: usize,
    pub widths: Vec<usize>,
    index_shifts: Vec<usize>,
    word_shifts: Vec<usize>,
}

impl Chunks {
    pub fn new(width: usize) -> Self {
        assert!(
            (1..=MAX_WIDTH).contains(&width),
            "Hotlink: a {width}-bit pointer does not fit one word"
        );
        let widths = (1..=width)
            .map(|d| {
                (0..d)
                    .map(|i| width / d + usize::from(i < width % d))
                    .collect::<Vec<_>>()
            })
            .find(|widths| widths.iter().map(|&w| 1usize << w).sum::<usize>() <= 64)
            .expect("one chunk per bit fits");
        let prefix = |f: &dyn Fn(usize) -> usize| {
            widths
                .iter()
                .scan(0, |at, &w| {
                    let start = *at;
                    *at += f(w);
                    Some(start)
                })
                .collect()
        };
        Self {
            width,
            index_shifts: prefix(&|w| w),
            word_shifts: prefix(&|w| 1 << w),
            widths,
        }
    }

    pub fn d(&self) -> usize {
        self.widths.len()
    }

    /// The widest chunk, which sets the length of the certificate point.
    pub fn max_width(&self) -> usize {
        self.widths[0]
    }

    /// The word of index `sigma`.
    pub fn encode(&self, sigma: usize) -> u64 {
        debug_assert!(sigma >> self.width == 0, "index {sigma} past 2^{}", self.width);
        (0..self.d()).fold(0, |word, i| {
            let v = (sigma >> self.index_shifts[i]) & ((1 << self.widths[i]) - 1);
            word | 1 << (self.word_shifts[i] + v)
        })
    }

    /// The indicator bits of chunk `i` of `word`, at bit 0.
    #[inline(always)]
    pub fn bits(&self, word: u64, i: usize) -> u64 {
        (word >> self.word_shifts[i]) & (u64::MAX >> (64 - (1 << self.widths[i])))
    }

    /// The index of a word whose chunks are all one-hot.
    #[inline(always)]
    pub fn decode(&self, word: u64) -> Option<usize> {
        let mut sigma = 0;
        for i in 0..self.d() {
            let b = self.bits(word, i);
            if !b.is_power_of_two() {
                return None;
            }
            sigma |= (b.trailing_zeros() as usize) << self.index_shifts[i];
        }
        Some(sigma)
    }

    /// Every index of a word: one per choice of a set bit in each chunk, so none when a chunk has no bit set.
    pub fn indices(&self, word: u64, f: &mut impl FnMut(usize)) {
        fn walk(chunks: &Chunks, word: u64, i: usize, sigma: usize, f: &mut impl FnMut(usize)) {
            if i == chunks.d() {
                return f(sigma);
            }
            let mut b = chunks.bits(word, i);
            while b != 0 {
                let v = b.trailing_zeros() as usize;
                walk(chunks, word, i + 1, sigma | v << chunks.index_shifts[i], f);
                b &= b - 1;
            }
        }
        walk(self, word, 0, 0, f);
    }

    /// Bit `b` of a word: its chunk and the value it indicates, `None` past the chunks.
    pub fn slot(&self, b: usize) -> Option<(usize, usize)> {
        (0..self.d())
            .find(|&i| b >= self.word_shifts[i] && b < self.word_shifts[i] + (1 << self.widths[i]))
            .map(|i| (i, b - self.word_shifts[i]))
    }

    /// Chunk `i`'s coordinates of a `W`-point.
    pub fn part<'a>(&self, point: &'a [F192], i: usize) -> &'a [F192] {
        &point[self.index_shifts[i]..self.index_shifts[i] + self.widths[i]]
    }
}

/// The pointer shifted by one, `Φ_1 = (1 0)·Λ_1⋯Λ_d·(1 0)ᵀ`, from each chunk's `D_i`, `A_i` and `B_i`: the
/// state is the pair (no carry, carry), which chunk 0 starts at `(A_0, B_0)`.
pub fn shifted(d: &[F192], a: &[F192], b: &[F192]) -> F192 {
    let (mut x, mut y) = (a[0], b[0]);
    for i in 1..d.len() {
        (x, y) = (x * d[i] + y * a[i], y * b[i]);
    }
    x
}

/// `Π_h ρ_h·(1 + ρ_h)`: what `D(ρ)·D(1 + ρ)` is for every one-hot chunk (the mirror test of Emile Wambsgans).
pub fn mirror(point: &[F192]) -> F192 {
    point.iter().map(|&x| x * (F192::ONE + x)).product()
}

/// One eq table per chunk: the functionals `D_i` at one point.
#[derive(Clone, Debug)]
pub struct ChunkEq {
    pub tables: Vec<Vec<F192>>,
}

impl ChunkEq {
    /// At a `W`-point, chunk `i` at its own coordinates.
    pub fn at(chunks: &Chunks, point: &[F192]) -> Self {
        assert_eq!(point.len(), chunks.width);
        Self {
            tables: (0..chunks.d()).map(|i| eq_table(chunks.part(point, i))).collect(),
        }
    }

    /// At the certificate point: chunk `i` at the first `w_i` coordinates of `rho`, or of `1 + rho` when `mirrored`.
    pub fn certificate(chunks: &Chunks, rho: &[F192], mirrored: bool) -> Self {
        let rho: Vec<F192> = rho.iter().map(|&x| if mirrored { F192::ONE + x } else { x }).collect();
        Self {
            tables: chunks.widths.iter().map(|&w| eq_table(&rho[..w])).collect(),
        }
    }

    /// The carry-matrix entries of the shift by one at a `W`-point (Binary Jolt's precompile link): chunk `i`
    /// with carry in moves `v` to `v + 1`, `A_i = Σ_{v + 1 < 2^{w_i}} e_v·eq(r_i, v + 1)` without a carry out and
    /// `B_i = e_{2^{w_i} − 1}·eq(r_i, 0)` with one. Chunk 0 always takes the carry in, which is the `+1`.
    pub fn shift(chunks: &Chunks, point: &[F192]) -> (Self, Self) {
        let eq = Self::at(chunks, point);
        let low = eq
            .tables
            .iter()
            .map(|t| {
                (0..t.len())
                    .map(|v| t.get(v + 1).copied().unwrap_or(F192::ZERO))
                    .collect()
            })
            .collect();
        let carry = eq
            .tables
            .iter()
            .map(|t| {
                (0..t.len())
                    .map(|v| if v + 1 == t.len() { t[0] } else { F192::ZERO })
                    .collect()
            })
            .collect();
        (Self { tables: low }, Self { tables: carry })
    }

    /// `D_i` of a word: a lookup when the chunk is one-hot, the sum over its set bits otherwise.
    #[inline(always)]
    pub fn eval(&self, chunks: &Chunks, word: u64, i: usize) -> F192 {
        let bits = chunks.bits(word, i);
        let table = &self.tables[i];
        if bits & bits.wrapping_sub(1) == 0 {
            if bits == 0 {
                F192::ZERO
            } else {
                table[bits.trailing_zeros() as usize]
            }
        } else {
            let mut sum = F192::ZERO;
            let mut rest = bits;
            while rest != 0 {
                sum += table[rest.trailing_zeros() as usize];
                rest &= rest - 1;
            }
            sum
        }
    }

    /// `D_i` from a word's 64 bit-slices: the functional is linear in the bits.
    pub fn from_slices(&self, chunks: &Chunks, slices: &[F192], i: usize) -> F192 {
        let start = chunks.word_shifts[i];
        self.tables[i].iter().zip(&slices[start..]).map(|(&e, &s)| e * s).sum()
    }

    /// The bit-slice weights: bit `b` of a word weighs `tables[i][v]` in `D_i` when `b` is chunk `i`'s value `v`.
    pub fn slice_weights(&self, chunks: &Chunks) -> Vec<(usize, usize, F192)> {
        (0..64)
            .filter_map(|b| chunks.slot(b).map(|(i, v)| (b, i, self.tables[i][v])))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::multilinear::eq_eval;
    use primitives::test_rng::Rng;

    #[test]
    fn layouts_are_the_fewest_chunks_that_fit() {
        let widths = |w| Chunks::new(w).widths;
        assert_eq!(widths(22), vec![4, 3, 3, 3, 3, 3, 3]);
        assert_eq!(widths(23), vec![3, 3, 3, 3, 3, 3, 3, 2]);
        assert_eq!(widths(24), vec![3; 8]);
        assert_eq!(widths(21), vec![3; 7]);
        assert_eq!(widths(6), vec![6]);
        assert_eq!(widths(7), vec![4, 3]);
        for w in 1..=MAX_WIDTH {
            let chunks = Chunks::new(w);
            assert_eq!(chunks.widths.iter().sum::<usize>(), w);
            assert!(chunks.widths.iter().map(|&w| 1usize << w).sum::<usize>() <= 64);
        }
    }

    #[test]
    fn a_word_at_a_point_is_eq_of_its_index() {
        let mut rng = Rng::new(7);
        for w in [5, 13, 22, 24] {
            let chunks = Chunks::new(w);
            let r = rng.ext_vec(w);
            let eqs = ChunkEq::at(&chunks, &r);
            for k in 0..40 {
                let sigma = match k {
                    0 => (1 << w) - 1,
                    1 => (1 << chunks.widths[0]) - 1,
                    _ => rng.next_u64() as usize & ((1 << w) - 1),
                };
                let word = chunks.encode(sigma);
                let bits: Vec<F192> = (0..w).map(|h| F192::new(((sigma >> h) & 1) as u64, 0, 0)).collect();
                let phi: F192 = (0..chunks.d()).map(|i| eqs.eval(&chunks, word, i)).product();
                assert_eq!(phi, eq_eval(&r, &bits));
                let rho = rng.ext_vec(chunks.max_width());
                let (d, dbar) = (
                    ChunkEq::certificate(&chunks, &rho, false),
                    ChunkEq::certificate(&chunks, &rho, true),
                );
                let (low, carry) = ChunkEq::shift(&chunks, &r);
                let at = |t: &ChunkEq| (0..chunks.d()).map(|i| t.eval(&chunks, word, i)).collect::<Vec<_>>();
                let next = sigma + 1;
                let expected = if next >> w == 0 {
                    let bits: Vec<F192> = (0..w).map(|h| F192::new(((next >> h) & 1) as u64, 0, 0)).collect();
                    eq_eval(&r, &bits)
                } else {
                    F192::ZERO
                };
                assert_eq!(
                    shifted(&at(&eqs), &at(&low), &at(&carry)),
                    expected,
                    "sigma + 1 = {next}"
                );
                for i in 0..chunks.d() {
                    let product = d.eval(&chunks, word, i) * dbar.eval(&chunks, word, i);
                    assert_eq!(product, mirror(&rho[..chunks.widths[i]]));
                }
            }
        }
    }
}

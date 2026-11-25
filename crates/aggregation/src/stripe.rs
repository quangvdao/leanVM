//! The binary stripe as the guest certifies it (`certify_stack`): the stack's block groups and the witness that
//! places each block.

use super::cells::count;
use primitives::field::F128;
use verifier::layout::{MIN_STACKED_LOG, SizeSource, StackBlock, stack_blocks};
use verifier::{INITIAL_FOLDING_FACTOR, MAX_STRIPE_WIDTH};

/// The guest's lane-group ids for a lane count `L`: id `j < LANE_LOW` is the group of `2^j` lanes for bit `j` of `L`,
/// and id `LANE_LOW + k` is the `k`-th group of `2^MAX_STRIPE_WIDTH` lanes, there when `k < L >> MAX_STRIPE_WIDTH`.
/// `L` is at most `2^INITIAL_FOLDING_FACTOR`, so its bits from `MAX_STRIPE_WIDTH` up are `LANE_FULL_BITS` many.
pub(super) const LANE_LOW: usize = if INITIAL_FOLDING_FACTOR >= MAX_STRIPE_WIDTH {
    MAX_STRIPE_WIDTH
} else {
    INITIAL_FOLDING_FACTOR + 1
};
pub(super) const LANE_FULL_BITS: usize = (INITIAL_FOLDING_FACTOR + 1).saturating_sub(MAX_STRIPE_WIDTH);
pub(super) const LANE_FULL: usize = if LANE_FULL_BITS == 0 {
    0
} else {
    1 << (LANE_FULL_BITS - 1)
};
const LANE_GROUPS: usize = LANE_LOW + LANE_FULL;

/// The guest's lane groups for `lanes` filled lanes, by id: whether the group is there, its first lane and its log
/// width. The groups that are there are `stack_offsets`'s.
pub(super) fn lane_groups(lanes: u64) -> [(bool, u64, usize); LANE_GROUPS] {
    let mut groups = [(false, 0, 0); LANE_GROUPS];
    for (j, group) in groups.iter_mut().enumerate().take(LANE_LOW) {
        *group = (lanes >> j & 1 == 1, (lanes >> (j + 1)) << (j + 1), j);
    }
    for k in 0..LANE_FULL {
        let k64 = k as u64;
        groups[LANE_LOW + k] = (
            k64 < lanes >> MAX_STRIPE_WIDTH,
            k64 << MAX_STRIPE_WIDTH,
            MAX_STRIPE_WIDTH,
        );
    }
    groups
}

/// A block's kappa source in the guest's `kappa_base`: 1 for the memory, `2 + t` for table `t`.
pub(super) fn source_index(source: SizeSource) -> usize {
    match source {
        SizeSource::Memory => 1,
        SizeSource::Table(table) => 2 + table,
    }
}

/// The first block of each run of consecutive blocks sharing a source, an adjustment and a ring flag. A run's blocks
/// have one size and one ring flag, so they are consecutive in the stack's order.
pub(super) fn block_groups(blocks: &[StackBlock]) -> Vec<usize> {
    let key = |block: &StackBlock| (block.source, block.adj, block.ring);
    (0..blocks.len())
        .filter(|&i| i == 0 || key(&blocks[i]) != key(&blocks[i - 1]))
        .collect()
}

/// The stack's log size for blocks of log sizes `kappas`: every block takes `2^max(kappa, MAX_STRIPE_WIDTH)` words,
/// and the stack is the least power of two holding them, at least `2^MIN_STACKED_LOG`.
pub(super) fn stack_log(kappas: &[usize]) -> usize {
    let total: u64 = kappas.iter().map(|&k| 1u64 << k.max(MAX_STRIPE_WIDTH)).sum();
    (64 - (total - 1).leading_zeros() as usize).max(MIN_STACKED_LOG)
}

/// The witness streams `certify_stack` reads for blocks of log sizes `kappas` (in [`stack_blocks`] order): the group
/// order, the lane count's bits, and per block its lane group, whether it stays whole, whether its log size is below
/// its group's width, and its in-lane offset over the lanes of its group.
pub(super) fn stack_hints(kappas: &[usize]) -> Vec<(&'static str, Vec<F128>)> {
    let blocks = stack_blocks();
    assert_eq!(kappas.len(), blocks.len(), "one log size per stack block");
    let taken = |b: usize| kappas[b].max(MAX_STRIPE_WIDTH);
    let starts = block_groups(&blocks);
    let mut order: Vec<usize> = (0..starts.len()).collect();
    order.sort_by_key(|&g| (std::cmp::Reverse(taken(starts[g])), !blocks[starts[g]].ring, starts[g]));

    let total: u64 = (0..blocks.len()).map(|b| 1u64 << taken(b)).sum();
    let log = stack_log(kappas);
    let lane = log - INITIAL_FOLDING_FACTOR;
    let lanes = total.div_ceil(1 << lane);
    let groups = lane_groups(lanes);

    let bit = |b: bool| if b { F128::ONE } else { F128::ZERO };
    let mut block_hints = vec![[F128::ZERO; 4]; blocks.len()];
    let mut at = 0u64;
    for &g in &order {
        let end = starts.get(g + 1).copied().unwrap_or(blocks.len());
        for b in starts[g]..end {
            let j = (0..LANE_GROUPS)
                .find(|&j| {
                    groups[j].0 && (at >> lane) >= groups[j].1 && (at >> lane) < groups[j].1 + (1 << groups[j].2)
                })
                .expect("every block starts in a lane group");
            let (_, first, width) = groups[j];
            let whole = taken(b) >= width + lane;
            let w_eff = if whole { 0 } else { width };
            let into = at - (first << lane);
            block_hints[b] = [
                count(j),
                bit(whole),
                bit(kappas[b] < w_eff),
                count((into >> w_eff) as usize),
            ];
            at += 1 << taken(b);
        }
    }
    vec![
        ("stk_group_order", order.iter().map(|&g| count(g)).collect()),
        (
            "stk_lane_bits",
            (0..=INITIAL_FOLDING_FACTOR).map(|k| bit(lanes >> k & 1 == 1)).collect(),
        ),
        ("stk_block", block_hints.into_iter().flatten().collect()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guest's groups that are there are `stack_offsets`'s, the powers of two in the lane count, largest first,
    /// none wider than `2^MAX_STRIPE_WIDTH` lanes, for every lane count.
    #[test]
    fn lane_groups_match_the_native_stripe() {
        for lanes in 1..=1u64 << INITIAL_FOLDING_FACTOR {
            let mut native = Vec::new();
            let mut first = 0u64;
            while first < lanes {
                let width = ((lanes - first).ilog2() as usize).min(MAX_STRIPE_WIDTH);
                native.push((first, width));
                first += 1 << width;
            }
            let mut guest: Vec<(u64, usize)> = lane_groups(lanes)
                .into_iter()
                .filter(|g| g.0)
                .map(|(_, first, width)| (first, width))
                .collect();
            guest.sort_unstable();
            assert_eq!(guest, native, "{lanes} lanes");
        }
    }
}

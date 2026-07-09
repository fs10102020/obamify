//! Forward auction algorithm with ε-scaling, plus a reusable sparse-auction
//! primitive.
//!
//! The forward auction is a natural fit for the assignment problem: each
//! source pixel ("bidder") bids for the target position it most wants, and
//! target positions acquire prices as bidders compete. With ε-scaling the
//! solver starts with coarse price increments and progressively refines.
//!
//! Two entry points:
//! - [`process_auction`] — dense forward auction over all (source, target)
//!   pairs for small images. It refuses unsafe sizes rather than allocating an
//!   infeasible dense graph.
//! - [`sparse_auction`] — the same algorithm but each bidder is restricted to
//!   a candidate list of targets. Used by [`super::multiscale`],
//!   [`super::sinkhorn`], and [`super::patchmatch`] as the discrete
//!   refinement / repair stage.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, atomic::AtomicBool};

use crate::app::calculate::ProgressMsg;
#[cfg(not(target_arch = "wasm32"))]
use crate::app::calculate::algorithms::check_cancel;
use crate::app::calculate::algorithms::{
    CostLookup, build_problem, emit_preview, finalize_preset, validate_permutation,
};
#[cfg(test)]
use crate::app::calculate::algorithms::{assert_valid_permutation, total_cost};
use crate::app::calculate::util::GenerationSettings;
use crate::app::calculate::util::ProgressSink;
use crate::app::preset::UnprocessedPreset;

/// Default ε-scaling factor (geometric decay between phases).
const EPSILON_SCALING_FACTOR: f64 = 5.0;

/// Default number of ε-scaling phases.
const DEFAULT_EPSILON_PHASES: usize = 5;
pub const MAX_DENSE_AUCTION_N: usize = 4096;

/// Sparse auction: only a subset of candidate sources per target is considered.
/// Sparse forward auction with ε-scaling.
///
/// `candidates[dst]` is the list of source indices that target `dst` is
/// allowed to consider. Internally we transpose this to per-source candidate
/// lists (the forward auction has sources as bidders). The function returns
/// `assignments[dst] = src` as a permutation of `0..n`, where `n =
/// cost.n_dst()`. Any target whose candidate list is empty will be assigned
/// via a fallback (any unused source), so the result is always a valid
/// permutation.
///
/// `initial_prices[dst]` is an optional warm-start target-price vector (from a
/// coarser level). Pass `None` to start from zero prices.
///
/// `epsilon_phases` controls the ε-scaling schedule: the solver runs multiple
/// phases, each with ε shrinking by `EPSILON_SCALING_FACTOR`. More phases =
/// higher quality, slower.
pub fn sparse_auction(
    cost: &CostLookup,
    candidates: &[Vec<usize>],
    initial_prices: Option<&[f64]>,
    epsilon_phases: usize,
) -> Vec<usize> {
    let n = cost.n_dst();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Transpose: candidates_by_src[src] = list of targets src can bid on.
    let mut candidates_by_src: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (dst, srcs) in candidates.iter().enumerate() {
        for &src in srcs {
            if src < n {
                candidates_by_src[src].push(dst);
            }
        }
    }
    for cands in &mut candidates_by_src {
        cands.sort_unstable();
        cands.dedup();
    }

    let mut prices: Vec<f64> = match initial_prices {
        Some(p) if p.len() == n => p.to_vec(),
        _ => vec![0.0; n],
    };

    // assignment[src] = dst, i.e. which target each source is currently
    // assigned to. usize::MAX = unassigned.
    let mut assigned_dst: Vec<usize> = vec![usize::MAX; n];
    // assigned_src[dst] = src (inverse).
    let mut assigned_src: Vec<usize> = vec![usize::MAX; n];

    // Initial ε: a fraction of the typical cost magnitude.
    let max_cost = estimate_max_cost(cost, &candidates_by_src);
    let mut epsilon = (max_cost as f64 / 4.0).max(1.0);

    let phases = epsilon_phases.max(1);

    for _phase in 0..phases {
        let eps = epsilon.max(1.0 / (n as f64 + 1.0));

        // Reconsider all bidders at the new ε. Keeping prices carries the
        // dual information forward; clearing assignments prevents later
        // ε-scaling phases from becoming no-ops once everyone is assigned.
        assigned_dst.fill(usize::MAX);
        assigned_src.fill(usize::MAX);

        // Run auction to convergence at this ε.
        auction_phase(
            cost,
            &candidates_by_src,
            &mut prices,
            &mut assigned_dst,
            &mut assigned_src,
            eps,
        );

        // Decay ε for the next phase.
        epsilon /= EPSILON_SCALING_FACTOR;
    }

    // Final repair: any unassigned targets get filled with unused sources.
    repair_to_permutation(&mut assigned_src, n);

    assigned_src
}

/// One phase of the forward auction at a fixed ε.
fn auction_phase(
    cost: &CostLookup,
    candidates: &[Vec<usize>],
    prices: &mut [f64],
    assigned_dst: &mut [usize],
    assigned_src: &mut [usize],
    epsilon: f64,
) {
    let n = cost.n_dst();

    // Collect the set of unassigned sources. We iterate: each unassigned
    // source bids for its best target; the target's current holder is evicted
    // (becomes unassigned) and the new bidder takes the spot.
    let mut unassigned_sources: std::collections::VecDeque<usize> = (0..n)
        .filter(|&src| assigned_dst[src] == usize::MAX)
        .collect();

    let mut iter_guard = 0;
    let edge_count = candidates.iter().map(Vec::len).sum::<usize>().max(n);
    let max_iters = edge_count
        .saturating_mul(16)
        .saturating_add(n.saturating_mul(4))
        .max(n); // safety bound without 32-bit overflow

    while !unassigned_sources.is_empty() && iter_guard < max_iters {
        iter_guard += 1;

        // Pick the next unassigned source (FIFO for determinism).
        let Some(src) = unassigned_sources.pop_front() else {
            break;
        };

        // Find the best and second-best target for this source, restricted to
        // its candidate list. Value = -cost (we maximize value = minimize cost).
        let cands = &candidates[src];
        if cands.is_empty() {
            // No candidates: skip, will be handled by repair.
            continue;
        }

        let mut best_dst = cands[0];
        let mut best_value = -(cost.cost(best_dst, src) as f64) - prices[best_dst];
        let mut second_value = f64::MIN;

        for &dst in &cands[1..] {
            let val = -(cost.cost(dst, src) as f64) - prices[dst];
            if val > best_value {
                second_value = best_value;
                best_value = val;
                best_dst = dst;
            } else if val > second_value {
                second_value = val;
            }
        }

        if second_value == f64::MIN {
            second_value = best_value - epsilon;
        }

        // Bid: raise the best target's price so its value drops to second-best.
        let bid_increment = best_value - second_value + epsilon;
        prices[best_dst] += bid_increment.max(0.0);

        // Evict the current holder of best_dst (if any).
        let old_src = assigned_src[best_dst];
        if old_src != usize::MAX {
            assigned_dst[old_src] = usize::MAX;
            unassigned_sources.push_back(old_src);
        }

        // Assign the bidder.
        assigned_dst[src] = best_dst;
        assigned_src[best_dst] = src;
    }
}

/// Estimate the maximum absolute cost over the candidate edges. Used to set
/// the initial ε.
fn estimate_max_cost(cost: &CostLookup, candidates: &[Vec<usize>]) -> i64 {
    let mut max_c: i64 = 0;
    for (src, cands) in candidates.iter().enumerate() {
        for &dst in cands {
            let c = cost.cost(dst, src).abs();
            if c > max_c {
                max_c = c;
            }
        }
    }
    max_c.max(1)
}

/// Fill any unassigned targets with unused sources to guarantee a permutation.
fn repair_to_permutation(assigned_src: &mut [usize], n: usize) {
    let mut used_src = vec![false; n];
    for &s in assigned_src.iter() {
        if s != usize::MAX {
            used_src[s] = true;
        }
    }
    let mut next_unused = 0;
    for val in assigned_src.iter_mut() {
        if *val == usize::MAX {
            while next_unused < n && used_src[next_unused] {
                next_unused += 1;
            }
            if next_unused < n {
                *val = next_unused;
                used_src[next_unused] = true;
            }
        }
    }
}

/// Solve the dense linear assignment problem via forward auction with ε-scaling.
/// Dense forward auction: every source considers every target. Used as a
/// standalone exact/approximate solver and as a baseline.
pub fn solve_dense(cost: &CostLookup, epsilon_phases: usize) -> Option<Vec<usize>> {
    let n = cost.n_dst();
    if n > MAX_DENSE_AUCTION_N {
        return None;
    }
    // Full candidate lists.
    let candidates: Vec<Vec<usize>> = (0..n).map(|_| (0..n).collect()).collect();
    Some(sparse_auction(cost, &candidates, None, epsilon_phases))
}

/// Dense forward auction with ε-scaling. Capped at `MAX_DENSE_AUCTION_N`.
/// Entry point mirroring the legacy `process_optimal` signature.
pub fn process_auction<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    #[cfg(not(target_arch = "wasm32"))] cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (cost, source_pixels, _target_pixels) = build_problem(&unprocessed, &settings)?;
    let n = cost.n_dst();

    tx.send(ProgressMsg::Progress(0.0));
    let Some(assignments) = solve_dense(&cost, DEFAULT_EPSILON_PHASES) else {
        tx.send(ProgressMsg::Error(format!(
            "dense auction is limited to {MAX_DENSE_AUCTION_N} pixels; use multiscale sparse auction for this resolution"
        )));
        return Ok(());
    };
    if let Err(err) = validate_permutation(&assignments, n) {
        tx.send(ProgressMsg::Error(format!(
            "auction produced invalid assignment: {err}"
        )));
        return Ok(());
    }
    emit_preview(tx, &source_pixels, &assignments, settings.sidelen, 0.95);

    #[cfg(not(target_arch = "wasm32"))]
    {
        if check_cancel(&cancel, tx) {
            return Ok(());
        }
    }

    finalize_preset(
        tx,
        unprocessed.name,
        &source_pixels,
        assignments,
        settings.sidelen,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::calculate::algorithms::brute_force_assignment;

    fn identical_pixel_lookup(n: usize, proximity: i64) -> CostLookup {
        let mut source = vec![(0u8, 0u8, 0u8); n];
        let mut target = vec![(0u8, 0u8, 0u8); n];
        for i in 0..n {
            source[i] = (i as u8, 0, 0);
            target[i] = (i as u8, 0, 0);
        }
        let sidelen = (n as f32).sqrt() as u32;
        CostLookup::new(source, target, vec![255; n], sidelen, proximity)
    }

    #[test]
    fn test_auction_dense_valid_permutation_4x4() {
        let cost = identical_pixel_lookup(4, 13);
        let assignments = solve_dense(&cost, 5).unwrap();
        assert_eq!(assignments.len(), 4);
        assert_valid_permutation(&assignments, 4);
    }

    #[test]
    fn test_auction_dense_identity_optimal_4x4() {
        let cost = identical_pixel_lookup(4, 13);
        let assignments = solve_dense(&cost, 8).unwrap();
        let identity: Vec<usize> = (0..4).collect();
        let cost_solve = total_cost(&cost, &assignments);
        let cost_identity = total_cost(&cost, &identity);
        assert!(
            cost_solve <= cost_identity,
            "auction cost {cost_solve} should be <= identity cost {cost_identity}"
        );
    }

    #[test]
    fn test_auction_dense_identity_optimal_9x9() {
        let cost = identical_pixel_lookup(9, 13);
        let assignments = solve_dense(&cost, 8).unwrap();
        let identity: Vec<usize> = (0..9).collect();
        let cost_solve = total_cost(&cost, &assignments);
        let cost_identity = total_cost(&cost, &identity);
        assert!(
            cost_solve <= cost_identity,
            "auction cost {cost_solve} should be <= identity cost {cost_identity}"
        );
    }

    #[test]
    fn test_auction_empty_input() {
        let cost = CostLookup::new(vec![], vec![], vec![], 0, 0);
        let assignments = solve_dense(&cost, 5).unwrap();
        assert!(assignments.is_empty());
    }

    #[test]
    fn test_auction_1x1() {
        let cost = CostLookup::new(vec![(1, 2, 3)], vec![(4, 5, 6)], vec![255], 1, 13);
        let assignments = solve_dense(&cost, 5).unwrap();
        assert_eq!(assignments, vec![0]);
    }

    #[test]
    fn test_sparse_auction_restricted_candidates_valid_permutation() {
        let cost = identical_pixel_lookup(4, 13);
        // candidates[dst] = list of source indices dst can be assigned to.
        // Each target can only take 2 of 4 sources. Still must produce a
        // valid permutation via repair.
        let candidates = vec![
            vec![0, 3], // dst 0 <- src 0 or 3
            vec![0, 1], // dst 1 <- src 0 or 1
            vec![1, 2], // dst 2 <- src 1 or 2
            vec![2, 3], // dst 3 <- src 2 or 3
        ];
        let assignments = sparse_auction(&cost, &candidates, None, 5);
        assert_eq!(assignments.len(), 4);
        assert_valid_permutation(&assignments, 4);
    }

    #[test]
    fn test_sparse_auction_empty_candidates_repaired() {
        let cost = identical_pixel_lookup(4, 13);
        let candidates = vec![vec![], vec![], vec![], vec![]];
        let assignments = sparse_auction(&cost, &candidates, None, 3);
        assert_valid_permutation(&assignments, 4);
    }

    #[test]
    fn test_sparse_auction_warm_start_prices() {
        let cost = identical_pixel_lookup(4, 13);
        // Full candidate lists (every dst can take any src).
        let candidates: Vec<Vec<usize>> = (0..4).map(|_| (0..4).collect()).collect();
        let prices = vec![10.0, 20.0, 30.0, 40.0];
        let assignments = sparse_auction(&cost, &candidates, Some(&prices), 5);
        assert_valid_permutation(&assignments, 4);
    }

    #[test]
    fn test_auction_epsilon_scaling_monotone_cost() {
        // More ε phases should produce <= cost of fewer phases (or equal).
        let cost = identical_pixel_lookup(9, 13);
        let coarse = solve_dense(&cost, 1).unwrap();
        let fine = solve_dense(&cost, 8).unwrap();
        let cost_coarse = total_cost(&cost, &coarse);
        let cost_fine = total_cost(&cost, &fine);
        assert!(
            cost_fine <= cost_coarse,
            "fine (8 phases) cost {cost_fine} should be <= coarse (1 phase) cost {cost_coarse}"
        );
    }

    #[test]
    fn test_auction_dense_matches_bruteforce_oracle_on_permuted_colors() {
        let source = vec![(0, 0, 0), (40, 0, 0), (90, 0, 0), (150, 0, 0)];
        let target = vec![source[2], source[0], source[3], source[1]];
        let cost = CostLookup::new(source, target, vec![255; 4], 2, 0);
        let oracle = brute_force_assignment(&cost);
        let got = solve_dense(&cost, 8).unwrap();
        assert_valid_permutation(&got, 4);
        assert_eq!(total_cost(&cost, &got), total_cost(&cost, &oracle));
    }

    #[test]
    fn test_dense_auction_refuses_unsafe_size() {
        let n = MAX_DENSE_AUCTION_N + 1;
        let cost = CostLookup::new(vec![(0, 0, 0); n], vec![(0, 0, 0); n], vec![255; n], 1, 0);
        assert!(solve_dense(&cost, 1).is_none());
    }

    #[test]
    fn test_sparse_auction_256_sidelen_guard_does_not_overflow() {
        let sidelen = 256;
        let n = sidelen * sidelen;
        let cost = CostLookup::new(
            vec![(0, 0, 0); n],
            vec![(0, 0, 0); n],
            vec![255; n],
            sidelen as u32,
            0,
        );
        let candidates: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let assignments = sparse_auction(&cost, &candidates, None, 1);
        assert_valid_permutation(&assignments, n);
        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[n - 1], n - 1);
    }
}

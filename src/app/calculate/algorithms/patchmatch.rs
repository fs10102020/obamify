//! PatchMatch-style correspondence search + repair to a permutation.
//!
//! PatchMatch discovers image correspondences via:
//!   1. Propagation: each target pixel tests the source assignments of its
//!      spatial neighbors (shifted by their displacement) and keeps the
//!      lowest-cost candidate.
//!   2. Random search: test random source candidates at exponentially
//!      shrinking radii.
//!
//! Ordinary PatchMatch allows many targets to choose the same source, so we
//! add a repair stage: use each raw correspondence plus nearby sources as a
//! sparse candidate graph, then run sparse auction to produce a valid
//! permutation.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, atomic::AtomicBool};

use crate::app::calculate::ProgressMsg;
use crate::app::calculate::algorithms::auction::sparse_auction;
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

/// Number of PatchMatch iterations (propagation + random search passes).
const PATCHMATCH_ITERS: usize = 4;

/// Initial random search radius (as a fraction of sidelen).
const INITIAL_RADIUS_FRAC: f32 = 0.5;

/// Random search shrink factor.
const RADIUS_SHRINK: f32 = 0.5;

/// Number of random search attempts per pixel per iteration.
const RANDOM_ATTEMPTS: usize = 8;

/// Run PatchMatch-style search and return a (non-unique) assignment where
/// `assignments[dst] = src`. Multiple dsts may map to the same src.
fn patchmatch_search(cost: &CostLookup, sidelen: u32, iters: usize) -> Vec<usize> {
    let n = cost.n_dst();
    if n == 0 {
        return Vec::new();
    }
    let s = sidelen as i32;

    // Initialize with nearest-color (cheap init: for each target, find the
    // closest source by color among a random sample).
    let mut rng = frand::Rand::with_seed(42424);
    let mut assignments: Vec<usize> = Vec::with_capacity(n);
    for dst in 0..n {
        let target_col = cost.target[dst];
        let mut best_src = 0;
        let mut best_dist = i64::MAX;
        for _ in 0..16 {
            let sample = rng.gen_range(0..n as u64) as usize;
            let src_col = cost.source[sample];
            let dist = (target_col.0 as i64 - src_col.0 as i64).pow(2)
                + (target_col.1 as i64 - src_col.1 as i64).pow(2)
                + (target_col.2 as i64 - src_col.2 as i64).pow(2);
            if dist < best_dist {
                best_dist = dist;
                best_src = sample;
            }
        }
        assignments.push(best_src);
    }

    let initial_radius = (s as f32 * INITIAL_RADIUS_FRAC).max(1.0) as i32;

    for iter in 0..iters {
        // Propagation pass: raster order (odd iters) or reverse (even iters).
        let forward = iter % 2 == 0;
        let order: Vec<usize> = if forward {
            (0..n).collect()
        } else {
            (0..n).rev().collect()
        };
        let neighbor_offsets: &[(i32, i32)] = if forward {
            &[(0, -1), (-1, 0)]
        } else {
            &[(0, 1), (1, 0)]
        };

        for &dst in &order {
            let dx_pos = dst % sidelen as usize;
            let dy_pos = dst / sidelen as usize;
            let current_cost = cost.cost(dst, assignments[dst]);

            // Test neighbor's assignment (propagation).
            let mut best_cost = current_cost;
            let mut best_src = assignments[dst];

            for &(ndx, ndy) in neighbor_offsets {
                let nx = dx_pos as i32 + ndx;
                let ny = dy_pos as i32 + ndy;
                if nx < 0 || ny < 0 || nx >= s || ny >= s {
                    continue;
                }
                let neighbor_dst = (ny as usize) * sidelen as usize + nx as usize;
                let neighbor_src = assignments[neighbor_dst];
                // The neighbor's displacement: neighbor_src - neighbor_dst.
                // Apply the same displacement to our dst.
                let ndx_src = neighbor_src % sidelen as usize;
                let ndy_src = neighbor_src / sidelen as usize;
                let disp_x = ndx_src as i32 - nx;
                let disp_y = ndy_src as i32 - ny;
                let candidate_x = (dx_pos as i32 + disp_x).clamp(0, s - 1);
                let candidate_y = (dy_pos as i32 + disp_y).clamp(0, s - 1);
                let candidate_src =
                    (candidate_y as usize) * sidelen as usize + candidate_x as usize;
                let candidate_cost = cost.cost(dst, candidate_src);
                if candidate_cost < best_cost {
                    best_cost = candidate_cost;
                    best_src = candidate_src;
                }
            }

            // Random search: test candidates at shrinking radii.
            let mut radius = initial_radius;
            while radius > 0 {
                let best_x = best_src % sidelen as usize;
                let best_y = best_src / sidelen as usize;
                for _ in 0..RANDOM_ATTEMPTS {
                    let rx = rng.gen_range(-radius..(radius + 1));
                    let ry = rng.gen_range(-radius..(radius + 1));
                    let cx = (best_x as i32 + rx).clamp(0, s - 1);
                    let cy = (best_y as i32 + ry).clamp(0, s - 1);
                    let candidate_src = (cy as usize) * sidelen as usize + cx as usize;
                    let candidate_cost = cost.cost(dst, candidate_src);
                    if candidate_cost < best_cost {
                        best_cost = candidate_cost;
                        best_src = candidate_src;
                    }
                }
                radius = (radius as f32 * RADIUS_SHRINK) as i32;
            }

            assignments[dst] = best_src;
        }
    }

    assignments
}

/// Repair a non-unique assignment to a valid permutation.
/// Use the raw PatchMatch source plus nearby sources as sparse candidates,
/// then let sparse auction resolve collisions globally.
fn repair_permutation(cost: &CostLookup, raw: &[usize], sidelen: u32) -> Vec<usize> {
    let n = cost.n_dst();
    if n == 0 {
        return Vec::new();
    }

    // Build candidates for the sparse auction repair: for each target, use
    // the raw PatchMatch assignment's source plus spatially nearby sources.
    let s = sidelen as i32;
    let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); n];
    for dst in 0..n {
        let raw_src = raw[dst];
        let sx_pos = raw_src % sidelen as usize;
        let sy_pos = raw_src / sidelen as usize;
        let mut cand_set: Vec<usize> = Vec::with_capacity(16);
        // Include the raw assignment's source.
        cand_set.push(raw_src);
        // Spatial neighborhood.
        for ddy in -3..=3 {
            for ddx in -3..=3 {
                let nx = (sx_pos as i32 + ddx).clamp(0, s - 1);
                let ny = (sy_pos as i32 + ddy).clamp(0, s - 1);
                cand_set.push((ny as usize) * sidelen as usize + nx as usize);
            }
        }
        cand_set.sort_unstable();
        cand_set.dedup();
        candidates[dst] = cand_set;
    }

    // Run sparse auction over the candidate lists to get a full permutation.
    sparse_auction(cost, &candidates, None, 3)
}

/// Run the full PatchMatch + repair pipeline and return the assignment.
pub fn solve(cost: &CostLookup, sidelen: u32) -> Vec<usize> {
    let n = cost.n_dst();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let raw = patchmatch_search(cost, sidelen, PATCHMATCH_ITERS);
    repair_permutation(cost, &raw, sidelen)
}

/// PatchMatch propagation/random-search correspondence + sparse auction repair.
/// Entry point mirroring the legacy `process_optimal` signature.
pub fn process_patchmatch<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    #[cfg(not(target_arch = "wasm32"))] cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (cost, source_pixels, _target_pixels) = build_problem(&unprocessed, &settings)?;
    let n = cost.n_dst();

    tx.send(ProgressMsg::Progress(0.0));
    let assignments = solve(&cost, settings.sidelen);
    if let Err(err) = validate_permutation(&assignments, n) {
        tx.send(ProgressMsg::Error(format!(
            "PatchMatch produced invalid assignment: {err}"
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
    fn test_patchmatch_4x4_valid_permutation() {
        let cost = identical_pixel_lookup(4, 13);
        let assignments = solve(&cost, 2);
        assert_eq!(assignments.len(), 4);
        assert_valid_permutation(&assignments, 4);
    }

    #[test]
    fn test_patchmatch_9x9_valid_permutation() {
        let cost = identical_pixel_lookup(9, 13);
        let assignments = solve(&cost, 3);
        assert_eq!(assignments.len(), 9);
        assert_valid_permutation(&assignments, 9);
    }

    #[test]
    fn test_patchmatch_identity_near_optimal_4x4() {
        let cost = identical_pixel_lookup(4, 13);
        let assignments = solve(&cost, 2);
        let identity: Vec<usize> = (0..4).collect();
        let cost_solve = total_cost(&cost, &assignments);
        let cost_identity = total_cost(&cost, &identity);
        // PatchMatch is a heuristic; allow generous slack.
        let slack = cost_identity.abs() + 1;
        assert!(
            cost_solve <= cost_identity + slack,
            "patchmatch cost {cost_solve} should be reasonable vs identity cost {cost_identity}"
        );
    }

    #[test]
    fn test_patchmatch_cost_monotone_decreasing() {
        let cost = identical_pixel_lookup(9, 13);
        // More iterations should not increase cost (monotone non-increasing
        // during the search phase, before repair).
        let raw1 = patchmatch_search(&cost, 3, 1);
        let raw4 = patchmatch_search(&cost, 3, 4);
        let cost1: i64 = (0..9).map(|dst| cost.cost(dst, raw1[dst])).sum();
        let cost4: i64 = (0..9).map(|dst| cost.cost(dst, raw4[dst])).sum();
        assert!(
            cost4 <= cost1,
            "more iters should not increase raw cost: {cost4} > {cost1}"
        );
    }

    #[test]
    fn test_patchmatch_empty_input() {
        let cost = CostLookup::new(vec![], vec![], vec![], 0, 0);
        let assignments = solve(&cost, 0);
        assert!(assignments.is_empty());
    }

    #[test]
    fn test_patchmatch_1x1() {
        let cost = CostLookup::new(vec![(1, 2, 3)], vec![(4, 5, 6)], vec![255], 1, 13);
        let assignments = solve(&cost, 1);
        assert_eq!(assignments, vec![0]);
    }

    #[test]
    fn test_repair_permutation_valid() {
        let cost = identical_pixel_lookup(4, 13);
        // Raw assignment with collisions: all targets claim source 0.
        let raw = vec![0, 0, 0, 0];
        let assignments = repair_permutation(&cost, &raw, 2);
        assert_valid_permutation(&assignments, 4);
    }

    #[test]
    fn test_patchmatch_repeated_runs_are_deterministic() {
        let cost = identical_pixel_lookup(16, 13);
        let a = solve(&cost, 4);
        let b = solve(&cost, 4);
        assert_eq!(a, b);
    }
}

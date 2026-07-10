//! Multiscale sparse auction — the headline algorithm.
//!
//! Solves the assignment problem at a hierarchy of resolutions
//! (16 → 32 → 64 → … → sidelen), using an exact JV solve at the coarsest
//! level and sparse ε-scaling auction at each finer level. At each level,
//! the candidate set for a target pixel is built from:
//!   - children of the parent's matched source block
//!   - spatially nearby sources (within a radius of the predicted position)
//!   - nearest-color sources (via coarse color bucketing)
//!   - a small number of random "escape" candidates
//!
//! Prices are warm-started from the coarser level. After the auction, a
//! 2-opt local swap pass cleans up residual sub-optimality.

use crate::app::calculate::ProgressMsg;
use crate::app::calculate::algorithms::auction::sparse_auction_with_checkpoint;
use crate::app::calculate::algorithms::checkpoint;
use crate::app::calculate::algorithms::jonker_volgenant::solve_with_checkpoint as jv_solve;
use crate::app::calculate::algorithms::{
    CostLookup, build_problem, emit_preview, finalize_preset, validate_permutation,
};
#[cfg(test)]
use crate::app::calculate::algorithms::{assert_valid_permutation, total_cost};
use crate::app::calculate::util::GenerationSettings;
use crate::app::calculate::util::ProgressSink;
use crate::app::calculate::util::SolverControl;
use crate::app::preset::UnprocessedPreset;

/// Coarsest pyramid level. Must be a power of two and <= sidelen.
const COARSEST_SIDELEN: u32 = 16;

/// Number of ε-scaling phases per pyramid level.
const PHASES_PER_LEVEL: usize = 4;

/// Spatial candidate radius (in fine-level pixels) around the predicted position.
const SPATIAL_RADIUS: i32 = 4;

/// Number of nearest-color candidates to include.
const NEAREST_COLOR_CANDIDATES: usize = 8;

/// Number of random escape candidates.
const RANDOM_CANDIDATES: usize = 4;

/// Build the pyramid of sidelens from COARSEST_SIDELEN up to `target_sidelen`.
/// Always includes the target sidelen as the finest level.
fn build_pyramid(target_sidelen: u32) -> Vec<u32> {
    let mut levels = Vec::new();
    let mut s = COARSEST_SIDELEN;
    while s < target_sidelen {
        levels.push(s);
        s *= 2;
    }
    levels.push(target_sidelen);
    levels
}

/// Average-pool a pixel array from `from_sidelen` down to `to_sidelen`.
/// Works for arbitrary square sizes by assigning each output pixel the average
/// of the covered integer source-pixel rectangle.
fn downsample_pixels(
    pixels: &[(u8, u8, u8)],
    from_sidelen: u32,
    to_sidelen: u32,
) -> Vec<(u8, u8, u8)> {
    if from_sidelen == to_sidelen {
        return pixels.to_vec();
    }
    let mut out = Vec::with_capacity((to_sidelen * to_sidelen) as usize);
    for oy in 0..to_sidelen {
        for ox in 0..to_sidelen {
            let x0 = ox * from_sidelen / to_sidelen;
            let x1 = ((ox + 1) * from_sidelen).div_ceil(to_sidelen).max(x0 + 1);
            let y0 = oy * from_sidelen / to_sidelen;
            let y1 = ((oy + 1) * from_sidelen).div_ceil(to_sidelen).max(y0 + 1);
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut count = 0u32;
            for iy in y0..y1.min(from_sidelen) {
                for ix in x0..x1.min(from_sidelen) {
                    let idx = (iy * from_sidelen + ix) as usize;
                    let (r, g, b) = pixels[idx];
                    r_sum += r as u32;
                    g_sum += g as u32;
                    b_sum += b as u32;
                    count += 1;
                }
            }
            out.push((
                (r_sum / count) as u8,
                (g_sum / count) as u8,
                (b_sum / count) as u8,
            ));
        }
    }
    out
}

/// Downsample weights similarly (averaging).
fn downsample_weights(weights: &[i64], from_sidelen: u32, to_sidelen: u32) -> Vec<i64> {
    if from_sidelen == to_sidelen {
        return weights.to_vec();
    }
    let mut out = Vec::with_capacity((to_sidelen * to_sidelen) as usize);
    for oy in 0..to_sidelen {
        for ox in 0..to_sidelen {
            let x0 = ox * from_sidelen / to_sidelen;
            let x1 = ((ox + 1) * from_sidelen).div_ceil(to_sidelen).max(x0 + 1);
            let y0 = oy * from_sidelen / to_sidelen;
            let y1 = ((oy + 1) * from_sidelen).div_ceil(to_sidelen).max(y0 + 1);
            let mut sum = 0i64;
            let mut count = 0i64;
            for iy in y0..y1.min(from_sidelen) {
                for ix in x0..x1.min(from_sidelen) {
                    let idx = (iy * from_sidelen + ix) as usize;
                    sum += weights[idx];
                    count += 1;
                }
            }
            out.push(sum / count);
        }
    }
    out
}

/// Upsample an assignment from `from_sidelen` to `to_sidelen`.
/// `coarse_assignments[dst] = src` at the coarse level. Each fine target
/// pixel maps to a coarse target, and its predicted source is the child of
/// the coarse source. Returns `predicted_src[fine_dst] = fine_src`.
fn upsample_assignment(
    coarse_assignments: &[usize],
    from_sidelen: u32,
    to_sidelen: u32,
) -> Vec<usize> {
    let mut out = Vec::with_capacity((to_sidelen * to_sidelen) as usize);
    for fine_y in 0..to_sidelen {
        for fine_x in 0..to_sidelen {
            let coarse_x = fine_x * from_sidelen / to_sidelen;
            let coarse_y = fine_y * from_sidelen / to_sidelen;
            let coarse_dst = (coarse_y * from_sidelen + coarse_x) as usize;
            let coarse_src = coarse_assignments[coarse_dst];
            // Map coarse source to fine source: place at the same relative
            // sub-position within the source block.
            let target_block_x0 = coarse_x * to_sidelen / from_sidelen;
            let target_block_y0 = coarse_y * to_sidelen / from_sidelen;
            let coarse_src_x = coarse_src % from_sidelen as usize;
            let coarse_src_y = coarse_src / from_sidelen as usize;
            let source_block_x0 = coarse_src_x as u32 * to_sidelen / from_sidelen;
            let source_block_y0 = coarse_src_y as u32 * to_sidelen / from_sidelen;
            let source_block_x1 = ((coarse_src_x as u32 + 1) * to_sidelen)
                .div_ceil(from_sidelen)
                .max(source_block_x0 + 1);
            let source_block_y1 = ((coarse_src_y as u32 + 1) * to_sidelen)
                .div_ceil(from_sidelen)
                .max(source_block_y0 + 1);
            let fine_src_x = (source_block_x0 + fine_x.saturating_sub(target_block_x0))
                .min(source_block_x1.saturating_sub(1))
                .min(to_sidelen - 1) as usize;
            let fine_src_y = (source_block_y0 + fine_y.saturating_sub(target_block_y0))
                .min(source_block_y1.saturating_sub(1))
                .min(to_sidelen - 1) as usize;
            out.push(fine_src_y * to_sidelen as usize + fine_src_x);
        }
    }
    out
}

/// Build sparse candidate source sets for each target from coarser-level predictions.
/// Build candidate lists for each fine target, given the predicted source
/// positions from the coarse level.
fn build_candidates(cost: &CostLookup, predicted: &[usize], sidelen: u32) -> Vec<Vec<usize>> {
    let n = cost.n_dst();
    let s = sidelen as i32;
    let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut rng = frand::Rand::with_seed(98765);

    for dst in 0..n {
        let mut cand_set: Vec<usize> = Vec::with_capacity(32);

        // 1. Children of the predicted source block (the predicted source +
        // its immediate neighbors).
        let pred = predicted[dst];
        cand_set.push(pred);
        let pred_x = pred % sidelen as usize;
        let pred_y = pred / sidelen as usize;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = pred_x as i32 + dx;
                let ny = pred_y as i32 + dy;
                if nx >= 0 && nx < s && ny >= 0 && ny < s {
                    cand_set.push((ny as usize) * sidelen as usize + nx as usize);
                }
            }
        }

        // 2. Spatially nearby sources (within SPATIAL_RADIUS of predicted).
        let radius = SPATIAL_RADIUS;
        let step_x = ((radius as usize) / 2).max(1);
        let step_y = ((radius as usize) / 2).max(1);
        let r = radius;
        let mut dy = -r;
        while dy <= r {
            let mut dx = -r;
            while dx <= r {
                let nx = pred_x as i32 + dx;
                let ny = pred_y as i32 + dy;
                if nx >= 0 && nx < s && ny >= 0 && ny < s {
                    cand_set.push((ny as usize) * sidelen as usize + nx as usize);
                }
                dx += step_x as i32;
            }
            dy += step_y as i32;
        }

        // 3. Nearest-color candidates: sample a few random sources and keep
        //    the closest by color. This is a cheap approximation of true
        //    nearest-color search.
        let target_color = cost.target[dst];
        let mut color_candidates: Vec<(i64, usize)> = Vec::new();
        for _ in 0..(NEAREST_COLOR_CANDIDATES * 4) {
            let sample = rng.gen_range(0..n as u64) as usize;
            let src_col = cost.source[sample];
            let color_dist = (target_color.0 as i64 - src_col.0 as i64).pow(2)
                + (target_color.1 as i64 - src_col.1 as i64).pow(2)
                + (target_color.2 as i64 - src_col.2 as i64).pow(2);
            color_candidates.push((color_dist, sample));
        }
        color_candidates.sort_by_key(|(d, _)| *d);
        for (_, src) in color_candidates.into_iter().take(NEAREST_COLOR_CANDIDATES) {
            cand_set.push(src);
        }

        // 4. Random escape candidates.
        for _ in 0..RANDOM_CANDIDATES {
            cand_set.push(rng.gen_range(0..n as u64) as usize);
        }

        cand_set.sort_unstable();
        cand_set.dedup();
        candidates[dst] = cand_set;
    }

    candidates
}

/// 2-opt local swap refinement over a small neighborhood.
/// 2-opt local swap refinement: for each pair of nearby targets, check if
/// swapping their assigned sources reduces total cost. A few sweeps.
pub fn local_swap_refinement(
    cost: &CostLookup,
    assignments: &mut [usize],
    sidelen: u32,
    sweeps: usize,
) {
    let _ = local_swap_refinement_with_checkpoint(cost, assignments, sidelen, sweeps, || true);
}

pub(crate) fn local_swap_refinement_with_checkpoint<F>(
    cost: &CostLookup,
    assignments: &mut [usize],
    sidelen: u32,
    sweeps: usize,
    mut checkpoint: F,
) -> bool
where
    F: FnMut() -> bool,
{
    let n = assignments.len();
    let s = sidelen as usize;

    for _ in 0..sweeps {
        if !checkpoint() {
            return false;
        }
        let mut improved = false;
        for dst_a in 0..n {
            if dst_a % 1024 == 0 && !checkpoint() {
                return false;
            }
            let ax = dst_a % s;
            let ay = dst_a / s;
            // Check a small neighborhood of nearby targets.
            for dy in 0..=2 {
                for dx in 0..=2 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = ax as i32 + dx - 1;
                    let ny = ay as i32 + dy - 1;
                    if nx < 0 || ny < 0 || nx >= s as i32 || ny >= s as i32 {
                        continue;
                    }
                    let dst_b = (ny as usize) * s + nx as usize;
                    if dst_b <= dst_a {
                        continue;
                    }
                    let src_a = assignments[dst_a];
                    let src_b = assignments[dst_b];
                    let current = cost.cost(dst_a, src_a) + cost.cost(dst_b, src_b);
                    let swapped = cost.cost(dst_a, src_b) + cost.cost(dst_b, src_a);
                    if swapped < current {
                        assignments[dst_a] = src_b;
                        assignments[dst_b] = src_a;
                        improved = true;
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }
    true
}

/// Build a `CostLookup` for a specific pyramid level by downsampling.
fn level_cost(
    source: &[(u8, u8, u8)],
    target: &[(u8, u8, u8)],
    weights: &[i64],
    full_sidelen: u32,
    level_sidelen: u32,
    proximity_importance: i64,
) -> CostLookup {
    if level_sidelen == full_sidelen {
        return CostLookup::new(
            source.to_vec(),
            target.to_vec(),
            weights.to_vec(),
            level_sidelen,
            proximity_importance,
        );
    }
    let ds_source = downsample_pixels(source, full_sidelen, level_sidelen);
    let ds_target = downsample_pixels(target, full_sidelen, level_sidelen);
    let ds_weights = downsample_weights(weights, full_sidelen, level_sidelen);
    CostLookup::new(
        ds_source,
        ds_target,
        ds_weights,
        level_sidelen,
        proximity_importance,
    )
}

/// Run the multiscale sparse auction solver and return the assignment.
pub fn solve(
    cost: &CostLookup,
    source_pixels: &[(u8, u8, u8)],
    target_pixels: &[(u8, u8, u8)],
    weights: &[i64],
    sidelen: u32,
    proximity_importance: i64,
) -> Vec<usize> {
    solve_with_checkpoint(
        cost,
        source_pixels,
        target_pixels,
        weights,
        sidelen,
        proximity_importance,
        || true,
    )
    .expect("uncancelled multiscale solve should complete")
}

pub(crate) fn solve_with_checkpoint<F>(
    cost: &CostLookup,
    source_pixels: &[(u8, u8, u8)],
    target_pixels: &[(u8, u8, u8)],
    weights: &[i64],
    sidelen: u32,
    proximity_importance: i64,
    mut checkpoint: F,
) -> Option<Vec<usize>>
where
    F: FnMut() -> bool,
{
    solve_inner(
        cost,
        source_pixels,
        target_pixels,
        weights,
        sidelen,
        proximity_importance,
        |_, _| checkpoint(),
    )
}

fn solve_inner<F>(
    cost: &CostLookup,
    source_pixels: &[(u8, u8, u8)],
    target_pixels: &[(u8, u8, u8)],
    weights: &[i64],
    sidelen: u32,
    proximity_importance: i64,
    mut on_level: F,
) -> Option<Vec<usize>>
where
    F: FnMut(usize, usize) -> bool,
{
    let n = cost.n_dst();
    if n == 0 {
        return Some(Vec::new());
    }
    if n == 1 {
        return Some(vec![0]);
    }

    let pyramid = build_pyramid(sidelen);
    let total_levels = pyramid.len();
    let mut prev_assignment: Option<Vec<usize>> = None;
    let mut prev_prices: Option<Vec<f64>> = None;
    let mut prev_sidelen: Option<u32> = None;

    for (level_idx, &level_sidelen) in pyramid.iter().enumerate() {
        if !on_level(level_idx, total_levels) {
            return None;
        }

        let level_cost = level_cost(
            source_pixels,
            target_pixels,
            weights,
            sidelen,
            level_sidelen,
            proximity_importance,
        );

        let assignment = match (level_sidelen == COARSEST_SIDELEN, prev_assignment.as_ref()) {
            (true, _) | (false, None) => {
                // Coarsest level or no previous: exact JV.
                jv_solve(&level_cost, || on_level(level_idx, total_levels))?
            }
            (false, Some(prev)) => {
                // Finer level: upsample prediction, build candidates, sparse auction.
                let from_sidelen = prev_sidelen.expect("previous level has assignment");
                let predicted = upsample_assignment(prev, from_sidelen, level_sidelen);
                let candidates = build_candidates(&level_cost, &predicted, level_sidelen);
                let warm_prices = prev_prices
                    .as_ref()
                    .map(|p| upsample_prices(p, from_sidelen, level_sidelen));
                sparse_auction_with_checkpoint(
                    &level_cost,
                    &candidates,
                    warm_prices.as_deref(),
                    PHASES_PER_LEVEL,
                    || on_level(level_idx, total_levels),
                )?
            }
        };

        prev_assignment = Some(assignment);
        prev_sidelen = Some(level_sidelen);
        // Reset prices for the next level (the auction re-derives them).
        prev_prices = Some(vec![0.0; level_cost.n_dst()]);
    }

    let mut final_assignment = prev_assignment.expect("pyramid produced no assignment");
    // 2-opt local refinement at the finest level.
    if !local_swap_refinement_with_checkpoint(cost, &mut final_assignment, sidelen, 3, || {
        on_level(total_levels, total_levels)
    }) {
        return None;
    }
    Some(final_assignment)
}

/// Upsample a price vector from `from_sidelen` to `to_sidelen` by
/// nearest-neighbor replication.
fn upsample_prices(prices: &[f64], from_sidelen: u32, to_sidelen: u32) -> Vec<f64> {
    if from_sidelen == to_sidelen {
        return prices.to_vec();
    }
    let mut out = Vec::with_capacity((to_sidelen * to_sidelen) as usize);
    for fine_y in 0..to_sidelen {
        for fine_x in 0..to_sidelen {
            let coarse_x = fine_x * from_sidelen / to_sidelen;
            let coarse_y = fine_y * from_sidelen / to_sidelen;
            let idx = (coarse_y * from_sidelen + coarse_x) as usize;
            out.push(prices[idx]);
        }
    }
    out
}

/// Coarse-to-fine sparse auction. The headline/default algorithm.
/// Entry point mirroring the legacy `process_optimal` signature.
pub fn process_multiscale<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    control: SolverControl,
) -> Result<(), Box<dyn std::error::Error>> {
    let (cost, source_pixels, target_pixels) = build_problem(&unprocessed, &settings)?;
    let n = cost.n_dst();
    let weights = &cost.weights;

    let mut cancelled = false;
    let assignments = solve_inner(
        &cost,
        &source_pixels,
        &target_pixels,
        weights,
        settings.sidelen,
        settings.proximity_importance,
        |level_idx, total_levels| {
            let progress = level_idx as f32 / total_levels as f32;
            tx.send(ProgressMsg::Progress(progress * 0.8));
            if checkpoint(&control, tx) {
                cancelled = true;
                return false;
            }
            true
        },
    );

    let Some(assignments) = assignments else {
        if !cancelled {
            tx.send(ProgressMsg::Error(
                "multiscale solve did not complete".to_string(),
            ));
        }
        return Ok(());
    };

    if let Err(err) = validate_permutation(&assignments, n) {
        tx.send(ProgressMsg::Error(format!(
            "multiscale produced invalid assignment: {err}"
        )));
        return Ok(());
    }
    emit_preview(tx, &source_pixels, &assignments, settings.sidelen, 0.95);

    if checkpoint(&control, tx) {
        return Ok(());
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

    // Test helper mirrors the production pixel triple shape.
    #[expect(clippy::type_complexity)]
    fn identical_pixels(n: usize) -> (Vec<(u8, u8, u8)>, Vec<(u8, u8, u8)>, Vec<i64>) {
        let mut source = vec![(0u8, 0u8, 0u8); n];
        let mut target = vec![(0u8, 0u8, 0u8); n];
        for i in 0..n {
            source[i] = (i as u8, 0, 0);
            target[i] = (i as u8, 0, 0);
        }
        (source, target, vec![255; n])
    }

    #[test]
    fn test_build_pyramid_128() {
        let p = build_pyramid(128);
        assert_eq!(p, vec![16, 32, 64, 128]);
    }

    #[test]
    fn test_build_pyramid_64() {
        let p = build_pyramid(64);
        assert_eq!(p, vec![16, 32, 64]);
    }

    #[test]
    fn test_build_pyramid_16() {
        let p = build_pyramid(16);
        assert_eq!(p, vec![16]);
    }

    #[test]
    fn test_downsample_pixels_4_to_2() {
        let pixels = vec![(10, 20, 30), (20, 30, 40), (30, 40, 50), (40, 50, 60)];
        let ds = downsample_pixels(&pixels, 2, 1);
        assert_eq!(ds, vec![(25, 35, 45)]); // average of all 4
    }

    #[test]
    fn test_downsample_pixels_identity() {
        let pixels = vec![(10, 20, 30), (40, 50, 60)];
        let ds = downsample_pixels(&pixels, 2, 2);
        assert_eq!(ds, pixels);
    }

    #[test]
    fn test_upsample_assignment_2_to_4() {
        // 2x2 coarse assignment: identity
        let coarse = vec![0, 1, 2, 3];
        let fine = upsample_assignment(&coarse, 2, 4);
        assert_eq!(fine.len(), 16);
        // Fine[0] should map to source 0 (top-left child of source 0)
        assert_eq!(fine[0], 0);
        // Fine[3] (top-right of top-left block) -> source 3
        assert_eq!(fine[3], 3);
    }

    #[test]
    fn test_multiscale_solve_16_valid_permutation() {
        let n = 16;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(source.clone(), target.clone(), weights.clone(), 4, 13);
        let assignments = solve(&cost, &source, &target, &weights, 4, 13);
        assert_eq!(assignments.len(), 16);
        assert_valid_permutation(&assignments, 16);
    }

    #[test]
    fn test_multiscale_solve_64_valid_permutation() {
        let n = 64;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(source.clone(), target.clone(), weights.clone(), 8, 13);
        let assignments = solve(&cost, &source, &target, &weights, 8, 13);
        assert_eq!(assignments.len(), 64);
        assert_valid_permutation(&assignments, 64);
    }

    #[test]
    fn test_multiscale_non_power_of_two_sidelen_valid_permutation() {
        let sidelen = 20;
        let n = sidelen * sidelen;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(
            source.clone(),
            target.clone(),
            weights.clone(),
            sidelen as u32,
            13,
        );
        let assignments = solve(&cost, &source, &target, &weights, sidelen as u32, 13);
        assert_eq!(assignments.len(), n);
        assert_valid_permutation(&assignments, n);
    }

    #[test]
    fn test_multiscale_repeated_runs_are_deterministic() {
        let sidelen = 20;
        let n = sidelen * sidelen;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(
            source.clone(),
            target.clone(),
            weights.clone(),
            sidelen as u32,
            13,
        );
        let a = solve(&cost, &source, &target, &weights, sidelen as u32, 13);
        let b = solve(&cost, &source, &target, &weights, sidelen as u32, 13);
        assert_eq!(a, b);
    }

    #[test]
    fn test_multiscale_identity_optimal_16() {
        let n = 16;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(source.clone(), target.clone(), weights.clone(), 4, 13);
        let assignments = solve(&cost, &source, &target, &weights, 4, 13);
        let identity: Vec<usize> = (0..n).collect();
        let cost_solve = total_cost(&cost, &assignments);
        let cost_identity = total_cost(&cost, &identity);
        // Multiscale should be close to optimal; allow small slack.
        let slack = cost_identity.abs() / 4 + 1;
        assert!(
            cost_solve <= cost_identity + slack,
            "multiscale cost {cost_solve} should be close to identity cost {cost_identity} (slack {slack})"
        );
    }

    #[test]
    fn test_local_swap_refinement_reduces_cost() {
        let n = 4;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(source, target, weights, 2, 13);
        // Start with a suboptimal assignment.
        let mut assignments = vec![1, 0, 3, 2]; // swapped pairs
        let before = total_cost(&cost, &assignments);
        local_swap_refinement(&cost, &mut assignments, 2, 5);
        let after = total_cost(&cost, &assignments);
        assert!(
            after <= before,
            "local swap should not increase cost: {after} > {before}"
        );
        assert_valid_permutation(&assignments, n);
    }

    #[test]
    fn test_build_candidates_nonempty() {
        let n = 16;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(source, target, weights, 4, 13);
        let predicted: Vec<usize> = (0..16).collect(); // identity prediction
        let candidates = build_candidates(&cost, &predicted, 4);
        for (dst, cands) in candidates.iter().enumerate() {
            assert!(!cands.is_empty(), "dst {dst} has empty candidates");
        }
    }

    #[test]
    fn test_multiscale_tiny_matches_bruteforce_oracle() {
        let source = vec![(0, 0, 0), (40, 0, 0), (90, 0, 0), (150, 0, 0)];
        let target = vec![source[2], source[0], source[3], source[1]];
        let weights = vec![255; 4];
        let cost = CostLookup::new(source.clone(), target.clone(), weights.clone(), 2, 0);
        let oracle = brute_force_assignment(&cost);
        let got = solve(&cost, &source, &target, &weights, 2, 0);
        assert_valid_permutation(&got, 4);
        assert_eq!(total_cost(&cost, &got), total_cost(&cost, &oracle));
    }
}

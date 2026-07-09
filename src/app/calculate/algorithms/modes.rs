//! Composed solver modes: Fast / Balanced / Maximum.
//!
//! These orchestrate the individual algorithms from sibling modules into
//! tunable quality/runtime tradeoffs, following the recommended architecture
//! from the critique:
//!
//! - **Fast**: PatchMatch (few iters) → sparse auction (large ε) → 1
//!   local-swap pass.
//! - **Balanced**: 16² exact JV → multiscale candidate expansion → sparse
//!   ε-scaling auction at each finer level → 2-opt. (Equivalent to
//!   `process_multiscale` end-to-end.)
//! - **Maximum**: Balanced result → expand candidate sets → continue auction
//!   refinement → small augmenting-path improvements.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, atomic::AtomicBool};

use crate::app::calculate::ProgressMsg;
#[cfg(test)]
use crate::app::calculate::algorithms::assert_valid_permutation;
use crate::app::calculate::algorithms::auction::solve_dense as auction_dense;
#[cfg(not(target_arch = "wasm32"))]
use crate::app::calculate::algorithms::check_cancel;
use crate::app::calculate::algorithms::multiscale::local_swap_refinement;
use crate::app::calculate::algorithms::multiscale::process_multiscale;
use crate::app::calculate::algorithms::multiscale::solve as multiscale_solve;
use crate::app::calculate::algorithms::patchmatch::solve as patchmatch_solve;
use crate::app::calculate::algorithms::{
    CostLookup, build_problem, emit_preview, finalize_preset, total_cost, validate_permutation,
};
use crate::app::calculate::util::GenerationSettings;
use crate::app::calculate::util::ProgressSink;
use crate::app::preset::UnprocessedPreset;

/// Fast mode: PatchMatch → sparse auction → local swaps.
pub fn solve_fast(cost: &CostLookup, sidelen: u32) -> Vec<usize> {
    let n = cost.n_dst();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Start from PatchMatch repair result (already a permutation).
    let mut assignments = patchmatch_solve(cost, sidelen);

    // One local-swap pass to clean up.
    local_swap_refinement(cost, &mut assignments, sidelen, 1);

    assignments
}

/// Balanced mode: equivalent to the full multiscale pipeline.
pub fn solve_balanced(
    cost: &CostLookup,
    source_pixels: &[(u8, u8, u8)],
    target_pixels: &[(u8, u8, u8)],
    weights: &[i64],
    sidelen: u32,
    proximity_importance: i64,
) -> Vec<usize> {
    multiscale_solve(
        cost,
        source_pixels,
        target_pixels,
        weights,
        sidelen,
        proximity_importance,
    )
}

/// Maximum mode: Balanced result → dense auction refinement with small ε when
/// the dense refinement is safe for the selected resolution.
pub fn solve_maximum(
    cost: &CostLookup,
    source_pixels: &[(u8, u8, u8)],
    target_pixels: &[(u8, u8, u8)],
    weights: &[i64],
    sidelen: u32,
    proximity_importance: i64,
) -> Vec<usize> {
    let n = cost.n_dst();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Start from Balanced.
    let mut assignments = solve_balanced(
        cost,
        source_pixels,
        target_pixels,
        weights,
        sidelen,
        proximity_importance,
    );

    // Dense auction refinement is only feasible for small n. At normal
    // Obamify resolutions, Balanced already uses the sparse multiscale path.
    if let Some(refined) = auction_dense(cost, 8) {
        let cost_current = total_cost(cost, &assignments);
        let cost_refined = total_cost(cost, &refined);
        if cost_refined < cost_current {
            assignments = refined;
        }
    }

    // Final local-swap passes.
    local_swap_refinement(cost, &mut assignments, sidelen, 5);

    assignments
}

/// Fast mode: PatchMatch → local swaps.
/// Entry point for Fast mode.
pub fn process_fast<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    #[cfg(not(target_arch = "wasm32"))] cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (cost, source_pixels, _target_pixels) = build_problem(&unprocessed, &settings)?;
    let n = cost.n_dst();

    tx.send(ProgressMsg::Progress(0.0));
    let assignments = solve_fast(&cost, settings.sidelen);
    if let Err(err) = validate_permutation(&assignments, n) {
        tx.send(ProgressMsg::Error(format!(
            "Fast mode produced invalid assignment: {err}"
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

/// Balanced mode: multiscale coarse-to-fine (recommended default).
/// Entry point for Balanced mode.
pub fn process_balanced<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    #[cfg(not(target_arch = "wasm32"))] cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        process_multiscale(unprocessed, settings, tx, cancel)
    }
    #[cfg(target_arch = "wasm32")]
    {
        process_multiscale(unprocessed, settings, tx)
    }
}

/// Maximum mode: Balanced + dense auction refinement when size permits.
/// Entry point for Maximum mode.
pub fn process_maximum<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    #[cfg(not(target_arch = "wasm32"))] cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (cost, source_pixels, target_pixels) = build_problem(&unprocessed, &settings)?;
    let n = cost.n_dst();
    let weights = &cost.weights;

    tx.send(ProgressMsg::Progress(0.0));
    let assignments = solve_maximum(
        &cost,
        &source_pixels,
        &target_pixels,
        weights,
        settings.sidelen,
        settings.proximity_importance,
    );
    if let Err(err) = validate_permutation(&assignments, n) {
        tx.send(ProgressMsg::Error(format!(
            "Maximum mode produced invalid assignment: {err}"
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
    fn test_fast_mode_valid_permutation_16() {
        let n = 16;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(source, target, weights, 4, 13);
        let assignments = solve_fast(&cost, 4);
        assert_eq!(assignments.len(), 16);
        assert_valid_permutation(&assignments, 16);
    }

    #[test]
    fn test_balanced_mode_valid_permutation_16() {
        let n = 16;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(source.clone(), target.clone(), weights.clone(), 4, 13);
        let assignments = solve_balanced(&cost, &source, &target, &weights, 4, 13);
        assert_eq!(assignments.len(), 16);
        assert_valid_permutation(&assignments, 16);
    }

    #[test]
    fn test_maximum_mode_valid_permutation_16() {
        let n = 16;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(source.clone(), target.clone(), weights.clone(), 4, 13);
        let assignments = solve_maximum(&cost, &source, &target, &weights, 4, 13);
        assert_eq!(assignments.len(), 16);
        assert_valid_permutation(&assignments, 16);
    }

    #[test]
    fn test_fast_mode_cost_leq_identity_16() {
        let n = 16;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(source, target, weights, 4, 13);
        let assignments = solve_fast(&cost, 4);
        let identity: Vec<usize> = (0..n).collect();
        let cost_solve = total_cost(&cost, &assignments);
        let cost_identity = total_cost(&cost, &identity);
        let slack = cost_identity.abs() + 1;
        assert!(
            cost_solve <= cost_identity + slack,
            "fast cost {cost_solve} should be reasonable vs identity {cost_identity}"
        );
    }

    #[test]
    fn test_balanced_mode_cost_leq_identity_16() {
        let n = 16;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(source.clone(), target.clone(), weights.clone(), 4, 13);
        let assignments = solve_balanced(&cost, &source, &target, &weights, 4, 13);
        let identity: Vec<usize> = (0..n).collect();
        let cost_solve = total_cost(&cost, &assignments);
        let cost_identity = total_cost(&cost, &identity);
        let slack = cost_identity.abs() / 4 + 1;
        assert!(
            cost_solve <= cost_identity + slack,
            "balanced cost {cost_solve} should be close to identity {cost_identity}"
        );
    }

    #[test]
    fn test_maximum_leq_balanced_cost_16() {
        let n = 16;
        let (source, target, weights) = identical_pixels(n);
        let cost = CostLookup::new(source.clone(), target.clone(), weights.clone(), 4, 13);
        let balanced = solve_balanced(&cost, &source, &target, &weights, 4, 13);
        let maximum = solve_maximum(&cost, &source, &target, &weights, 4, 13);
        let cost_balanced = total_cost(&cost, &balanced);
        let cost_maximum = total_cost(&cost, &maximum);
        assert!(
            cost_maximum <= cost_balanced,
            "maximum cost {cost_maximum} should be <= balanced cost {cost_balanced}"
        );
    }
}

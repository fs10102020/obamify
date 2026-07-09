//! Jonker-Volgenant linear assignment solver.
//!
//! Pure-Rust exact dense assignment baseline. It uses the same proven
//! shortest-augmenting-path/Kuhn-Munkres core as the legacy `process_optimal`,
//! adapted to the positive-cost [`super::CostLookup`] convention by maximizing
//! `-cost`. This gives a robust exact baseline for small/coarse levels without
//! materializing an explicit dense matrix.
//!
//! This module also serves as the coarse-level exact baseline for
//! [`super::multiscale`].

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, atomic::AtomicBool};

use crate::app::calculate::ProgressMsg;
#[cfg(not(target_arch = "wasm32"))]
use crate::app::calculate::algorithms::check_cancel;
use crate::app::calculate::algorithms::{
    CostLookup, MAX_EXACT_N, build_problem, emit_preview, finalize_preset, validate_permutation,
};
#[cfg(test)]
use crate::app::calculate::algorithms::{assert_valid_permutation, total_cost};
use crate::app::calculate::util::GenerationSettings;
use crate::app::calculate::util::ProgressSink;
use crate::app::preset::UnprocessedPreset;

/// Solve the linear assignment problem via shortest augmenting paths (Kuhn-Munkres core).
/// Solve the dense linear assignment problem and return `assignments[dst] = src`.
///
/// Exposed publicly so the multiscale backend can call it at coarse levels and
/// tests can verify it directly.
pub fn solve(cost: &CostLookup) -> Vec<usize> {
    let n = cost.n_dst();
    debug_assert_eq!(n, cost.n_src(), "obamify costs are square");
    jonker_volgenant_dense(cost, n)
}

/// Core dense exact LAP solver. Returns `assignments[dst] = src` as a
/// permutation of `0..n`. The positive `CostLookup` cost is minimized by
/// maximizing its negation.
fn jonker_volgenant_dense(cost: &CostLookup, n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }

    #[inline(always)]
    fn weight_at(cost: &CostLookup, row: usize, col: usize) -> i64 {
        -cost.cost(row, col)
    }

    let mut xy: Vec<Option<usize>> = vec![None; n];
    let mut yx: Vec<Option<usize>> = vec![None; n];
    let mut lx: Vec<i64> = (0..n)
        .map(|row| (0..n).map(|col| weight_at(cost, row, col)).max().unwrap())
        .collect();
    let mut ly: Vec<i64> = vec![0; n];
    let mut s_list: Vec<usize> = Vec::with_capacity(n);
    let mut s_set: Vec<bool> = vec![false; n];
    let mut alternating = Vec::with_capacity(n);
    let mut slack = vec![0; n];
    let mut slackx = Vec::with_capacity(n);

    for root in 0..n {
        alternating.clear();
        alternating.resize(n, None);
        s_list.clear();
        s_set.fill(false);
        s_list.push(root);
        s_set[root] = true;
        for col in 0..n {
            slack[col] = lx[root] + ly[col] - weight_at(cost, root, col);
        }
        slackx.clear();
        slackx.resize(n, root);

        let mut col = Some(loop {
            let mut delta = i64::MAX;
            let mut row = 0;
            let mut col = 0;
            for yy in 0..n {
                if alternating[yy].is_none() && slack[yy] < delta {
                    delta = slack[yy];
                    row = slackx[yy];
                    col = yy;
                }
            }

            if delta > 0 {
                for &x in &s_list {
                    lx[x] -= delta;
                }
                for y in 0..n {
                    if alternating[y].is_some() {
                        ly[y] += delta;
                    } else {
                        slack[y] -= delta;
                    }
                }
            }

            alternating[col] = Some(row);
            if yx[col].is_none() {
                break col;
            }

            let row = yx[col].unwrap();
            s_list.push(row);
            s_set[row] = true;
            for y in 0..n {
                if alternating[y].is_none() {
                    let alternate_slack = lx[row] + ly[y] - weight_at(cost, row, y);
                    if slack[y] > alternate_slack {
                        slack[y] = alternate_slack;
                        slackx[y] = row;
                    }
                }
            }
        });

        while let Some(y) = col {
            let x = alternating[y].unwrap();
            let prec = xy[x];
            yx[y] = Some(x);
            xy[x] = Some(y);
            col = prec;
        }
    }

    xy.into_iter().map(|o| o.unwrap()).collect()
}

/// Exact Jonker-Volgenant (KM shortest-augmenting-path) linear assignment solver.
/// Entry point mirroring the legacy `process_optimal` signature.
pub fn process_jonker_volgenant<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    #[cfg(not(target_arch = "wasm32"))] cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (cost, source_pixels, _target_pixels) = build_problem(&unprocessed, &settings)?;
    let n = cost.n_dst();
    if n > MAX_EXACT_N {
        tx.send(ProgressMsg::Error(format!(
            "Jonker-Volgenant exact solve is limited to {MAX_EXACT_N} pixels; use Multiscale or Balanced for this resolution"
        )));
        return Ok(());
    }

    // For obamify's sidelen <= 256 the solve is at most 65_536² evaluations.
    // Emit periodic progress; the 100-row cadence matches the legacy
    // `process_optimal` so the GUI bar moves similarly. The solver itself
    // runs to completion in one call, so we emit before/after.
    tx.send(ProgressMsg::Progress(0.0));
    let assignments = solve(&cost);
    if let Err(err) = validate_permutation(&assignments, n) {
        tx.send(ProgressMsg::Error(format!(
            "Jonker-Volgenant produced invalid assignment: {err}"
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
        // source == target: with proximity > 0, the minimum-cost (maximum
        // heuristic) assignment is the identity (zero displacement).
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
    fn test_jv_identity_is_optimal_for_identical_pixels() {
        let cost = identical_pixel_lookup(16, 13);
        let assignments = solve(&cost);
        assert_valid_permutation(&assignments, cost.n_dst());
        let identity: Vec<usize> = (0..cost.n_dst()).collect();
        let cost_solve = total_cost(&cost, &assignments);
        let cost_identity = total_cost(&cost, &identity);
        assert_eq!(
            cost_solve, cost_identity,
            "JV should find the identity optimum, got cost {cost_solve} vs identity {cost_identity}"
        );
    }

    #[test]
    fn test_jv_returns_valid_permutation_4x4() {
        let cost = identical_pixel_lookup(4, 13);
        let assignments = solve(&cost);
        assert_eq!(assignments.len(), 4);
        assert_valid_permutation(&assignments, 4);
    }

    #[test]
    fn test_jv_empty_input() {
        let cost = CostLookup::new(vec![], vec![], vec![], 0, 0);
        let assignments = solve(&cost);
        assert!(assignments.is_empty());
    }

    #[test]
    fn test_jv_1x1() {
        let cost = CostLookup::new(vec![(1, 2, 3)], vec![(4, 5, 6)], vec![255], 1, 13);
        let assignments = solve(&cost);
        assert_eq!(assignments, vec![0]);
    }

    #[test]
    fn test_jv_9x9_valid_permutation() {
        let n = 9;
        let mut source = vec![(0u8, 0u8, 0u8); n];
        let mut target = vec![(0u8, 0u8, 0u8); n];
        for i in 0..n {
            source[i] = ((i * 7) as u8, (i * 3) as u8, i as u8);
            target[i] = ((i * 5) as u8, i as u8, (i * 2) as u8);
        }
        let cost = CostLookup::new(source, target, vec![255; n], 3, 13);
        let assignments = solve(&cost);
        assert_valid_permutation(&assignments, n);
    }

    #[test]
    fn test_jv_optimal_cost_leq_identity_cost() {
        // For any cost matrix, the JV solution cost should be <= the identity
        // assignment cost (since JV finds the optimum).
        let cost = identical_pixel_lookup(16, 13);
        let assignments = solve(&cost);
        let identity: Vec<usize> = (0..16).collect();
        let cost_solve = total_cost(&cost, &assignments);
        let cost_identity = total_cost(&cost, &identity);
        assert!(
            cost_solve <= cost_identity,
            "JV cost {cost_solve} should be <= identity cost {cost_identity}"
        );
    }

    #[test]
    fn test_jv_4x4_known_permutation() {
        // If target[i] = source[perm[i]], the optimum (with proximity=0) is
        // assignments[i] = perm[i], and its cost equals the identity cost on
        // the un-permuted problem.
        let n = 4;
        let source = vec![(10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120)];
        let perm = [2, 0, 3, 1];
        let target: Vec<(u8, u8, u8)> = perm.iter().map(|&p| source[p]).collect();
        let cost = CostLookup::new(source, target, vec![255; n], 2, 0);
        let assignments = solve(&cost);
        assert_valid_permutation(&assignments, n);
        let identity: Vec<usize> = (0..n).collect();
        let cost_solve = total_cost(&cost, &assignments);
        let cost_identity = total_cost(&cost, &identity);
        assert!(
            cost_solve <= cost_identity,
            "JV cost {cost_solve} should be <= identity cost {cost_identity}"
        );
    }

    #[test]
    fn test_jv_matches_bruteforce_oracle_on_permuted_colors() {
        let source = vec![(0, 0, 0), (40, 0, 0), (90, 0, 0), (150, 0, 0)];
        let target = vec![source[2], source[0], source[3], source[1]];
        let cost = CostLookup::new(source, target, vec![255; 4], 2, 0);
        let oracle = brute_force_assignment(&cost);
        let got = solve(&cost);
        assert_valid_permutation(&got, 4);
        assert_eq!(total_cost(&cost, &got), total_cost(&cost, &oracle));
    }
}

//! Sinkhorn optimal transport + rounding to a hard permutation.
//!
//! Sinkhorn iteration solves an entropy-regularized transport problem: it
//! repeatedly normalizes rows and columns of a transport kernel
//! `K[i,j] = exp(-cost(i,j) / ε)`. The result is a *soft* transport plan
//! (each source distributes its mass across multiple targets). We then extract
//! top-k hard candidates from that plan and run sparse auction to produce a
//! valid permutation.
//!
//! The implementation uses the log-domain stabilized form to avoid numerical
//! underflow at small ε. Costs are shifted to non-negative before forming the
//! kernel (Sinkhorn requires non-negative costs). This dense implementation is
//! deliberately capped to small images; full-size Obamify runs should use the
//! multiscale sparse auction path.

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

/// Number of Sinkhorn iterations per ε level.
const SINKHORN_ITERS: usize = 50;

/// Number of ε-scaling levels (ε shrinks each level for sharper plans).
const SINKHORN_LEVELS: usize = 4;

/// Initial ε as a fraction of the max cost spread.
const INITIAL_EPS_FRAC: f64 = 0.5;

/// Geometric ε decay factor.
const EPS_DECAY: f64 = 0.5;

/// Number of top-k candidates per target to extract from the transport plan
/// for the sparse auction repair.
const TOP_K_CANDIDATES: usize = 8;
pub const MAX_DENSE_SINKHORN_N: usize = 1024;

/// Run Sinkhorn iteration in the log domain and return the soft transport
/// plan as a row-major `n × n` matrix of non-negative weights (probabilities).
///
/// `row_mass` and `col_mass` are the target marginals (both uniform = 1/n for
/// the assignment problem). The returned plan approximates the optimal
/// entropy-regularized transport.
fn sinkhorn_log_domain(cost: &CostLookup, epsilon: f64, iters: usize, plan: &mut Vec<Vec<f64>>) {
    let n = cost.n_dst();
    if n == 0 {
        plan.clear();
        return;
    }

    // Shift costs to non-negative. Find the min cost (most negative) and shift.
    let mut min_cost: f64 = 0.0;
    for dst in 0..n {
        for src in 0..n {
            let c = cost.cost(dst, src) as f64;
            if c < min_cost {
                min_cost = c;
            }
        }
    }
    let shift = -min_cost; // make all costs >= 0

    // Log-domain Sinkhorn: maintain log-u (row potentials) and log-v (col
    // potentials). The transport plan is exp(K[i,j] + u[i] + v[j]) where
    // K[i,j] = -cost_shifted(i,j) / epsilon.
    //
    // Sinkhorn updates:
    //   u[i] = log(row_mass[i]) - logsumexp_j(K[i,j] + v[j])
    //   v[j] = log(col_mass[j]) - logsumexp_i(K[i,j] + u[i])
    let log_row_mass = (1.0 / n as f64).ln(); // uniform row mass
    let log_col_mass = (1.0 / n as f64).ln(); // uniform col mass

    let mut log_u: Vec<f64> = vec![0.0; n];
    let mut log_v: Vec<f64> = vec![0.0; n];

    let mut terms = Vec::with_capacity(n);
    for _iter in 0..iters {
        // Update log_u: u[i] = log_row_mass - logsumexp_j(K[i,j] + v[j])
        for (i, log_u_i) in log_u.iter_mut().enumerate() {
            terms.clear();
            for (j, &lv) in log_v.iter().enumerate() {
                let k_ij = -((cost.cost(i, j) as f64 + shift) / epsilon);
                terms.push(k_ij + lv);
            }
            *log_u_i = log_row_mass - logsumexp(&terms);
        }

        // Update log_v: v[j] = log_col_mass - logsumexp_i(K[i,j] + u[i])
        for (j, log_v_j) in log_v.iter_mut().enumerate() {
            terms.clear();
            for (i, &lu) in log_u.iter().enumerate() {
                let k_ij = -((cost.cost(i, j) as f64 + shift) / epsilon);
                terms.push(k_ij + lu);
            }
            *log_v_j = log_col_mass - logsumexp(&terms);
        }
    }

    // Reconstruct the transport plan into the provided buffer.
    if plan.len() != n {
        plan.resize(n, Vec::new());
    }
    for row in plan.iter_mut() {
        if row.len() != n {
            row.resize(n, 0.0);
        }
    }
    for (i, row) in plan.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let k_ij = -((cost.cost(i, j) as f64 + shift) / epsilon);
            *cell = (k_ij + log_u[i] + log_v[j]).exp().max(0.0);
        }
    }
}

/// Numerically stable log-sum-exp.
fn logsumexp(terms: &[f64]) -> f64 {
    if terms.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = terms.iter().map(|&t| (t - max).exp()).sum();
    max + sum.ln()
}

/// Round a soft transport plan to a hard permutation.
///
/// Extract the top-k candidate sources per target from the transport plan,
/// then run sparse auction over those candidates to produce a valid
/// permutation. The auction naturally resolves collisions and fills gaps.
fn round_to_permutation(plan: &[Vec<f64>], cost: &CostLookup) -> Vec<usize> {
    let n = plan.len();
    if n == 0 {
        return Vec::new();
    }

    // Build candidate lists from the transport plan: for each target, use the
    // top-k sources by transport weight.
    let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); n];
    for dst in 0..n {
        let mut indexed: Vec<(f64, usize)> = (0..n).map(|src| (plan[dst][src], src)).collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates[dst] = indexed
            .into_iter()
            .take(TOP_K_CANDIDATES)
            .map(|(_, src)| src)
            .collect();
    }

    // Sparse auction over the top-k candidates produces a valid permutation.
    sparse_auction(cost, &candidates, None, 3)
}

/// Run the full Sinkhorn + rounding pipeline and return the assignment.
pub fn solve(cost: &CostLookup) -> Option<Vec<usize>> {
    let n = cost.n_dst();
    if n == 0 {
        return Some(Vec::new());
    }
    if n == 1 {
        return Some(vec![0]);
    }
    if n > MAX_DENSE_SINKHORN_N {
        return None;
    }

    // Estimate the cost spread for the initial ε.
    let mut min_c: f64 = 0.0;
    let mut max_c: f64 = 0.0;
    for dst in 0..n {
        for src in 0..n {
            let c = cost.cost(dst, src) as f64;
            if c < min_c {
                min_c = c;
            }
            if c > max_c {
                max_c = c;
            }
        }
    }
    let spread = (max_c - min_c).abs().max(1.0);
    let mut epsilon = spread * INITIAL_EPS_FRAC;

    let mut plan = vec![vec![0.0f64; n]; n];
    for _level in 0..SINKHORN_LEVELS {
        sinkhorn_log_domain(cost, epsilon, SINKHORN_ITERS, &mut plan);
        epsilon *= EPS_DECAY;
        epsilon = epsilon.max(1.0);
    }

    Some(round_to_permutation(&plan, cost))
}

/// Entropy-regularized OT (log-domain Sinkhorn) + top-k rounding via sparse auction.
/// Entry point mirroring the legacy `process_optimal` signature.
pub fn process_sinkhorn<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    #[cfg(not(target_arch = "wasm32"))] cancel: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (cost, source_pixels, _target_pixels) = build_problem(&unprocessed, &settings)?;
    let n = cost.n_dst();

    tx.send(ProgressMsg::Progress(0.0));
    let Some(assignments) = solve(&cost) else {
        tx.send(ProgressMsg::Error(format!(
            "dense Sinkhorn is limited to {MAX_DENSE_SINKHORN_N} pixels; use multiscale sparse auction for this resolution"
        )));
        return Ok(());
    };
    if let Err(err) = validate_permutation(&assignments, n) {
        tx.send(ProgressMsg::Error(format!(
            "Sinkhorn produced invalid assignment: {err}"
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
    fn test_logsumexp_empty() {
        assert!(logsumexp(&[]).is_infinite());
    }

    #[test]
    fn test_logsumexp_single() {
        let val = logsumexp(&[1.0]);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_multiple() {
        // logsumexp([0, 0]) = log(2) ≈ 0.693
        let val = logsumexp(&[0.0, 0.0]);
        assert!((val - 2f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_sinkhorn_2x2_valid_permutation() {
        let cost = identical_pixel_lookup(2, 13);
        let assignments = solve(&cost).unwrap();
        assert_eq!(assignments.len(), 2);
        assert_valid_permutation(&assignments, 2);
    }

    #[test]
    fn test_sinkhorn_4x4_valid_permutation() {
        let cost = identical_pixel_lookup(4, 13);
        let assignments = solve(&cost).unwrap();
        assert_eq!(assignments.len(), 4);
        assert_valid_permutation(&assignments, 4);
    }

    #[test]
    fn test_sinkhorn_identity_near_optimal_4x4() {
        let cost = identical_pixel_lookup(4, 13);
        let assignments = solve(&cost).unwrap();
        let identity: Vec<usize> = (0..4).collect();
        let cost_solve = total_cost(&cost, &assignments);
        let cost_identity = total_cost(&cost, &identity);
        // Sinkhorn + rounding should be close to optimal.
        let slack = cost_identity.abs() / 2 + 1;
        assert!(
            cost_solve <= cost_identity + slack,
            "sinkhorn cost {cost_solve} should be close to identity cost {cost_identity} (slack {slack})"
        );
    }

    #[test]
    fn test_sinkhorn_empty_input() {
        let cost = CostLookup::new(vec![], vec![], vec![], 0, 0);
        let assignments = solve(&cost).unwrap();
        assert!(assignments.is_empty());
    }

    #[test]
    fn test_sinkhorn_1x1() {
        let cost = CostLookup::new(vec![(1, 2, 3)], vec![(4, 5, 6)], vec![255], 1, 13);
        let assignments = solve(&cost).unwrap();
        assert_eq!(assignments, vec![0]);
    }

    #[test]
    fn test_round_to_permutation_resolves_collisions() {
        // Construct a plan where two rows claim the same column.
        let n = 3;
        let plan = vec![
            vec![0.9, 0.05, 0.05], // dst 0 -> src 0
            vec![0.8, 0.1, 0.1],   // dst 1 -> src 0 (collision!)
            vec![0.1, 0.1, 0.8],   // dst 2 -> src 2
        ];
        let cost = identical_pixel_lookup(n, 13);
        let assignments = round_to_permutation(&plan, &cost);
        assert_valid_permutation(&assignments, n);
    }

    #[test]
    fn test_sinkhorn_refuses_unsafe_dense_size() {
        let n = MAX_DENSE_SINKHORN_N + 1;
        let cost = CostLookup::new(vec![(0, 0, 0); n], vec![(0, 0, 0); n], vec![255; n], 1, 0);
        assert!(solve(&cost).is_none());
    }
}

//! Assignment-algorithm backends for obamify.
//!
//! This module hosts the new algorithms introduced alongside the legacy
//! `process_optimal` (Hungarian) and `process_genetic` (random-swap) solvers
//! in the parent module, plus three composed "mode" entry points.
//!
//! - [`jonker_volgenant`] — exact linear-assignment baseline (shortest
//!   augmenting path / Kuhn-Munkres adapted to the positive-cost convention).
//! - [`auction`] — dense forward auction with ε-scaling + a reusable
//!   sparse-auction primitive used by the multiscale, Sinkhorn, and PatchMatch
//!   backends. Dense auction is capped at [`MAX_DENSE_AUCTION_N`] to avoid
//!   allocating infeasible graphs in the UI.
//! - [`multiscale`] — coarse-to-fine sparse auction (the headline / default
//!   algorithm). Coarsest level uses exact JV, finer levels build sparse
//!   candidate sets from the coarser prediction and run sparse ε-scaling
//!   auction. Progress and cancellation are checked per pyramid level.
//! - [`sinkhorn`] — entropy-regularized OT (log-domain Sinkhorn) + top-k
//!   rounding via sparse auction. Capped at [`MAX_DENSE_SINKHORN_N`] because
//!   the transport plan is a dense n×n matrix.
//! - [`patchmatch`] — propagation/random-search correspondence + sparse
//!   auction repair (heuristic; no exact guarantee).
//! - [`modes`] — `Fast` (PatchMatch → local swaps), `Balanced` (same as
//!   `multiscale`; this is the new UI default), and `Maximum` (Balanced +
//!   dense auction refinement when size permits).
//!
//! Every backend mirrors the legacy signature: a `process_<name>` free function
//! that takes an `UnprocessedPreset`, `GenerationSettings`, a `ProgressSink`,
//! and a shared [`SolverControl`] handle. This makes them drop-in arms for the
//! `process` dispatcher in the parent module for both native and wasm builds.

pub mod auction;
pub mod jonker_volgenant;
pub mod modes;
pub mod multiscale;
pub mod patchmatch;
pub mod sinkhorn;

use crate::app::calculate::util::{GenerationSettings, ProgressSink, SolverControl};
use crate::app::calculate::{ProgressMsg, UnprocessedPreset, heuristic, make_new_img};

/// Dense exact solvers become impractical beyond 64×64 in the interactive UI.
pub const MAX_EXACT_N: usize = 4096;

/// Re-export of the shared cost function so backends share one definition.
// Kept for backend experiments and tests that need to assert the shared cost convention.
#[expect(unused)]
pub(crate) use crate::app::calculate::heuristic as cost_heuristic;

/// Read-only view over the cost matrix materialized on demand.
///
/// The legacy `ImgDiffWeights` builds the full `pathfinding::Weights` matrix
/// lazily through `at(row, col)`, but it is tied to the Hungarian solver's
/// `Weights` trait. `CostLookup` exposes the same per-pair cost as a plain
/// function so the new backends can use it without going through that trait.
///
/// Conventions match the legacy code: `dst_idx` is the row (target) and
/// `src_idx` is the column (source). The returned value is `+heuristic`, so
/// **lower is better**. The legacy Hungarian adapter returns `-heuristic`
/// because it maximizes weights; these new backends minimize costs directly.
#[derive(Clone)]
pub struct CostLookup {
    pub source: Vec<(u8, u8, u8)>,
    pub target: Vec<(u8, u8, u8)>,
    pub weights: Vec<i64>,
    pub sidelen: usize,
    pub proximity_importance: i64,
    pub dst_positions: Vec<(u16, u16)>,
    pub src_positions: Vec<(u16, u16)>,
}

impl CostLookup {
    /// Build a `CostLookup` from the already-resolved pixel arrays returned by
    /// `util::get_images`. The caller is expected to have produced these via
    /// the same path as the legacy solvers so the semantics stay identical.
    pub fn new(
        source: Vec<(u8, u8, u8)>,
        target: Vec<(u8, u8, u8)>,
        weights: Vec<i64>,
        sidelen: u32,
        proximity_importance: i64,
    ) -> Self {
        let sidelen_usize = sidelen as usize;
        let dst_positions: Vec<(u16, u16)> = (0..target.len())
            .map(|i| ((i % sidelen_usize) as u16, (i / sidelen_usize) as u16))
            .collect();
        let src_positions: Vec<(u16, u16)> = (0..source.len())
            .map(|i| ((i % sidelen_usize) as u16, (i / sidelen_usize) as u16))
            .collect();
        Self {
            source,
            target,
            weights,
            sidelen: sidelen_usize,
            proximity_importance,
            dst_positions,
            src_positions,
        }
    }

    /// Number of target pixels (rows).
    #[inline]
    pub fn n_dst(&self) -> usize {
        self.target.len()
    }

    /// Number of source pixels (columns). Always equals `n_dst` for obamify.
    #[inline]
    pub fn n_src(&self) -> usize {
        self.source.len()
    }

    /// Cost of assigning source pixel `src_idx` to target pixel `dst_idx`.
    /// Lower is better. Returns `+heuristic(...)` (positive cost) so that
    /// minimization-based solvers (JV, auction, etc.) directly minimize the
    /// matching cost. The legacy `ImgDiffWeights::at` returns `-heuristic`
    /// instead because the Hungarian *maximizes* weight; the two systems are
    /// independent.
    #[inline(always)]
    pub fn cost(&self, dst_idx: usize, src_idx: usize) -> i64 {
        let (x1, y1) = self.dst_positions[dst_idx];
        let (x2, y2) = self.src_positions[src_idx];
        let (r1, g1, b1) = self.target[dst_idx];
        let (r2, g2, b2) = self.source[src_idx];
        let weight = self.weights[dst_idx];
        heuristic(
            (x1, y1),
            (x2, y2),
            (r1, g1, b1),
            (r2, g2, b2),
            weight,
            self.proximity_importance,
        )
    }

    /// Position of a target pixel in grid coordinates.
    // Useful in solver diagnostics even though current production paths use indices directly.
    #[cfg_attr(not(test), expect(dead_code))]
    #[inline]
    pub fn dst_pos(&self, dst_idx: usize) -> (u16, u16) {
        (
            (dst_idx % self.sidelen) as u16,
            (dst_idx / self.sidelen) as u16,
        )
    }

    /// Position of a source pixel in grid coordinates.
    // Useful in solver diagnostics even though current production paths use indices directly.
    #[cfg_attr(not(test), expect(dead_code))]
    #[inline]
    pub fn src_pos(&self, src_idx: usize) -> (u16, u16) {
        (
            (src_idx % self.sidelen) as u16,
            (src_idx / self.sidelen) as u16,
        )
    }
}

/// Total cost of a full assignment. Used by tests and refinement passes.
pub fn total_cost(cost: &CostLookup, assignments: &[usize]) -> i64 {
    let mut sum: i64 = 0;
    for (dst_idx, &src_idx) in assignments.iter().enumerate() {
        sum = sum.saturating_add(cost.cost(dst_idx, src_idx));
    }
    sum
}

/// Runtime + test assertion: `assignments` is a permutation of `0..n`.
#[cfg(test)]
pub fn assert_valid_permutation(assignments: &[usize], n: usize) {
    assert_eq!(
        assignments.len(),
        n,
        "assignment length {} != expected {}",
        assignments.len(),
        n
    );
    let mut seen = vec![false; n];
    for &a in assignments {
        assert!(a < n, "assignment value {a} out of range 0..{n}");
        assert!(
            !std::mem::replace(&mut seen[a], true),
            "duplicate assignment target source {a}"
        );
    }
}

/// Returns `Ok(())` if `assignments` is a valid permutation of `0..n`, else an error string.
pub fn validate_permutation(assignments: &[usize], n: usize) -> Result<(), String> {
    if assignments.len() != n {
        return Err(format!(
            "assignment length {} != expected {}",
            assignments.len(),
            n
        ));
    }
    let mut seen = vec![false; n];
    for &a in assignments {
        if a >= n {
            return Err(format!("assignment value {a} out of range 0..{n}"));
        }
        if std::mem::replace(&mut seen[a], true) {
            return Err(format!("duplicate assignment source {a}"));
        }
    }
    Ok(())
}

#[cfg(test)]
pub fn brute_force_assignment(cost: &CostLookup) -> Vec<usize> {
    let n = cost.n_dst();
    assert!(n <= 9, "brute-force oracle is only for tiny tests");

    fn search(
        cost: &CostLookup,
        dst: usize,
        used: &mut [bool],
        current: &mut Vec<usize>,
        best_cost: &mut i64,
        best: &mut Vec<usize>,
    ) {
        if dst == cost.n_dst() {
            let total = total_cost(cost, current);
            if total < *best_cost {
                *best_cost = total;
                *best = current.clone();
            }
            return;
        }

        for src in 0..cost.n_src() {
            if used[src] {
                continue;
            }
            used[src] = true;
            current.push(src);
            search(cost, dst + 1, used, current, best_cost, best);
            current.pop();
            used[src] = false;
        }
    }

    let mut used = vec![false; n];
    let mut current = Vec::with_capacity(n);
    let mut best = Vec::new();
    let mut best_cost = i64::MAX;
    search(cost, 0, &mut used, &mut current, &mut best_cost, &mut best);
    best
}

/// Emit a preview frame + progress fraction to the sink. Shared by all
/// backends so the GUI sees a consistent stream of `ProgressMsg`s.
pub fn emit_preview<S: ProgressSink>(
    tx: &mut S,
    source_pixels: &[(u8, u8, u8)],
    assignments: &[usize],
    sidelen: u32,
    progress: f32,
) {
    let data = make_new_img(source_pixels, assignments, sidelen);
    tx.send(ProgressMsg::UpdatePreview {
        width: sidelen,
        height: sidelen,
        data,
    });
    tx.send(ProgressMsg::Progress(progress));
}

/// Finalise a solved assignment into the `Preset` the GUI expects and send
/// `ProgressMsg::Done`. Mirrors the legacy `Done` payload shape exactly.
pub fn finalize_preset<S: ProgressSink>(
    tx: &mut S,
    unprocessed_name: String,
    source_pixels: &[(u8, u8, u8)],
    assignments: Vec<usize>,
    sidelen: u32,
) {
    tx.send(ProgressMsg::Done(crate::app::preset::Preset {
        inner: UnprocessedPreset {
            name: unprocessed_name,
            width: sidelen,
            height: sidelen,
            source_img: source_pixels
                .iter()
                .flat_map(|(r, g, b)| [*r, *g, *b])
                .collect(),
        },
        assignments,
    }));
}

/// Reaches a cooperative solver checkpoint and reports cancellation.
pub fn checkpoint<S: ProgressSink>(control: &SolverControl, tx: &mut S) -> bool {
    if !control.checkpoint() {
        tx.send(ProgressMsg::Cancelled);
        true
    } else {
        false
    }
}

/// Pixel triple returned by `get_images`.
type PixelTriple = (Vec<(u8, u8, u8)>, Vec<(u8, u8, u8)>, Vec<i64>);

/// Resolve the `(source_pixels, target_pixels, weights)` triple for a preset
/// and settings, exactly like the legacy solvers do. Centralised here so every
/// backend starts from the same pixel data.
pub fn resolve_pixels(
    unprocessed: &UnprocessedPreset,
    settings: &GenerationSettings,
) -> Result<PixelTriple, Box<dyn std::error::Error>> {
    let source_img = image::ImageBuffer::from_vec(
        unprocessed.width,
        unprocessed.height,
        unprocessed.source_img.clone(),
    )
    .ok_or_else(|| {
        format!(
            "invalid source image buffer: {}x{} requires {} RGB bytes, got {}",
            unprocessed.width,
            unprocessed.height,
            unprocessed.width as usize * unprocessed.height as usize * 3,
            unprocessed.source_img.len()
        )
    })?;
    crate::app::calculate::util::get_images(source_img, settings)
}

/// Build a `CostLookup` from a preset + settings in one call.
#[expect(clippy::type_complexity)]
pub fn build_problem(
    unprocessed: &UnprocessedPreset,
    settings: &GenerationSettings,
) -> Result<(CostLookup, Vec<(u8, u8, u8)>, Vec<(u8, u8, u8)>), Box<dyn std::error::Error>> {
    let (source_pixels, target_pixels, weights) = resolve_pixels(unprocessed, settings)?;
    let cost = CostLookup::new(
        source_pixels.clone(),
        target_pixels.clone(),
        weights,
        settings.sidelen,
        settings.proximity_importance,
    );
    Ok((cost, source_pixels, target_pixels))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trivial_lookup(n: usize) -> CostLookup {
        // Uniform grey pixels, uniform weights, zero proximity: every cost is 0.
        CostLookup::new(
            vec![(128, 128, 128); n],
            vec![(128, 128, 128); n],
            vec![255; n],
            (n as f32).sqrt() as u32,
            0,
        )
    }

    #[test]
    fn test_cost_lookup_dimensions() {
        let c = trivial_lookup(16);
        assert_eq!(c.n_dst(), 16);
        assert_eq!(c.n_src(), 16);
        assert_eq!(c.sidelen, 4);
    }

    #[test]
    fn test_cost_lookup_zero_for_identical_pixels() {
        let c = trivial_lookup(16);
        for i in 0..16 {
            assert_eq!(c.cost(i, i), 0);
        }
    }

    #[test]
    fn test_total_cost_adds_assignment_costs() {
        let mut c = trivial_lookup(4);
        c.proximity_importance = 1;
        let assignments = vec![1, 0, 3, 2];
        assert_eq!(total_cost(&c, &assignments), 4);
    }

    #[test]
    fn test_validate_permutation_accepts_identity() {
        assert!(validate_permutation(&[0, 1, 2, 3], 4).is_ok());
    }

    #[test]
    fn test_validate_permutation_rejects_duplicate() {
        assert!(validate_permutation(&[0, 1, 1, 3], 4).is_err());
    }

    #[test]
    fn test_validate_permutation_rejects_out_of_range() {
        assert!(validate_permutation(&[0, 1, 2, 4], 4).is_err());
    }

    #[test]
    fn test_validate_permutation_rejects_wrong_length() {
        assert!(validate_permutation(&[0, 1, 2], 4).is_err());
    }

    #[test]
    fn test_dst_src_pos_round_trip() {
        let c = trivial_lookup(16);
        for i in 0..16 {
            let (x, y) = c.dst_pos(i);
            assert_eq!(y as usize * c.sidelen + x as usize, i);
            let (x, y) = c.src_pos(i);
            assert_eq!(y as usize * c.sidelen + x as usize, i);
        }
    }
}

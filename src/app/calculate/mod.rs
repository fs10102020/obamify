pub mod algorithms;
pub mod drawing_process;
pub mod util;

#[cfg(target_arch = "wasm32")]
pub mod worker;

use crate::app::calculate::util::Algorithm;
use crate::app::calculate::util::SolverControl;
use crate::app::{
    calculate::util::{GenerationSettings, ProgressSink},
    preset::{Preset, UnprocessedPreset},
};
use serde::{Deserialize, Serialize};

/// Cost heuristic: weighted sum of color distance and squared spatial distance.
#[inline(always)]
pub(crate) fn heuristic(
    apos: (u16, u16),
    bpos: (u16, u16),
    a: (u8, u8, u8),
    b: (u8, u8, u8),
    color_weight: i64,
    spatial_weight: i64,
) -> i64 {
    let dx = apos.0 as i64 - bpos.0 as i64;
    let dy = apos.1 as i64 - bpos.1 as i64;
    let spatial = dx * dx + dy * dy;
    let dr = a.0 as i64 - b.0 as i64;
    let dg = a.1 as i64 - b.1 as i64;
    let db = a.2 as i64 - b.2 as i64;
    let color = dr * dr + dg * dg + db * db;
    let weighted_spatial = spatial * spatial_weight;
    color * color_weight + weighted_spatial * weighted_spatial
}

struct ImgDiffWeights<'a> {
    source: Vec<(u8, u8, u8)>,
    target: Vec<(u8, u8, u8)>,
    weights: Vec<i64>,
    sidelen: usize,
    settings: &'a GenerationSettings,
}

impl ImgDiffWeights<'_> {
    fn rows(&self) -> usize {
        self.target.len()
    }

    fn columns(&self) -> usize {
        self.source.len()
    }

    #[inline(always)]
    fn at(&self, row: usize, col: usize) -> i64 {
        let (x1, y1) = (row % self.sidelen, row / self.sidelen);
        let (x2, y2) = (col % self.sidelen, col / self.sidelen);
        let (r1, g1, b1) = self.target[row];
        let (r2, g2, b2) = self.source[col];
        let weight = self.weights[row];
        -heuristic(
            (x1 as u16, y1 as u16),
            (x2 as u16, y2 as u16),
            (r1, g1, b1),
            (r2, g2, b2),
            weight,
            self.settings.proximity_importance,
        )
    }
}

/// Progress messages sent from background computation to the GUI.
#[derive(Serialize, Deserialize)]
pub enum ProgressMsg {
    Progress(f32),
    UpdatePreview {
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
    UpdateAssignments(Vec<usize>),
    Done(Preset), // result directory
    Error(String),
    Cancelled,
}

/// Legacy Hungarian (Kuhn-Munkres) exact assignment solver. Inlined from `pathfinding::kuhn_munkres`.
pub fn process_optimal<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    control: SolverControl,
) -> Result<(), Box<dyn std::error::Error>> {
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
    // let start_time = std::time::Instant::now();
    let (source_pixels, target_pixels, weights) = util::get_images(source_img, &settings)?;
    if target_pixels.len() > algorithms::MAX_EXACT_N {
        tx.send(ProgressMsg::Error(format!(
            "Hungarian exact solve is limited to {} pixels; use Multiscale or Balanced for this resolution",
            algorithms::MAX_EXACT_N
        )));
        return Ok(());
    }

    let weights = ImgDiffWeights {
        source: source_pixels.clone(),
        target: target_pixels,
        weights,
        sidelen: settings.sidelen as usize,
        settings: &settings,
    };

    // pathfinding::kuhn_munkres, inlined to allow for progress bar and cancelling
    let (_total_diff, assignments) = {
        // We call x the rows and y the columns. (nx, ny) is the size of the matrix.
        let nx = weights.rows();
        let ny = weights.columns();
        assert!(
            nx <= ny,
            "number of rows must not be larger than number of columns"
        );
        // xy represents matching for x, yz matching for y
        let mut xy: Vec<Option<usize>> = vec![None; nx];
        let mut yx: Vec<Option<usize>> = vec![None; ny];
        // lx is the labelling for x nodes, ly the labelling for y nodes. We start
        // with an acceptable labelling with the maximum possible values for lx
        // and 0 for ly.
        let mut lx: Vec<i64> = (0..nx)
            .map(|row| (0..ny).map(|col| weights.at(row, col)).max().unwrap())
            .collect::<Vec<_>>();
        let mut ly: Vec<i64> = vec![0; ny];
        // s, augmenting, and slack will be reset every time they are reused. augmenting
        // contains Some(prev) when the corresponding node belongs to the augmenting path.
        let mut s_list: Vec<usize> = Vec::with_capacity(nx);
        let mut s_set: Vec<bool> = vec![false; nx];
        let mut alternating = Vec::with_capacity(ny);
        let mut slack = vec![0; ny];
        let mut slackx = Vec::with_capacity(ny);
        for root in 0..nx {
            alternating.clear();
            alternating.resize(ny, None);
            // Find y such that the path is augmented. This will be set when breaking for the
            // loop below. Above the loop is some code to initialize the search.
            let mut y = {
                s_list.clear();
                s_set.fill(false);
                s_list.push(root);
                s_set[root] = true;
                // Slack for a vertex y is, initially, the margin between the
                // sum of the labels of root and y, and the weight between root and y.
                // As we add x nodes to the alternating path, we update the slack to
                // represent the smallest margin between one of the x nodes and y.
                for y in 0..ny {
                    slack[y] = lx[root] + ly[y] - weights.at(root, y);
                }
                slackx.clear();
                slackx.resize(ny, root);
                Some(loop {
                    let mut delta = i64::MAX;
                    let mut x = 0;
                    let mut y = 0;
                    // Select one of the smallest slack delta and its edge (x, y)
                    // for y not in the alternating path already.
                    for yy in 0..ny {
                        if alternating[yy].is_none() && slack[yy] < delta {
                            delta = slack[yy];
                            x = slackx[yy];
                            y = yy;
                        }
                    }
                    // If some slack has been found, remove it from x nodes in the
                    // alternating path, and add it to y nodes in the alternating path.
                    // The slack of y nodes outside the alternating path will be reduced
                    // by this minimal slack as well.
                    if delta > 0 {
                        for &x in &s_list {
                            lx[x] -= delta;
                        }
                        for y in 0..ny {
                            if alternating[y].is_some() {
                                ly[y] += delta;
                            } else {
                                slack[y] -= delta;
                            }
                        }
                    }
                    // Add (x, y) to the alternating path.
                    alternating[y] = Some(x);
                    if yx[y].is_none() {
                        // We have found an augmenting path.
                        break y;
                    }
                    // This y node had a predecessor, add it to the set of x nodes
                    // in the augmenting path.
                    let x = yx[y].unwrap();
                    s_list.push(x);
                    s_set[x] = true;
                    // Update slack because of the added vertex in s might contain a
                    // greater slack than with previously inserted x nodes in the augmenting
                    // path.
                    for y in 0..ny {
                        if alternating[y].is_none() {
                            let alternate_slack = lx[x] + ly[y] - weights.at(x, y);
                            if slack[y] > alternate_slack {
                                slack[y] = alternate_slack;
                                slackx[y] = x;
                            }
                        }
                    }
                })
            };
            // Inverse edges along the augmenting path.
            while y.is_some() {
                let x = alternating[y.unwrap()].unwrap();
                let prec = xy[x];
                yx[y.unwrap()] = Some(x);
                xy[x] = y;
                y = prec;
            }
            if !control.checkpoint() {
                tx.send(ProgressMsg::Cancelled);
                return Ok(());
            }
            if root % 100 == 0 {
                // send progress
                tx.send(ProgressMsg::Progress(root as f32 / nx as f32));

                let data = make_new_img(
                    &source_pixels,
                    &xy.clone()
                        .into_iter()
                        .map(|a| a.unwrap_or(0))
                        .collect::<Vec<_>>(),
                    settings.sidelen,
                );

                tx.send(ProgressMsg::UpdatePreview {
                    width: settings.sidelen,
                    height: settings.sidelen,
                    data,
                });
            }
        }
        (
            lx.into_iter().sum::<i64>() + ly.into_iter().sum::<i64>(),
            xy.into_iter().map(Option::unwrap).collect::<Vec<_>>(),
        )
    };

    tx.send(ProgressMsg::Done(Preset {
        inner: UnprocessedPreset {
            name: unprocessed.name,
            width: settings.sidelen,
            height: settings.sidelen,
            source_img: source_pixels
                .into_iter()
                .flat_map(|(r, g, b)| [r, g, b])
                .collect(),
        },
        assignments,
    }));

    // println!(
    //     "finished in {:.2?} seconds",
    //     std::time::Instant::now().duration_since(start_time)
    // );
    Ok(())
}

/// Build a preview image by rearranging source pixels according to assignments.
pub(crate) fn make_new_img(
    source_pixels: &[(u8, u8, u8)],
    assignments: &[usize],
    sidelen: u32,
) -> Vec<u8> {
    let mut img = vec![0; (sidelen * sidelen * 3) as usize];
    for (target_idx, source_idx) in assignments.iter().enumerate() {
        let (r, g, b) = source_pixels[*source_idx];
        let base = target_idx * 3;
        img[base] = r;
        img[base + 1] = g;
        img[base + 2] = b;
    }
    img
}

#[derive(Clone, Copy)]
struct Pixel {
    src_x: u16,
    src_y: u16,
    rgb: (u8, u8, u8),
    h: i64, // current heuristic value
}

impl Pixel {
    fn new(src_x: u16, src_y: u16, rgb: (u8, u8, u8), h: i64) -> Self {
        Self {
            src_x,
            src_y,
            rgb,
            h,
        }
    }

    fn update_heuristic(&mut self, new_h: i64) {
        self.h = new_h;
    }

    #[inline(always)]
    fn calc_heuristic(
        &self,
        target_pos: (u16, u16),
        target_col: (u8, u8, u8),
        weight: i64,
        proximity_importance: i64,
    ) -> i64 {
        heuristic(
            (self.src_x, self.src_y),
            target_pos,
            self.rgb,
            target_col,
            weight,
            proximity_importance,
        )
    }
}

const SWAPS_PER_GENERATION_PER_PIXEL: usize = 128;

/// Random pair-swap annealing optimizer. Fast and approximate.
pub fn process_genetic<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    control: SolverControl,
) -> Result<(), Box<dyn std::error::Error>> {
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
    // let start_time = std::time::Instant::now();
    let (source_pixels, target_pixels, weights) = util::get_images(source_img, &settings)?;

    let mut pixels = source_pixels
        .iter()
        .enumerate()
        .map(|(i, &(r, g, b))| {
            let x = (i as u32 % settings.sidelen) as u16;
            let y = (i as u32 / settings.sidelen) as u16;
            let mut p = Pixel::new(x, y, (r, g, b), 0);
            let h = p.calc_heuristic(
                (x, y),
                target_pixels[i],
                weights[i],
                settings.proximity_importance,
            );
            p.update_heuristic(h);
            p
        })
        .collect::<Vec<_>>();

    let mut rng = frand::Rand::with_seed(12345);
    let swaps_per_generation = SWAPS_PER_GENERATION_PER_PIXEL * pixels.len();

    let mut max_dist = settings.sidelen;
    let mut generation = 0u32;
    loop {
        generation += 1;
        if generation > 5000 {
            tx.send(ProgressMsg::Error(
                "Genetic: exceeded maximum generations without converging".into(),
            ));
            return Ok(());
        }
        let mut swaps_made = 0;
        for swap_index in 0..swaps_per_generation {
            if swap_index % 4096 == 0 && !control.checkpoint() {
                tx.send(ProgressMsg::Cancelled);
                return Ok(());
            }
            let apos = rng.gen_range(0..pixels.len() as u32) as usize;
            let ax = apos as u16 % settings.sidelen as u16;
            let ay = apos as u16 / settings.sidelen as u16;
            let bx = (ax as i16 + rng.gen_range(-(max_dist as i16)..(max_dist as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let by = (ay as i16 + rng.gen_range(-(max_dist as i16)..(max_dist as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let bpos = by as usize * settings.sidelen as usize + bx as usize;

            let t_a = target_pixels[apos];
            let t_b = target_pixels[bpos];

            let a_on_b_h = pixels[apos].calc_heuristic(
                (bx, by),
                t_b,
                weights[bpos],
                settings.proximity_importance,
            );

            let b_on_a_h = pixels[bpos].calc_heuristic(
                (ax, ay),
                t_a,
                weights[apos],
                settings.proximity_importance,
            );

            let improvement_a = pixels[apos].h - b_on_a_h;
            let improvement_b = pixels[bpos].h - a_on_b_h;
            if improvement_a + improvement_b > 0 {
                // swap
                pixels.swap(apos, bpos);
                pixels[apos].update_heuristic(b_on_a_h);
                pixels[bpos].update_heuristic(a_on_b_h);
                swaps_made += 1;
            }
        }

        if !control.checkpoint() {
            tx.send(ProgressMsg::Cancelled);
            return Ok(());
        }

        let assignments = pixels
            .iter()
            .map(|p| p.src_y as usize * settings.sidelen as usize + p.src_x as usize)
            .collect::<Vec<_>>();
        //debug_print(format!("max_dist = {max_dist}, swaps made = {swaps_made}"));
        if max_dist < 4 && swaps_made < 10 {
            //let dir_name = util::save_result(target, base_name, source, assignments, img)?;
            tx.send(ProgressMsg::Done(Preset {
                inner: UnprocessedPreset {
                    name: unprocessed.name,
                    width: settings.sidelen,
                    height: settings.sidelen,
                    source_img: source_pixels
                        .iter()
                        .flat_map(|(r, g, b)| [*r, *g, *b])
                        .collect(),
                },
                assignments,
            }));
            return Ok(());
        }
        let data = make_new_img(&source_pixels, &assignments, settings.sidelen);
        tx.send(ProgressMsg::UpdatePreview {
            width: settings.sidelen,
            height: settings.sidelen,
            data,
        });
        tx.send(ProgressMsg::Progress(
            1.0 - max_dist as f32 / settings.sidelen as f32,
        ));

        max_dist = (max_dist as f32 * 0.99).max(2.0) as u32;
    }
}

/// Dispatch to the selected algorithm backend (native).
#[cfg(not(target_arch = "wasm32"))]
pub fn process<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    control: SolverControl,
) -> Result<(), Box<dyn std::error::Error>> {
    use algorithms::*;
    match settings.algorithm {
        Algorithm::Optimal => process_optimal(unprocessed, settings, tx, control),
        Algorithm::Genetic => process_genetic(unprocessed, settings, tx, control),
        Algorithm::JonkerVolgenant => {
            jonker_volgenant::process_jonker_volgenant(unprocessed, settings, tx, control)
        }
        Algorithm::Auction => auction::process_auction(unprocessed, settings, tx, control),
        Algorithm::Multiscale => multiscale::process_multiscale(unprocessed, settings, tx, control),
        Algorithm::Sinkhorn => sinkhorn::process_sinkhorn(unprocessed, settings, tx, control),
        Algorithm::PatchMatch => patchmatch::process_patchmatch(unprocessed, settings, tx, control),
        Algorithm::Fast => modes::process_fast(unprocessed, settings, tx, control),
        Algorithm::Balanced => modes::process_balanced(unprocessed, settings, tx, control),
        Algorithm::Maximum => modes::process_maximum(unprocessed, settings, tx, control),
    }
}

/// Dispatch to the selected algorithm backend (WASM).
#[cfg(target_arch = "wasm32")]
pub fn process<S: ProgressSink>(
    unprocessed: UnprocessedPreset,
    settings: GenerationSettings,
    tx: &mut S,
    control: SolverControl,
) -> Result<(), Box<dyn std::error::Error>> {
    use algorithms::*;
    match settings.algorithm {
        Algorithm::Optimal => process_optimal(unprocessed, settings, tx, control),
        Algorithm::Genetic => process_genetic(unprocessed, settings, tx, control),
        Algorithm::JonkerVolgenant => {
            jonker_volgenant::process_jonker_volgenant(unprocessed, settings, tx, control)
        }
        Algorithm::Auction => auction::process_auction(unprocessed, settings, tx, control),
        Algorithm::Multiscale => multiscale::process_multiscale(unprocessed, settings, tx, control),
        Algorithm::Sinkhorn => sinkhorn::process_sinkhorn(unprocessed, settings, tx, control),
        Algorithm::PatchMatch => patchmatch::process_patchmatch(unprocessed, settings, tx, control),
        Algorithm::Fast => modes::process_fast(unprocessed, settings, tx, control),
        Algorithm::Balanced => modes::process_balanced(unprocessed, settings, tx, control),
        Algorithm::Maximum => modes::process_maximum(unprocessed, settings, tx, control),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::preset::UnprocessedPreset;
    use util::GenerationSettings;
    use uuid::Uuid;

    #[cfg(not(target_arch = "wasm32"))]
    use crate::app::calculate::util::SolverControl;

    // --- Tests 1-3: heuristic function ---

    #[test]
    fn test_heuristic_same_position_same_color_is_zero() {
        let h = heuristic((5, 5), (5, 5), (100, 100, 100), (100, 100, 100), 255, 13);
        assert_eq!(h, 0);
    }

    #[test]
    fn test_heuristic_color_dominates_with_high_color_weight() {
        let h_low_weight = heuristic((0, 0), (1, 1), (0, 0, 0), (255, 0, 0), 1, 0);
        let h_high_weight = heuristic((0, 0), (1, 1), (0, 0, 0), (255, 0, 0), 255, 0);
        // With higher color weight, the heuristic should be larger.
        assert!(
            h_high_weight > h_low_weight,
            "high color weight {h_high_weight} should be > low {h_low_weight}"
        );
    }

    #[test]
    fn test_heuristic_spatial_is_quadratic_in_spatial_weight() {
        // heuristic spatial term = (spatial * spatial_weight)^2 = spatial^2 * weight^2
        // So doubling spatial_weight should quadruple the spatial contribution.
        let h1 = heuristic((0, 0), (10, 0), (0, 0, 0), (0, 0, 0), 0, 1);
        let h2 = heuristic((0, 0), (10, 0), (0, 0, 0), (0, 0, 0), 0, 2);
        // spatial = 100, h1 = (100*1)^2 = 10000, h2 = (100*2)^2 = 40000
        assert_eq!(h1, 10_000);
        assert_eq!(h2, 40_000);
    }

    // --- Helper: build a minimal UnprocessedPreset + GenerationSettings ---
    fn make_test_preset(sidelen: u32, n: usize) -> (UnprocessedPreset, GenerationSettings) {
        let source_img: Vec<u8> = (0..n)
            .flat_map(|i| {
                let v = (i % 256) as u8;
                [v, v, v]
            })
            .collect();
        let unprocessed = UnprocessedPreset {
            name: "test".to_string(),
            width: sidelen,
            height: sidelen,
            source_img,
        };
        let mut settings = GenerationSettings::default(Uuid::new_v4(), "test".to_string());
        settings.sidelen = sidelen;
        settings.proximity_importance = 0; // avoid spatial dominance in small tests
        settings.algorithm = Algorithm::Optimal;
        (unprocessed, settings)
    }

    fn msg_type(msg: &ProgressMsg) -> &'static str {
        match msg {
            ProgressMsg::Progress(_) => "progress",
            ProgressMsg::UpdatePreview { .. } => "update_preview",
            ProgressMsg::UpdateAssignments(_) => "update_assignments",
            ProgressMsg::Done(_) => "done",
            ProgressMsg::Error(_) => "error",
            ProgressMsg::Cancelled => "cancelled",
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_process_dispatches_every_algorithm_variant() {
        for algorithm in Algorithm::ALL {
            let (unprocessed, mut settings) = make_test_preset(4, 16);
            settings.algorithm = algorithm;
            settings.proximity_importance = 1;
            let control = SolverControl::default();
            let mut msgs: Vec<ProgressMsg> = Vec::new();
            let mut sink = |msg: ProgressMsg| {
                msgs.push(msg);
            };

            let result = process(unprocessed, settings, &mut sink, control);
            assert!(result.is_ok(), "{} returned an error", algorithm.label());
            assert!(
                msgs.iter().any(|m| matches!(m, ProgressMsg::Done(_))),
                "{} did not emit Done; got {:?}",
                algorithm.label(),
                msgs.iter().map(msg_type).collect::<Vec<_>>()
            );
        }
    }

    // --- Test 4: process_optimal 4x4 emits Done preset ---
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_process_optimal_4x4_emits_done_preset() {
        let (unprocessed, mut settings) = make_test_preset(4, 16);
        settings.algorithm = Algorithm::Optimal;
        let control = SolverControl::default();
        let mut msgs: Vec<ProgressMsg> = Vec::new();
        let mut sink = |msg: ProgressMsg| {
            msgs.push(msg);
        };
        let result = process_optimal(unprocessed, settings, &mut sink, control);
        assert!(result.is_ok(), "process_optimal should succeed");
        let has_done = msgs.iter().any(|m| matches!(m, ProgressMsg::Done(_)));
        assert!(
            has_done,
            "should emit Done message, got: {:?}",
            msgs.iter().map(msg_type).collect::<Vec<_>>()
        );
    }

    // --- Test 5: process_genetic 2x2 converges to Done ---
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_process_genetic_2x2_converges_to_done() {
        let (unprocessed, mut settings) = make_test_preset(2, 4);
        settings.algorithm = Algorithm::Genetic;
        settings.proximity_importance = 1;
        let control = SolverControl::default();
        let mut msgs: Vec<ProgressMsg> = Vec::new();
        let mut sink = |msg: ProgressMsg| {
            msgs.push(msg);
        };
        let result = process_genetic(unprocessed, settings, &mut sink, control);
        assert!(result.is_ok());
        let has_done = msgs.iter().any(|m| matches!(m, ProgressMsg::Done(_)));
        assert!(has_done, "genetic should converge and emit Done");
    }

    // --- Test 6: process_genetic emits progress messages ---
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_process_genetic_emits_progress_messages() {
        let (unprocessed, mut settings) = make_test_preset(4, 16);
        settings.algorithm = Algorithm::Genetic;
        settings.proximity_importance = 1;
        let control = SolverControl::default();
        let mut msgs: Vec<ProgressMsg> = Vec::new();
        let mut sink = |msg: ProgressMsg| {
            msgs.push(msg);
        };
        let result = process_genetic(unprocessed, settings, &mut sink, control);
        assert!(result.is_ok());
        // Should emit at least one Progress or UpdatePreview message.
        let has_progress = msgs.iter().any(|m| {
            matches!(
                m,
                ProgressMsg::Progress(_) | ProgressMsg::UpdatePreview { .. }
            )
        });
        assert!(
            has_progress,
            "genetic should emit progress messages, got: {:?}",
            msgs.iter().map(msg_type).collect::<Vec<_>>()
        );
    }

    // --- Test 7: make_new_img rearranges pixels ---
    #[test]
    fn test_make_new_img_rearranges_pixels() {
        let source_pixels = vec![(10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120)];
        // assignments[dst] = src: dst 0 -> src 1, dst 1 -> src 0, etc.
        let assignments = vec![1, 0, 3, 2];
        let img = make_new_img(&source_pixels, &assignments, 2);
        // dst 0 should get source_pixels[1] = (40, 50, 60)
        assert_eq!(img[0], 40);
        assert_eq!(img[1], 50);
        assert_eq!(img[2], 60);
        // dst 1 should get source_pixels[0] = (10, 20, 30)
        assert_eq!(img[3], 10);
        assert_eq!(img[4], 20);
        assert_eq!(img[5], 30);
    }

    // --- Test 8: make_new_img output size matches sidelen ---
    #[test]
    fn test_make_new_img_output_size_matches_sidelen() {
        let source_pixels = vec![(0, 0, 0); 16];
        let assignments = vec![0; 16];
        let img = make_new_img(&source_pixels, &assignments, 4);
        assert_eq!(img.len(), 4 * 4 * 3);
    }
}

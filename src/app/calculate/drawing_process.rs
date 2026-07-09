use crate::app::SeedColor;
use crate::app::calculate;
#[cfg(not(target_arch = "wasm32"))]
use crate::app::calculate::SWAPS_PER_GENERATION_PER_PIXEL;
use crate::app::preset::UnprocessedPreset;

use std::error::Error;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::AtomicU32;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;

#[cfg(not(target_arch = "wasm32"))]
use super::ProgressMsg;

use super::GenerationSettings;

#[derive(Clone, Copy)]
pub struct PixelData {
    pub stroke_id: u32,
}
impl PixelData {
    pub(crate) fn init_canvas() -> Vec<PixelData> {
        vec![PixelData { stroke_id: 0 }; DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE]
    }
}

pub const DRAWING_CANVAS_SIZE: usize = 128;

use super::heuristic;

#[derive(Clone, Copy)]
pub(crate) struct DrawingPixel {
    pub(crate) src_x: u16,
    pub(crate) src_y: u16,
    pub(crate) h: i64, // current heuristic value
}

impl DrawingPixel {
    pub(crate) fn new(src_x: u16, src_y: u16, h: i64) -> Self {
        Self { src_x, src_y, h }
    }

    pub(crate) fn update_heuristic(&mut self, new_h: i64) {
        self.h = new_h;
    }

    #[inline(always)]
    pub(crate) fn calc_drawing_heuristic(
        &self,
        target_pos: (u16, u16),
        target_col: (u8, u8, u8),
        weight: i64,
        colors: &[SeedColor],
        proximity_importance: i64,
    ) -> i64 {
        heuristic(
            (self.src_x, self.src_y),
            target_pos,
            {
                let rgba =
                    colors[self.src_y as usize * DRAWING_CANVAS_SIZE + self.src_x as usize].rgba;
                (
                    (rgba[0] * 256.0) as u8,
                    (rgba[1] * 256.0) as u8,
                    (rgba[2] * 256.0) as u8,
                )
            },
            target_col,
            weight,
            proximity_importance,
        )
    }
}

pub(crate) const STROKE_REWARD: i64 = -10000000000;

pub(crate) struct DrawingOptimizer {
    settings: GenerationSettings,
    target_pixels: Vec<(u8, u8, u8)>,
    weights: Vec<i64>,
    pixels: Vec<DrawingPixel>,
    rng: frand::Rand,
}

impl DrawingOptimizer {
    pub(crate) fn new(
        source: UnprocessedPreset,
        settings: GenerationSettings,
        colors: &[SeedColor],
    ) -> Result<Self, Box<dyn Error>> {
        let source_img =
            image::ImageBuffer::from_raw(source.width, source.height, source.source_img.clone())
                .ok_or_else(|| {
                    format!(
                        "invalid canvas source buffer: {}x{} RGB image got {} bytes",
                        source.width,
                        source.height,
                        source.source_img.len()
                    )
                })?;
        let (source_pixels, target_pixels, weights) =
            calculate::util::get_images(source_img, &settings)?;

        let pixels = source_pixels
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let x = (i as u32 % settings.sidelen) as u16;
                let y = (i as u32 / settings.sidelen) as u16;
                let mut p = DrawingPixel::new(x, y, 0);
                let h = p.calc_drawing_heuristic(
                    (x, y),
                    target_pixels[i],
                    weights[i],
                    colors,
                    settings.proximity_importance,
                ) + STROKE_REWARD;
                p.update_heuristic(h);
                p
            })
            .collect::<Vec<_>>();

        Ok(Self {
            settings,
            target_pixels,
            weights,
            pixels,
            rng: frand::Rand::with_seed(12345),
        })
    }

    pub(crate) fn step(
        &mut self,
        colors: &[SeedColor],
        pixel_data: &[PixelData],
        swap_budget: usize,
    ) -> Option<Vec<usize>> {
        let mut swaps_made = 0;
        let max_search = (DRAWING_CANVAS_SIZE / 4) as i16;

        for _ in 0..swap_budget {
            let apos = self.rng.gen_range(0..self.pixels.len() as u64) as usize;
            let ax = apos as u16 % self.settings.sidelen as u16;
            let ay = apos as u16 / self.settings.sidelen as u16;

            let bx = (ax as i16 + self.rng.gen_range(-max_search..(max_search + 1)))
                .clamp(0, self.settings.sidelen as i16 - 1) as u16;
            let by = (ay as i16 + self.rng.gen_range(-max_search..(max_search + 1)))
                .clamp(0, self.settings.sidelen as i16 - 1) as u16;
            let bpos = by as usize * self.settings.sidelen as usize + bx as usize;

            let t_a = self.target_pixels[apos];
            let t_b = self.target_pixels[bpos];

            let a_on_b_h = self.pixels[apos].calc_drawing_heuristic(
                (bx, by),
                t_b,
                self.weights[bpos],
                colors,
                self.settings.proximity_importance,
            ) + stroke_reward(bpos, apos, pixel_data, &self.pixels);

            let b_on_a_h = self.pixels[bpos].calc_drawing_heuristic(
                (ax, ay),
                t_a,
                self.weights[apos],
                colors,
                self.settings.proximity_importance,
            ) + stroke_reward(apos, bpos, pixel_data, &self.pixels);

            let improvement_a = self.pixels[apos].h - b_on_a_h;
            let improvement_b = self.pixels[bpos].h - a_on_b_h;
            if improvement_a + improvement_b > 0 {
                self.pixels.swap(apos, bpos);
                self.pixels[apos].update_heuristic(b_on_a_h);
                self.pixels[bpos].update_heuristic(a_on_b_h);
                swaps_made += 1;
            }
        }

        (swaps_made > 0).then(|| self.assignments())
    }

    fn assignments(&self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.pixels.len());
        result.extend(
            self.pixels
                .iter()
                .map(|p| p.src_y as usize * self.settings.sidelen as usize + p.src_x as usize),
        );
        result
    }
}

pub(crate) fn stroke_reward(
    newpos: usize,
    oldpos: usize,
    pixel_data: &[PixelData],
    pixels: &[DrawingPixel],
) -> i64 {
    let x = (newpos % DRAWING_CANVAS_SIZE) as u16;
    let y = (newpos / DRAWING_CANVAS_SIZE) as u16;
    // look at 8-connected neighbors
    // if any has the same stroke_id, return true
    let data = pixel_data
        [pixels[oldpos].src_x as usize + pixels[oldpos].src_y as usize * DRAWING_CANVAS_SIZE];
    let stroke_id = data.stroke_id;

    for (dx, dy) in [
        //(-1, -1),
        (0, -1),
        //(1, -1),
        (-1, 0),
        (1, 0),
        //(-1, 1),
        (0, 1),
        //(1, 1),
    ] {
        let nx = x as i16 + dx;
        let ny = y as i16 + dy;
        if nx < 0 || nx >= DRAWING_CANVAS_SIZE as i16 || ny < 0 || ny >= DRAWING_CANVAS_SIZE as i16
        {
            continue;
        }
        let npos = ny as usize * DRAWING_CANVAS_SIZE + nx as usize;
        if pixel_data
            [pixels[npos].src_x as usize + pixels[npos].src_y as usize * DRAWING_CANVAS_SIZE]
            .stroke_id
            == stroke_id
        {
            return STROKE_REWARD;
        }
    }
    0
}

// Thread entry point takes the state it needs explicitly so ownership across the spawned thread is clear.
#[cfg(not(target_arch = "wasm32"))]
pub fn drawing_process_genetic(
    source: UnprocessedPreset,
    settings: GenerationSettings,
    tx: mpsc::SyncSender<ProgressMsg>,
    colors: Arc<std::sync::RwLock<Vec<SeedColor>>>,
    pixel_data: Arc<std::sync::RwLock<Vec<PixelData>>>,
    my_id: u32,
    current_id: Arc<AtomicU32>,
) -> Result<(), Box<dyn Error>> {
    let read_colors = colors.read().unwrap_or_else(|e| e.into_inner()).clone();
    let mut optimizer = DrawingOptimizer::new(source, settings, &read_colors)?;
    let swaps_per_generation = SWAPS_PER_GENERATION_PER_PIXEL * read_colors.len();

    let mut colors_buf: Vec<SeedColor> = read_colors;
    let mut pixel_data_buf: Vec<PixelData> =
        pixel_data.read().unwrap_or_else(|e| e.into_inner()).clone();

    loop {
        {
            let r = colors.read().unwrap_or_else(|e| e.into_inner());
            colors_buf.clear();
            colors_buf.extend_from_slice(&r);
        }
        {
            let r = pixel_data.read().unwrap_or_else(|e| e.into_inner());
            pixel_data_buf.clear();
            pixel_data_buf.extend_from_slice(&r);
        }

        if let Some(assignments) =
            optimizer.step(&colors_buf, &pixel_data_buf, swaps_per_generation)
        {
            tx.send(ProgressMsg::UpdateAssignments(assignments))?;
        }
        if my_id != current_id.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = tx.send(ProgressMsg::Cancelled);
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_colors(n: usize) -> Vec<SeedColor> {
        (0..n)
            .map(|i| {
                let v = (i % 256) as f32 / 255.0;
                SeedColor {
                    rgba: [v, v, v, 1.0],
                }
            })
            .collect()
    }

    #[test]
    fn test_pixel_data_init_canvas_correct_count() {
        let canvas = PixelData::init_canvas();
        assert_eq!(canvas.len(), DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE);
        for pd in &canvas {
            assert_eq!(pd.stroke_id, 0);
        }
    }

    #[test]
    fn test_drawing_pixel_new_stores_fields() {
        let p = DrawingPixel::new(10, 20, 999);
        assert_eq!(p.src_x, 10);
        assert_eq!(p.src_y, 20);
        assert_eq!(p.h, 999);
    }

    #[test]
    fn test_drawing_pixel_update_heuristic() {
        let mut p = DrawingPixel::new(0, 0, 100);
        p.update_heuristic(200);
        assert_eq!(p.h, 200);
    }

    #[test]
    fn test_drawing_pixel_calc_drawing_heuristic_zero_for_matching() {
        let colors = make_colors(DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE);
        let p = DrawingPixel::new(5, 5, 0);
        // When source color == target color and position matches, heuristic should be 0
        let target_col = (colors[5 * DRAWING_CANVAS_SIZE + 5].rgba[0] * 256.0) as u8;
        let h = p.calc_drawing_heuristic(
            (5, 5),
            (target_col, target_col, target_col),
            255,
            &colors,
            0,
        );
        assert_eq!(
            h, 0,
            "heuristic should be 0 for matching color and position"
        );
    }

    #[test]
    fn test_stroke_reward_returns_reward_with_matching_neighbors() {
        let n = DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE;
        let pixel_data: Vec<PixelData> = vec![PixelData { stroke_id: 999 }; n];
        let pixels: Vec<DrawingPixel> = (0..n)
            .map(|i| {
                let x = (i % DRAWING_CANVAS_SIZE) as u16;
                let y = (i / DRAWING_CANVAS_SIZE) as u16;
                DrawingPixel::new(x, y, 0)
            })
            .collect();
        let reward = stroke_reward(DRAWING_CANVAS_SIZE / 2, 0, &pixel_data, &pixels);
        assert_eq!(reward, STROKE_REWARD);
    }

    #[test]
    fn test_stroke_reward_returns_zero_with_different_stroke_ids() {
        let n = DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE;
        let mut pixel_data: Vec<PixelData> = vec![PixelData { stroke_id: 0 }; n];
        // Give the oldpos pixel a unique stroke_id
        pixel_data[0] = PixelData { stroke_id: 42 };
        let pixels: Vec<DrawingPixel> = (0..n)
            .map(|i| {
                let x = (i % DRAWING_CANVAS_SIZE) as u16;
                let y = (i / DRAWING_CANVAS_SIZE) as u16;
                DrawingPixel::new(x, y, 0)
            })
            .collect();
        // oldpos=0 → src_x=0, src_y=0 → pixel_data[0].stroke_id = 42
        // Check neighbors of newpos — they all have stroke_id 0, not 42 → return 0
        let reward = stroke_reward(
            DRAWING_CANVAS_SIZE + 1, // position (1,1)
            0,                       // oldpos 0 → stroke_id 42
            &pixel_data,
            &pixels,
        );
        assert_eq!(
            reward, 0,
            "should return 0 when neighbors have different stroke_id"
        );
    }

    #[test]
    fn test_drawing_optimizer_new_succeeds() {
        let sidelen = DRAWING_CANVAS_SIZE as u32;
        let source_img: Vec<u8> = (0..(sidelen * sidelen * 3) as usize)
            .map(|i| (i % 256) as u8)
            .collect();
        let source = UnprocessedPreset {
            name: "test".to_string(),
            width: sidelen,
            height: sidelen,
            source_img,
        };
        let settings = GenerationSettings::default(uuid::Uuid::new_v4(), "test".to_string());
        let colors = make_colors(DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE);
        let result = DrawingOptimizer::new(source, settings, &colors);
        assert!(
            result.is_ok(),
            "DrawingOptimizer::new should succeed with valid input"
        );
    }

    #[test]
    fn test_drawing_optimizer_new_rejects_invalid_buffer() {
        let source = UnprocessedPreset {
            name: "test".to_string(),
            width: 128,
            height: 128,
            source_img: vec![0; 10], // way too small
        };
        let settings = GenerationSettings::default(uuid::Uuid::new_v4(), "test".to_string());
        let colors = make_colors(DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE);
        let result = DrawingOptimizer::new(source, settings, &colors);
        assert!(result.is_err(), "should reject invalid source buffer");
    }

    #[test]
    fn test_drawing_optimizer_step_returns_assignments_on_swaps() {
        let sidelen = DRAWING_CANVAS_SIZE as u32;
        // Create a source where pixels are clearly mismatched with target
        let source_img: Vec<u8> = (0..(sidelen * sidelen * 3) as usize)
            .map(|i| ((i * 7) % 256) as u8)
            .collect();
        let source = UnprocessedPreset {
            name: "test".to_string(),
            width: sidelen,
            height: sidelen,
            source_img,
        };
        let mut settings = GenerationSettings::default(uuid::Uuid::new_v4(), "test".to_string());
        settings.proximity_importance = 1;
        let colors = make_colors(DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE);
        let mut optimizer = DrawingOptimizer::new(source, settings, &colors).unwrap();
        let pixel_data = PixelData::init_canvas();
        // Run several steps — at least one should produce assignments
        let mut got_assignments = false;
        for _ in 0..10 {
            if optimizer.step(&colors, &pixel_data, 4096).is_some() {
                got_assignments = true;
                break;
            }
        }
        assert!(
            got_assignments,
            "optimizer step should eventually produce assignments"
        );
    }

    #[test]
    fn test_drawing_optimizer_assignments_is_valid_permutation() {
        let sidelen = DRAWING_CANVAS_SIZE as u32;
        let source_img: Vec<u8> = vec![128; (sidelen * sidelen * 3) as usize];
        let source = UnprocessedPreset {
            name: "test".to_string(),
            width: sidelen,
            height: sidelen,
            source_img,
        };
        let settings = GenerationSettings::default(uuid::Uuid::new_v4(), "test".to_string());
        let colors = make_colors(DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE);
        let mut optimizer = DrawingOptimizer::new(source, settings, &colors).unwrap();
        let pixel_data = PixelData::init_canvas();
        // Run steps until we get assignments
        for _ in 0..20 {
            if let Some(assignments) = optimizer.step(&colors, &pixel_data, 8192) {
                let n = DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE;
                // Verify it's a valid permutation
                assert_eq!(assignments.len(), n);
                let mut seen = vec![false; n];
                for &a in &assignments {
                    assert!(a < n, "assignment out of range: {a}");
                    assert!(!std::mem::replace(&mut seen[a], true), "duplicate: {a}");
                }
                return;
            }
        }
        panic!("optimizer never produced assignments in 20 steps");
    }
}

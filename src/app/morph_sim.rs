use std::mem;

use image::ImageBuffer;

use crate::app::preset::UnprocessedPreset;
use crate::app::{SeedColor, SeedPos, preset::Preset};

// const DST_FORCE: f32 = 0.2;
pub fn init_image(sidelen: u32, source: Preset) -> (u32, Vec<SeedPos>, Vec<SeedColor>, Sim) {
    let imgpath = image::ImageBuffer::from_vec(
        source.inner.width,
        source.inner.height,
        source.inner.source_img,
    )
    .expect("invalid preset image buffer dimensions");
    let assignments = source.assignments;

    let (seeds, colors, seeds_n) = init_colors(sidelen, imgpath);
    let mut sim = Sim::new(source.inner.name);
    sim.cells = vec![CellBody::new(0.0, 0.0, 0.0, 0.0, 0.0); seeds_n];

    sim.set_assignments(assignments, sidelen);
    for cell in &mut sim.cells {
        cell.dst_force = 0.13;
    }
    (seeds_n as u32, seeds, colors, sim)
}

pub fn init_canvas(
    sidelen: u32,
    source: UnprocessedPreset,
) -> (u32, Vec<SeedPos>, Vec<SeedColor>, Sim) {
    use crate::app::calculate::drawing_process::DRAWING_CANVAS_SIZE;
    let imgpath = image::ImageBuffer::from_vec(source.width, source.height, source.source_img)
        .expect("invalid canvas image buffer dimensions");
    let assignments = (0..(DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE)).collect::<Vec<usize>>();

    let (seeds, colors, seeds_n) = init_colors(sidelen, imgpath);
    let mut sim = Sim::new(source.name);
    sim.cells = vec![CellBody::new(0.0, 0.0, 0.0, 0.0, 0.0); seeds_n];

    sim.set_assignments(assignments, sidelen);
    (seeds_n as u32, seeds, colors, sim)
}

fn init_colors(
    sidelen: u32,
    source: ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> (Vec<SeedPos>, Vec<SeedColor>, usize) {
    let mut seeds = Vec::new();
    let mut colors = Vec::new();

    let width = source.width() as usize;
    let height = source.height() as usize;

    assert_eq!(width, height);

    let seeds_n = width * height;
    let pixelsize = sidelen as f32 / width as f32;

    for y in 0..width {
        for x in 0..width {
            let p = source.get_pixel(x as u32, y as u32);
            seeds.push(SeedPos {
                xy: [(x as f32 + 0.5) * pixelsize, (y as f32 + 0.5) * pixelsize],
            });
            colors.push(SeedColor {
                rgba: [
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                    1.0,
                ],
            });
        }
    }
    (seeds, colors, seeds_n)
}

#[derive(Clone, Copy)]
pub struct CellBody {
    srcx: f32,
    srcy: f32,
    dstx: f32,
    dsty: f32,

    velx: f32,
    vely: f32,

    accx: f32,
    accy: f32,

    dst_force: f32,
    age: u32,
    stroke_id: u32,
}

const PERSONAL_SPACE: f32 = 0.95;
const MAX_VELOCITY: f32 = 6.0;
const ALIGNMENT_FACTOR: f32 = 0.8;

fn factor_curve(x: f32) -> f32 {
    (x * x * x).min(1000.0)
}

impl CellBody {
    fn new(srcx: f32, srcy: f32, dstx: f32, dsty: f32, dst_force: f32) -> Self {
        Self {
            srcx,
            srcy,
            dstx,
            dsty,
            dst_force,
            velx: 0.0,
            vely: 0.0,
            accx: 0.0,
            accy: 0.0,
            age: 0,
            stroke_id: 0,
        }
    }
    pub fn set_age(&mut self, age: u32) {
        self.age = age;
    }
    pub fn set_dst_force(&mut self, force: f32) {
        self.dst_force = force;
    }
    pub fn set_stroke_id(&mut self, stroke_id: u32) {
        self.stroke_id = stroke_id;
    }

    fn update(&mut self, pos: &mut SeedPos) {
        self.velx += self.accx;
        self.vely += self.accy;

        self.accx = 0.0;
        self.accy = 0.0;

        self.velx *= 0.97;
        self.vely *= 0.97;

        // pos.xy[0] += self.velx;
        // pos.xy[1] += self.vely;

        pos.xy[0] += self.velx.clamp(-MAX_VELOCITY, MAX_VELOCITY);
        pos.xy[1] += self.vely.clamp(-MAX_VELOCITY, MAX_VELOCITY);

        self.age += 1;
    }

    fn apply_dst_force(&mut self, pos: &SeedPos, sidelen: f32) {
        let elapsed = self.age as f32 / 60.0;
        let factor = if self.dst_force == 0.0 {
            0.1
        } else {
            factor_curve(elapsed * self.dst_force)
        };

        let dx = self.dstx - pos.xy[0];
        let dy = self.dsty - pos.xy[1];
        let dist = (dx * dx + dy * dy).sqrt();

        self.accx += (dx * dist * factor) / sidelen;
        self.accy += (dy * dist * factor) / sidelen;
    }

    fn apply_neighbour_force(&mut self, pos: &SeedPos, other: &SeedPos, pixel_size: f32) -> f32 {
        let dx = other.xy[0] - pos.xy[0];
        let dy = other.xy[1] - pos.xy[1];
        let dist = (dx * dx + dy * dy).sqrt();
        let personal_space = pixel_size * PERSONAL_SPACE;

        let weight = (1.0 / dist) * (personal_space - dist) / personal_space;

        if dist > 0.0 && dist < personal_space {
            self.accx -= dx * weight;
            self.accy -= dy * weight;
        } else if dist.abs() < f32::EPSILON {
            // if they are exactly on top of each other, push in a random direction
            let seed = (pos.xy[0].to_bits() as u64) ^ ((pos.xy[1].to_bits() as u64) << 32);
            let mut rng = frand::Rand::with_seed(seed);

            let r1 = rng.gen_range(0.0..1.0);
            let r2 = rng.gen_range(0.0..1.0);

            self.accx += (r1 - 0.5) * 0.1;
            self.accy += (r2 - 0.5) * 0.1;
        }

        weight.max(0.0)
    }

    fn apply_wall_force(&mut self, pos: &SeedPos, sidelen: f32, pixel_size: f32) {
        let personal_space = pixel_size * PERSONAL_SPACE * 0.5;

        if pos.xy[0] < personal_space {
            self.accx += (personal_space - pos.xy[0]) / personal_space;
        } else if pos.xy[0] > sidelen - personal_space {
            self.accx -= (pos.xy[0] - (sidelen - personal_space)) / personal_space;
        }

        if pos.xy[1] < personal_space {
            self.accy += (personal_space - pos.xy[1]) / personal_space;
        } else if pos.xy[1] > sidelen - personal_space {
            self.accy -= (pos.xy[1] - (sidelen - personal_space)) / personal_space;
        }
    }

    fn apply_stroke_attraction(&mut self, i: SeedPos, other_cell: SeedPos, weight: f32) {
        self.accx += (other_cell.xy[0] - i.xy[0]) * weight * 0.8;
        self.accy += (other_cell.xy[1] - i.xy[1]) * weight * 0.8;
    }
}

pub struct Sim {
    //elapsed_frames: u32,
    pub cells: Vec<CellBody>,
    name: String,
    reversed: bool,
    grid: Vec<Vec<usize>>,
}

impl Sim {
    pub fn new(name: String) -> Self {
        Self {
            cells: Vec::new(),
            //elapsed_frames: 0,
            name,
            reversed: false,
            grid: Vec::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    // pub fn source_path(&self) -> PathBuf {
    //     self.source.clone()
    // }

    pub fn switch(&mut self) {
        for cell in &mut self.cells {
            mem::swap(&mut cell.srcx, &mut cell.dstx);
            mem::swap(&mut cell.srcy, &mut cell.dsty);
            cell.age = 0;
        }
        self.reversed = !self.reversed;
    }

    pub fn update(&mut self, positions: &mut [SeedPos], sidelen: u32) {
        let grid_size = (self.cells.len() as f32).sqrt();
        let pixel_size = sidelen as f32 / grid_size;
        //dbg!(grid_size, pixel_size);

        if self.grid.len() != self.cells.len() {
            self.grid = vec![vec![]; self.cells.len()];
        } else {
            for g in &mut self.grid {
                g.clear();
            }
        }
        let grid = &mut self.grid;

        for (i, p) in positions.iter().enumerate() {
            let x = p.xy[0] / pixel_size;
            let y = p.xy[1] / pixel_size;

            let index = (y.floor().clamp(0.0, grid_size - 1.0) * grid_size) as usize
                + (x.floor().clamp(0.0, grid_size - 1.0) as usize);
            //
            grid[index].push(i);
        }

        for (i, cell) in self.cells.iter_mut().enumerate() {
            cell.apply_wall_force(&positions[i], sidelen as f32, pixel_size);
            cell.apply_dst_force(&positions[i], sidelen as f32);
        }

        for i in 0..self.cells.len() {
            let pos = positions[i].xy;
            let col = (pos[0] / pixel_size) as usize;
            let row = (pos[1] / pixel_size) as usize;
            let mut avg_xvel = 0.0;
            let mut avg_yvel = 0.0;
            let mut count = 0.0;
            for dy in 0..=2 {
                for dx in 0..=2 {
                    if col + dx == 0
                        || row + dy == 0
                        || col + dx >= grid_size as usize
                        || row + dy >= grid_size as usize
                    {
                        continue;
                    }
                    let ncol = col + dx - 1;
                    let nrow = row + dy - 1;
                    let nindex = nrow * (grid_size as usize) + ncol;
                    for other in grid[nindex].iter() {
                        if other == &i {
                            continue;
                        }
                        let other_cell = positions[*other];
                        let weight = self.cells[i].apply_neighbour_force(
                            &positions[i],
                            &other_cell,
                            pixel_size,
                        );

                        if self.cells[i].stroke_id == self.cells[*other].stroke_id
                        // && self.cells[i].stroke_id != 0
                        {
                            // stronger attraction to same stroke
                            self.cells[i].apply_stroke_attraction(positions[i], other_cell, weight);
                        }

                        avg_xvel += self.cells[*other].velx * weight;
                        avg_yvel += self.cells[*other].vely * weight;
                        count += weight;
                    }
                }
            }

            if count > 0.0 {
                avg_xvel /= count;
                avg_yvel /= count;

                self.cells[i].accx += (avg_xvel - self.cells[i].velx) * ALIGNMENT_FACTOR;
                self.cells[i].accy += (avg_yvel - self.cells[i].vely) * ALIGNMENT_FACTOR;
            }
        }

        for (index, cell) in self.cells.iter_mut().enumerate() {
            cell.update(&mut positions[index]);
        }
    }

    pub fn set_assignments(&mut self, assignments: Vec<usize>, sidelen: u32) {
        let width = (self.cells.len() as f32).sqrt();
        let pixelsize = sidelen as f32 / width;

        for (dst_idx, src_idx) in assignments.iter().enumerate() {
            let src_x = (src_idx % width as usize) as f32;
            let src_y = (src_idx / width as usize) as f32;
            let dst_x = (dst_idx % width as usize) as f32;
            let dst_y = (dst_idx / width as usize) as f32;
            let prev = self.cells[*src_idx];

            self.cells[*src_idx] = CellBody::new(
                (src_x + 0.5) * pixelsize,
                (src_y + 0.5) * pixelsize,
                (dst_x + 0.5) * pixelsize,
                (dst_y + 0.5) * pixelsize,
                prev.dst_force,
            );

            self.cells[*src_idx].age = prev.age;
            self.cells[*src_idx].stroke_id = prev.stroke_id;
        }
    }

    pub(crate) fn prepare_play(&mut self, positions: &mut [SeedPos], reverse: bool) {
        if self.reversed == reverse {
            for (i, cell) in self.cells.iter_mut().enumerate() {
                positions[i].xy[0] = cell.srcx;
                positions[i].xy[1] = cell.srcy;
                cell.age = 0;
            }
        } else {
            for (i, cell) in self.cells.iter().enumerate() {
                positions[i].xy[0] = cell.dstx;
                positions[i].xy[1] = cell.dsty;
            }
            self.switch();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::SeedPos;
    use crate::app::calculate::drawing_process::DRAWING_CANVAS_SIZE;
    use crate::app::preset::Preset;

    // --- Tests 22-25: CellBody force methods ---

    #[test]
    fn test_cell_body_apply_dst_force_pushes_toward_target() {
        let mut cell = CellBody::new(0.0, 0.0, 10.0, 10.0, 0.13);
        // apply_dst_force uses factor_curve(age/60 * dst_force); with age=0
        // the factor is 0 and no force is applied. Set a non-zero age so the
        // factor is positive.
        cell.set_age(60);
        let pos = SeedPos { xy: [5.0, 5.0] };
        cell.apply_dst_force(&pos, 128.0);
        // dst is at (10,10), pos is at (5,5), so force should push towards (10,10).
        assert!(
            cell.accx > 0.0,
            "accx {} should be positive (toward dst)",
            cell.accx
        );
        assert!(
            cell.accy > 0.0,
            "accy {} should be positive (toward dst)",
            cell.accy
        );
    }

    #[test]
    fn test_cell_body_apply_wall_force_reverses_near_left_edge() {
        let mut cell = CellBody::new(0.0, 0.0, 10.0, 10.0, 0.13);
        let pos = SeedPos { xy: [0.01, 64.0] }; // near left edge
        cell.apply_wall_force(&pos, 128.0, 1.0);
        // Near the left wall, force should push right (positive accx).
        assert!(
            cell.accx > 0.0,
            "wall force should push away from left edge, accx={}",
            cell.accx
        );
    }

    #[test]
    fn test_cell_body_apply_neighbour_force_repels_when_close() {
        let mut cell = CellBody::new(0.0, 0.0, 10.0, 10.0, 0.13);
        let pos = SeedPos { xy: [5.0, 5.0] };
        let other = SeedPos { xy: [5.1, 5.0] }; // very close
        let weight = cell.apply_neighbour_force(&pos, &other, 1.0);
        // Should repel: accx should be negative (pushed away from other which
        // is to the right).
        assert!(
            cell.accx < 0.0,
            "neighbour force should repel, accx={}",
            cell.accx
        );
        assert!(
            weight > 0.0,
            "weight should be positive when within personal space"
        );
    }

    #[test]
    fn test_cell_body_apply_neighbour_force_zero_distance_random_jitter() {
        let mut cell = CellBody::new(0.0, 0.0, 10.0, 10.0, 0.13);
        let pos = SeedPos { xy: [5.0, 5.0] };
        let other = SeedPos { xy: [5.0, 5.0] }; // exactly overlapping
        let _weight = cell.apply_neighbour_force(&pos, &other, 1.0);
        // When dist == 0, random jitter is applied. The force may be small
        // but the function should not panic or produce NaN.
        assert!(
            cell.accx.is_finite(),
            "random jitter accx should be finite, got {}",
            cell.accx
        );
        assert!(
            cell.accy.is_finite(),
            "random jitter accy should be finite, got {}",
            cell.accy
        );
    }

    // --- Helper: build a small Sim ---
    fn make_sim(n: usize) -> Sim {
        let mut sim = Sim::new("test".to_string());
        sim.cells = vec![CellBody::new(0.0, 0.0, 0.0, 0.0, 0.13); n];
        sim
    }

    // --- Test 26: Sim::update produces no NaN ---
    #[test]
    fn test_sim_update_produces_no_nan() {
        let n = 4; // 2x2 grid
        let mut sim = make_sim(n);
        let mut positions: Vec<SeedPos> = vec![
            SeedPos { xy: [0.5, 0.5] },
            SeedPos { xy: [1.5, 0.5] },
            SeedPos { xy: [0.5, 1.5] },
            SeedPos { xy: [1.5, 1.5] },
        ];
        sim.update(&mut positions, 2);
        for p in &positions {
            assert!(p.xy[0].is_finite(), "pos x should be finite");
            assert!(p.xy[1].is_finite(), "pos y should be finite");
        }
        for cell in &sim.cells {
            assert!(cell.velx.is_finite(), "velx should be finite");
            assert!(cell.vely.is_finite(), "vely should be finite");
        }
    }

    // --- Test 27: Sim::set_assignments remaps cells ---
    #[test]
    fn test_sim_set_assignments_remaps_cells() {
        let n = 4;
        let mut sim = make_sim(n);
        // Assign: dst 0 -> src 2, dst 1 -> src 0, dst 2 -> src 3, dst 3 -> src 1
        let assignments = vec![2, 0, 3, 1];
        sim.set_assignments(assignments, 2);
        // After set_assignments, cell[src].dstx should correspond to where src
        // was assigned. src 2 is assigned to dst 0, so cell[2].dstx should be
        // (0 + 0.5) * pixelsize where pixelsize = 2.0/2.0 = 1.0.
        assert!(
            (sim.cells[2].dstx - 0.5).abs() < 0.01,
            "cell[2].dstx should be ~0.5, got {}",
            sim.cells[2].dstx
        );
        assert!(
            (sim.cells[0].dstx - 1.5).abs() < 0.01,
            "cell[0].dstx should be ~1.5 (assigned to dst 1), got {}",
            sim.cells[0].dstx
        );
    }

    // --- Test 28: Sim::switch toggles src/dst and reversed flag ---
    #[test]
    fn test_sim_switch_toggles_src_dst_and_reversed_flag() {
        let n = 4;
        let mut sim = make_sim(n);
        // Set some distinct src/dst values.
        for i in 0..n {
            sim.cells[i].srcx = i as f32;
            sim.cells[i].srcy = 0.0;
            sim.cells[i].dstx = (i + 10) as f32;
            sim.cells[i].dsty = 0.0;
        }
        // Before switch: not reversed.
        // We can't directly check `reversed` (private), but we can check that
        // src and dst are swapped after switch().
        sim.switch();
        for i in 0..n {
            assert_eq!(
                sim.cells[i].srcx,
                (i + 10) as f32,
                "src/dst should be swapped"
            );
            assert_eq!(sim.cells[i].dstx, i as f32, "src/dst should be swapped");
        }
        // Switch back.
        sim.switch();
        for i in 0..n {
            assert_eq!(sim.cells[i].srcx, i as f32, "double switch should restore");
            assert_eq!(
                sim.cells[i].dstx,
                (i + 10) as f32,
                "double switch should restore"
            );
        }
    }

    // --- Test 29: Sim::prepare_play resets seeds to source ---
    #[test]
    fn test_sim_prepare_play_resets_seeds_to_source() {
        let n = 4;
        let mut sim = make_sim(n);
        for i in 0..n {
            sim.cells[i].srcx = i as f32;
            sim.cells[i].srcy = (i * 2) as f32;
            sim.cells[i].dstx = (i + 100) as f32;
            sim.cells[i].dsty = (i + 200) as f32;
        }
        let mut positions: Vec<SeedPos> = vec![SeedPos { xy: [99.0, 99.0] }; n];
        sim.prepare_play(&mut positions, false);
        // After prepare_play (not reversed), positions should be at src.
        for (i, pos) in positions.iter().enumerate().take(n) {
            assert_eq!(pos.xy[0], sim.cells[i].srcx, "pos should reset to src");
            assert_eq!(pos.xy[1], sim.cells[i].srcy, "pos should reset to src");
        }
    }

    // --- Test 30: init_image builds correct cell count ---
    #[test]
    fn test_init_image_builds_correct_cell_count() {
        // Build a minimal Preset: 4x4 source image with identity assignments.
        let sidelen = 4;
        let source_img: Vec<u8> = (0..16).flat_map(|i| [i as u8, i as u8, i as u8]).collect();
        let preset = Preset {
            inner: crate::app::preset::UnprocessedPreset {
                name: "test".to_string(),
                width: sidelen,
                height: sidelen,
                source_img,
            },
            assignments: (0..16).collect(),
        };
        let (seeds_n, seeds, _colors, sim) = init_image(100, preset);
        assert_eq!(seeds_n, 16, "should have 16 seeds for 4x4 image");
        assert_eq!(seeds.len(), 16);
        assert_eq!(sim.cells.len(), 16);
    }

    #[test]
    fn test_init_canvas_creates_identity_assignments() {
        let sidelen = DRAWING_CANVAS_SIZE as u32;
        let n = sidelen * sidelen;
        let source_img: Vec<u8> = (0..n).flat_map(|i| [(i % 256) as u8; 3]).collect();
        let source = UnprocessedPreset {
            name: "canvas".to_string(),
            width: sidelen,
            height: sidelen,
            source_img,
        };
        let (seeds_n, _seeds, _colors, sim) = init_canvas(100, source);
        assert_eq!(seeds_n as usize, n as usize);
        assert_eq!(sim.cells.len(), n as usize);
    }

    #[test]
    fn test_factor_curve_grows_cubically() {
        assert_eq!(factor_curve(0.0), 0.0);
        assert_eq!(factor_curve(1.0), 1.0);
        assert_eq!(factor_curve(2.0), 8.0);
        assert_eq!(factor_curve(3.0), 27.0);
    }

    #[test]
    fn test_factor_curve_clamps_at_1000() {
        assert_eq!(factor_curve(100.0), 1000.0);
        assert_eq!(factor_curve(1000.0), 1000.0);
    }

    #[test]
    fn test_cell_body_update_dampens_velocity() {
        let mut cell = CellBody::new(0.0, 0.0, 10.0, 10.0, 0.13);
        cell.velx = 10.0;
        cell.vely = 5.0;
        let mut pos = SeedPos { xy: [0.0, 0.0] };
        cell.update(&mut pos);
        // velocity should be dampened by 0.97
        assert!(
            cell.velx < 10.0,
            "velocity should be dampened, got {}",
            cell.velx
        );
        assert!(
            cell.vely < 5.0,
            "velocity should be dampened, got {}",
            cell.vely
        );
        // position should have moved
        assert!(pos.xy[0] > 0.0, "position should advance");
    }

    #[test]
    fn test_cell_body_update_increments_age() {
        let mut cell = CellBody::new(0.0, 0.0, 0.0, 0.0, 0.13);
        cell.set_age(5);
        let mut pos = SeedPos { xy: [0.0, 0.0] };
        cell.update(&mut pos);
        assert_eq!(cell.age, 6);
    }

    #[test]
    fn test_sim_update_moves_cell_toward_destination() {
        let mut sim = Sim::new("test".to_string());
        sim.cells = vec![CellBody::new(0.0, 0.0, 100.0, 100.0, 0.13)];
        sim.cells[0].set_age(60); // non-zero age so dst_force kicks in
        let mut positions = vec![SeedPos { xy: [0.0, 0.0] }];
        sim.update(&mut positions, 200);
        // Cell should have moved toward (100, 100), so both x and y should increase
        assert!(
            positions[0].xy[0] > 0.0,
            "cell should move toward dst x, got {}",
            positions[0].xy[0]
        );
        assert!(
            positions[0].xy[1] > 0.0,
            "cell should move toward dst y, got {}",
            positions[0].xy[1]
        );
    }

    #[test]
    fn test_sim_switch_resets_age() {
        let mut sim = make_sim(4);
        for i in 0..4 {
            sim.cells[i].set_age(99);
        }
        sim.switch();
        for cell in &sim.cells {
            assert_eq!(cell.age, 0, "age should reset after switch");
        }
    }

    #[test]
    fn test_sim_name_returns_str() {
        let sim = Sim::new("hello".to_string());
        assert_eq!(sim.name(), "hello");
    }
}

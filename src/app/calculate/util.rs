use crate::app::calculate::ProgressMsg;

use image::imageops;
use serde::Deserialize;
use serde::Serialize;
use uuid::Uuid;

use std::error::Error;

/// Trait for receiving progress messages from background computation.
pub trait ProgressSink {
    fn send(&mut self, msg: ProgressMsg);
}
// Native-friendly adapter
impl ProgressSink for std::sync::mpsc::SyncSender<ProgressMsg> {
    fn send(&mut self, msg: ProgressMsg) {
        let _ = std::sync::mpsc::SyncSender::send(self, msg);
    }
}

// Allow using closures as progress sinks in WASM
impl<T> ProgressSink for T
where
    T: FnMut(crate::app::calculate::ProgressMsg),
{
    fn send(&mut self, msg: crate::app::calculate::ProgressMsg) {
        self(msg);
    }
}

// This tuple is the shared image-processing payload and is immediately destructured by callers.
/// Resolve source/target pixels and weights from a source image and settings.
#[expect(clippy::type_complexity)]
pub(crate) fn get_images(
    source: SourceImg,
    settings: &GenerationSettings,
) -> Result<(Vec<(u8, u8, u8)>, Vec<(u8, u8, u8)>, Vec<i64>), Box<dyn Error>> {
    let source = settings.source_crop_scale.apply(&source, settings.sidelen);
    let source_pixels = source
        .pixels()
        .map(|p| (p[0], p[1], p[2]))
        .collect::<Vec<_>>();

    let (target, weights) = settings.get_target()?;
    let target_pixels = target
        .pixels()
        .map(|p| (p[0], p[1], p[2]))
        .collect::<Vec<_>>();
    if source_pixels.len() != target_pixels.len() {
        return Err(format!(
            "source/target pixel counts differ: {} vs {}",
            source_pixels.len(),
            target_pixels.len()
        )
        .into());
    }
    Ok((source_pixels, target_pixels, weights))
}

/// Crop and scale parameters for source/target image selection.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct CropScale {
    pub x: f32,     // -1: all left, 0: center, 1: all right
    pub y: f32,     // -1: all top, 0: center, 1: all bottom
    pub scale: f32, // 1: fit within frame, >1: zoom in, <1: not allowed
}

impl CropScale {
    /// Returns the identity crop (centered, no zoom).
    pub fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            scale: 1.0,
        }
    }

    /// Apply crop and scale to an image, producing a square `sidelen`×`sidelen` result.
    pub fn apply(&self, img: &SourceImg, sidelen: u32) -> SourceImg {
        let (w, h) = img.dimensions();

        let s = self.scale.max(1.0);

        let base_side = w.min(h) as f32;
        let mut crop_side = (base_side / s).floor().max(1.0);

        crop_side = crop_side.min(w as f32).min(h as f32);

        let max_x_off = (w as f32 - crop_side).max(0.0);
        let max_y_off = (h as f32 - crop_side).max(0.0);

        let xn = (self.x.clamp(-1.0, 1.0) + 1.0) * 0.5;
        let yn = (self.y.clamp(-1.0, 1.0) + 1.0) * 0.5;

        let x0 = (xn * max_x_off).floor() as u32;
        let y0 = (yn * max_y_off).floor() as u32;
        let cs = crop_side as u32;
        let cropped = imageops::crop_imm(img, x0, y0, cs, cs).to_image();

        if cs == sidelen {
            cropped
        } else {
            imageops::resize(&cropped, sidelen, sidelen, imageops::FilterType::Lanczos3)
        }
    }
}

/// Available assignment algorithm backends.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum Algorithm {
    /// Legacy inlined Kuhn-Munkres (Hungarian). Exact, intractable past ~128 sidelen.
    Optimal,
    /// Legacy random pair-swap annealing. Fast, approximate.
    Genetic,
    /// Exact Jonker-Volgenant linear assignment. Faster constants than Hungarian.
    JonkerVolgenant,
    /// Dense forward auction with ε-scaling. Approximate-to-exact, tunable.
    Auction,
    /// Multiscale sparse auction (headline algorithm). Coarse exact JV → candidate
    /// expansion → sparse ε-scaling auction at each finer level → 2-opt refinement.
    #[default]
    Multiscale,
    /// Entropy-regularized Sinkhorn OT on the GPU-friendly transport kernel, then
    /// rounding + sparse-auction repair to a hard permutation.
    Sinkhorn,
    /// PatchMatch-style correspondence propagation + random search, then repair to
    /// a permutation via sparse auction.
    PatchMatch,
    /// Composed mode: PatchMatch (few iters) → sparse auction (large ε) → 1 local-swap pass.
    Fast,
    /// Composed mode: 16² exact JV → multiscale candidate expansion → sparse ε-scaling
    /// auction at 32²/64²/128² → 2-opt. Equivalent to Multiscale end-to-end.
    Balanced,
    /// Composed mode: Balanced result → expand candidate sets → continue auction
    /// refinement → small augmenting-path improvements.
    Maximum,
}

/// UI grouping: composed modes vs. individual solvers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgorithmGroup {
    Mode,
    Algorithm,
}

/// Metadata for an algorithm variant, used by the UI picker.
pub struct AlgorithmInfo {
    pub algorithm: Algorithm,
    pub group: AlgorithmGroup,
    pub label: &'static str,
    pub short_label: &'static str,
    pub description: &'static str,
    pub max_sidelen: Option<u32>,
}

impl Algorithm {
    pub const ALL: [Algorithm; 10] = [
        Algorithm::Fast,
        Algorithm::Balanced,
        Algorithm::Maximum,
        Algorithm::Multiscale,
        Algorithm::Auction,
        Algorithm::JonkerVolgenant,
        Algorithm::Sinkhorn,
        Algorithm::PatchMatch,
        Algorithm::Optimal,
        Algorithm::Genetic,
    ];

    pub const fn info(self) -> AlgorithmInfo {
        match self {
            Algorithm::Optimal => AlgorithmInfo {
                algorithm: self,
                group: AlgorithmGroup::Algorithm,
                label: "Hungarian exact",
                short_label: "Hungarian",
                description: "Legacy exact assignment. Best for tiny sources; capped to avoid dense blowups.",
                max_sidelen: Some(64),
            },
            Algorithm::Genetic => AlgorithmInfo {
                algorithm: self,
                group: AlgorithmGroup::Algorithm,
                label: "Genetic",
                short_label: "Genetic",
                description: "Original random-swap optimizer. Fast, approximate, and good for comparison.",
                max_sidelen: None,
            },
            Algorithm::JonkerVolgenant => AlgorithmInfo {
                algorithm: self,
                group: AlgorithmGroup::Algorithm,
                label: "Jonker-Volgenant exact",
                short_label: "JV exact",
                description: "Exact shortest-augmenting-path solver with better constants than Hungarian.",
                max_sidelen: Some(64),
            },
            Algorithm::Auction => AlgorithmInfo {
                algorithm: self,
                group: AlgorithmGroup::Algorithm,
                label: "Auction dense",
                short_label: "Auction",
                description: "Dense epsilon-scaling auction. Strong small-image baseline.",
                max_sidelen: Some(64),
            },
            Algorithm::Multiscale => AlgorithmInfo {
                algorithm: self,
                group: AlgorithmGroup::Algorithm,
                label: "Multiscale sparse auction",
                short_label: "Multiscale",
                description: "Default coarse-to-fine sparse auction. Reliable for normal web sizes.",
                max_sidelen: None,
            },
            Algorithm::Sinkhorn => AlgorithmInfo {
                algorithm: self,
                group: AlgorithmGroup::Algorithm,
                label: "Sinkhorn OT",
                short_label: "Sinkhorn",
                description: "Entropy-regularized transport rounded into a hard permutation.",
                max_sidelen: Some(32),
            },
            Algorithm::PatchMatch => AlgorithmInfo {
                algorithm: self,
                group: AlgorithmGroup::Algorithm,
                label: "PatchMatch + repair",
                short_label: "PatchMatch",
                description: "Propagation and random search, repaired by sparse auction.",
                max_sidelen: None,
            },
            Algorithm::Fast => AlgorithmInfo {
                algorithm: self,
                group: AlgorithmGroup::Mode,
                label: "Fast mode",
                short_label: "Fast",
                description: "Quick approximate pass for previews and experimentation.",
                max_sidelen: None,
            },
            Algorithm::Balanced => AlgorithmInfo {
                algorithm: self,
                group: AlgorithmGroup::Mode,
                label: "Balanced mode",
                short_label: "Balanced",
                description: "Recommended quality/speed tradeoff. Uses the same solver as Multiscale.",
                max_sidelen: None,
            },
            Algorithm::Maximum => AlgorithmInfo {
                algorithm: self,
                group: AlgorithmGroup::Mode,
                label: "Maximum mode",
                short_label: "Maximum",
                description: "Balanced result plus extra refinement when the selected size is safe.",
                max_sidelen: None,
            },
        }
    }

    pub const fn label(self) -> &'static str {
        self.info().label
    }

    pub const fn short_label(self) -> &'static str {
        self.info().short_label
    }

    pub const fn max_sidelen(self) -> Option<u32> {
        self.info().max_sidelen
    }

    pub const fn supports_sidelen(self, sidelen: u32) -> bool {
        match self.max_sidelen() {
            Some(max) => sidelen <= max,
            None => true,
        }
    }
}

/// Full configuration for a single obamification run.
#[derive(Serialize, Deserialize, Clone)]
pub struct GenerationSettings {
    pub id: Uuid,
    pub name: String,

    pub proximity_importance: i64,
    #[serde(default)]
    pub algorithm: Algorithm,

    pub sidelen: u32,
    custom_target: Option<(u32, u32, Vec<u8>)>,
    pub target_crop_scale: CropScale,
    pub source_crop_scale: CropScale,
}

/// Type alias for the source image format used throughout the crate.
pub type SourceImg = image::RgbImage;

impl GenerationSettings {
    /// Create default settings with the given ID and name.
    pub fn default(id: Uuid, name: String) -> Self {
        Self {
            name,
            proximity_importance: 13, // 20
            algorithm: Algorithm::Multiscale,
            id,
            sidelen: 128,
            custom_target: None,
            target_crop_scale: CropScale::identity(),
            source_crop_scale: CropScale::identity(),
        }
    }

    /// Resolve the target image and per-pixel weights for this settings.
    pub fn get_target(&self) -> Result<(SourceImg, Vec<i64>), Box<dyn std::error::Error>> {
        let target = self.get_raw_target();
        let target = self.target_crop_scale.apply(&target, self.sidelen);
        let weights = if self.custom_target.is_some() {
            vec![255; (self.sidelen * self.sidelen) as usize] // uniform weights
        } else {
            let target_weights =
                image::load_from_memory(include_bytes!("weights256.png"))?.to_rgb8();
            let target_weights = self.target_crop_scale.apply(&target_weights, self.sidelen);
            load_weights(target_weights)
        };

        Ok((target, weights))
    }

    pub(crate) fn get_raw_target(&self) -> SourceImg {
        if let Some((w, h, data)) = &self.custom_target {
            image::ImageBuffer::from_vec(*w, *h, data.clone())
                .expect("custom target image buffer dimensions must match data length")
        } else {
            image::load_from_memory(include_bytes!("target256.png"))
                .expect("embedded target256.png must be a valid PNG")
                .to_rgb8()
        }
    }

    pub(crate) fn set_raw_target(&mut self, img: SourceImg) {
        let (w, h) = img.dimensions();
        let data = img.into_raw();
        self.custom_target = Some((w, h, data));
    }

    /// Clone with a fresh UUID and an incremented version suffix on the name.
    pub fn clone_with_new_id(&self) -> Self {
        let mut new = self.clone();
        new.id = Uuid::new_v4();

        new.name = if let Some(v_pos) = self.name.rfind(" v") {
            let potential_version = &self.name[v_pos + 2..];
            if let Ok(version) = potential_version.parse::<u32>() {
                let base_name = &self.name[..v_pos];
                format!("{} v{}", base_name, version + 1)
            } else {
                format!("{} v2", self.name)
            }
        } else {
            format!("{} v2", self.name)
        };

        new
    }
}

/// Extract per-pixel weights from the red channel of an image.
pub fn load_weights(source: SourceImg) -> Vec<i64> {
    let (width, height) = source.dimensions();
    let mut weights = vec![0; (width * height) as usize];
    for (x, y, pixel) in source.enumerate_pixels() {
        let weight = pixel[0] as i64;
        weights[(y * width + x) as usize] = weight;
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    // --- Tests 10-15: CropScale ---

    #[test]
    fn test_crop_scale_identity_returns_default() {
        let id = CropScale::identity();
        assert_eq!(id.x, 0.0);
        assert_eq!(id.y, 0.0);
        assert_eq!(id.scale, 1.0);
    }

    #[test]
    fn test_crop_scale_identity_apply_returns_image() {
        let img = image::RgbImage::from_pixel(10, 10, image::Rgb([50, 100, 150]));
        let result = CropScale::identity().apply(&img, 10);
        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_crop_scale_zoom_in_crops_smaller_window() {
        let mut img = image::RgbImage::new(20, 20);
        for y in 0..20 {
            for x in 0..20 {
                img.put_pixel(x, y, image::Rgb([x as u8, y as u8, 0]));
            }
        }
        let zoomed = CropScale {
            x: 0.0,
            y: 0.0,
            scale: 2.0,
        }
        .apply(&img, 10);
        // With scale=2, crop_side = 20/2 = 10, then resize to 10 → same size.
        assert_eq!(zoomed.dimensions(), (10, 10));
        // The cropped region should be the center 10x10 (x0=5, y0=5).
        // Check that the top-left pixel comes from the center region.
        let px = zoomed.get_pixel(0, 0);
        // With Lanczos3 the exact values are interpolated, but the center
        // crop starts at (5,5) so the R channel should be around 5.
        assert!(
            (px[0] as i16 - 5).abs() <= 3,
            "R channel {} should be near 5",
            px[0]
        );
    }

    #[test]
    fn test_crop_scale_offset_shifts_window() {
        let mut img = image::RgbImage::new(20, 20);
        // Fill left half with red, right half with blue.
        for y in 0..20 {
            for x in 0..20 {
                if x < 10 {
                    img.put_pixel(x, y, image::Rgb([255, 0, 0]));
                } else {
                    img.put_pixel(x, y, image::Rgb([0, 0, 255]));
                }
            }
        }
        // Offset all the way left: should get red region.
        let left = CropScale {
            x: -1.0,
            y: 0.0,
            scale: 2.0,
        }
        .apply(&img, 10);
        let left_px = left.get_pixel(0, 0);
        assert!(
            left_px[0] > 200,
            "left offset should select red region, got R={}",
            left_px[0]
        );

        // Offset all the way right: should get blue region.
        let right = CropScale {
            x: 1.0,
            y: 0.0,
            scale: 2.0,
        }
        .apply(&img, 10);
        let right_px = right.get_pixel(0, 0);
        assert!(
            right_px[2] > 200,
            "right offset should select blue region, got B={}",
            right_px[2]
        );
    }

    #[test]
    fn test_crop_scale_clamps_zoom_to_one() {
        let img = image::RgbImage::from_pixel(10, 10, image::Rgb([50, 100, 150]));
        // scale < 1 should be clamped to 1 (no zoom out).
        let result = CropScale {
            x: 0.0,
            y: 0.0,
            scale: 0.5,
        }
        .apply(&img, 10);
        // With clamped scale=1, crop_side = 10, no cropping → full image.
        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_crop_scale_resizes_when_target_sidelen_differs() {
        let img = image::RgbImage::from_pixel(20, 20, image::Rgb([50, 100, 150]));
        let result = CropScale::identity().apply(&img, 10);
        assert_eq!(result.dimensions(), (10, 10));
    }

    // --- Tests 16-18: GenerationSettings ---

    #[test]
    fn test_generation_settings_default_fields() {
        let settings = GenerationSettings::default(Uuid::new_v4(), "test".to_string());
        assert_eq!(settings.name, "test");
        assert_eq!(settings.proximity_importance, 13);
        assert_eq!(settings.sidelen, 128);
        // NOTE: the default algorithm is now Algorithm::Multiscale (changed
        // from Algorithm::Genetic as part of the new-algorithm addition).
        assert_eq!(settings.algorithm, Algorithm::Multiscale);
        assert!(settings.custom_target.is_none());
        assert_eq!(settings.target_crop_scale, CropScale::identity());
        assert_eq!(settings.source_crop_scale, CropScale::identity());
    }

    #[test]
    fn test_algorithm_catalog_exposes_all_variants_once() {
        assert_eq!(Algorithm::ALL.len(), 10);
        for (idx, algorithm) in Algorithm::ALL.iter().enumerate() {
            assert_eq!(Algorithm::ALL.iter().filter(|a| *a == algorithm).count(), 1);
            assert!(!algorithm.label().is_empty(), "missing label at {idx}");
            assert!(
                !algorithm.short_label().is_empty(),
                "missing short label at {idx}"
            );
            assert!(
                !algorithm.info().description.is_empty(),
                "missing description at {idx}"
            );
        }
    }

    #[test]
    fn test_algorithm_size_caps_match_dense_backend_guardrails() {
        assert_eq!(Algorithm::Auction.max_sidelen(), Some(64));
        assert_eq!(Algorithm::Sinkhorn.max_sidelen(), Some(32));
        assert_eq!(Algorithm::Optimal.max_sidelen(), Some(64));
        assert_eq!(Algorithm::JonkerVolgenant.max_sidelen(), Some(64));
        assert!(Algorithm::Multiscale.supports_sidelen(256));
        assert!(!Algorithm::Sinkhorn.supports_sidelen(64));
    }

    #[test]
    fn test_clone_with_new_id_no_version_appends_v2() {
        let settings = GenerationSettings::default(Uuid::new_v4(), "mypreset".to_string());
        let cloned = settings.clone_with_new_id();
        assert_eq!(cloned.name, "mypreset v2");
        assert_ne!(cloned.id, settings.id);
    }

    #[test]
    fn test_clone_with_new_id_increments_existing_version() {
        let settings = GenerationSettings::default(Uuid::new_v4(), "mypreset v3".to_string());
        let cloned = settings.clone_with_new_id();
        assert_eq!(cloned.name, "mypreset v4");
        assert_ne!(cloned.id, settings.id);
    }

    // --- Tests 19-21: get_target and load_weights ---

    #[test]
    fn test_get_target_default_loads_target_and_weights() {
        // CORRECTION: The repo overview (gotcha #1) claims that
        // GenerationSettings::get_target returns a 256×256 weights vector even
        // when sidelen=128. This is INCORRECT for the current code: util.rs
        // line 163 resizes the weights image to `sidelen` via
        // `target_crop_scale.apply(&target_weights, self.sidelen)` before
        // calling `load_weights`. This test asserts the CORRECT behavior:
        // weights.len() == sidelen².
        let settings = GenerationSettings::default(Uuid::new_v4(), "test".to_string());
        let (target, weights) = settings.get_target().unwrap();
        assert_eq!(target.dimensions(), (settings.sidelen, settings.sidelen));
        assert_eq!(
            weights.len(),
            (settings.sidelen * settings.sidelen) as usize,
            "weights length {} should equal sidelen² = {}",
            weights.len(),
            settings.sidelen * settings.sidelen
        );
    }

    #[test]
    fn test_get_target_custom_uses_uniform_weights() {
        let mut settings = GenerationSettings::default(Uuid::new_v4(), "test".to_string());
        settings.sidelen = 64;
        let custom = image::RgbImage::from_pixel(64, 64, image::Rgb([128, 128, 128]));
        settings.set_raw_target(custom);
        let (target, weights) = settings.get_target().unwrap();
        assert_eq!(target.dimensions(), (64, 64));
        assert_eq!(weights.len(), 64 * 64);
        // All weights should be 255 (uniform).
        for &w in &weights {
            assert_eq!(w, 255, "custom target should have uniform weight 255");
        }
    }

    #[test]
    fn test_load_weights_uses_red_channel() {
        let mut img = image::RgbImage::new(3, 1);
        img.put_pixel(0, 0, image::Rgb([10, 20, 30]));
        img.put_pixel(1, 0, image::Rgb([40, 50, 60]));
        img.put_pixel(2, 0, image::Rgb([70, 80, 90]));
        let weights = load_weights(img);
        assert_eq!(weights, vec![10, 40, 70]); // only red channel
    }
}

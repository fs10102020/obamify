use serde::{Deserialize, Serialize};

/// A saved obamify preset: the source image pixels plus the computed
/// assignment permutation that maps source pixels to target positions.
#[derive(Clone, Serialize, Deserialize)]
pub struct Preset {
    /// The raw source image data and metadata.
    pub inner: UnprocessedPreset,
    /// Permutation vector: `assignments[dst_idx] = src_idx`.
    pub assignments: Vec<usize>,
}

/// Raw, unprocessed preset data: the source image as RGB bytes plus dimensions.
#[derive(Clone, Serialize, Deserialize)]
pub struct UnprocessedPreset {
    /// Human-readable preset name.
    pub name: String,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Raw RGB pixel data (3 bytes per pixel, row-major).
    pub source_img: Vec<u8>,
}

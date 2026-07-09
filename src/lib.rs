//! Public entry points for the obamify egui application.

#![warn(clippy::all, missing_docs, rust_2018_idioms)]

mod app;

/// Main egui application type used by native `eframe` and the WASM runner.
pub use app::ObamifyApp;
#[cfg(target_arch = "wasm32")]
/// Installs the web worker message handler for WASM processing jobs.
pub use app::worker_entry;

# obamify Agent Notes

Repo root is `obamify/` under the outer `Obamify-testing/` folder. Single Rust 2024 crate: egui + wgpu native app and Trunk/WASM web app. Rust pinned by `rust-toolchain` to 1.85 with `rustfmt`, `clippy`, and `wasm32-unknown-unknown`.

## Verify

Run from `obamify/`. Same checks as CI:

```bash
cargo fmt --all -- --check
cargo check --all-features
cargo test --lib
cargo clippy --all-targets -- -D warnings
cargo check --all-features --lib --target wasm32-unknown-unknown
RUSTFLAGS='-D warnings' trunk build
node --test sites/worker.test.mjs
./scripts/package-sites.sh
```

Focused test: `cargo test --lib test_name`. Native run: `cargo run --release`. Web run: `trunk serve --release --open`.

Linux CI needs: `libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev`. Nix dev shell is also available.

## Entry Points

- `src/main.rs`: native `eframe::run_native`; WASM dispatches page startup vs dedicated-worker startup.
- `src/lib.rs`: public API is `ObamifyApp`; `worker_entry` is WASM-only.
- `src/app.rs`: app state, wgpu resources, JFA render pipeline, native/WASM job startup, preset validation.
- `src/app/gui.rs`: egui UI, file prompts, algorithm picker, drawing/GIF controls, per-job cancel tokens.
- `src/app/calculate/worker/mod.rs`: WASM worker message handler; `worker.js` only imports the generated module.

## Algorithm Context

This project is intentionally an algorithm showcase. Do not collapse the 10 algorithm variants unless explicitly asked.

- Variants in `src/app/calculate/algorithms/*.rs`, dispatched by `Algorithm` in `src/app/calculate/util.rs`.
- `Balanced` is intentionally exposed even though it uses the same backend as `Multiscale`.
- `CostLookup::cost()` returns positive cost to minimize. `ImgDiffWeights::at()` returns negative cost for the inlined Hungarian maximize path. `ImgDiffWeights` has inherent methods, not a trait impl — the `pathfinding`/`indexmap` dependencies were removed.
- Hungarian and JV solvers use a `Vec<usize>` + `Vec<bool>` pair for the augmenting-path set instead of `IndexSet`.

## Rendering Gotchas

- `run_gpu` pipeline order: `LoadOp::Clear` ID texture A, seed splat, JFA ping-pong, shade.
- JFA params must use staging-buffer `copy_buffer_to_buffer` inside the pass loop. `queue.write_buffer` is deferred and would make every pass see the last step value.
- Seed positions and colors are WebGL-compatible textures, not storage buffers.
- WASM forces WebGL backend in `Cargo.toml`/`src/main.rs`; check `wasm32-unknown-unknown` after renderer changes.
- Color lookup texture uses `Rgba32Float`; shade pass loads seed position but doesn't use it (dead load kept for bind-group symmetry).

## Drawing Mode

- `DRAWING_CANVAS_SIZE` is 128. Drawing mode identity assignments and tests assume 128x128 source.
- Native drawing runs on a thread. WASM advances `DrawingOptimizer` per frame.
- Shared state uses `Arc<RwLock<Vec<T>>>`; poisoning recovered with `.unwrap_or_else(|e| e.into_inner())`.
- When leaving Draw mode, `current_drawing_id` is incremented so the native thread self-cancels. `UpdateAssignments` messages are filtered outside Draw mode.
- `PixelData.last_edited` was removed — no `frame_count` in drawing process. `max_dist` uses `DRAWING_CANVAS_SIZE / 4`.

## Solver Job Control

- `GuiState.solver_control` is `Option<SolverControl>`. Create a fresh handle per job and clear it when the progress modal hides.
- Every solver checkpoints through `SolverControl`; cancellation, pause/resume, frame-step, and the displayed frame counter share that path.
- Native control uses an atomic cancel flag plus a condition variable. WASM uses `SharedArrayBuffer`/`Atomics`; the UI disables Pause/Step when shared memory is unavailable.
- Native drawing and generation use separate progress channels so a stale drawing cancellation cannot cancel a new generation job.
- The `ProgressMsg` modal is keyed by the UUID from `GenerationSettings.id`.

## Genetic Solver

- Unbounded loop capped at 5000 generations (emits `ProgressMsg::Error` if exceeded).
- Cancel check runs every 5000 swaps *inside* the swap loop, not just per generation.

## WASM Worker

- Worker creation, serialization, and `postMessage` all use fallible paths — failures push `ProgressMsg::Error` to the inbox instead of panicking.
- Generation requests send an envelope containing the serialized request and the shared `SolverControl` buffer.
- Drawing mode stays on the animation loop with a bounded per-frame optimizer budget; generation algorithms run in the dedicated worker.

## Preset Persistence

- Persisted presets from `eframe::get_value` are validated by `validate_presets()` before use: checks RGB byte count, square dimensions, assignment length, and valid permutation. Invalid local storage falls back to built-in presets.
- Built-in presets live in `presets/{name}/source.png` + `assignments.json` and are loaded by the `include_presets!` macro.

## GIF Recording

- `GifRecorder.should_stop` is reset in `init_encoder`, `stop`, and `finish` so a prior stop request doesn't abort the next recording.
- Max 140 frames, 10MB cap. NeuQuant palette from active seed colors.

## Tests And Lints

- Tests under `#[cfg(test)] mod tests`; no `tests/` directory. **118 active lib tests, 0 ignored.**
- `unsafe_code = "deny"`, `redundant_clone = "deny"`. CI sets `RUSTFLAGS=-D warnings`; `missing_docs = "warn"` becomes an error — new public items need docs.
- Heuristic solver tests check valid permutations and cost bounds, not exact optimality. Exact solvers have brute-force oracle tests for tiny cases.
- Use `let _ =` not `.ok()` for must-use returns.

## Web/PWA Files

- `index.html` registers `sw.js`; `#dev` skips service-worker registration during cache debugging.
- `assets/sw.js` (v4): network-first for navigations so users always get fresh HTML; cache-first for other same-origin GETs. It also preserves the shared-memory isolation fallback for hosts that cannot set response headers directly.
- `assets/manifest.json`: relative paths throughout (`assets/...`, `./index.html`, `./`) for subpath deploy compatibility.
- `Trunk.toml` enables file hashing; use Trunk for web builds.

## OpenAI Sites Deployment

- Keep the Rust/WASM implementation. Sites compatibility is handled at the deployment boundary by `sites/worker.mjs` and `scripts/package-sites.sh`.
- Sites static assets must be under `dist/client/`; emit the ES-module worker at `dist/server/index.js` and keep `dist/index.js` for legacy compatibility. Copy the hosting manifest to `dist/.openai/hosting.json`.
- The worker adds COOP/COEP, origin-agent-cluster, CORP, and nosniff headers. Do not remove these without rechecking WASM Pause/Step on Safari and Chromium.
- `scripts/package-sites.sh` builds release WASM, omits the README-only `example.gif`, validates the worker and required artifacts, and produces `target/obamify-sites.tar.gz`.
- Push the exact staged source tree to the Sites-managed `main` branch before saving a version with its commit SHA. Never persist the short-lived repository credential.

## Removed Dependencies

`egui_extras`, `pathfinding`, and `indexmap` were dropped. An SVG arrow image in `gui.rs` was replaced with text `"→"`. Do not re-add these deps without strong justification.

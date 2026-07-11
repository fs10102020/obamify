[(try it here)](https://obamify.com/)
# obamify
revolutionary new technology that turns any image into obama

![example](example.gif)

# How to use

**Use the studio controls to play transformations, choose saved presets, generate new ones, or switch into drawing mode.** You can change the source image and target image, choose how they are cropped to a square (tip: if both the images are faces, try making the eyes overlap), and pick the assignment algorithm directly in the creation panel. Drawing mode is available on native and web builds; the web build advances the drawing optimizer incrementally to keep the browser responsive.
| Setting               | Description                                                                                     |
|-----------------------|-------------------------------------------------------------------------------------------------|
| resolution            | How many cells the images will be divided into. Higher resolution will capture more high frequency details. |
| proximity importance  | How much the algorithm changes the original image to make it look like the target image. Increase this if you want a more subtle transformation. |
| algorithm             | The algorithm used to calculate the assignment of each pixel. See the algorithm table below. |

## Algorithm selection

The creation panel exposes three **composed modes** and seven individual algorithms:

| Mode / Algorithm | Description | Performance |
|------------------|-------------|-------------|
| **Fast mode** | PatchMatch correspondence search → sparse auction → local swaps | Fastest; good for preview |
| **Balanced mode** | Multiscale sparse auction: 16×16 exact JV → candidate expansion → sparse ε-scaling auction through the selected resolution → 2-opt | Best speed/quality compromise |
| **Maximum mode** | Balanced result → dense auction refinement (small images only) → extended swaps | Highest quality, small images only |
| Multiscale (sparse auction) | Same as Balanced mode — coarse-to-fine sparse candidate matching | Fast, high quality |
| Auction (ε-scaling) | Dense forward auction with ε-scaling. Limited to small images (≤64 sidelen) to avoid memory issues. | Approximate-to-exact |
| Jonker-Volgenant (exact) | Proven Kuhn-Munkres exact baseline. Limited to ≤64 sidelen in the UI. | Exact, use for small grids |
| Hungarian (exact, slow) | Legacy inlined Hungarian algorithm. Limited to ≤64 sidelen in the UI. | Exact, use for small grids |
| Genetic (legacy fast) | Legacy random pair-swap annealing. | Fast, approximate |
| Sinkhorn OT | Entropy-regularized optimal transport + rounding to a permutation. Limited to small images (≤32 sidelen). | Novelty; experimental |
| PatchMatch + repair | Propagation/random search correspondence, then sparse auction repair. | Fast heuristic |

The default algorithm is **Multiscale sparse auction**, which is equivalent to Balanced mode and supports arbitrary square resolutions in the UI range.

# Installations

Install the latest version in [releases](https://github.com/Spu7Nix/obamify/releases). Unzip and run the .exe file inside!
**Note for macOS users:**
Run 'xattr -C <path/to/app.app>' in your terminal to remove the damaged app warning. 
### Building from source

1. Install [Rust](https://www.rust-lang.org/tools/install)
2. Run `cargo run --release` in the project folder

#### Running the web version locally
1. Install [Rust](https://www.rust-lang.org/tools/install)
2. Install the required target with `rustup target add wasm32-unknown-unknown`
3. Install Trunk with `cargo install --locked trunk`
4. Run `trunk serve --release --open`

#### Packaging for OpenAI Sites

The Sites deployment keeps the Rust algorithms and compiles the app to WASM. A small
JavaScript worker serves the Trunk output and adds the cross-origin isolation headers
needed by Pause and frame-step controls.

```bash
./scripts/package-sites.sh
```

The command runs a release Trunk build and creates:

- `target/sites-package/`: a Sites-ready source tree with static assets under
  `dist/client/` and the worker entry point at `dist/index.js`.
- `target/obamify-sites.tar.gz`: the archive passed to Sites when saving a version.

`example.gif` is intentionally omitted because it is README media rather than a runtime
asset. Set `TRUNK_BIN=/path/to/trunk` when Trunk is not on `PATH`. For a previously built
root `dist/`, use `SITES_SKIP_BUILD=1 ./scripts/package-sites.sh`.

The Sites-managed source branch must point at the same staged tree used to create the
archive before saving a version. Never commit or persist the short-lived Sites Git token.

# Contributing

Please open an issue or a pull request if you have any suggestions or find any bugs :)

# How it works

magic

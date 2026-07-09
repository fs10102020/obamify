# obamify TODO

No active fix checklist.

The previous 5-skill review fixes were applied and should be re-verified with:

```bash
cargo test --lib
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo check --all-features --lib --target wasm32-unknown-unknown
```

Add new tasks here only when they are still open.

#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target_root="$repo_root/target"
stage_dir="${SITES_STAGE_DIR:-$target_root/sites-package}"
trunk_dist="${SITES_TRUNK_DIST:-$target_root/sites-trunk-dist}"
archive="${SITES_ARCHIVE:-$target_root/obamify-sites.tar.gz}"
trunk_bin="${TRUNK_BIN:-trunk}"

require_target_path() {
    case "$1" in
        "$target_root"/*) ;;
        *)
            echo "refusing to clean a Sites output outside $target_root: $1" >&2
            exit 1
            ;;
    esac
}

require_target_path "$stage_dir"
require_target_path "$trunk_dist"
require_target_path "$archive"

if [[ ! -f "$repo_root/.openai/hosting.json" ]]; then
    echo "missing .openai/hosting.json" >&2
    exit 1
fi

rm -rf -- "$stage_dir" "$trunk_dist" "$archive"
mkdir -p "$stage_dir" "$trunk_dist"

if [[ "${SITES_SKIP_BUILD:-0}" != "1" ]]; then
    "$trunk_bin" build --release --public-url / --dist "$trunk_dist"
elif [[ -d "$repo_root/dist" ]]; then
    cp -R "$repo_root/dist/." "$trunk_dist/"
else
    echo "SITES_SKIP_BUILD=1 requires an existing dist/ directory" >&2
    exit 1
fi

git -C "$repo_root" archive HEAD | tar --exclude="example.gif" -x -C "$stage_dir"
# The packaged tree is committed to the Sites source repository, so its generated
# deployment artifacts must not inherit the development checkout's dist ignore.
sed -i '/^dist$/d' "$stage_dir/.gitignore"
mkdir -p "$stage_dir/dist/client" "$stage_dir/dist/server" "$stage_dir/dist/.openai"
cp -R "$trunk_dist/." "$stage_dir/dist/client/"
cp "$repo_root/sites/worker.mjs" "$stage_dir/dist/index.js"
cp "$repo_root/sites/worker.mjs" "$stage_dir/dist/server/index.js"
cp "$repo_root/.openai/hosting.json" "$stage_dir/dist/.openai/hosting.json"

node --check "$stage_dir/dist/index.js"
node --check "$stage_dir/dist/server/index.js"
test -f "$stage_dir/dist/client/index.html"
test -n "$(find "$stage_dir/dist/client" -maxdepth 1 -name '*_bg.wasm' -print -quit)"
test ! -e "$stage_dir/example.gif"

tar -czf "$archive" -C "$stage_dir" .

echo "Sites stage: $stage_dir"
echo "Sites archive: $archive"

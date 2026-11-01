#!/usr/bin/env bash
# Build the recorder's runtime on ext4, away from the repo. Needs no sudo.
# Run from the repo in WSL:  bash deploy/install.sh
# A code change reaches the running service only when this runs and the service restarts.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$HOME/lob-recorder"
sha="$(git -C "$repo" rev-parse HEAD)"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT

mkdir -p "$target/config"
git -C "$repo" archive "$sha" | tar -x -C "$build"
uv venv --quiet --allow-existing --python 3.12 "$target/venv"
(cd "$build" && uv export --quiet --frozen --no-dev --no-emit-project -o requirements.txt)
uv pip install --quiet --python "$target/venv/bin/python" -r "$build/requirements.txt"
uv pip install --quiet --python "$target/venv/bin/python" --no-deps --reinstall "$build"
cp "$build"/config/recorder.yaml "$build"/config/instruments.yaml "$target/config/"
echo "$sha" > "$target/DEPLOYED_SHA"
if [ ! -f "$target/.env" ]; then
  cp "$build/.env.example" "$target/.env"
  sed -i "s|^LOB_LOG_DIR=.*|LOB_LOG_DIR=|" "$target/.env"
fi

data_root="$HOME/data/crypto-lob-alpha"
mkdir -p "$data_root"
if [ ! -e "$repo/data" ]; then
  ln -s "$data_root" "$repo/data" || echo "could not link $repo/data; the recorder does not need it"
fi

echo "installed $sha to $target"

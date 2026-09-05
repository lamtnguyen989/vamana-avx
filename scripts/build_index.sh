#!/usr/bin/env bash

# Automating building graph index for data (shard)

set -euo pipefail
SHARD_DIR="${1:?usage: build_index.sh <shard_dir> <index_dir> [R] [L] [alpha] [n_threads] [seed]}"
INDEX_DIR="${2:?usage: build_index.sh <shard_dir> <index_dir> [R] [L] [alpha] [n_threads] [seed]}"
R="${3:-32}"
L="${4:-64}"
ALPHA="${5:-1.20}"
THREADS="${6:-4}"
SEED="$7"
BUILD_BIN="$(dirname "$0")/../build/build_index"

pids=()
for f in "$SHARD_DIR"/shard_*.vecf; do
    n=$(basename "$f" .vecf | sed 's/shard_//')
    out="$INDEX_DIR/index_${n}.vamindx"
    echo "building shard index $n -> $out"
    "$BUILD_BIN" "$f" "$out" "$R" "$L" "$ALPHA" "$THREADS" "$SEED" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "All data graph index built."
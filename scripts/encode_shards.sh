#!/usr/bin/env bash

set -euo pipefail

# Setting up encodings of all data shard against a trained quantization codebook
# Note hard-coding almost everything due to this is written when we are still bootstrapping the pipeline


# Setting up background stuff
SHARD_DIR="${1:?usage: encode_shards.sh <shard_dir> <encoding_dir> <codebook> [threads_per_shard=4]}"
CODEBOOK="${2:?usage: encode_shards.sh <shard_dir> <encoding_dir> <codebook> [threads_per_shard=4]}"
ENCODING_DIR="${3:?usage: encode_shards.sh <shard_dir> <encoding_dir> <codebook> [threads_per_shard=4]}"
THREADS="${3:-4}"
ENCODER="$(dirname "$0")/../build/encode"

mkdir -p $ENCODING_DIR

# Checking codebooks
if [ ! -f "$CODEBOOK" ]; then
    echo "Error: $CODEBOOK not found. Please train a quantization codebook before encoding" >&2
    exit 1
fi

# Encoding 
pids=()
for f in "$SHARD_DIR"/shard_*.vecf; do
    n=$(basename "$f" .vecf | sed 's/shard_//')
    out="$ENCODING_DIR/pq_codes_${n}.pqbin"
    echo "Encoding shard $n -> $out"
    "$ENCODER" "$f" "$CODEBOOK" "$out" "$THREADS" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "All shards encoded against $CODEBOOK"

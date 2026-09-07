#!/usr/bin/env bash

mkdir -p data/
mkdir -p data/shards/

# Generate and sample training data
cargo run -r -p generate_data   -- data/vectors.vecf \
                                --n-vectors 2000000 \
                                --dim 512 \
                                --n-clusters 200 \
                                --cluster-std-dev 10.5 \
                                --threads 4 \
                                --seed 69
                                
python3 scripts/sample.py data/vectors.vecf data/sample.vecf --sample-size 500000

# Shard
python3 scripts/shard_vecfile.py data/vectors.vecf data/shards --num-shards 4
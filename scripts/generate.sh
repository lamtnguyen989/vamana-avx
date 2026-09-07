#!/usr/bin/env bash

mkdir -p data/
mkdir -p data/shards/

# Generate and sample training data
cargo run -r -p generate_data -- data/vectors.vecf -n 2000000 -d 512 -c 200 -s 10.5 \
                                --threads 4 \
                                --seed 69
                                
python3 scripts/sample.py data/vectors.vecf data/sample.vecf --sample-size 500000

# Shard
python3 scripts/shard_vecfile.py data/vectors.vecf data/shards --num-shards 4
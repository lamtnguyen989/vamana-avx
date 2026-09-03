#!/usr/bin/env bash

mkdir -p data/
mkdir -p data/shards/

# Generate and sample training data
python3 scripts/generate_data.py data/vectors.vecf --num-vectors 2000000 --dim 512 --cluster-st-dev 10.5
python3 scripts/sample.py data/vectors.vecf data/sample.vecf --sample-size 500000

# Shard
python3 scripts/shard_vecfile.py data/vectors.vecf data/shards --num-shards 4
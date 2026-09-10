#!/usr/bin/env bash

mkdir -p data/
mkdir -p data/shards/

DIM=${1:-128}

# Generate and sample training data
cargo run -r -p generate_data -- data/vectors.vecf -n 2000000 -d "$DIM" -c 200 -s 10.5 \
                                --threads 4 \
                                --seed 69
                                
python3 scripts/sample.py data/vectors.vecf data/sample.vecf --sample-size 500000

#!/usr/bin/env bash

# Just a simple script to automate the training pipeline for developing

mkdir -p data/
mkdir -p data/shards/
make train

python3 scripts/generate_data.py data/vectors.vecf --num-vectors 2000000 --dim 512 --cluster-st-dev 10.5
python3 scripts/sample.py data/vectors.vecf data/sample.vecf --sample-size 500000

mpiexec -n 4 build/train data/sample.vecf data/shards/pq_codebook.pqbook 32 256 100 8 0.0 69

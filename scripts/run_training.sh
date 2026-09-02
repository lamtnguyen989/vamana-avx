#!/usr/bin/env bash

# Just a simple script to automate the training pipeline for developing

mkdir -p data/
mkdir -p data/shards/
make train

python3 scripts/generate_data.py data/vectors.vecf --num-vectors 250000 --dim 512
python3 scripts/sample.py data/vectors.vecf data/sample.vecf --sample-size 55000

mpiexec -n 4 build/train data/sample.vecf data/shards/pq_codebook.pqbook 32 256 100 4 0.0 69

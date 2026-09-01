#!/usr/bin/env bash

# Just a simple script to automate the training pipeline for developing

mkdir -p data/
mkdir -p data/shards/
make train

python3 scripts/generate_data.py data/vectors.bin --num-vectors 50000 --dim 128
python3 scripts/sample.py data/vectors.bin data/sample.bin --sample-size 20000

mpiexec -n 4 build/train data/sample.bin data/shards/pq_codebook.bin 32 256 50 4 0.0 69

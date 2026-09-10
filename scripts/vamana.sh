#!/usr/bin/env bash

# Creating output directories
mkdir -p data/shards/index 
mkdir -p data/shards/encodings

# Run
make vamana
mpiexec -n 4 build/vamana data/vectors.vecf 4 data/shards/index data/shards/encodings data/shards/pq_codebook.pqbook 32 64 1.20 4 69

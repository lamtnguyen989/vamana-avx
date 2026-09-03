#!/usr/bin/env bash

# Just a simple script to automate the training pipeline for developing

mkdir -p data/
mkdir -p data/shards/
make train

mpiexec -n 4 build/train data/sample.vecf data/shards/pq_codebook.pqbook 32 256 100 8 0.0 69

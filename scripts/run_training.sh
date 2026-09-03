#!/usr/bin/env bash

# Just a simple script to automate the training pipeline for developing
# Make sure you have the Spack environment already activated before running this script

make train
mpiexec -n 4 build/train data/sample.vecf data/shards/pq_codebook.pqbook 32 256 100 4 0.0 69

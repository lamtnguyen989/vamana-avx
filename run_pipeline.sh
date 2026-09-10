#!/usr/bin/env bash

# Create high dimensional dataset and serialize to `.vecf` format (along with the shardings)
./scripts/generate.sh

# Create queries and associated ground truths to the generated dataset
./scripts/ground_truths.sh

# Compile the codebook training binary and train 
./scripts/run_training.sh

# Build Vamana graph index and encode the data shards using the trained codebook
./scripts/vamana.sh

# Search
./scripts/search.sh
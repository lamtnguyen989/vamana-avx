#!/usr/bin/env bash

# Generate queries and compute ground truths from the dataset
cargo run -r -p queries -- data/vectors.vecf data/queries.vecf data/gt/ground_truths.csv \
                            -n 1000 -k 100 -s 0.5 -t 8 --seed 42 -b 5000 

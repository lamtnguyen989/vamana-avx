#!/usr/bin/env bash

mpiexec -n 4 build/search data/queries.vecf 20 data/shards/index data/shards/pq_codebook.pqbook data/shards/encodings data/result/results.csv

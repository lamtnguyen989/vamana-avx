#!/usr/bin/env bash

make search
mpiexec -n 4 build/search data/queries.vecf 20 data/shards/index data/shards/pq_codebook.pqbook data/shards/encodings data/result/results.csv 256 64 4

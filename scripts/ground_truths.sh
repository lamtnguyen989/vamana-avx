#!/usr/bin/env bash

python3 -m venv .venv
source .venv/bin/activate 
pip install -r environments/requirements.txt
echo

mkdir -p data/result data/gt


# Query and generate ground truths from dataset with respect to the queries
python3 scripts/query.py data/vectors.vecf data/queries.vecf data/gt/ground_truths.csv --k 100 --std-dev 0.5

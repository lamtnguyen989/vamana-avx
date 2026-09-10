#!/usr/bin/env bash

python3 -m venv .venv
source .venv/bin/activate 
pip install -r environments/requirements.txt
echo

mkdir -p data/result data/gt


# Current still bootstrapping logic with python
python3 queries/query.py data/vectors.vecf data/queries.vecf data/gt/ground_truths.csv --k 100 --std-dev 0.5

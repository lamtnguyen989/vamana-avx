#!/usr/bin/env bash

python3 -m venv .venv
source .venv/bin/activate 
pip install -r environments/requirements.txt
echo

python3 scripts/query.py data/vectors.vecf

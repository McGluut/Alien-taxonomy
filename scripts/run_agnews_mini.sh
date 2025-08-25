#!/usr/bin/env bash
set -e
source .venv/bin/activate || true
python -m alien_taxonomy.alien_taxonomy \
  --domain text \
  --models sentence-transformers/all-mpnet-base-v2 intfloat/e5-base-v2 \
  --clusterer hdbscan \
  --limit 200 \
  --output_dir runs/agnews_mini

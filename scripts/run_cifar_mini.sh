#!/usr/bin/env bash
set -e
source .venv/bin/activate || true
python -m alien_taxonomy.alien_taxonomy \
  --domain image \
  --models openai/clip-vit-base-patch32 google/vit-base-patch16-224 \
  --clusterer kmeans \
  --limit 500 \
  --output_dir runs/cifar_mini

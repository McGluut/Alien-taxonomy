# Alien Taxonomy via Latent Spaces — Pilot
Pipeline to discover and evaluate ‘alien taxonomies’ in latent spaces. Compare model-derived clusters with human categories using prediction, information-theoretic metrics, redundancy, and mediation checks. Includes AG News and CIFAR-100 pilots

Discover **alien taxonomies** (clusters from model latent spaces) and compare them to **human taxonomies** on prediction, information, stability, and natural‑latents checks.

## Domains
- **Text:** AG News (4 classes)
- **Images:** CIFAR‑100 (100 fine classes, 20 superclasses)

## Quickstart

```bash
# 0) Optional: create & activate a venv
bash setup.sh
source .venv/bin/activate

# 1) AG News mini run (CPU ok, ~200 samples)
python -m alien_taxonomy.alien_taxonomy \
  --domain text \
  --models sentence-transformers/all-mpnet-base-v2 intfloat/e5-base-v2 \
  --clusterer hdbscan \
  --limit 200 \
  --output_dir runs/agnews_mini

# 2) CIFAR‑100 mini run (GPU recommended, ~500 samples)
python -m alien_taxonomy.alien_taxonomy \
  --domain image \
  --models openai/clip-vit-base-patch32 google/vit-base-patch16-224 \
  --clusterer kmeans \
  --limit 500 \
  --output_dir runs/cifar_mini

import os, json, argparse, random, math, pathlib
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, mutual_info_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors

import umap
import hdbscan

# -------------------- helpers --------------------

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def cosine_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms

def embed_text(texts: List[str], model_name: str, batch_size=64, device=None) -> np.ndarray:
    try:
        model = SentenceTransformer(model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        return np.array(embs, dtype=np.float32)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name).to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        mdl.eval()
        out = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                reps = mdl(**enc).last_hidden_state.mean(dim=1).detach().cpu().numpy()
            out.append(reps)
        return cosine_normalize(np.concatenate(out, axis=0).astype(np.float32))

def embed_images(images, model_name: str, batch_size=64, device=None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    out = []
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i+batch_size]
        enc = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            reps = model(**enc).last_hidden_states.mean(dim=1).detach().cpu().numpy()
        out.append(reps)
    return cosine_normalize(np.concatenate(out, axis=0).astype(np.float32))

def load_text_agnews(n=5000):
    ds = load_dataset("ag_news", split="train")
    texts = [f"{t} {d}" for t, d in zip(ds["title"], ds["description"])]
    labels = np.array(ds["label"], dtype=int)
    if n and n < len(texts):
        texts, labels = texts[:n], labels[:n]
    return texts, labels, ["World", "Sports", "Business", "Sci/Tech"]

def load_images_cifar100(n=5000):
    ds = load_dataset("cifar100", split="test")
    images = ds["img"]
    fine = np.array(ds["fine_label"], dtype=int)
    coarse = np.array(ds["coarse_label"], dtype=int)
    fine_names = ds.features["fine_label"].names
    coarse_names = ds.features["coarse_label"].names
    if n and n < len(images):
        images, fine, coarse = images[:n], fine[:n], coarse[:n]
    return images, fine, coarse, fine_names, coarse_names

def cluster_embeddings(X: np.ndarray, method="hdbscan", k=50, min_cluster_size=30):
    if method == "kmeans":
        k = min(k, max(2, len(X)//100))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_
        return labels, centers
    elif method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(X)
        centers = []
        for c in sorted(set(labels)):
            if c < 0: continue
            centers.append(X[labels==c].mean(axis=0))
        centers = np.stack(centers, axis=0) if centers else np.zeros((0, X.shape[1]), dtype=np.float32)
        return labels, centers
    else:
        raise ValueError("Unknown clusterer")

def prototypes(X, labels, centers, k=10):
    protos = {}
    if len(centers)==0: return protos
    nn = NearestNeighbors(n_neighbors=min(k, len(X)), metric="cosine").fit(X)
    for idx, c in enumerate(sorted(set(labels))):
        if c < 0: continue
        center = centers[idx:idx+1]
        _, inds = nn.kneighbors(center, return_distance=True)
        protos[int(c)] = inds[0].tolist()
    return protos

def fit_lr(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")

def mutual_info_discrete(a, b):
    return float(mutual_info_score(a, b))

def save_umap(X, labels, out_png):
    reducer = umap.UMAP(n_components=2, random_state=42)
    Z = reducer.fit_transform(X)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.scatter(Z[:,0], Z[:,1], s=4, c=labels, alpha=0.7)
    plt.title("UMAP of embeddings (clusters)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["text","image"], required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--clusterer", choices=["hdbscan","kmeans"], default="hdbscan")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--min_cluster_size", type=int, default=30)
    parser.add_argument("--kmeans_k", type=int, default=50)
    parser.add_argument("--keep_outliers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load
    if args.domain == "text":
        texts, y_human, _ = load_text_agnews(n=args.limit)
        observables = [embed_text(texts, m, device=device) for m in args.models]
        human_taxo = y_human
        item_ids = np.arange(len(texts))
    else:
        images, fine, coarse, _, _ = load_images_cifar100(n=args.limit)
        observables = [embed_images(images, m, device=device) for m in args.models]
        human_taxo = fine
        item_ids = np.arange(len(images))

    # 2) Cluster
    X_concat = np.concatenate(observables, axis=1)
    labels, centers = cluster_embeddings(X_concat, method=args.clusterer,
                                         k=args.kmeans_k, min_cluster_size=args.min_cluster_size)
    keep = labels != -1 if not args.keep_outliers and (labels==-1).any() else np.ones(len(labels), bool)
    item_ids, alien, X_concat, human_taxo = item_ids[keep], labels[keep], X_concat[keep], human_taxo[keep]
    observables = [obs[keep] for obs in observables]

    # 3) Prototypes
    protos = prototypes(X_concat, alien, centers, k=10)

    # 4) Evaluation
    metrics = {}
    Xtr, Xte, ytr, yte = train_test_split(X_concat, human_taxo, test_size=0.2,
                                          random_state=args.seed, stratify=human_taxo)
    acc_embed, f1_embed = fit_lr(Xtr, ytr, Xte, yte)
    metrics["baseline_acc"], metrics["baseline_f1"] = acc_embed, f1_embed

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    A = enc.fit_transform(alien.reshape(-1,1))
    H = enc.fit_transform(human_taxo.reshape(-1,1))

    XtrA, XteA, ytrA, yteA = train_test_split(A, human_taxo, test_size=0.2, stratify=human_taxo)
    acc_alien, f1_alien = fit_lr(XtrA, ytrA, XteA, yteA)
    metrics["alien_acc"], metrics["alien_f1"] = acc_alien, f1_alien

    XtrH, XteH, ytrH, yteH = train_test_split(H, human_taxo, test_size=0.2, stratify=human_taxo)
    acc_human, f1_human = fit_lr(XtrH, ytrH, XteH, yteH)
    metrics["human_acc"], metrics["human_f1"] = acc_human, f1_human

    Xboth = np.concatenate([A,H], axis=1)
    XtrB, XteB, ytrB, yteB = train_test_split(Xboth, human_taxo, test_size=0.2, stratify=human_taxo)
    acc_both, f1_both = fit_lr(XtrB, ytrB, XteB, yteB)
    metrics["both_acc"], metrics["both_f1"] = acc_both, f1_both

    metrics["mi"]  = mutual_info_discrete(alien, human_taxo)
    metrics["nmi"] = float(normalized_mutual_info_score(human_taxo, alien))
    metrics["ari"] = float(adjusted_rand_score(human_taxo, alien))

    # Redundancy
    red = {}
    for i, obs in enumerate(observables):
        strat = alien if len(set(alien))>1 else None
        Xtr, Xte, ytr, yte = train_test_split(obs, alien, test_size=0.2,
                                              random_state=args.seed, stratify=strat)
        acc, f1 = fit_lr(Xtr, ytr, Xte, yte)
        red[f"encoder_{i}"] = {"acc": acc, "f1": f1}
    metrics["redundancy"] = red

    # 5) Save
    outdir = pathlib.Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": item_ids, "alien_cluster": alien, "human_label": human_taxo}).to_csv(outdir/"clusters.csv", index=False)
    with open(outdir/"metrics.json","w") as f: json.dump(metrics, f, indent=2)
    save_umap(X_concat, alien, str(outdir/"umap_clusters.png"))
    with open(outdir/"prototypes.json","w") as f: json.dump(protos,f,indent=2)

    print("Done. Results in", outdir)

if __name__ == "__main__":
    main()

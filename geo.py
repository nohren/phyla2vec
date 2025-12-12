"""
eval_continent_geo.py

Evaluate continent prediction using:
 - PCoA (UniFrac distance geometry baseline)
 - Transformer Encoder
 - MLP Encoder
 - VAE (backbone)
 - UniFrac VAE (backbone)

Continents:
  class 0: Africa
  class 1: Europe
  class 2: North America
"""

import os
import numpy as np
import pandas as pd
import biom
import torch
from torch.utils.data import DataLoader

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa

from train import (
    SampleSequenceDataset,
    prepare_epoch_data,
    UniFracEncoder,
    MLPUniFracEncoder,
    UniFracVAE,
    SEQ_LEN,
    EMBED_DIM,
    MODEL_READS,
    MAX_SAMPLES,
    DEVICE,
)

from datetime import datetime


RESULTS_FILE = "continent_results_geo.txt"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BIOM_PATH = "test_filtered/test.biom"
CONTINENT_PATH = "test_filtered/continent_labels.txt"
UNIFRAC_DM_PATH = "test_filtered/distance-matrix.tsv"

RAREFY_DEPTH_EVAL = 5000
BATCH_SIZE_EVAL = 32

N_RUNS = 6
BASE_SEED = 42

CONTINENT_TO_ID = {
    "africa": 0,
    "europe": 1,
    "north america": 2,
}
ID_TO_CONTINENT = ["Africa", "Europe", "North America"]


# ============================================================
# SELECT EXACTLY ONE MODEL TO EVALUATE
# ============================================================

# --- Transformer Encoder ---
# CKPT = "/data/nicklas/scratch/pres_transformer_encoder_epoch_20.pt"
# MODEL_NAME = "TransformerEncoder"
# MODEL_BUILDER = lambda: UniFracEncoder()
# ENCODER_TYPE = "base"

# --- MLP Encoder ---
# CKPT = "/data/nicklas/scratch/pres_mlp_encoder_epoch_20.pt"
# MODEL_NAME = "MLPEncoder"
# MODEL_BUILDER = lambda: MLPUniFracEncoder()
# ENCODER_TYPE = "base"

# --- VAE (encoder backbone) ---
CKPT = "/data/nicklas/scratch/pres_vae_epoch_20.pt"
MODEL_NAME = "VAE_EncoderBackbone"
MODEL_BUILDER = lambda: UniFracVAE()
ENCODER_TYPE = "vae_backbone"

# --- UniFrac VAE (encoder backbone) ---
# CKPT = "/data/nicklas/scratch/pres_unifrac_vae_epoch_20.pt"
# MODEL_NAME = "UniFracVAE_EncoderBackbone"
# MODEL_BUILDER = lambda: UniFracVAE()
# ENCODER_TYPE = "vae_backbone"


def load_continent_labels(path):
    label_dict = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            sid = parts[0]
            cont = " ".join(parts[1:]).lower()

            if cont in CONTINENT_TO_ID:
                label_dict[sid] = CONTINENT_TO_ID[cont]

    return label_dict


@torch.no_grad()
def compute_embeddings(model, dataloader, encoder_type="base"):
    model.eval()
    N = len(dataloader.dataset)
    emb = np.zeros((N, EMBED_DIM), dtype=np.float32)

    for batch in dataloader:
        seq = batch["seq"].to(DEVICE)
        idx = batch["idx"].cpu().numpy()

        if encoder_type == "vae_backbone":
            out = model.encoder_backbone(seq)
        else:
            out = model(seq)

        emb[idx] = out.cpu().numpy()
    return emb


def compute_pcoa_embeddings(sample_ids, dm_path=UNIFRAC_DM_PATH, dim=EMBED_DIM):
    """
    Returns: emb_pcoa (N_samples, EMBED_DIM)
    in same order as sample_ids (BIOM rarefaction order).
    """
    print(f"Loading UniFrac distance matrix from: {dm_path}")
    df = pd.read_csv(dm_path, sep="\t", index_col=0)

    missing = [sid for sid in sample_ids if sid not in df.index]
    if missing:
        raise RuntimeError(
            f"{len(missing)} samples from BIOM not found in UniFrac DM. "
            f"Example missing: {missing[:5]}"
        )

    df = df.loc[sample_ids, sample_ids]

    print("Running PCoA...")
    dm = DistanceMatrix(df.values, ids=df.index.tolist())
    pcoa_res = pcoa(dm)
    coords = pcoa_res.samples

    k = min(dim, coords.shape[1])
    arr = coords.values[:, :k].astype(np.float32)

    if k < dim:
        pad = np.zeros((arr.shape[0], dim - k), dtype=np.float32)
        arr = np.hstack([arr, pad])

    return arr


def train_and_eval_xgb(X, y, name, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    clf = XGBClassifier(
        n_estimators=24,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=8,
        random_state=seed,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except:
        auc = float("nan")

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {f1_macro:.4f}")
    print(f"Macro AUC: {auc:.4f}")

    return acc, f1_macro, auc


def repeated_eval(X, y, name):
    accs, f1s, aucs = [], [], []

    for i in range(N_RUNS):
        acc, f1, auc = train_and_eval_xgb(
            X, y, f"{name}_run{i}", seed=BASE_SEED + i
        )
        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)

    accs, f1s, aucs = map(np.array, (accs, f1s, aucs))

    acc_m, acc_s = accs.mean(), accs.std(ddof=1)
    f1_m, f1_s = f1s.mean(), f1s.std(ddof=1)
    auc_m, auc_s = np.nanmean(aucs), np.nanstd(aucs, ddof=1)

    print(f"\n=== SUMMARY {name} ===")
    print(f"Accuracy:  {acc_m:.4f} ± {acc_s:.4f}")
    print(f"Macro F1:  {f1_m:.4f} ± {f1_s:.4f}")
    print(f"Macro AUC: {auc_m:.4f} ± {auc_s:.4f}")

    with open(RESULTS_FILE, "a") as f:
        f.write("\n" + "-" * 70 + "\n")
        f.write(f"SUMMARY {name}\n")
        f.write(f"Accuracy:  {acc_m:.4f} ± {acc_s:.4f}\n")
        f.write(f"Macro F1:  {f1_m:.4f} ± {f1_s:.4f}\n")
        f.write(f"Macro AUC: {auc_m:.4f} ± {auc_s:.4f}\n")


def eval_continent_geo():
    with open(RESULTS_FILE, "a") as f:
        f.write("\n" + "#" * 80 + "\n")
        f.write(f"NEW CONTINENT EVAL RUN at {datetime.now()}\n")
        f.write("#" * 80 + "\n")

    print(f"Loading BIOM table from: {BIOM_PATH}")
    full_table = biom.load_table(BIOM_PATH)

    sample_tokens, rare_table, sample_ids, obs_ids = prepare_epoch_data(
        full_table,
        depth=RAREFY_DEPTH_EVAL,
        seq_len=SEQ_LEN,
        model_reads=MODEL_READS,
        max_samples=MAX_SAMPLES,
        seed=42,
    )

    dataset = SampleSequenceDataset(sample_tokens)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    labels = load_continent_labels(CONTINENT_PATH)
    labeled_idx = [i for i, sid in enumerate(sample_ids) if sid in labels]
    y = np.array([labels[sample_ids[i]] for i in labeled_idx], dtype=np.int64)
    labeled_idx = np.array(labeled_idx)

    print("\n--- Evaluating PCoA UniFrac geometry baseline ---")
    emb_pcoa = compute_pcoa_embeddings(sample_ids, UNIFRAC_DM_PATH, EMBED_DIM)
    repeated_eval(emb_pcoa[labeled_idx], y, name="PCoA_UniFrac")

    if not os.path.isfile(CKPT):
        raise FileNotFoundError(f"Missing ckpt: {CKPT}")

    print(f"\n--- Evaluating {MODEL_NAME} ---")
    model = MODEL_BUILDER().to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    emb = compute_embeddings(model, dataloader, ENCODER_TYPE)
    repeated_eval(emb[labeled_idx], y, name=MODEL_NAME)


if __name__ == "__main__":
    eval_continent_geo()

import os
import argparse

import numpy as np
import biom
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

BIOM_PATH = "feature-table-2500.biom"
PCOA_TSV_PATH = "pcoa_unifrac_128d.tsv"

ACID_REFLUX_PATH = "acid_reflux.txt"
AGE_PATH = "host_age.txt"

RAREFY_DEPTH_EVAL = 5000
BATCH_SIZE_EVAL = 32

CKPT_TRANSFORMER = "/data/nicklas/scratch/unifrac_encoder_epoch_20.pt"
CKPT_MLP = "/data/nicklas/scratch/unifrac_mlp_encoder_epoch_20.pt"
CKPT_VAE = "/data/nicklas/scratch/unifrac_vae_epoch_20.pt"


def load_acid_reflux_labels(path: str):
    labels = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            sid, lab = line.split()
            labels[sid] = int(lab)
    return labels


def age_to_bucket(age: float) -> int:
    if age < 20:
        return 0
    if age < 60:
        return 1
    return 2


def load_age_buckets(path: str):
    labels = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            sid = parts[0]
            raw = parts[1]

            if "not" in raw.lower():
                continue
            if raw.lower() in {"na", "nan", "none"}:
                continue

            try:
                age = float(raw)
            except ValueError:
                continue

            labels[sid] = age_to_bucket(age)

    return labels


@torch.no_grad()
def compute_embeddings(model, dataloader, encoder_type="base"):
    model.eval()
    N = len(dataloader.dataset)
    out = np.zeros((N, EMBED_DIM), dtype=np.float32)

    for batch in dataloader:
        seq = batch["seq"].to(DEVICE)
        idx = batch["idx"].cpu().numpy()

        if encoder_type == "vae_backbone":
            emb = model.encoder_backbone(seq)
        else:
            emb = model(seq)

        out[idx] = emb.cpu().numpy()

    return out


def load_pcoa_embeddings(path):
    emb = {}
    with open(path, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 129:
                continue
            sid = parts[0]
            vals = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            emb[sid] = vals
    return emb


def save_plot(Z, y, title, task, outfile):
    plt.figure(figsize=(6, 5))

    if task == "acid":
        classes = [0, 1]
        labels = ["no reflux", "reflux"]
        colors = ["tab:blue", "tab:red"]
    else:
        classes = [0, 1, 2]
        labels = ["<20", "20-59", ">=60"]
        colors = ["tab:blue", "tab:orange", "tab:green"]

    for c, lab, col in zip(classes, labels, colors):
        mask = y == c
        if mask.sum() == 0:
            continue
        plt.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.7, label=lab)

    plt.title(title)
    plt.legend(markerscale=1.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")


def visualize(task, perplexity=30.0):
    print(f"\n=== Visualizing task: {task} ===")

    table = biom.load_table(BIOM_PATH)
    sample_tokens, _, sample_ids, _ = prepare_epoch_data(
        table,
        depth=RAREFY_DEPTH_EVAL,
        seq_len=SEQ_LEN,
        model_reads=MODEL_READS,
        max_samples=MAX_SAMPLES,
        seed=42,
    )
    dataset = SampleSequenceDataset(sample_tokens)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    if task == "acid":
        label_dict = load_acid_reflux_labels(ACID_REFLUX_PATH)
    else:
        label_dict = load_age_buckets(AGE_PATH)

    labeled_idx = []
    y = []

    for i, sid in enumerate(sample_ids):
        if sid in label_dict:
            labeled_idx.append(i)
            y.append(label_dict[sid])

    labeled_idx = np.array(labeled_idx)
    y = np.array(y)

    print("\nLoading PCoA embeddings...")
    pcoa = load_pcoa_embeddings(PCOA_TSV_PATH)
    X_pcoa = []
    y_pcoa = []

    for i in labeled_idx:
        sid = sample_ids[i]
        if sid in pcoa:
            X_pcoa.append(pcoa[sid])
            y_pcoa.append(label_dict[sid])

    if len(X_pcoa) > 0:
        X_pcoa = np.stack(X_pcoa)
        y_pcoa = np.array(y_pcoa)

        Zp = PCA(n_components=2).fit_transform(X_pcoa)
        save_plot(Zp, y_pcoa, f"PCoA ({task}) - PCA", task, f"PCoA_{task}_PCA.png")

        print("Running t-SNE on PCoA embeddings...")
        Zt = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate=200,
            verbose=1,
        ).fit_transform(X_pcoa)

        save_plot(
            Zt,
            y_pcoa,
            f"PCoA ({task}) - t-SNE",
            task,
            f"PCoA_{task}_TSNE.png",
        )

    models = [
        ("TransformerEncoder", UniFracEncoder, CKPT_TRANSFORMER, "base"),
        ("MLPEncoder", MLPUniFracEncoder, CKPT_MLP, "base"),
        ("VAE_EncoderBackbone", UniFracVAE, CKPT_VAE, "vae_backbone"),
    ]

    for name, builder, ckpt, enc_type in models:
        print(f"\n--- Visualizing {name} ---")

        if not os.path.exists(ckpt):
            print(f"Checkpoint missing: {ckpt}")
            continue

        model = builder().to(DEVICE)
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

        emb_all = compute_embeddings(model, dataloader, encoder_type=enc_type)
        X = emb_all[labeled_idx]

        Zp = PCA(n_components=2).fit_transform(X)
        save_plot(Zp, y, f"{name} ({task}) - PCA", task, f"{name}_{task}_PCA.png")

        print(f"Running t-SNE on {name} embeddings...")
        Zt = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate=200,
            verbose=1,
        ).fit_transform(X)

        save_plot(
            Zt,
            y,
            f"{name} ({task}) - t-SNE",
            task,
            f"{name}_{task}_TSNE.png",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["acid", "age"])
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    visualize(args.task, perplexity=args.perplexity)


if __name__ == "__main__":
    main()

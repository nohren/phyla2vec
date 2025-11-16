import os
import numpy as np
import biom
import torch
from torch.utils.data import DataLoader

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
)
from datetime import datetime

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

RESULTS_FILE = "acid_results_2.txt"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
BIOM_PATH = "feature-table-2500.biom"
PCOA_TSV_PATH = "pcoa_unifrac_128d.tsv"
ACID_REFLUX_PATH = "acid_reflux.txt"
RAREFY_DEPTH_EVAL = 5000
BATCH_SIZE_EVAL = 32
CKPT_TRANSFORMER = "/data/nicklas/scratch/unifrac_encoder_epoch_20.pt"
CKPT_MLP = "/data/nicklas/scratch/unifrac_mlp_encoder_epoch_20.pt"
CKPT_VAE = "/data/nicklas/scratch/unifrac_vae_epoch_20.pt"
N_RUNS = 6
BASE_SEED = 0


def load_acid_reflux_labels(path: str):
    labels = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            sid, lab = parts[0], parts[1]
            try:
                lab_int = int(lab)
            except ValueError:
                continue
            labels[sid] = lab_int
    return labels


@torch.no_grad()
def compute_embeddings(model, dataloader, encoder_type="base"):
    model.eval()
    dataset = dataloader.dataset
    N = len(dataset)
    emb_all = np.zeros((N, EMBED_DIM), dtype=np.float32)

    for batch in dataloader:
        seq = batch["seq"].to(DEVICE)
        idx = batch["idx"].cpu().numpy()

        if encoder_type == "vae_backbone":
            emb = model.encoder_backbone(seq)
        else:
            emb = model(seq)

        emb_all[idx] = emb.cpu().numpy()

    return emb_all


def load_pcoa_embeddings(tsv_path):
    emb_dict = {}
    dim = None

    with open(tsv_path, "r") as f:
        header = f.readline().strip().split("\t")

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            sid = parts[0]
            vals = np.array([float(x) for x in parts[1:]], dtype=np.float32)

            if dim is None:
                dim = len(vals)
            else:
                if len(vals) != dim:
                    raise ValueError(
                        f"Inconsistent embedding dimension for sample {sid}: expected {dim}, got {len(vals)}"
                    )

            emb_dict[sid] = vals

    if dim is None:
        raise RuntimeError(f"No embeddings found in {tsv_path}")

    return emb_dict, dim


def train_and_eval_xgb(X, y, name="model", seed=0, log_to_file=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    if pos == 0:
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = neg / pos

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=8,
        random_state=seed,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print(f"\n=== {name} embeddings (XGBoost) ===")
    print(f"Seed: {seed}")
    print(f"scale_pos_weight = {scale_pos_weight:.3f}  (neg={neg}, pos={pos})")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {auc:.4f}")
    print(f"F1      : {f1:.4f}")
    print("Classification report:")
    print(report)

    if log_to_file:
        with open(RESULTS_FILE, "a") as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Model: {name}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"scale_pos_weight = {scale_pos_weight:.3f}  (neg={neg}, pos={pos})\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"ROC-AUC : {auc:.4f}\n")
            f.write(f"F1      : {f1:.4f}\n")
            f.write("Classification report:\n")
            f.write(report)
            f.write("\n")

    return acc, auc, f1


def repeated_xgb_eval(X, y, name, n_runs=N_RUNS, base_seed=BASE_SEED):
    accs, aucs, f1s = [], [], []

    for i in range(n_runs):
        seed = base_seed + i
        run_name = f"{name}_run{i}"
        acc, auc, f1 = train_and_eval_xgb(
            X,
            y,
            name=run_name,
            seed=seed,
            log_to_file=True,
        )
        accs.append(acc)
        aucs.append(auc)
        f1s.append(f1)

    accs = np.array(accs, dtype=np.float64)
    aucs = np.array(aucs, dtype=np.float64)
    f1s = np.array(f1s, dtype=np.float64)

    acc_mean, acc_std = accs.mean(), accs.std(ddof=1)
    auc_mean = np.nanmean(aucs)
    auc_std = np.nanstd(aucs, ddof=1)
    f1_mean, f1_std = f1s.mean(), f1s.std(ddof=1)

    print(f"\n=== SUMMARY over {n_runs} runs for {name} ===")
    print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"ROC-AUC : {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"F1      : {f1_mean:.4f} ± {f1_std:.4f}")

    with open(RESULTS_FILE, "a") as f:
        f.write("\n" + "-" * 70 + "\n")
        f.write(f"SUMMARY over {n_runs} runs for {name}\n")
        f.write(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}\n")
        f.write(f"ROC-AUC : {auc_mean:.4f} ± {auc_std:.4f}\n")
        f.write(f"F1      : {f1_mean:.4f} ± {f1_std:.4f}\n")

    return {
        "acc": (acc_mean, acc_std),
        "auc": (auc_mean, auc_std),
        "f1": (f1_mean, f1_std),
    }


def eval_acid_reflux():
    with open(RESULTS_FILE, "a") as f:
        f.write("\n" + "#" * 80 + "\n")
        f.write(f"NEW ACID REFLUX EVAL RUN at {datetime.now()}\n")
        f.write("#" * 80 + "\n")

    print(f"Loading BIOM table from: {BIOM_PATH}")
    full_table = biom.load_table(BIOM_PATH)

    print("Preparing sample tokens (rarefaction + subsampling)...")
    sample_tokens, rare_table, sample_ids, obs_ids = prepare_epoch_data(
        full_table,
        depth=RAREFY_DEPTH_EVAL,
        seq_len=SEQ_LEN,
        model_reads=MODEL_READS,
        max_samples=MAX_SAMPLES,
        seed=42,
    )
    print(f"Num samples after filtering+subsample: {len(sample_ids)}")

    dataset = SampleSequenceDataset(sample_tokens)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
    )

    print(f"Loading acid reflux labels from: {ACID_REFLUX_PATH}")
    label_dict = load_acid_reflux_labels(ACID_REFLUX_PATH)

    labeled_indices = []
    y_list = []
    for i, sid in enumerate(sample_ids):
        if sid in label_dict:
            labeled_indices.append(i)
            y_list.append(label_dict[sid])

    labeled_indices = np.array(labeled_indices, dtype=np.int64)
    y = np.array(y_list, dtype=np.int64)

    print(f"Total samples with labels: {len(y)}")
    if len(y) == 0:
        raise RuntimeError(
            "No overlap between BIOM sample IDs and acid_reflux.txt SampleIDs."
        )

    print(f"\nLoading PCoA UniFrac embeddings from: {PCOA_TSV_PATH}")
    pcoa_dict, pcoa_dim = load_pcoa_embeddings(PCOA_TSV_PATH)
    print(f"  Embedding dimension from TSV: {pcoa_dim}")

    pcoa_rows = []
    y_pcoa = []

    for i in labeled_indices:
        sid = sample_ids[i]
        if sid not in pcoa_dict:
            continue
        pcoa_rows.append(pcoa_dict[sid])
        y_pcoa.append(label_dict[sid])

    if len(pcoa_rows) > 0:
        X_pcoa = np.stack(pcoa_rows, axis=0)
        y_pcoa = np.array(y_pcoa, dtype=np.int64)

        print(f"  PCoA samples with labels and embeddings: {X_pcoa.shape[0]}")
        repeated_xgb_eval(X_pcoa, y_pcoa, name="PCoA_UniFrac_128D")

    models_cfg = [
        {
            "name": "TransformerEncoder",
            "builder": lambda: UniFracEncoder(),
            "ckpt": CKPT_TRANSFORMER,
            "encoder_type": "base",
        },
        {
            "name": "MLPEncoder",
            "builder": lambda: MLPUniFracEncoder(),
            "ckpt": CKPT_MLP,
            "encoder_type": "base",
        },
        {
            "name": "VAE_EncoderBackbone",
            "builder": lambda: UniFracVAE(),
            "ckpt": CKPT_VAE,
            "encoder_type": "vae_backbone",
        },
    ]

    for cfg in models_cfg:
        print(f"\n--- Evaluating {cfg['name']} ---")

        if not os.path.isfile(cfg["ckpt"]):
            print(f"  [WARN] Checkpoint not found: {cfg['ckpt']}. Skipping.")
            continue

        model = cfg["builder"]().to(DEVICE)

        print(f"  Loading checkpoint: {cfg['ckpt']}")
        ckpt = torch.load(cfg["ckpt"], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        emb_all = compute_embeddings(
            model,
            dataloader,
            encoder_type=cfg["encoder_type"],
        )

        X = emb_all[labeled_indices]

        repeated_xgb_eval(X, y, name=cfg["name"])


if __name__ == "__main__":
    eval_acid_reflux()

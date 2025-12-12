import os
import numpy as np
import biom
import torch
from torch.utils.data import DataLoader
import pandas as pd
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)

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


RESULTS_FILE = "age_results_3.txt"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

BIOM_PATH = "test_filtered/test.biom"
DM_PATH = "test_filtered/distance-matrix.tsv"
AGE_PATH = "test_filtered/age_labels.txt"

RAREFY_DEPTH_EVAL = 5000
BATCH_SIZE_EVAL = 32

N_RUNS = 6
BASE_SEED = 0


# CKPT = "pres_transformer_encoder_epoch_20_attentionpoollinear_rarepoch1_3800reads.pt"
# MODEL_NAME = "TransformerEncoder"
# MODEL_BUILDER = lambda: UniFracEncoder()
# ENCODER_TYPE = "base"

# CKPT = "/data/nicklas/scratch/pres_mlp_encoder_epoch_20.pt"
# MODEL_NAME = "MLPEncoder"
# MODEL_BUILDER = lambda: MLPUniFracEncoder()
# ENCODER_TYPE = "base"

CKPT = "/data/nicklas/scratch/pres_vae_epoch_20.pt"
MODEL_NAME = "VAE_EncoderBackbone"
MODEL_BUILDER = lambda: UniFracVAE()
ENCODER_TYPE = "vae_backbone"

# CKPT = "/data/nicklas/scratch/pres_unifrac_vae_epoch_20.pt"
# MODEL_NAME = "UniFracVAE_EncoderBackbone"
# MODEL_BUILDER = lambda: UniFracVAE()
# ENCODER_TYPE = "vae_backbone"


def get_study_id(sample_id: str) -> str:
    """
    Extract study ID as the prefix before the first period in the SampleID.
    Example: "11888.2066" -> "11888"
    """
    return sample_id.split(".", 1)[0]


def age_to_bucket(age: float) -> int:
    """
    Map a numeric age to a bucket (binary):
      0: age < 30
      1: age >= 30
    """
    if age < 30.0:
        return 0
    else:
        return 1


def load_age_buckets_with_study(path: str):
    """
    Returns:
        label_dict: {sample_id (str): bucket_label (int)}
        study_dict: {sample_id (str): study_id (str)}
    """
    label_dict = {}
    study_dict = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            sid, raw_val = parts[0], parts[1]

            lower = raw_val.lower()
            if lower in {"not", "not_provided", "not_provided,", "na"}:
                continue
            if "not" in lower:
                continue

            try:
                age = float(raw_val)
            except ValueError:
                continue

            bucket = age_to_bucket(age)
            label_dict[sid] = bucket
            study_dict[sid] = get_study_id(sid)

    return label_dict, study_dict


@torch.no_grad()
def compute_embeddings(model, dataloader, encoder_type="base"):
    """
    Returns:
        np.ndarray of shape (N_samples, EMBED_DIM)
    """
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


def train_and_eval_xgb_multiclass_fixed_split(
    X_train,
    y_train,
    X_test,
    y_test,
    name="model",
    seed=0,
    log_to_file=False,
):
    """
    Train an XGBoost binary classifier on age buckets with a FIXED
    train/test split (studies already separated).
    """
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))

    total = len(y_train)
    n_classes = len(unique)
    if n_classes < 2:
        print(
            f"[WARN] Only {n_classes} class present in training labels "
            f"({unique}). Skipping training for {name}."
        )
        return np.nan, np.nan, np.nan, np.nan

    class_weights = {c: total / (n_classes * count) for c, count in class_counts.items()}
    sample_weight = np.array([class_weights[c] for c in y_train], dtype=np.float32)

    clf = XGBClassifier(
        n_estimators=35,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=8,
        random_state=seed,
    )

    clf.fit(X_train, y_train, sample_weight=sample_weight)

    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    auc_macro = float("nan")
    try:
        auc_macro = roc_auc_score(y_test, y_prob[:, 1])
    except Exception:
        pass

    report = classification_report(
        y_test,
        y_pred,
        digits=4,
        target_names=["<30", ">=30"],
    )

    print(f"\n=== {name} embeddings (XGBoost, age buckets) ===")
    print(f"Seed: {seed}")
    print("Train class distribution:")
    for c in sorted(class_counts.keys()):
        print(f"  class {c}: count={class_counts[c]}, weight={class_weights[c]:.3f}")
    print(f"Train size: {len(y_train)}, Test size (held-out study): {len(y_test)}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Macro F1:       {f1_macro:.4f}")
    print(f"AUC:            {auc_macro:.4f}")
    print("Classification report (held-out study):")
    print(report)

    return train_acc, test_acc, f1_macro, auc_macro


def repeated_xgb_eval_fixed_split(
    X_train,
    y_train,
    X_test,
    y_test,
    name,
    n_runs=N_RUNS,
    base_seed=BASE_SEED,
):
    """
    Only the summary across runs is written to RESULTS_FILE.
    """
    train_accs, test_accs, f1s, aucs = [], [], [], []

    for i in range(n_runs):
        seed = base_seed + i
        run_name = f"{name}_run{i}"
        train_acc, test_acc, f1, auc = train_and_eval_xgb_multiclass_fixed_split(
            X_train,
            y_train,
            X_test,
            y_test,
            name=run_name,
            seed=seed,
            log_to_file=False,
        )
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        f1s.append(f1)
        aucs.append(auc)

    train_accs = np.array(train_accs, dtype=np.float64)
    test_accs = np.array(test_accs, dtype=np.float64)
    f1s = np.array(f1s, dtype=np.float64)
    aucs = np.array(aucs, dtype=np.float64)

    train_acc_mean, train_acc_std = train_accs.mean(), train_accs.std(ddof=1)
    test_acc_mean, test_acc_std = test_accs.mean(), test_accs.std(ddof=1)
    f1_mean, f1_std = f1s.mean(), f1s.std(ddof=1)
    auc_mean = np.nanmean(aucs)
    auc_std = np.nanstd(aucs, ddof=1)

    print(f"\n=== SUMMARY over {n_runs} runs for {name} (held-out study) ===")
    print(f"Train Accuracy: {train_acc_mean:.4f} ± {train_acc_std:.4f}")
    print(f"Test Accuracy:  {test_acc_mean:.4f} ± {test_acc_std:.4f}")
    print(f"Macro F1:       {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"AUC:            {auc_mean:.4f} ± {auc_std:.4f}")

    with open(RESULTS_FILE, "a") as f:
        f.write("\n" + "-" * 70 + "\n")
        f.write(f"SUMMARY over {n_runs} runs for {name} (held-out study)\n")
        f.write(f"Train Accuracy: {train_acc_mean:.4f} ± {train_acc_std:.4f}\n")
        f.write(f"Test Accuracy:  {test_acc_mean:.4f} ± {test_acc_std:.4f}\n")
        f.write(f"Macro F1:       {f1_mean:.4f} ± {f1_std:.4f}\n")
        f.write(f"AUC:            {auc_mean:.4f} ± {auc_std:.4f}\n")

    return {
        "train_acc": (train_acc_mean, train_acc_std),
        "test_acc": (test_acc_mean, test_acc_std),
        "f1": (f1_mean, f1_std),
        "auc": (auc_mean, auc_std),
    }


def compute_pcoa_embeddings_for_samples(
    sample_ids,
    dm_path=DM_PATH,
    n_components=EMBED_DIM,
):
    """
    Returns:
      emb_all: np.ndarray of shape (N_samples, n_components)
               in the same order as sample_ids.
    """
    print(f"Loading distance matrix from: {dm_path}")
    df = pd.read_table(dm_path, sep="\t", index_col=0)

    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Distance matrix must be square, got {df.shape}")

    missing = [sid for sid in sample_ids if sid not in df.index]
    if missing:
        raise RuntimeError(
            f"{len(missing)} sample IDs from BIOM/rarefaction not found "
            f"in distance matrix. First few missing: {missing[:5]}"
        )

    df_sub = df.loc[sample_ids, sample_ids]

    dm = DistanceMatrix(df_sub.values, list(df_sub.index))
    print("Running PCoA on distance matrix ...")
    ord_res = pcoa(dm)

    coords = ord_res.samples

    k = min(n_components, coords.shape[1])
    coords = coords.iloc[:, :k]

    emb_all = coords.loc[sample_ids].to_numpy(dtype=np.float32)

    if k < n_components:
        print(
            f"[WARN] PCoA returned only {k} components; "
            f"padding to {n_components} with zeros."
        )
        pad = np.zeros((emb_all.shape[0], n_components - k), dtype=np.float32)
        emb_all = np.concatenate([emb_all, pad], axis=1)

    return emb_all


def eval_age_holdout_study():
    with open(RESULTS_FILE, "a") as f:
        f.write("\n" + "#" * 80 + "\n")
        f.write(f"NEW AGE EVAL RUN (held-out study, binary buckets) at {datetime.now()}\n")
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

    print(f"Loading age labels from: {AGE_PATH}")
    label_dict, study_dict = load_age_buckets_with_study(AGE_PATH)

    study_to_indices = {}
    study_to_labels = {}

    for i, sid in enumerate(sample_ids):
        if sid not in label_dict:
            continue
        study = study_dict.get(sid, get_study_id(sid))
        bucket = label_dict[sid]

        if study not in study_to_indices:
            study_to_indices[study] = []
            study_to_labels[study] = []
        study_to_indices[study].append(i)
        study_to_labels[study].append(bucket)

    total_labeled = sum(len(v) for v in study_to_indices.values())
    print(f"Total samples with age buckets (across all studies): {total_labeled}")
    if total_labeled == 0:
        raise RuntimeError(
            "No overlap between BIOM sample IDs and age_labels.txt SampleIDs (with valid ages)."
        )

    print("\nStudies with labeled samples and counts:")
    for study, idxs in study_to_indices.items():
        print(f"  Study {study}: {len(idxs)} samples")

    studies = list(study_to_indices.keys())
    held_out_study = min(studies, key=lambda s: len(study_to_indices[s]))
    held_out_count = len(study_to_indices[held_out_study])

    print(
        f"\nHeld-out study for TEST: {held_out_study} "
        f"(num labeled samples: {held_out_count})"
    )

    train_indices_list = []
    train_labels_list = []
    test_indices_list = []
    test_labels_list = []

    for study in studies:
        idxs = study_to_indices[study]
        labels = study_to_labels[study]
        if study == held_out_study:
            test_indices_list.extend(idxs)
            test_labels_list.extend(labels)
        else:
            train_indices_list.extend(idxs)
            train_labels_list.extend(labels)

    if len(test_indices_list) == 0:
        raise RuntimeError("Held-out study has zero labeled samples; cannot evaluate.")
    if len(train_indices_list) == 0:
        raise RuntimeError("No training samples outside the held-out study; cannot train classifier.")

    train_indices = np.array(train_indices_list, dtype=np.int64)
    test_indices = np.array(test_indices_list, dtype=np.int64)
    y_train = np.array(train_labels_list, dtype=np.int64)
    y_test = np.array(test_labels_list, dtype=np.int64)

    print(
        f"\nTrain samples (all other studies): {len(y_train)}, "
        f"Test samples (study {held_out_study}): {len(y_test)}"
    )

    print("\n--- Evaluating PCoA embeddings from distance matrix ---")
    emb_pcoa = compute_pcoa_embeddings_for_samples(
        sample_ids,
        dm_path=DM_PATH,
        n_components=EMBED_DIM,
    )

    X_train_pcoa = emb_pcoa[train_indices]
    X_test_pcoa = emb_pcoa[test_indices]

    repeated_xgb_eval_fixed_split(
        X_train_pcoa,
        y_train,
        X_test_pcoa,
        y_test,
        name=f"PCoA_distance_matrix_heldout_{held_out_study}",
    )

    if not os.path.isfile(CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    print(f"\n--- Evaluating {MODEL_NAME} (held-out study {held_out_study}) ---")
    model = MODEL_BUILDER().to(DEVICE)

    print(f"Loading checkpoint: {CKPT}")
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    emb_all = compute_embeddings(
        model,
        dataloader,
        encoder_type=ENCODER_TYPE,
    )

    X_train_emb = emb_all[train_indices]
    X_test_emb = emb_all[test_indices]

    repeated_xgb_eval_fixed_split(
        X_train_emb,
        y_train,
        X_test_emb,
        y_test,
        name=f"{MODEL_NAME}_heldout_{held_out_study}",
    )


if __name__ == "__main__":
    eval_age_holdout_study()

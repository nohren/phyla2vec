import os
import numpy as np
import pandas as pd
import biom
import torch
from torch.utils.data import DataLoader

# simply file to compute embeddings for all samples in same order as unifrac table 
# this matches biom table
# retain all data processing steps from training (rarefaction + subsampling)

from train import (
    SampleSequenceDataset,
    prepare_epoch_data,
    UniFracEncoder,
    SEQ_LEN,
    EMBED_DIM,
    MODEL_READS,
    MAX_SAMPLES,
    DEVICE,
    RAREFY_DEPTH
)

# ----- CONFIG -----
BIOM_PATH = "train_filtered/train.biom"          # same as training
CKPT_PATH = "pres_transformer_encoder_epoch_20.pt"       # your 20-epoch checkpoint
OUT_TSV = "embeddings_epoch20.tsv"
BATCH_SIZE = 8                              

@torch.no_grad()
def main():
    print(f"Loading BIOM table from {BIOM_PATH} ...")
    full_table = biom.load_table(BIOM_PATH)

    print("Preparing sample tokens (rarefaction + subsampling)...")
    sample_tokens, rare_table, sample_ids, obs_ids = prepare_epoch_data(
        full_table,
        depth=RAREFY_DEPTH,
        seq_len=SEQ_LEN,
        model_reads=MODEL_READS,
        max_samples=None,   # None to keep all samples
        seed=123,           # fixed so it’s reproducible
    )
    print(f"Num samples after filtering+subsample: {len(sample_ids)}")

    dataset = SampleSequenceDataset(sample_tokens)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # important to keep order to output tsv matching unifrac matrix
    )

    print(f"Loading UniFracEncoder checkpoint from {CKPT_PATH} ...")
    model = UniFracEncoder().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    N = len(dataset)
    emb_all = np.zeros((N, EMBED_DIM), dtype=np.float32)

    print("Computing embeddings...")
    for batch in dataloader:
        seq = batch["seq"].to(DEVICE)       # (B, R, L)
        idx = batch["idx"].cpu().numpy()    # (B,)
        emb = model(seq)                    # (B, D)
        emb_all[idx] = emb.cpu().numpy()

    # Build TSV compatible with Procrustes script
    cols = [f"e{i+1}" for i in range(emb_all.shape[1])]
    emb_df = pd.DataFrame(emb_all, index=sample_ids, columns=cols)
    emb_df.index.name = "#SampleID"

    print(f"Writing embeddings to {OUT_TSV} ...")
    emb_df.to_csv(OUT_TSV, sep="\t")
    print("Done.")

if __name__ == "__main__":
    main()

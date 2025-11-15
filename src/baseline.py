#!/usr/bin/env python

import argparse
import pandas as pd
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa


def main():
    p = argparse.ArgumentParser(
        description="Compute PCoA embeddings from a precomputed distance matrix"
    )
    p.add_argument("--dm_in", required=True,
                   help="Input distance matrix TSV (square, samples x samples)")
    p.add_argument("--n_components", type=int, default=10,
                   help="Number of PCoA axes to keep in the output")
    p.add_argument("--out", default="pcoa_embeddings.tsv",
                   help="Output TSV with sample embeddings")
    args = p.parse_args()

    # ---- 1. Load distance matrix ----
    # Assumes first column is sample IDs, header row has same IDs
    print(f"Loading distance matrix from {args.dm_in}")
    df = pd.read_table(args.dm_in, sep="\t", index_col=0)

    ids = df.index.tolist()
    data = df.values

    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Distance matrix must be square, got {df.shape}")

    print(f"n_samples in distance matrix: {len(ids)}")

    dm = DistanceMatrix(data, ids)

    # ---- 2. PCoA ----
    print("Running PCoA ...")
    ord_res = pcoa(dm)

    coords = ord_res.samples  # DataFrame: index = sample_id, columns = PC1, PC2, ...

    if args.n_components is not None:
        n = min(args.n_components, coords.shape[1])
        coords = coords.iloc[:, :n]

    # ---- 3. Save embeddings ----
    coords.index.name = "#SampleID"
    coords.to_csv(args.out, sep="\t")

    print(f"Saved PCoA embeddings for {coords.shape[0]} samples "
          f"with {coords.shape[1]} components to {args.out}")


if __name__ == "__main__":
    main()

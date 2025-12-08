import numpy as np
import pandas as pd
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
EMBEDDINGS_FP = "random_128d_embeddings.tsv" # need to mock this
UNIFRAC_FP    = "distance-matrix.tsv"
N_PROCRUSTES_DIMS = 128   # number of dims to align in Procrustes (<= 128 and <= n_samples-1)

# =====================
# 1. Load data
# =====================
# Embeddings file format (TSV example):
# #SampleID    e1    e2    ...    e128
# S1           ...
# S2           ...
emb_df = pd.read_csv(EMBEDDINGS_FP, sep="\t")
# assume first column is sample ID
emb_df = emb_df.set_index(emb_df.columns[0])
print("Embeddings shape:", emb_df.shape)  # (n_samples, 128)

# UniFrac distance matrix format (TSV):
# first row:   sampleID  S1    S2    ...
# subsequent:  S1        0     d12   ...
#              S2        d21   0     ...
dist_df = pd.read_csv(UNIFRAC_FP, sep="\t", index_col=0)
print("UniFrac matrix shape:", dist_df.shape)  # (n_samples, n_samples)

# =====================
# 2. Align sample IDs
# =====================
common_ids = emb_df.index.intersection(dist_df.index)
if len(common_ids) < 3:
    raise ValueError("Need at least 3 overlapping samples for Procrustes.")

dist_df = dist_df.loc[common_ids, common_ids]
emb_df = emb_df.loc[common_ids]

print("Number of common samples:", len(common_ids))
assert dist_df.index.equals(emb_df.index), "Sample ID order mismatch after alignment."


print("Any NaNs in full matrix?", dist_df.isna().any().any())
print("How many rows have NaNs?", dist_df.isna().any(axis=1).sum())
print("Symmetric (within 1e-12)?",
      np.allclose(dist_df.values, dist_df.values.T, atol=1e-12, equal_nan=True))


# =====================
# 3. PCoA on UniFrac distances
# =====================
dm = DistanceMatrix(dist_df.values, ids=dist_df.index.tolist())
pcoa_res = pcoa(dm)
pcoa_coords = pcoa_res.samples.values  # shape: (n_samples, n_dims_pcoa)

print("PCoA coord shape:", pcoa_coords.shape)

# =====================
# 4. Reduce embeddings to same dimension with PCA
# =====================
n_samples, emb_dim = emb_df.shape
n_pcoa_dim = pcoa_coords.shape[1]

k = min(N_PROCRUSTES_DIMS, emb_dim, n_pcoa_dim, n_samples - 1)
print(f"Using {k} dimensions for Procrustes.")

pca = PCA(n_components=k)
emb_k = pca.fit_transform(emb_df.values)      # (n_samples, k)
pcoa_k = pcoa_coords[:, :k]                   # take first k PCoA axes

# =====================
# 5. Procrustes alignment
# =====================
mtx1, mtx2, disparity = procrustes(pcoa_k, emb_k)
# mtx1: transformed pcoa_k
# mtx2: transformed emb_k

print("Procrustes disparity (lower = better alignment):", disparity)

# =====================
# 6. Optional: 2D visualization of aligned coordinates
# =====================
# Plot first 2 dimensions of the Procrustes-aligned configurations
if k >= 2:
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(mtx1[:, 0], mtx1[:, 1], marker='o', alpha=0.6, label="PCoA (UniFrac)")
    ax.scatter(mtx2[:, 0], mtx2[:, 1], marker='x', alpha=0.6, label="Your 128D embeddings (PCA+Procrustes)")

    ax.set_xlabel("PC 1 (aligned)")
    ax.set_ylabel("PC 2 (aligned)")
    ax.set_title(f"Procrustes alignment (disparity = {disparity:.4f})")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("k < 2, skipping 2D plot.")

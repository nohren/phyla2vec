#!/usr/bin/env python3
import sys
from collections import Counter

try:
    from biom import load_table
except ImportError:
    sys.exit("Missing dependency: pip install biom-format")

if len(sys.argv) < 2:
    sys.exit("Usage: python inspect_biom.py <table.biom>")

path = sys.argv[1]
tbl = load_table(path)

# pick up to 5 safely, even if fewer exist
k_obs = min(5, tbl.shape[0])
k_sam = min(5, tbl.shape[1])
sub_obs = list(tbl.ids(axis='observation')[:k_obs])
sub_samp = list(tbl.ids(axis='sample')[:k_sam])

submat = tbl.filter(sub_obs, axis='observation', inplace=False)
submat = submat.filter(sub_samp, axis='sample', inplace=False)

dense = submat.matrix_data.toarray() 
print(f"5x5 preview (features x samples) — actual: {submat.shape}")
print("features:", sub_obs)
print("samples :", sub_samp)
print(dense)

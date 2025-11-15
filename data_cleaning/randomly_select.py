#!/usr/bin/env python

import argparse
import random
from biom import load_table, Table
from biom.util import biom_open

def main():
    p = argparse.ArgumentParser(
        description="Randomly select up to N samples from a BIOM table."
    )
    p.add_argument("--in_biom", required=True, help="Input BIOM filepath")
    p.add_argument("--out_biom", default="subset_5000_samples.biom",
                   help="Output BIOM filepath (subset of samples)")
    p.add_argument("--n", type=int, default=5000,
                   help="Number of samples to randomly select (default: 5000)")
    args = p.parse_args()

    # 1. Load table
    table = load_table(args.in_biom)

    # 2. Get all sample IDs
    sample_ids = list(table.ids(axis='sample'))
    total = len(sample_ids)
    n_keep = min(args.n, total)

    print(f"Total samples in table: {total}")
    print(f"Randomly selecting: {n_keep}")

    # 3. Randomly choose sample IDs
    random.shuffle(sample_ids)
    keep_ids = sample_ids[:n_keep]

    # 4. Filter table to these samples
    subset = table.filter(keep_ids, axis='sample', inplace=False)

    # 5. Save new BIOM
    with biom_open(args.out_biom, 'w') as f:
        subset.to_hdf5(f, "random_5000_subset")

    print(f"Wrote subset BIOM to: {args.out_biom}")

if __name__ == "__main__":
    main()

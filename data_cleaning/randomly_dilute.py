#!/usr/bin/env python
# dilute large study 10317 from biom table to keep invariance

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
    p.add_argument("--target_study_id", required=True, help="Target study ID to dilute")

    non_target_ids = []
    target_ids = []

    args = p.parse_args()

    # 1. Load table
    table = load_table(args.in_biom)

    # 2. Get all sample IDs
    sample_ids = list(table.ids(axis='sample'))
    total = len(sample_ids)
    n_keep = min(args.n, total)

    print(f"Total samples in table: {total}")
    print(f"diluting to keep: {n_keep}")
    
    target_study_id = str(args.target_study_id)

    non_target_ids = []
    target_ids = []
    
    # shuffle sample ids
    random.shuffle(sample_ids)
    
    for sid in sample_ids:
        # if sid matches target study id, add to target_ids, else to non_target_ids
        # if count of target_ids < n_keep, stop adding to target_ids
        if sid.startswith(target_study_id) and len(target_ids) < n_keep:
            target_ids.append(sid)
        
        if not sid.startswith(target_study_id):
            non_target_ids.append(sid)
            
        
    # 3. Randomly choose sample IDs
    keep_ids = target_ids + non_target_ids
    random.shuffle(keep_ids)

    # 4. Filter table to these samples
    subset = table.filter(keep_ids, axis='sample', inplace=False)
    
    print(f"Total samples after dilution: {len(subset.ids(axis='sample'))}")
    print(f"Total target samples after dilution: {len(target_ids)}")
    print(f"Total non-target samples after dilution: {len(non_target_ids)}")

    # 5. Save new BIOM
    with biom_open(args.out_biom, 'w') as f:
        subset.to_hdf5(f, "random_5000_subset")

    print(f"Wrote subset BIOM to: {args.out_biom}")

if __name__ == "__main__":
    main()

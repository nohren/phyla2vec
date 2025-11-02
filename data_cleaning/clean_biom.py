#!/usr/bin/env python3
"""
Collapse technical replicates (keep highest-depth run) and remove blanks/controls.

Usage:
  python filter_blanks_replicates.py \
    --table roseburia_example.biom \
    --ambiguities roseburia_example.biom.ambiguities \
    --out keep_highest_depth.txt

Then subset:
  biom filter -i roseburia_example.biom \
    --sample-ids keep_highest_depth.txt \
    -o roseburia_dedup.biom
"""
import argparse
import json
import re
import sys

from matplotlib.pyplot import table
from biom import load_table
from biom.util import biom_open

def main():
    p = argparse.ArgumentParser(
        description="Keep highest-depth replicate per sample and drop blanks/controls."
    )
    p.add_argument("--table", required=True, help="Input BIOM table filepath")
    p.add_argument("--ambiguities", required=True, help=".ambiguities JSON from redbiom")
    p.add_argument("--out", default="keep_highest_depth.txt", help="Output keep-list (one sample ID per line)")
    p.add_argument(
        "--control-pattern",
        default=r"(blank|control|neg|ntc|water|reagent)",
        help="Regex (case-insensitive) to identify blanks/controls in sample IDs (default: '(blank|control|neg|ntc|water|reagent)')",
    )
    p.add_argument("--clip_count", default=5000, type=int, help="Minimum sequencing depth to keep a sample (default: 5000)")
    args = p.parse_args()

    # Compile control regex
    is_control = re.compile(args.control_pattern, re.I).search

    # Load table
    try:
        table = load_table(args.table)
    except Exception as e:
        sys.exit(f"[error] failed to load BIOM table '{args.table}': {e}")

    # Load ambiguities mapping
    try:
        with open(args.ambiguities) as f:
            ambig = json.load(f)
    except Exception as e:
        sys.exit(f"[error] failed to read ambiguities JSON '{args.ambiguities}': {e}")

    # Current sample IDs present in table
    present = set(table.ids(axis="sample"))

    keep = set()
    sample_to_idx = {sample_id: i for i, sample_id in enumerate(table.ids(axis="sample"))}
    sample_depths = table.sum(axis='sample')

    # Helper: total observation count (sequencing depth) for a sample
    def depth(sample_id: str) -> float:
        # table.sum(axis='sample', ids=[id]) returns a 1-element ndarray
        return float(sample_depths[sample_to_idx[sample_id]])
    
    # kept items from replicate sets
    kept_from_replicates = 0
    replicate_total = len([run for runs in ambig.values() for run in runs])
    dropped_controls_counts = 0
    for canon, runs in ambig.items():
        
        if is_control(canon):
            print(f"  skipping control run: {canon}")
            dropped_controls_counts += 1
            continue
        avail = [run for run in runs if run in present]
        if not avail:
            print(f"  no available samples for run: {canon}")
            continue
        best = max(avail, key=depth)
        if depth(best) < args.clip_count:
            print(f"  skipping low-depth run: {best} (depth: {depth(best)})")
            dropped_controls_counts += 1
            continue
        keep.add(best)

        kept_from_replicates += 1
    print(f"kept {kept_from_replicates} samples out of {replicate_total} ambiguous samples.")

    # subtract all ambiguous sample from present, then add back kept ones
    ambig_runs = {run for runs in ambig.values() for run in runs}
    unambiguous = present - ambig_runs

    added_unambiguous = 0
    
    for sample_id in unambiguous:
        if is_control(sample_id):
            print(f"skipping control run: {sample_id}")
            dropped_controls_counts += 1
            continue
        if depth(sample_id) < args.clip_count:
            print(f"skipping low-depth run: {sample_id} (depth: {depth(sample_id)})")
            dropped_controls_counts += 1
            continue
        keep.add(sample_id)
        added_unambiguous += 1

    print(f"Writing new table to: {args.out}")
    # 3) Write keep list
    filter_fn = lambda val, id_, md: id_ in keep
    new_table = table.filter(filter_fn, inplace=False)
    with biom_open(args.out, "w") as f:
        new_table.to_hdf5(f, "cleaned table")

    # 4) Print a small QC summary to stderr
    total_present = len(present)
    total_kept = len(keep)
    total_removed = total_present - total_kept
    print(
        f"        table samples present: {total_present}\n"
        f"        table unambiguous samples: {len(unambiguous)}\n"
        f"        table ambiguous samples: {replicate_total}\n"
        f"        kept from replicate groups: {kept_from_replicates}\n"
        f"        removed (controls + depth): {dropped_controls_counts}\n"
        f"        removed (controls + depth + non-kept replicates): {total_removed}\n",
        f"       new table samples size: {total_kept}\n",
        file=sys.stderr,
        
    )

if __name__ == "__main__":
    main()

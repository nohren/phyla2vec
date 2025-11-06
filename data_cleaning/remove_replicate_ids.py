
import argparse
import sys
from biom import load_table
from biom.util import biom_open

def main():
    p = argparse.ArgumentParser(
        description="Keep highest-depth replicate per sample and drop blanks/controls."
    )
    p.add_argument("--table", required=True, help="Input BIOM table filepath")
    p.add_argument("--out", default="keep_highest_depth.txt", help="Output keep-list (one sample ID per line)")
    args = p.parse_args()

    # Load table
    try:
        table = load_table(args.table)
    except Exception as e:
        sys.exit(f"[error] failed to load BIOM table '{args.table}': {e}")

    # Current sample IDs present in table and mapping to renamed IDs
    present = table.ids(axis="sample")
    mapping = {c: ".".join(c.split(".")[:-1]) for c in present}
    updated_table = table.update_ids(mapping, axis='sample', inplace=False)

    with biom_open(args.out, "w") as f:
        updated_table.to_hdf5(f, "remove_replicate_ids")


if __name__ == "__main__":
    main()
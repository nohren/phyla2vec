import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser(
        description="Clean metadata to only include samples kept after replicate-resolution."
    )
    p.add_argument("--sample_ids", required=True, help="Input sampleid filepath")
    p.add_argument("--meta", required=True, help="Input metadata filepath")
    p.add_argument("--out", default="meta_cleaned.txt", help="Output keep-list (one sample ID per line)")
    args = p.parse_args()
    # 1. Canonical sample IDs you kept after replicate-resolution
    kept_ids = None

    #read text from file into an array
    with open(args.sample_ids, "r") as f:
        kept_ids = pd.Index([line.strip() for line in f])

    # 2. Load metadata
    meta = pd.read_table(args.meta, sep='\t', dtype=str)
    # assume metadata has a sample ID column, e.g. 'sample_name' or '#SampleID'
    id_col = '#SampleID'  # adjust to whatever it is

    meta_samples = meta[id_col].astype(str)

    # 3. Filter metadata to only kept samples
    meta_filtered = meta[meta_samples.isin(kept_ids)].copy()

    print("Original metadata rows:", len(meta))
    print("After intersect with kept replicates:", len(meta_filtered))

    # 4. (Sanity) Check for biom-kept samples with missing metadata
    missing_meta = kept_ids.difference(meta_filtered[id_col])
    print("Kept samples with no metadata:", len(missing_meta))
    print(list(missing_meta[:10]))

    # remove replicate id form to become sample id form
    meta_filtered[id_col] = [".".join(c.split(".")[:-1]) for c in meta_filtered[id_col]]

    # 5. Save cleaned metadata
    meta_filtered.to_csv(args.out, sep='\t', index=False)

if __name__ == "__main__":
    main()
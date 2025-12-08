import argparse
import pandas as pd
from biom import load_table

continent_map = {
"PT": "Europe",
"CY": "Middle East",          # Cyprus – politically EU; you can relabel to 'Middle East' if you prefer
"AU": "Oceania",
"AT": "Europe",
"FI": "Europe",
"DE": "Europe",
"US": "North America",
"USA": "North America",
"MX": "North America",
"PL": "Europe",
"TH": "Asia",
"NZ": "Oceania",
"United Kingdom": "Europe",
"GB": "Europe",
"Israel": "Middle East",        # or 'Middle East' if you want that granularity
"FR": "Europe",
"not collected": "Unknown",
"BE": "Europe",
"MT": "Europe",
"SK": "Europe",
"HU": "Europe",
"JE": "Europe",          # Jersey
"SG": "Asia",
"PH": "Asia",
"PR": "North America",   # Puerto Rico (Caribbean, but NA for your purposes)
"IE": "Europe",
"CH": "Europe",
"ES": "Europe",
"JP": "Asia",
"SE": "Europe",
"CZ": "Europe",
"CA": "North America",
"NO": "Europe",
"IT": "Europe",
"NL": "Europe",
"DK": "Europe",
"CN": "Asia",
"RSA": "Africa",
"Ghana": "Africa",
"Jamaica": "North America",
}
 
#adapters
def acid_reflux_label(x: str) -> int:
    positive = {
        "Diagnosed by a medical professional (doctor, physician assistant)",
        "Diagnosed by an alternative medicine practitioner",
        "Self-diagnosed",
    }
    # x might be NaN, so guard with isinstance
    return 1 if isinstance(x, str) and x in positive else 0

def get_country(geo_loc_name):
    return geo_loc_name.split(":")[0].strip()

def geo_loc_name_label(x: str) -> str:
    country = get_country(x)
    return continent_map.get(country, "Other")

def main():
    p = argparse.ArgumentParser(
        description="Output sample,label pairs for samples in a BIOM table"
    )
    p.add_argument("--biom_in", required=True, help="Input BIOM filepath")
    p.add_argument("--meta_in", required=True, help="Input metadata filepath")
    p.add_argument("--label_id", required=True, help="Column name in metadata to use as label")
    p.add_argument("--out", default="labels.txt", help="Labels output filepath")
    args = p.parse_args()

    # 1. Load BIOM table and get sample IDs
    table = load_table(args.biom_in)
    biom_sample_ids = list(table.ids(axis='sample'))
    print("first id is ", biom_sample_ids[0])
    print("last id is ", biom_sample_ids[-1])
    biom_sample_set = set(biom_sample_ids)
    print("Samples in BIOM table:", len(biom_sample_ids))

    # 2. Load metadata
    meta = pd.read_table(args.meta_in, sep='\t', dtype=str)

    # Assume metadata has a sample ID column named '#SampleID'
    id_col = '#SampleID'
    if id_col not in meta.columns:
        raise ValueError(f"Metadata is missing required ID column '{id_col}'")

    if args.label_id not in meta.columns:
        raise ValueError(f"Metadata is missing label column '{args.label_id}'")

    # 3. Keep only metadata rows whose IDs are in the BIOM table
    # i might wanted to start with which labels are the most common... oh well
    in_biom_mask = meta[id_col].isin(biom_sample_set)
    meta_filtered = meta.loc[in_biom_mask, [id_col, args.label_id]].copy()

    print("Metadata rows total:", len(meta))
    print("Metadata rows with IDs in BIOM:", len(meta_filtered))

    # 4. Align order to BIOM sample order
    meta_filtered = (
        meta_filtered
        .set_index(id_col)
        .reindex(biom_sample_ids)  # rows for samples not in metadata become NaN
    )

    missing_meta_ids = meta_filtered[meta_filtered[args.label_id].isna()].index.tolist()
    if missing_meta_ids:
        print("WARNING: samples in BIOM with no metadata label:", len(missing_meta_ids))
        print("First few missing:", missing_meta_ids[:10])

    if args.label_id == "acid_reflux":
        meta_filtered[args.label_id] = meta_filtered[args.label_id].map(acid_reflux_label)
        
    if args.label_id == "geo_loc_name":
        # Simplify to continent
        meta_filtered[args.label_id] = meta_filtered[args.label_id].map(geo_loc_name_label)

    # Drop samples without labels
    meta_filtered = meta_filtered.dropna(subset=[args.label_id])

    # 5. Reset index and rename label column to 'label'
    meta_filtered = meta_filtered.reset_index()
    meta_filtered.rename(columns={args.label_id: "label", id_col: id_col}, inplace=True)

    # 6. Save sample_id,label pairs
    meta_filtered.to_csv(args.out, sep='\t', index=False)
    print(f"Wrote {len(meta_filtered)} sample,label pairs to {args.out}")


if __name__ == "__main__":
    main()
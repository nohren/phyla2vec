#!/usr/bin/env bash

echo "fetch stool samples per ctx"

while IFS= read -r ctx; do
  # Extract everything between '16S-' and '-150nt' (e.g., V3, V1-V3, V3V4, V4-V5, etc.)
  region=$(printf "%s" "$ctx" | perl -ne 'print $1 if /16S-(.*?)-150nt/')

  # Fallback if the pattern isn't found
  if [ -z "$region" ]; then
    region=$(printf "%s" "$ctx" | tr -cd 'A-Za-z0-9._-')
  fi
  dir_name="${region}_stool"
  prefix="data/$dir_name"
  mkdir -p "$prefix"
  out="$prefix/${region}_stool.biom"
  echo "ctx=$ctx → region=$region → $out"

  redbiom search metadata "where sample_type in ('Stool','stool')" \
    | redbiom fetch samples --context "$ctx" --output "$out"

  biom summarize-table -i "$out" -o "$prefix/${region}_summary.txt"
done < data/ctx_list.txt
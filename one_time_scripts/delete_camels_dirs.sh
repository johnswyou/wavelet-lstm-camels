#!/usr/bin/env bash
# delete_camels_dirs.sh
# Deletes ~/scratch/<code>/ for every <code> found in csv_filenames_v4.txt

set -euo pipefail                       # fail fast on any error
mapfile -t codes < <(                   # read codes into an array
  awk -F'_' '{print $1}' csv_filenames_v4.txt
)

scratch_dir="$HOME/scratch"

for code in "${codes[@]}"; do
  target="$scratch_dir/$code"
  if [[ -d "$target" ]]; then
    echo "Removing directory: $target"
    rm -rf -- "$target"
  else
    echo "Directory not found (skipping): $target"
  fi
done

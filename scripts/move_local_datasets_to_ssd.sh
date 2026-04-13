#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/Users/jeongjihoon/Desktop/Capstone"
DST_ROOT="/Volumes/SAMSUNG/CapstoneData/local_datasets"
STAMP="20260407"

DATASETS=(
  "UCF101-Action Recognition"
  "VoxCeleb2"
  "FaceForensics++_C23"
  "Celeb_V2"
  "gen_img"
)

mkdir -p "$DST_ROOT"

count_files() {
  local dir="$1"
  find "$dir" -type f ! -name '._*' | wc -l | tr -d ' '
}

for name in "${DATASETS[@]}"; do
  src="$SRC_ROOT/$name"
  dst="$DST_ROOT/$name"

  if [[ ! -e "$src" ]]; then
    echo "[WARN] missing source, skip: $src"
    continue
  fi

  if [[ -L "$src" ]]; then
    echo "[INFO] already symlinked, skip: $src -> $(readlink "$src")"
    continue
  fi

  echo "[INFO] syncing: $src -> $dst"
  mkdir -p "$dst"
  rsync -a "$src/" "$dst/"

  src_count=$(count_files "$src")
  dst_count=$(count_files "$dst")
  echo "[INFO] verify file_count src=$src_count dst=$dst_count"
  if [[ "$src_count" != "$dst_count" ]]; then
    echo "[ERROR] file count mismatch for $name"
    exit 1
  fi

  echo "[INFO] replacing local directory with symlink: $src"
  rm -rf "$src"
  ln -s "$dst" "$src"
  echo "[INFO] done: $src -> $dst"
  echo "---"
done

echo "[INFO] completed dataset move to SSD"

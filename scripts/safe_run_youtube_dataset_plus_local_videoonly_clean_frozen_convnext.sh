#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export KMP_DUPLICATE_LIB_OK=TRUE
export KMP_USE_SHM=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

CFG="${1:-configs/protocol_youtube_dataset_plus_local_videoonly_clean_frozen_convnext.yaml}"

python train.py --config "$CFG"
python eval.py --config "$CFG" --split test
python scripts/eval_lowfpr_calibration.py --config "$CFG" --split test

echo "[INFO] clean frozen convnext run complete"

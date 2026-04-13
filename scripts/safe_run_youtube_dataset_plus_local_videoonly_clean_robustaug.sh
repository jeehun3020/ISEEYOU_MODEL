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

CONFIG="${1:-configs/protocol_youtube_dataset_plus_local_videoonly_clean_robustaug.yaml}"

python train.py --config "$CONFIG"
python eval.py --config "$CONFIG" --checkpoint outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_frame/best.pt --split test
python scripts/eval_lowfpr_calibration.py --config "$CONFIG" --checkpoint outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_frame/best.pt --split test

echo "[INFO] video-only clean robustaug complete for $CONFIG"

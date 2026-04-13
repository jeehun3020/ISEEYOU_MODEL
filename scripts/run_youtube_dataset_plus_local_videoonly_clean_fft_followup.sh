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

ROBUST_CFG="configs/protocol_youtube_dataset_plus_local_videoonly_clean_robustaug.yaml"
ROBUST_CKPT="outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_frame/best.pt"

AMP_CFG="configs/protocol_youtube_dataset_plus_local_videoonly_clean_fft_amplitude.yaml"
AMP_CKPT="outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_fft_amplitude/best.pt"

PHASE_CFG="configs/protocol_youtube_dataset_plus_local_videoonly_clean_fft_phase.yaml"
PHASE_CKPT="outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_fft_phase/best.pt"

if [[ ! -f "$ROBUST_CKPT" ]]; then
  echo "[ERROR] missing robust checkpoint: $ROBUST_CKPT" >&2
  exit 1
fi

if [[ -f "$AMP_CKPT" ]]; then
  python scripts/tune_multicue_ensemble.py \
    --component robust:frame:"$ROBUST_CFG":"$ROBUST_CKPT" \
    --component fft_amp:frame:"$AMP_CFG":"$AMP_CKPT" \
    --val-split val \
    --test-split test \
    --samples 600 \
    --monitor f1 \
    --output-json outputs/eval/multicue_videoonly_clean_robust_fft_amp.json
else
  echo "[WARN] skipping robust+fft_amp; checkpoint not found: $AMP_CKPT"
fi

if [[ -f "$PHASE_CKPT" ]]; then
  python scripts/tune_multicue_ensemble.py \
    --component robust:frame:"$ROBUST_CFG":"$ROBUST_CKPT" \
    --component fft_phase:frame:"$PHASE_CFG":"$PHASE_CKPT" \
    --val-split val \
    --test-split test \
    --samples 600 \
    --monitor f1 \
    --output-json outputs/eval/multicue_videoonly_clean_robust_fft_phase.json
else
  echo "[WARN] skipping robust+fft_phase; checkpoint not found: $PHASE_CKPT"
fi

python scripts/tune_final_video_pipeline.py \
  --primary-config configs/protocol_youtube_dataset_plus_local_videoonly_clean_robustaug.yaml \
  --primary-checkpoint outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_frame/best.pt \
  --verifier-config configs/protocol_youtube_dataset_plus_local_videoonly_clean_frozen_convnext.yaml \
  --verifier-checkpoint outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_frozen_convnext_frame/best.pt \
  --val-split val \
  --test-split test \
  --output-json outputs/eval/final_video_pipeline_policy.json

echo "[INFO] fft follow-up complete"

#!/bin/zsh
set -euo pipefail

ROOT="/Users/jeongjihoon/Desktop/Capstone"
cd "$ROOT"

source .venv/bin/activate

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/youtube_mega_context_bundle_${timestamp}.log"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}" | tee -a "$master_log"
  set +e
  "$@" 2>&1 | tee -a "logs/${name}_${timestamp}.log" | tee -a "$master_log"
  local cmd_status=$?
  set -e
  if [[ $cmd_status -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAIL ${name} exit=${cmd_status}" | tee -a "$master_log"
    exit "$cmd_status"
  fi
  echo "[$(date '+%F %T')] DONE ${name}" | tee -a "$master_log"
}

run_step mega_face_blackout_prepare \
  python prepare_data.py \
  --config configs/baseline_binary_allvideo_textmask_protocol_youtube_mega_face_blackout.yaml

run_step mega_face_blackout_train \
  python train.py \
  --config configs/baseline_binary_allvideo_textmask_protocol_youtube_mega_face_blackout.yaml

run_step mega_face_blackout_eval \
  python eval.py \
  --config configs/baseline_binary_allvideo_textmask_protocol_youtube_mega_face_blackout.yaml \
  --checkpoint outputs/checkpoints_allvideo_binary_textmask_protocol_youtube_mega_face_blackout/best.pt \
  --split test

run_step mega_background_only_prepare \
  python prepare_data.py \
  --config configs/baseline_binary_allvideo_textmask_protocol_youtube_mega_background_only.yaml

run_step mega_background_only_train \
  python train.py \
  --config configs/baseline_binary_allvideo_textmask_protocol_youtube_mega_background_only.yaml

run_step mega_background_only_eval \
  python eval.py \
  --config configs/baseline_binary_allvideo_textmask_protocol_youtube_mega_background_only.yaml \
  --checkpoint outputs/checkpoints_allvideo_binary_textmask_protocol_youtube_mega_background_only/best.pt \
  --split test

run_step mega_framediff_temporal_train \
  python train_temporal.py \
  --config configs/temporal_binary_allvideo_textmask_protocol_youtube_mega_framediff.yaml

run_step mega_framediff_temporal_eval \
  python eval_temporal.py \
  --config configs/temporal_binary_allvideo_textmask_protocol_youtube_mega_framediff.yaml \
  --checkpoint outputs/checkpoints_temporal_allvideo_binary_textmask_protocol_youtube_mega_framediff/best.pt \
  --split test

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$master_log"

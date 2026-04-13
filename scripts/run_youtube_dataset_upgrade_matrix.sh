#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
YouTube Dataset Upgrade Matrix

1) Baseline full-frame:
   configs/protocol_youtube_dataset_fullframe_baseline.yaml

2) Robust spatial regularization:
   configs/protocol_youtube_dataset_fullframe_robustaug.yaml

3) Temporal conv head:
   configs/protocol_youtube_dataset_temporalconv.yaml

4) Frozen ConvNeXt shallow head:
   configs/protocol_youtube_dataset_frozen_convnext.yaml

5) FFT amplitude:
   configs/protocol_youtube_dataset_fft_amplitude.yaml

6) FFT phase:
   configs/protocol_youtube_dataset_fft_phase.yaml

Recommended order:
  a. scripts/run_youtube_dataset_protocol_preflight.sh configs/protocol_youtube_dataset_fullframe_baseline.yaml
  b. python train.py --config configs/protocol_youtube_dataset_fullframe_baseline.yaml
  c. python eval.py --config configs/protocol_youtube_dataset_fullframe_baseline.yaml --split test
  d. python train.py --config configs/protocol_youtube_dataset_fullframe_robustaug.yaml
  e. python train_temporal.py --config configs/protocol_youtube_dataset_temporalconv.yaml
EOF

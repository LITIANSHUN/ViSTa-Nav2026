#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/train/checkpoint/best.pt

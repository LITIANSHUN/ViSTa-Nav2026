#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python scripts/inspect_dataset.py --data-root /home/tianshun/Downloads/aeronet_repro_code/data/data
python scripts/train.py --config configs/default.yaml

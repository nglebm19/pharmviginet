#!/usr/bin/env bash
# Train text + mol models. Run inside tmux/screen so SSH disconnect is safe.
#
# Usage:
#   bash scripts/run_pipeline.sh                  # full training
#   bash scripts/run_pipeline.sh --sample 500000  # quick run
#   bash scripts/run_pipeline.sh --only text      # text model only

set -euo pipefail

python -m pharmviginet.train.train "$@"

#!/usr/bin/env bash
# The two module-2 experiments: dissimilarity threshold and template transfer.
#
# Usage (runnable from anywhere — it cd's to the repo root itself):
#   bash scripts/experiments/run_experiments.sh
#
# Both reuse the detectors saved by the main runs and fit any that are missing,
# so run the replicates first to control how they are trained:
#   python scripts/anomaly_detection/run_replicates.py --dataset both --methods ...

set -euo pipefail

# Anchor to the repo root (this script lives in scripts/experiments/).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

step() { echo; echo "── $1 ──────────────────────────────────────────────"; }

START=$(date +%s)

step "1/2  Dissimilarity threshold"
uv run python -m src.experiments.threshold

step "2/2  Template transfer (NHTS -> Geolife)"
uv run python -m src.experiments.transfer

echo
echo "Experiments complete in $(( $(date +%s) - START ))s"

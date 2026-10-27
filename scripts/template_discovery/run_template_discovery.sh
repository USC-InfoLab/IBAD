#!/usr/bin/env bash
# Behavioural template discovery (module 1) for every dataset.
#
# Usage (runnable from anywhere — it cd's to the repo root itself):
#   bash scripts/template_discovery/run_template_discovery.sh           # fit at the configured K
#   bash scripts/template_discovery/run_template_discovery.sh --sweep   # also run the K sweep
#
# Prerequisites:
#   nhts    : data/raw/nhts_2017/trippub.csv
#   numosim : none — parquet files download from OSF on first run
#   geolife : module 0 output (bash scripts/geolife_poi_attribution/run_geolife_poi_attribution.sh)

set -euo pipefail

# Anchor to the repo root (this script lives in scripts/template_discovery/).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

step() { echo; echo "── $1 ──────────────────────────────────────────────"; }

START=$(date +%s)

for dataset in nhts numosim geolife; do
    step "$dataset"
    uv run python -m src.templates.discover --dataset "$dataset" "$@"
done

echo
echo "Template discovery complete in $(( $(date +%s) - START ))s"

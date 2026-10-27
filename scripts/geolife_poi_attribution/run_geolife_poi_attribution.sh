#!/usr/bin/env bash
# Geolife stay-point extraction + LLM labelling pipeline (module 0).
#
# Usage (runnable from anywhere — it cd's to the repo root itself):
#   bash scripts/geolife_poi_attribution/run_geolife_poi_attribution.sh
#
# Authentication (pick one before running — see README.md for details):
#   API key:   export GEMINI_API_KEY=...
#   Vertex AI: set gcp_project/gcp_location in configs/data/geolife_poi.yaml
#              and export GCS_BUCKET=...

set -euo pipefail

# Anchor to the repo root (this script lives in scripts/geolife_poi_attribution/).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

step() { echo; echo "── $1 ──────────────────────────────────────────────"; }
elapsed() { echo "done in $(( $(date +%s) - START ))s"; }

START=$(date +%s)

step "1/4  Extract stay-points"
uv run python -m src.data.geolife.extract_staypoints
elapsed

step "2/4  Fetch OSM primary tags (Nominatim)"
uv run python -m src.data.geolife.fetch_osm_primary
elapsed

step "3/4  Fetch OSM neighbors (Overpass)"
uv run python -m src.data.geolife.fetch_osm_neighbors
elapsed

step "4/4  Label with LLM (NHTS, pass 1 + 2)"
uv run python -m src.data.geolife.label_with_llm
elapsed

echo
echo "Pipeline complete in $(( $(date +%s) - START ))s"

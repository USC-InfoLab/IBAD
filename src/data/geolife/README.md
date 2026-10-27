# Module 0 — Geolife POI Extraction

Turns the raw [Microsoft Geolife GPS Trajectories v1.3](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
archive into a **labelled stay-point dataset**: each GPS stay-point is
reverse-geocoded against OpenStreetMap, then classified into one of the 18
[NHTS 2017](https://nhts.ornl.gov/) activity categories by an LLM judge (Gemini 2.0).

The output dataset will be used by behavioral template
discovery module.

---

## Pipeline

| # | Stage | Script | Service | Output |
|---|-------|--------|---------|--------|
| 1 | Extract stay-points | `extract_staypoints.py` | — | `data/interim/geolife/staypoints.tsv` |
| 2 | OSM primary tag | `fetch_osm_primary.py` | Nominatim (1 req/s) | `data/interim/osm/osm_cache.sqlite` |
| 3 | OSM neighbors | `fetch_osm_neighbors.py` | Overpass (~1 req/s) | `data/interim/osm/osm_cache.sqlite` |
| 4 | LLM label | `label_with_llm.py` | Gemini API | `data/interim/geolife/geolife_nhts_llm_labeled.tsv` |
| 5 | Split (optional) | `split_dataset.py` | — | `data/processed/geolife/` |

- **Output level:** stay-point. The labelled file (stage 4) lives in
  `data/interim/` and is the canonical input for downstream modules.
- **`data/processed/`** holds only the optional per-agent chronological
  80/10/10 train/val/test split (stage 5).
- Stages 2–4 are **resumable** — cached results (SQLite) are skipped on re-runs,
  so interrupted jobs continue without re-querying APIs.

---

## Setup

### 1 · Raw data

Download Geolife Trajectories 1.3 and extract so that:

```
data/raw/Geolife Trajectories 1.3/Data/000/Trajectory/…
```

### 2 · Dependencies

Managed with [uv](https://docs.astral.sh/uv/) (see the
[root README](../../../README.md)):

```bash
uv sync                 # core
uv sync --extra vertex  # + google-cloud-storage for Vertex AI batch mode
```

### 3 · Gemini authentication (stage 4 only — pick ONE)

Stages 1–3 need no LLM credentials. Only stage 4 (`label_with_llm`) needs
one of:

**API key (real-time calls):**

```bash
export GEMINI_API_KEY=your_key_here
```

**Vertex AI + GCS batch (≈50 % cheaper at scale):**

```bash
gcloud auth application-default login
export GCS_BUCKET=your-gcs-bucket
# set gcp_project / gcp_location in configs/data/geolife_poi.yaml
```

The labelling stage auto-detects which is active: if `GEMINI_API_KEY` (or
`GOOGLE_API_KEY`) is set it uses real-time calls; otherwise it requires
`GCS_BUCKET` and submits Vertex AI batch jobs. Setting neither makes stage 4
exit with an error telling you to pick one.

---

## Running

```bash
# all stages 1–4, from anywhere (script cd's to repo root)
bash scripts/geolife_poi_attribution/run_geolife_poi_attribution.sh
```

Individual stages:

```bash
uv run python -m src.data.geolife.extract_staypoints
uv run python -m src.data.geolife.fetch_osm_primary
uv run python -m src.data.geolife.fetch_osm_neighbors
uv run python -m src.data.geolife.label_with_llm
uv run python -m src.data.geolife.split_dataset   # optional
```

---

## Configuration

All tunable parameters live in **`configs/data/geolife_poi.yaml`** — no constant
is hardcoded in any script. Key groups: stay-point extraction thresholds, OSM nearby-place search radius (radius_primary_m, radius_fallback_m) / precision, and LLM labelling (`llm_model`, `max_workers`,
`top_k_frequent`).

---

## Output columns (`geolife_nhts_llm_labeled.tsv`)

| Column | Description |
|--------|-------------|
| `AgentID` | Geolife user ID |
| `ArrivingTime` / `LeavingTime` | Stay start / end (Asia/Shanghai, tz-aware) |
| `Latitude` / `Longitude` | Stay centroid |
| `Duration_Min` | Dwell time (minutes) |
| `ActivityCategory` | NHTS 2017 label assigned by the LLM |
| `LLMReasoning` | One-sentence justification |
| `Confidence` | `high` (1 candidate) / `low` (2 candidates) |
| `Pass` | `1` or `2` — which LLM pass produced the final label |

The 18 NHTS categories: `Home`, `Work`, `Education`, `Shopping`, `Restaurant`,
`Exercise`, `Health`, `Recreation`, `Social`, `Religious`, `Services`,
`Errands`, `Childcare`, `Adultcare`, `Volunteer`, `Pickup_Dropoff`,
`Transportation`, `Other`.

---

## LLM classification

`src/utils/nhts_llm_judge.py` implements a two-pass LLM-as-judge classifier:

- **Pass 1** — every agent-day goes to Gemini with its chronological stay
  sequence (OSM primary tag + top-3 nearest POIs), the agent's top-5 most-visited
  locations with temporal fingerprints, and transit distance/time between stays.
- **Pass 2** — stays labelled `Other` in pass 1 are re-submitted with the pass-1
  modal label per frequent location as a prior, to resolve ambiguous cases.

All responses are cached in SQLite (`nhts_llm_cache.sqlite`); re-runs make zero
API calls for already-labelled agent-days.

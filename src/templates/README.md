# Module 1 — Template discovery

Learns **behavioural templates**: recurring daily activity patterns, discovered
without labels.

Each agent-day is encoded as a document of `(time slot, activity)` tokens, and
LDA is fit over those documents. Every topic is a template, and every day gets a
*soft membership* — a mixture over templates rather than one hard cluster id.

```
Activity labelled mobility data
      │
      ▼  src/data/*         load + harmonise activity labels
      ▼  src/features/*     encode each agent-day as a one-hot document
      ▼  src/templates/lda  fit LDA (K topics = K templates)
      ▼  src/templates/*    figures, artifacts
```

---

## Datasets

| Dataset | Documents | Vocabulary | Needs |
|---|---|---|---|
| `nhts` | ~213k person-days | 18 activities | `data/raw/nhts_2017/trippub.csv` |
| `numosim` | agent-days in the train window | POI activity *bundles* | nothing — downloads from OSF on first run |
| `geolife` | agent-days in the train split | activities present in the split | module 0 output in `data/processed/geolife/` |

Each dataset contributes one adapter under `src/features/` returning the same
`Documents` object — a slot-major one-hot matrix, its vocabulary, and an
agent-day index — so the discovery core never branches on the dataset. Adding a
dataset means writing one adapter and one config entry, nothing else.

Columns are laid out slot-major (`column = slot * n_acts + activity`), which is
what lets any topic matrix reshape to `(K, n_slots, n_acts)` for plotting.

---

## Running

```bash
# every dataset, fitting at the configured K
bash scripts/template_discovery/run_template_discovery.sh

# one dataset, plus the perplexity sweep over K
uv run python -m src.templates.discover --dataset geolife --sweep
```

`notebooks/01_template_discovery.ipynb` steps through the same calls with
commentary. All logic lives here in `src/`, so the notebook and the CLI produce
identical artifacts.

---

## Module layout

| File | Role |
|---|---|
| `lda.py` | fit, transform, topic-word matrix, K sweep |
| `diagnostics.py` | membership statistics and per-topic usage |
| `artifacts.py` | save/load a fitted run |
| `plots.py` | topic heatmaps, sparklines, elbow, membership histograms |
| `discover.py` | orchestrates the above; CLI entry point |

---

## Configuration

Everything tunable lives in `configs/templates/default.yaml`: K per dataset, the
sweep range, LDA settings, and output paths.

```yaml
seed: 42
n_slots: 96            # 15-minute slots per day
lda:
  max_iter: 3          # low so the 13-point K sweep stays tractable
  batch_size: 4096
k_sweep: {k_min: 3, k_max: 15}
```

Raise `max_iter` for a final publication-quality fit. Per-dataset keys override
the shared ones — Geolife sets `batch_size: 64` because it only has a few
thousand documents.

---

## Outputs

| Path | Contents |
|---|---|
| `artifacts/templates/<dataset>_lda.pkl` | fitted model, per-day mixtures (`theta`), agent-day index, vocabulary |
| `output/template_discovery/<dataset>_templates.png` | one heatmap per template: activities × time of day |
| `output/template_discovery/<dataset>_membership.png` | membership entropy and top-weight distributions |
| `output/template_discovery/<dataset>_k_sweep.{csv,png}` | perplexity elbow (only with `--sweep`) |

The vocabulary is stored beside the model because a topic matrix is only
interpretable against the column order it was fitted on.

---

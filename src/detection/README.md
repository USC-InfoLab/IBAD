# Module 2 — Anomaly detection

Trains **IBAD**, a hierarchical behavioural forecaster, to predict what an
agent's expected behavioral template mixture in a target day should looks like. A day is anomalous when what actually happened diverges from the expectation.

```
visit sequence in a day   ──►  visit embedder + intra-day encoder  ──►  daily representation.
Sequence of daily encodings of one agent   ──►  causal continuous-time transformer conditioned on target day ──►  Predicted target day mixture over k templates.
prediction vs observed ─────────────────────────────────────────►  anomaly score
```
---

## Anomaly score

A day's anomaly score is the KL divergence of its observed mixture `θ` from the
template mixture  `p`, KL(θ ‖ p).

Scores are always z-scored per agent before ranking. Without it a pooled AUROC
partly ranks *agents* rather than days: people differ in how many stays they log
and how regular they are, so their score distributions sit at different offsets.
Statistics come from each agent's training-window days, so an injected day never
shifts its own baseline.

AUROC and AP are threshold-free. F1 is measured on test at the threshold that
maximises F1 on validation, which is spliced the same way as test.

---

## Benchmark

Both datasets use the same splice benchmark: one random test day per agent takes a donor day's visits (activities, arrival and departure times). The day keeps its own date, weekday and agent id. The donor depends on the policy:

| Policy | Description |
|---|---|
| `intra` | the same agent's opposite week-part (a weekday day receiving a weekend, or the reverse) |
| `inter` | a different agent's day regardless of weekday or weekend.|
| `combined` | a fair coin flip per agent, either do `intra` or `inter`. |

A swap day qualifies only if its template mixture is at least $\epsilon$ nats
of Jensen–Shannon divergence from the target day's to avoid classifying near-identical behaviors as anomalies.

**Why NUMOSIM is spliced too.** NUMOSIM's own anomalies (`test_anomalous`) are
single unusual stays, anomalous in place or time. IBAD targets daily behvaioral anomalies in a day-level, so it is spliced.

---

## Running

```bash
uv sync --extra notebook --extra detection     # torch + hmmlearn; keep notebook if you use the kernel

# one method, one dataset, one seed (--method defaults to ibad)
uv run python -m src.detection.run --dataset geolife --seed 42

# five seeds (42–46) with mean ± sd; --dataset both runs Geolife then NUMOSIM
uv run python scripts/anomaly_detection/run_replicates.py --dataset geolife
uv run python scripts/anomaly_detection/run_replicates.py --dataset geolife \
    --methods ibad histmean ghmm iforest ocsvm deepsvdd bestad tod4traj icad
```

Both commands take `--retrain` and `--device` (GPU by default when available).
On a GPU, IBAD trains in ~2 min per seed on Geolife and much longer on NUMOSIM.
Scoring a saved model takes seconds on Geolife and a few minutes on NUMOSIM,
plus ~5 min to build its cohort. Each seed draws its own splices, so the spread
covers model and benchmark variance.

---

## Module layout

| File | Role |
|---|---|
| `src/models/ibad.py` | the IBAD model and its anomaly score. |
| `src/features/visits.py` | per-agent visit sequences and the chronological split. |
| `src/detection/data.py` | builds each dataset's cohort: visit sequences and template mixtures. |
| `src/detectors/` | the `Detector` interface, IBAD's wrapper, and the baselines. |
| `src/training/dataset.py` | per-agent batch data loader. |
| `src/training/loop.py` | KL training loop, keeping the best-validation weights |
| `src/evaluation/benchmark.py` | the splice benchmark: which days are spliced, and the cohort with the splices in place |
| `src/evaluation/scoring.py` | per-agent standardisation, AUROC / AP / F1. |
| `src/detection/run.py` | one (method, dataset, seed) end to end; CLI entry point |
| `scripts/anomaly_detection/run_replicates.py` | several methods and seeds, and the mean ± sd. |

---

## Outputs

| Path | Contents |
|---|---|
| `artifacts/detection/<dataset>_<method>_seed<N>.*` | a fitted detector |
| `output/detection/<dataset>_<method>_seed<N>.json` | per-policy metrics for one run, with the validation-tuned thresholds |
| `output/detection/replicates.csv` | one row per (method, dataset, seed, policy), rebuilt from all the run files |
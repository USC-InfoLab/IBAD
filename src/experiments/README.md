# Experiments

Three experiments, all run from the repo root. Each needs module 1's templates
for the datasets it uses (`python -m src.templates.discover --dataset <name>`).

Methods: `ibad histmean ghmm iforest ocsvm deepsvdd bestad tod4traj icad trajgen`.
Datasets: `geolife`, `numosim`.

A fitted model is saved under `artifacts/detection/` and reused by every later
run; add `--retrain` to refit it.

---

## 1 · Main experiment (ε = 0.05)

The splice threshold ε is `min_shift` in `configs/detection/benchmark.yaml`
(0.05 for both datasets).

One method, one seed:

```bash
uv run python -m src.detection.run --method bestad --dataset geolife --seed 42
```

Several methods, seeds 42–46, with mean ± sd:

```bash
uv run python scripts/anomaly_detection/run_replicates.py --dataset geolife \
    --methods ibad histmean ghmm iforest ocsvm deepsvdd bestad tod4traj icad trajgen
```

`--dataset both` runs Geolife and NUMOSIM; `--seeds` overrides the seeds.
TrajGenAgent needs its LLM chains generated first; see
[src/detectors/](../detectors/README.md#trajgenagents-generation-step).

Outputs in `output/detection/`: one `<dataset>_<method>_seed<N>.json` per run
and `replicates.csv` for all of them.

---

## 2 · Effect of ε

Scores the saved detectors at every ε in `configs/experiments/threshold.yaml`
(`epsilons`, `methods`, `datasets`), so run the main experiment first.

All datasets and methods in the config:

```bash
uv run python -m src.experiments.threshold
```

A subset:

```bash
uv run python -m src.experiments.threshold --datasets geolife --methods ibad histmean
```

Redraw the figure from the saved tables only:

```bash
uv run python -m src.experiments.threshold --plot-only
```

Outputs in `output/experiments/threshold/`: `threshold_<dataset>.csv` and
`threshold.{png,pdf}`.

---

## 3 · Transferability (NHTS → Geolife)

Needs NHTS and Geolife templates. Source, target and settings are in
`configs/experiments/transfer.yaml`.

```bash
uv run python -m src.experiments.transfer                  # alignment and utility
uv run python -m src.experiments.transfer --part alignment # template alignment only, no training
uv run python -m src.experiments.transfer --part utility   # IBAD on NHTS templates vs Geolife's own
```

Outputs in `output/experiments/transfer/`; IBAD trained on the NHTS templates is
saved under `artifacts/experiments/transfer/`.

---

## Experiments 2 and 3 together

```bash
bash scripts/experiments/run_experiments.sh
```

[`notebooks/02_experiments.ipynb`](../../notebooks/02_experiments.ipynb) shows
the results of experiments 2 and 3.

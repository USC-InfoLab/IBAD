# IBAD: Interpretable Behavioral Anomaly Detection on Human Mobility Data

## Modules

| Module | Description | Path |
|---|---|---|
| **0 · Geolife POI attribution** | Raw GPS trajectories → labelled stay-points. Each stay is reverse-geocoded against OpenStreetMap, then an LLM judge (Gemini 2.0) assigns one of the 18 [NHTS 2017](https://nhts.ornl.gov/) activity categories. | [src/data/geolife/](src/data/geolife/README.md). The preprocessed data with splits can be found at [this link](https://drive.google.com/drive/folders/1hjZywpjDLZ7Av3ziSt8Q9kLUoIPLtL4l?usp=drive_link). Copy the `.tsv` files to `data/processed/geolife/` to skip this step. |
| **1 · Template discovery** | Labelled days → interpretable behavioural templates. Each agent-day becomes a bag of `(time slot, activity)` tokens; LDA over those documents makes every topic a template and gives each day a soft mixture over them. | [src/templates/](src/templates/README.md) |
| **2 · Anomaly detection** | Behavioral Templates + longitudinal mobility anomaly detection pipeline. **IBAD** predicts behavioral template mixture (e.g. $70\%$ work, $20\%$ stay-at-home, $10\%$ leisure) of an agent on a target date from its historical behavioral characteristic.  A day scores anomalous when what actually happened diverges significantly from the predicted template mixture. | [src/detection/](src/detection/README.md) |
| **Baselines** | Nine comparison methods, similar experimental setup as IBAD. | [src/detectors/](src/detectors/README.md) |
| **Experiments** | Behavioral anomaly detection results + transferability experiment + effect of dissimilarity threshold $\epsilon$. | [src/experiments/](src/experiments/README.md) |

## Quickstart

**1 · Install.** Dependencies are managed with [uv](https://docs.astral.sh/uv/),
which pins Python 3.12. Prefix commands with `uv run`, or activate `.venv`.

```bash
uv sync --extra notebook --extra detection    # core + ipykernel + PyTorch
```

Extras are additive and each sync reconciles against exactly the ones you name,
so pass them together — `uv sync --extra detection` on its own uninstalls
ipykernel. `detection` pulls PyTorch with the CUDA runtime and is only needed for
module 2. Add `--extra vertex` only for module 0's Vertex AI batch labelling if not using API Key for Gemini.

**2 · Get the data.** 

*NHTS 2017* is an 84 MB zip of four CSVs, of which only `trippub.csv` is read:

```bash
mkdir -p data/raw/nhts_2017
curl -L https://nhts.ornl.gov/assets/2016/download/csv.zip -o /tmp/nhts.zip
unzip -j /tmp/nhts.zip trippub.csv -d data/raw/nhts_2017
```
*NUMOSIM* Downloaded from [OSF](https://osf.io/sjyfr/)
into `data/raw/numosim/` on first run.

*Geolife*: module 1 reads the labelled stay-point splits in
`data/processed/geolife/`. You can get them in one of two ways:

- **Download preprocessed.** Put the three `.tsv` files from
  [Google Drive](https://drive.google.com/drive/folders/1hjZywpjDLZ7Av3ziSt8Q9kLUoIPLtL4l?usp=sharing)
  into `data/processed/geolife/`.
- **Rebuild them from raw GPS.** Download the 313 MB
  [Geolife Trajectories 1.3](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
  archive from Microsoft Research, extract it to
  `data/raw/Geolife Trajectories 1.3/`, and follow
  [module 0 docs](src/data/geolife/README.md).

**3 · Discover templates.** Open
[`notebooks/01_template_discovery.ipynb`](notebooks/01_template_discovery.ipynb). Alternatively, you can run template discovery script from:

```bash
uv run python -m src.templates.discover --dataset geolife
```

**4 · Anomaly Detection.** Trains IBAD on top of step 3's templates and scores the
dataset's benchmark.

```bash
uv run python -m src.detection.run --dataset geolife --seed 42             # one run
uv run python scripts/anomaly_detection/run_replicates.py --dataset geolife  # seeds 42–46, mean ± sd
```

A trained model is reused on the next run with the same dataset and seed, so add
`--retrain` after refitting the templates. Benchmarks, outputs and results are in
the [module 2 docs](src/detection/README.md).

**5 · Baselines and experiments.** The same runner scores any method by name, and
the experiments reuse every saved model:

```bash
uv run python scripts/anomaly_detection/run_replicates.py --dataset geolife \
    --methods ibad histmean ghmm iforest ocsvm deepsvdd bestad tod4traj icad
uv run python -m src.experiments.threshold     # dissimilarity-threshold sweep
uv run python -m src.experiments.transfer      # NHTS -> Geolife templates
```

[`notebooks/02_experiments.ipynb`](notebooks/02_experiments.ipynb) walks through
both experiments' results.

TrajGenAgent additionally needs an LLM to generate its chains first; see
[src/detectors/](src/detectors/README.md#trajgenagents-generation-step).

## Repository layout

```
IBAD/
├── configs/          YAML, one file per concern; no constants in code
├── data/
│   ├── raw/          immutable inputs
│   ├── interim/      module 0 intermediates + caches
│   └── processed/    labelled datasets ready for modelling
├── src/
│   ├── data/         loading + label harmonisation (module 0 lives here)
│   ├── features/     agent-day encodings: one-hot documents, visit sequences
│   ├── templates/    module 1: LDA discovery, diagnostics, plots
│   ├── models/       module 2: IBAD, and the baselines' networks
│   ├── training/     module 2: batching + training loop
│   ├── evaluation/   module 2: benchmarks + scoring, shared by every method
│   ├── detection/    module 2: cohort assembly + CLI
│   ├── detectors/    IBAD and the baselines behind one interface
│   ├── experiments/  threshold sweep, template transfer, their figures
│   └── utils/        config loader, LLM judge, shared helpers
├── notebooks/        narrative walkthroughs; all logic stays in src/
├── scripts/          one end-to-end runner per module
├── artifacts/        fitted models
└── output/           figures and logs
```
## Citation

If you find our work or the Geolife activity labels useful, please cite our paper:

```bibtex
@inproceedings{azarijoo2026ibad,
  title     = {{IBAD}: Interpretable Behavioral Anomaly Detection on Human Mobility Data},
  author    = {Azarijoo, Bita and Krumm, John and Shahabi, Cyrus},
  booktitle = {Proceedings of the 34th ACM International Conference on Advances in Geographic Information Systems},
  series    = {SIGSPATIAL '26},
  year      = {2026},
  address   = {Riverside, CA, USA},
  publisher = {ACM},
  doi       = {10.1145/3841645.3842984}
}
```
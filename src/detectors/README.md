# Detectors — IBAD and the baselines

A **detector** turns an agent's days into one anomaly score per day: higher
means more unusual. IBAD is one detector; the baselines it is compared against
are the others. They all implement the same small interface
([`base.py`](base.py)): `fit()` learns from the cohort's train days, and
`score(sequences)` returns a score for every day of the cohort — of its real
days, or of a copy with anomalies injected.

Because every detector speaks this interface, one piece of code evaluates them
all, and a comparison between methods is a comparison on identical days:

- **Same cohort, same splits.** Every method is fitted on the train split only;
  validation picks the F1 threshold, test is only ever scored.
- **Same anomalies.** At a given seed every method scores the same injected
  days (see [`src/evaluation/benchmark.py`](../evaluation/benchmark.py)).
- **Same per-agent standardisation.** Each method's scores are z-scored against
  that agent's own train-day scores before ranking, so no method is rewarded
  for merely ranking busy agents above quiet ones.
- **Same information.** No method sees a location. Methods that read visits see
  the same visits, activities and times IBAD sees.

---

## The methods

| Method | Sees | Learns | A day's score |
|---|---|---|---|
| **IBAD** | visits (activity, arrival, departure) + day-of-week, the agent's whole history | a hierarchical forecaster of the next day's template mixture | KL(observed mixture ‖ forecast) |
| HistMean | the day's template mixture, the agent's earlier mixtures | nothing | KL(mixture ‖ mean of the agent's earlier mixtures) |
| GHMM | the agent's sequence of mixtures | a Gaussian hidden Markov model over mixtures | −log p(mixture \| earlier mixtures) |
| IF | the day's mixture | an Isolation Forest over all train days | its outlierness |
| OCSVM | the day's mixture | a one-class SVM (RBF) over all train days | its outlierness |
| DeepSVDD | the day's mixture | a network mapping normal days near a centre | squared distance from the centre |
| BeSTAD | the day's stays (start time, duration, activity) | an autoencoder that clusters days into templates | how the day's template departs from the agent's template profile |
| TOD4Traj | the day's activities, in the context of the agent's other days | contrastive day embeddings | 1 − cosine to the agent's prototype for that weekday |
| ICAD | the agent's visit sequence (activity, arrival, departure) | an autoregressive transformer over visits | mean per-visit surprise |
| TrajGenAgent | the day's activity chain | nothing (an LLM generates a normal chain per day) | edit distance to the generated chain |

The first five are **θ-space** baselines: generic detectors fed only the
template mixture θ each day gets from module 1. They measure how much of the
signal is already in the mixtures. The last four are **sequence** baselines
from the mobility anomaly-detection literature, each a port of its reference
implementation with location inputs removed.

### θ-space ([`theta.py`](theta.py))

- **HistMean** — the agent's running mean mixture is its "normal day"; a day
  scores by how far it departs from it. Strictly causal, so an injected day
  enters the history of later days exactly as it does for IBAD.
- **GHMM** — one hidden state per template, diagonal Gaussian emissions over
  mixtures, fitted on each agent's train days as separate sequences. A day
  scores by its one-step-ahead negative log-likelihood, with each day's
  emission densities rescaled so its best-fitting state has density 1: the
  score says how unexpected the day's best state is after the days before it.
- **IF / OCSVM** — standard scikit-learn detectors fitted on the pooled train
  days of all agents.
- **DeepSVDD** — one-class Deep SVDD (Ruff et al., 2018): a two-layer MLP and
  a fixed centre at the mean initial embedding of the first 10,000 train days,
  trained to pull train days towards it.

On NUMOSIM the four that are fitted (all but HistMean) learn from the train
days of 5,000 sampled agents rather than all 200,000, because the one-class SVM
is quadratic in the number of days; whole agents are sampled so each agent's
days stay contiguous for the GHMM.

### Sequence baselines

- **BeSTAD** ([`bestad.py`](bestad.py), model in
  [`src/models/bestad.py`](../models/bestad.py)) — an SJCAE clustering
  autoencoder (BiLSTM encoder, Gumbel-softmax cluster head) assigns every day
  one of six learned templates. Each agent's profile is its distribution of
  templates over its train days, and a day scores by BeSTAD's behaviour-change
  score: the Jensen-Shannon divergence between the profile and the day's
  template, whether the template was never seen in train, and whether it
  differs from the agent's dominant one (the three components defined for a
  single day, with the reference weights).
- **TOD4Traj** ([`tod4traj.py`](tod4traj.py)) — a transformer over the
  agent's activity tokens, trained so that days on the same weekday embed
  close together. Each agent gets one prototype per weekday from its train
  and validation days, and a day scores by its cosine distance from the
  prototype of its weekday. On NUMOSIM, days run from UTC midnight (16:00 local time), and a day's label is judged against
  the score of the UTC day after it.
- **ICAD** ([`icad.py`](icad.py)) — an agent's visits form one causal
  sequence with a separator closing each day. ICAD predicts each visit's
  activity, travel time and departure time of day from the visits before it;
  a visit scores by ICAD's "relative" surprise, a day by the mean over its
  visits, and a day without visits of its own scores 0. Times are fractional
  days since the cohort's earliest train stay. Sequences are scored in
  left-padded batches without a padding mask, a spliced day keeps the donor
  day's own timestamps, and on Geolife visits are grouped into UTC days.
- **TrajGenAgent** ([`trajgen.py`](trajgen.py)) — an LLM is prompted with the
  agent's own train days on the same weekday and generates a normal activity
  chain for each date; the day scores by its normalised edit distance to that
  chain. Generation is a separate GPU job, see below.

### Train days that judge themselves

BeSTAD, TOD4Traj and TrajGenAgent build a per-agent reference from the agent's
train days. A train day would then be judged against a reference it is itself
part of, and its score would come out too small — which matters because train
days calibrate the per-agent standardisation. Each of them therefore judges a
train day against its reference *with that day removed* (TrajGenAgent: with the
day left out of its own prompt). Days outside train are unaffected.

---

## Running

Every method runs through the same entry points as IBAD:

```bash
# one method, one dataset, one seed
uv run python -m src.detection.run --method bestad --dataset geolife --seed 42

# several methods, five seeds each, into the shared replicate table
uv run python scripts/anomaly_detection/run_replicates.py --dataset geolife \
    --methods ibad histmean ghmm iforest ocsvm deepsvdd bestad tod4traj icad
```

A fitted detector is saved as `artifacts/detection/<dataset>_<method>_seed<N>.*`
and reused on the next run; `--retrain` refits it. Hyper-parameters live in
`configs/baselines/<family>.yaml` (θ-space methods share `theta.yaml`).

### TrajGenAgent's generation step

TrajGenAgent needs an LLM (Qwen2.5-32B-Instruct by default) to write its normal
chains, once per cohort:

```bash
# 1. prompts for every agent-day (this repo's environment)
uv run python -m src.detectors.trajgen_prompts --dataset geolife

# 2. generation (an environment with vLLM; resumable)
CUDA_VISIBLE_DEVICES=3,4,5,6 python scripts/baselines/generate_trajgen_chains.py \
    --dir artifacts/trajgen/geolife

# 3. scoring, like any other method
uv run python scripts/anomaly_detection/run_replicates.py --dataset geolife --methods trajgen
```

---

## Adding a detector

Write a module here with a `Detector` subclass (`fit`, `score`, `save`,
`load`), a config file under `configs/baselines/`, and one line in
`REGISTRY` ([`__init__.py`](__init__.py)). The evaluation, the replicate
script and both experiments pick it up from there.

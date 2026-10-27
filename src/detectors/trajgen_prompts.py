"""Build TrajGenAgent's generation prompts for every agent-day of a cohort.

Port of TrajGenAgent's phase 2 (``llm_activity_chain_gen.py``). For each target
agent-day the prompt summarises the agent's own *train* days on the same weekday
(falling back to the same day type, weekday or weekend, when there are fewer
than two): how often each activity occurs, the most common next activity after
each one, and a few example chains. The target day itself is always left out
of its own history, which matters for train-window targets.

Writes ``<chains.dir>/<dataset>/prompts_<window>.jsonl``, one record per day,
plus ``vocabulary.json``. The LLM step reads these files
(``scripts/baselines/generate_trajgen_chains.py``).

Run from the repo root:
    python -m src.detectors.trajgen_prompts --dataset geolife
"""

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.detection.data import load_cohort
from src.detection.run import CONFIG_PATH
from src.detectors.trajgen import WINDOWS, activity_token, chains_path, observed_chains
from src.utils.config import load_cfg

TEMPLATE = """You are a Stochastic Human Mobility Simulator. Generate a daily activity chain for a specific agent.

**Simulation Context:**
- **Target Date:** {target_date} ({weekday})
- **Day Type:** {day_type}
- **Random Seed:** {random_id}

**Agent Profile (derived from {day_type}s):**
- **Activity Likelihood:** {occurrence}
- **Transition Patterns:**
{transitions}

**Historical Patterns (STRONG REFERENCE):**
Use the following past examples as a strong template for the daily structure.
{examples}

**Mandatory Constraints:**
1. **Vocabulary**: You must ONLY use the following {n_vocab} activity types:
   {vocabulary}
2. **No Repetition**: Adjacent activities MUST NOT be the same (e.g., ['Home', 'Home'] is invalid).
3. **Home Logic**:
   - 'Home' usually appears at the end of the day.
   - However, be flexible and adapt to the specific agent's historical patterns if they deviate from this general rule.

**Task:**
Generate a NEW activity chain for {target_date}.
Output ONLY a valid Python list of strings.

**Output:**
"""


def _occurrence(chains: list[list[str]]) -> str:
    counts = Counter(a for chain in chains for a in set(chain))
    return ", ".join(f"{a}({100 * n / len(chains):.0f}%)" for a, n in counts.most_common(8))


def _transitions(chains: list[list[str]]) -> str:
    following = defaultdict(list)
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            following[a].append(b)
    lines = []
    for start in sorted(following, key=lambda a: len(following[a]), reverse=True):
        counts = Counter(following[start])
        top = ", ".join(f"{b}({100 * n / len(following[start]):.0f}%)"
                        for b, n in counts.most_common(3))
        lines.append(f"  {start} -> [{top}]")
    return "\n".join(lines) or "  (Insufficient Data)"


def build_prompts(dataset: str, bench_cfg: dict, cfg: dict) -> dict[str, list[dict]]:
    """Prompt records per window for every recorded day of the cohort."""
    cohort = load_cohort(dataset, bench_cfg)
    sequences = cohort.sequences
    chains = observed_chains(sequences, cohort.activities)
    vocabulary = sorted({activity_token(a) for a in cohort.activities})
    rng = np.random.default_rng(cfg["prompt"]["seed"])
    n_examples = cfg["prompt"]["history_examples"]

    # Each agent's history: its train days, with weekday and day type.
    history = defaultdict(list)
    for (row, day), chain in chains.items():
        if sequences.split[row, day] == 0 and chain:
            weekday = pd.Timestamp(sequences.date[row, day]).day_name()
            history[row].append((day, weekday, chain))

    prompts = {window: [] for window in WINDOWS}
    for (row, day), _ in sorted(chains.items()):
        window = WINDOWS[sequences.split[row, day]]
        date = pd.Timestamp(sequences.date[row, day])
        weekday = date.day_name()
        day_type = "Weekend" if weekday in ("Saturday", "Sunday") else "Weekday"
        own = [(d, w, c) for d, w, c in history[row] if d != day]
        relevant = [c for _, w, c in own if w == weekday]
        if len(relevant) < 2:
            relevant = [c for _, w, c in own
                        if ("Weekend" if w in ("Saturday", "Sunday") else "Weekday") == day_type]
        if not relevant:
            relevant = [c for _, _, c in own]
        if not relevant:
            continue
        picks = (rng.choice(len(relevant), n_examples, replace=False)
                 if len(relevant) > n_examples else range(len(relevant)))
        examples = "\n".join(f"- [{weekday} Pattern]: " + " -> ".join(relevant[i]) for i in picks)
        prompts[window].append({
            "agent_id": int(sequences.agent_ids[row]),
            "date": str(sequences.date[row, day]),
            "prompt": TEMPLATE.format(
                target_date=str(sequences.date[row, day]), weekday=weekday, day_type=day_type,
                random_id=f"{int(rng.integers(16 ** 6)):06x}",
                occurrence=_occurrence(relevant), transitions=_transitions(relevant),
                examples=examples, n_vocab=len(vocabulary), vocabulary=vocabulary,
            ),
        })
    return {"vocabulary": vocabulary, **prompts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("geolife", "numosim"))
    args = parser.parse_args()

    cfg = load_cfg("configs/baselines/trajgen.yaml")
    built = build_prompts(args.dataset, load_cfg(CONFIG_PATH), cfg)
    out_dir = chains_path(cfg, args.dataset, "prompts", "train").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocabulary.json").write_text(json.dumps(built["vocabulary"]))
    for window in WINDOWS:
        path = chains_path(cfg, args.dataset, "prompts", window)
        path.write_text("".join(json.dumps(r) + "\n" for r in built[window]))
        print(f"  {window}: {len(built[window]):,} prompts -> {path}")


if __name__ == "__main__":
    main()

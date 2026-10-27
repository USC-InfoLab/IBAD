"""TrajGenAgent baseline: how far a day is from an LLM's normal version of it.

TrajGenAgent (IEEE MDM 2026) is an LLM trajectory generator, used here as a
detector:

1. For every agent-day, an LLM is prompted with statistics and examples of the
   agent's own train days on the same weekday and generates a *normal* activity
   chain for that date (``src/detectors/trajgen_prompts.py`` builds the prompts;
   ``scripts/baselines/generate_trajgen_chains.py`` runs the LLM). No location
   is involved.
2. A day's observed chain is its stays' activities in time order, with adjacent
   repeats collapsed. A day with no stay of its own (spent inside an overnight
   stay) has no chain, and is not scored.
3. The day scores by the normalised edit distance between the two chains.

Nothing is fitted here: the generated chains are the model, produced once per
cohort and reused by every seed. Train days are generated with the day itself
left out of its own history, so the train days that calibrate the per-agent
baseline are measured the way held-out days are — all of them, the first
included, since every day is judged on its own.
"""

import json
from pathlib import Path

import numpy as np

from src.detectors.base import Detector
from src.features.visits import VisitSequences
from src.utils.config import ROOT

WINDOWS = ("train", "val", "test")

# NUMOSIM stays carry a *set* of activities; TrajGenAgent works with one per
# stay. The reference collapses a set to its most specific member, and spells
# two activities differently from NUMOSIM's data dictionary.
_PRIORITY = ["School", "ChildCare", "HealthCare", "Religious", "EatOut", "BuyGoods",
             "Recreation", "Exercise", "Services", "Errands", "Home", "Work", "Visit",
             "DropOff", "SomethingElse", "Transportation"]
_SPELLING = {"Childcare": "ChildCare", "Excercise": "Exercise"}


def activity_token(label: str) -> str:
    """The one TrajGenAgent token for an activity (or activity-set) label."""
    if label == "∅":
        return "Home"
    names = [_SPELLING.get(name, name) for name in label.split("+")]
    rank = {name: i for i, name in enumerate(_PRIORITY)}
    return min(names, key=lambda name: rank.get(name, len(rank)))


def observed_chains(sequences: VisitSequences, activities: list[str],
                    agents: np.ndarray | None = None) -> dict[tuple[int, int], list[str]]:
    """``{(agent, day): chain}`` for every day of ``sequences`` with stays of its own.

    ``agents`` (a boolean mask over rows) limits the chains to some agents.
    """
    tokens = np.array([None] + [activity_token(a) for a in activities], dtype=object)
    chains = {}
    has_stays = sequences.visit_mask.any(axis=2)
    if agents is not None:
        has_stays &= agents[:, None]
    rows, days = np.nonzero(has_stays)
    for row, day in zip(rows, days):
        visits = list(tokens[sequences.bundle_id[row, day][sequences.visit_mask[row, day]]])
        chains[(int(row), int(day))] = [visits[0]] + [b for a, b in zip(visits, visits[1:])
                                                      if b != a]
    return chains


def edit_distance(a: list[str], b: list[str]) -> float:
    """Levenshtein distance between two token sequences, divided by the longer length."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, token in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, other in enumerate(b, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1,
                             previous[j - 1] + (token != other))
        previous = current
    return previous[-1] / max(len(a), len(b))


def chains_path(cfg: dict, dataset: str, kind: str, window: str) -> Path:
    return ROOT / cfg["chains"]["dir"] / dataset / f"{kind}_{window}.jsonl"


class TrajGenDetector(Detector):
    name = "trajgen"
    display_name = "TrajGenAgent"
    config_path = "configs/baselines/trajgen.yaml"
    checkpoint_ext = ".json"
    # A day that matches its generated chain scores 0, a real score.
    zero_marks_unscored = False

    def _read_generated(self) -> dict[tuple[int, str], list[str]]:
        """Generated chains by (agent id, date)."""
        generated, files = {}, []
        for window in WINDOWS:
            path = chains_path(self.cfg, self.cohort.dataset, "chains", window)
            if not path.exists():
                continue
            files.append(path.name)
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if record["chain"] is not None:
                    generated[(int(record["agent_id"]), record["date"])] = record["chain"]
        if not generated:
            raise FileNotFoundError(
                f"no generated chains for {self.cohort.dataset} under {self.cfg['chains']['dir']}; "
                "build prompts with `python -m src.detectors.trajgen_prompts` and run "
                "scripts/baselines/generate_trajgen_chains.py")
        self.files = files
        return generated

    def fit(self) -> dict:
        self.generated = self._read_generated()
        return {"generated_days": len(self.generated), "files": self.files}

    def _distances(self, sequences: VisitSequences) -> np.ndarray:
        """``(n_agents, n_days)`` distance of each day from its generated chain; NaN without one."""
        grid = np.full(sequences.template.shape, np.nan)
        generated_for = np.isin(np.asarray(sequences.agent_ids, dtype=np.int64),
                                np.array(sorted({agent for agent, _ in self.generated})))
        for (row, day), chain in observed_chains(sequences, self.cohort.activities,
                                                 generated_for).items():
            generated = self.generated.get((int(sequences.agent_ids[row]),
                                            str(sequences.date[row, day])))
            if generated is not None:
                grid[row, day] = edit_distance(chain, generated)
        return grid

    def score(self, sequences: VisitSequences) -> np.ndarray:
        return self._distances(sequences)[:, 1:]

    def calibration_scores(self, sequences: VisitSequences, days: np.ndarray) -> np.ndarray:
        return np.where(days, self._distances(sequences), np.nan)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"generated_days": len(self.generated), "files": self.files}))

    def load(self, path: Path) -> None:
        self.generated = self._read_generated()

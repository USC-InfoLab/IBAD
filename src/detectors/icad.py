"""ICAD baseline: score each visit by how unexpected it is given the agent's history.

1. An agent's days become one causal token sequence: its visits in time order,
   each day closed by a separator token timed midway to the next day's first
   visit. A visit carries the agent, its activity, and its arrival and
   departure in fractional days since the cohort's earliest train stay — no
   location. Its travel time is the gap since the previous stay in the same
   window, 0 for the window's first.
2. ICAD (``src/models/icad.py``) is trained on each agent's train days to
   predict every visit's activity, travel time and departure time of day from
   the visits before it.
3. A visit scores by ICAD's ``relative`` score (top-3 activity margin plus the
   two time mode margins), a day by the mean score of its visits; a day
   without visits of its own scores 0, "no score".

Batching, splicing and days:

* Sequences are batched left-padded and the encoder is given no padding mask,
  so a visit also attends to the padding before it, and an agent's scores
  depend on the longest sequence in its batch.
* A spliced day takes the donor day's visits with their own timestamps, travel
  times and times of day; only their day index and agent change.
* With ``inputs.utc_days`` (Geolife), visits are grouped into UTC days and their
  departure time of day read on the UTC clock, and a benchmark day is looked up
  by its local day number among those UTC days.
"""

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.detectors.base import Detector
from src.evaluation.benchmark import Benchmark
from src.evaluation.scoring import Window
from src.features.visits import VisitSequences
from src.models.icad import ICAD, PAD, SEPARATOR, icad_loss, visit_scores

_FIELDS = ("agent", "activity", "arrival", "departure", "travel", "tod", "day")


@dataclass
class Visits:
    """A flat visit table; ``seq`` names the sequence a visit belongs to and
    ``day`` the key its day is grouped and looked up by."""

    seq: np.ndarray
    agent: np.ndarray
    activity: np.ndarray
    arrival: np.ndarray
    departure: np.ndarray
    travel: np.ndarray
    tod: np.ndarray
    day: np.ndarray

    def take(self, index) -> "Visits":
        return Visits(**{name: getattr(self, name)[index] for name in ("seq", *_FIELDS)})

    @staticmethod
    def concat(*parts: "Visits") -> "Visits":
        return Visits(**{name: np.concatenate([getattr(p, name) for p in parts])
                         for name in ("seq", *_FIELDS)})


def time_origin(sequences: VisitSequences) -> float:
    """The start of the cohort's earliest train stay, in days on its calendar —
    the fixed point ICAD's absolute times are measured from."""
    r, d, v = np.nonzero(sequences.visit_mask & (sequences.split == 0)[..., None])
    return float((sequences.day_ts[r, d] + sequences.arrival[r, d, v] / 24.0).min())


def window_visits(sequences: VisitSequences, origin: float, utc_offset: float | None,
                  days: np.ndarray | None = None) -> Visits:
    """The visits of ``sequences`` (of the ``(N, T)`` mask ``days`` if given), in
    agent and time order, keyed by their day: its position, or under the UTC
    clock its UTC day number."""
    mask = sequences.visit_mask if days is None else sequences.visit_mask & days[..., None]
    r, d, v = np.nonzero(mask)
    local = sequences.day_ts[r, d].astype(np.float64)
    arrival = local - origin + sequences.arrival[r, d, v].astype(np.float64) / 24.0
    departure = local - origin + sequences.departure[r, d, v].astype(np.float64) / 24.0
    clock = sequences.departure[r, d, v].astype(np.float64)
    if utc_offset is None:
        day = d
    else:
        start = np.floor(local + (sequences.arrival[r, d, v] - utc_offset) / 24.0).astype(np.int64)
        day = start - start.min()
        clock = clock - utc_offset
    follows = np.r_[False, r[1:] == r[:-1]]
    return Visits(
        seq=r, agent=r, activity=sequences.bundle_id[r, d, v] + 1,
        arrival=arrival, departure=departure,
        travel=np.where(follows, arrival - np.r_[0.0, departure[:-1]], 0.0),
        tod=np.mod(clock, 24.0) / 24.0, day=day,
    )


def position_keys(sequences: VisitSequences, utc_offset: float | None) -> np.ndarray:
    """``(N, T)`` the day key each benchmark day is looked up by."""
    if utc_offset is None:
        return np.broadcast_to(np.arange(sequences.n_days), sequences.day_ts.shape)
    return np.rint(sequences.day_ts).astype(np.int64)


def _tokens(visits: Visits) -> tuple[dict, np.ndarray]:
    """Visits sorted per sequence by (day, arrival), a separator closing each day.

    Returns the token fields (``real`` False for separators) and per-sequence
    ``offsets`` over ``np.unique(visits.seq)``.
    """
    visits = visits.take(np.lexsort((visits.arrival, visits.day, visits.seq)))
    s, d = visits.seq, visits.day
    first = np.flatnonzero(np.r_[True, (s[1:] != s[:-1]) | (d[1:] != d[:-1])])
    last = np.r_[first[1:], len(s)] - 1
    has_next = np.r_[s[first[1:]] == s[first[:-1]], False]
    next_arrival = np.r_[visits.arrival[first[1:]], 0.0]
    mid = np.where(has_next, 0.5 * (visits.departure[last] + next_arrival), visits.departure[last])

    n_seps = len(first)
    tokens = {
        "seq": np.r_[s, s[first]],
        "agent": np.r_[visits.agent, visits.agent[first]],
        "activity": np.r_[visits.activity, np.full(n_seps, SEPARATOR)],
        "arrival": np.r_[visits.arrival, mid],
        "departure": np.r_[visits.departure, mid],
        "travel": np.r_[visits.travel, np.zeros(n_seps)],
        "tod": np.r_[visits.tod, np.zeros(n_seps)],
        "day": np.r_[d, np.full(n_seps, -1)],
        "real": np.r_[np.ones(len(s), bool), np.zeros(n_seps, bool)],
    }
    # A day's separator follows its last visit.
    order = np.lexsort((np.r_[np.arange(len(s)), last + 0.5], tokens["seq"]))
    tokens = {name: values[order] for name, values in tokens.items()}
    _, counts = np.unique(tokens["seq"], return_counts=True)
    return tokens, np.r_[0, np.cumsum(counts)]


def _left_padded(tokens: dict, offsets: np.ndarray, spans: np.ndarray,
                 device: torch.device) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Sequences ``spans`` as left-padded ``(B, L)`` tensors.

    Also returns, per token, its batch row and column and its index into ``tokens``.
    Padding is agent 0, activity ``PAD`` and time 0.
    """
    lengths = offsets[spans + 1] - offsets[spans]
    width = int(lengths.max())
    flat = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in spans])
    row = np.repeat(np.arange(len(spans)), lengths)
    col = width - np.repeat(lengths, lengths) + (
        np.arange(len(flat)) - np.repeat(np.r_[0, np.cumsum(lengths)[:-1]], lengths))
    grid = {name: np.zeros((len(spans), width), dtype=np.float32)
            for name in ("arrival", "departure", "travel", "tod")}
    grid["agent"] = np.zeros((len(spans), width), dtype=np.int64)
    grid["activity"] = np.full((len(spans), width), PAD, dtype=np.int64)
    grid["real"] = np.zeros((len(spans), width), dtype=bool)
    for name in grid:
        grid[name][row, col] = tokens[name][flat]
    return ({name: torch.from_numpy(values).to(device) for name, values in grid.items()},
            row, col, flat)


class ICADDetector(Detector):
    name = "icad"
    display_name = "ICAD"
    config_path = "configs/baselines/icad.yaml"
    checkpoint_ext = ".pt"

    def __init__(self, cfg, cohort, seed, device):
        super().__init__(cfg, cohort, seed, device)
        sequences = cohort.sequences
        self.model = ICAD(sequences.n_agents, sequences.n_bundles - 1, **cfg["model"]).to(device)
        self.origin = time_origin(sequences)
        self.utc_offset = cohort.utc_offset_hours if cfg["inputs"]["utc_days"] else None

    def _visits(self, sequences: VisitSequences, days: np.ndarray | None = None) -> Visits:
        return window_visits(sequences, self.origin, self.utc_offset, days)

    def _forward(self, batch: dict) -> dict:
        return self.model(batch["agent"], batch["activity"], batch["arrival"], batch["departure"])

    # ---------------------------------------------------------------- fitting

    def fit(self) -> dict:
        params = self.cfg["training"]
        sequences = self.cohort.sequences
        tokens, offsets = _tokens(self._visits(sequences, sequences.split == 0))
        optimizer = torch.optim.Adafactor(self.model.parameters(), lr=params["lr"])

        best_loss, best_state = float("inf"), None
        for epoch in range(1, params["epochs"] + 1):
            self.model.train()
            losses = []
            for chunk in torch.randperm(len(offsets) - 1).split(params["batch_size"]):
                batch, *_ = _left_padded(tokens, offsets, chunk.numpy(), self.device)
                loss = icad_loss(self._forward(batch), batch["activity"][:, 1:],
                                 batch["travel"][:, 1:], batch["tod"][:, 1:], batch["real"][:, 1:])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), params["grad_clip"])
                optimizer.step()
                losses.append(loss.item())
            mean_loss = float(np.mean(losses))
            # The reference keeps the epoch with the lowest training loss.
            if mean_loss < best_loss:
                best_loss, best_state = mean_loss, copy.deepcopy(self.model.state_dict())
            if epoch == 1 or epoch % 50 == 0:
                print(f"    ICAD epoch {epoch:3d}/{params['epochs']}  "
                      f"loss={mean_loss:.4f}  best={best_loss:.4f}")
        self.model.load_state_dict(best_state)
        return {"best_loss": best_loss}

    # ---------------------------------------------------------------- scoring

    @torch.no_grad()
    def _day_scores(self, visits: Visits, n_keys: int) -> np.ndarray:
        """``(n_sequences, n_keys)`` mean visit score per day key, in ``seq`` order; 0 without visits."""
        self.model.eval()
        tokens, offsets = _tokens(visits)
        terms = self.cfg["scoring"]["weights"]
        weights = (terms["activity"], terms["travel"], terms["departure"])
        n_sequences = len(offsets) - 1
        totals = np.zeros((n_sequences, n_keys))
        counts = np.zeros((n_sequences, n_keys))
        batch_size = self.cfg["scoring"]["batch_size"]
        for start in range(0, n_sequences, batch_size):
            spans = np.arange(start, min(start + batch_size, n_sequences))
            batch, row, col, flat = _left_padded(tokens, offsets, spans, self.device)
            scores = visit_scores(self._forward(batch), batch["activity"][:, 1:],
                                  batch["travel"][:, 1:], batch["tod"][:, 1:],
                                  weights).cpu().numpy()
            keep = tokens["real"][flat] & (col >= 1)
            at = (start + row[keep], tokens["day"][flat][keep])
            np.add.at(totals, at, scores[row[keep], col[keep] - 1])
            np.add.at(counts, at, 1.0)
        return np.where(counts > 0, totals / np.maximum(counts, 1.0), 0.0).astype(np.float32)

    def _grid(self, visits: Visits, keys: np.ndarray) -> np.ndarray:
        """``(N, T - 1)`` scores of each agent's days 1.., read through ``keys``."""
        n_keys = int(max(visits.day.max(), keys.max())) + 1
        by_key = np.zeros((keys.shape[0], n_keys), dtype=np.float32)
        by_key[np.unique(visits.seq)] = self._day_scores(visits, n_keys)
        return np.take_along_axis(by_key, keys[:, 1:], axis=1)

    def score(self, sequences: VisitSequences) -> np.ndarray:
        return self._grid(self._visits(sequences), position_keys(sequences, self.utc_offset))

    def scored_window(self, cohort) -> Window:
        """As the default, except that the window's visits are kept for splicing,
        and a dataset scored as one sequence calibrates on its train-split days."""
        standalone = cohort.test_window is not None
        sequences = cohort.test_window if standalone else cohort.sequences
        window = self._visits(sequences)
        keys = position_keys(sequences, self.utc_offset)
        clean = self._grid(window, keys)
        calibration = (self.score(cohort.sequences) if standalone
                       else np.where(sequences.split[:, 1:] == 0, clean, np.nan))
        return Window(sequences, clean, calibration, standalone, state=(window, keys))

    def perturbed_scores(self, window: Window, bench: Benchmark) -> np.ndarray:
        visits, keys = window.state
        spliced = _splice(visits, keys, bench.plan)
        perturbed = window.clean.copy()
        agents = np.unique(spliced.seq)
        perturbed[agents] = self._grid(spliced, keys)[agents]
        return perturbed

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


def _splice(window: Visits, keys: np.ndarray, plan: np.ndarray) -> Visits:
    """The visits of every spliced sequence.

    A target keeps its visits outside the target day and receives the donor
    day's visits, relabelled onto its own agent and day with their times
    untouched. A target without visits is not re-scored, and one whose donor
    day has no visits keeps its own day.
    """
    n_keys = int(max(window.day.max(), keys.max())) + 1
    groups, starts, sizes = np.unique(window.seq * n_keys + window.day,
                                      return_index=True, return_counts=True)
    targets = np.flatnonzero(plan[:, 0] >= 0)
    targets = targets[np.isin(targets, window.seq)]
    target_key = keys[targets, plan[targets, 0]]
    wanted = plan[targets, 1] * n_keys + keys[plan[targets, 1], plan[targets, 2]]
    found = np.minimum(np.searchsorted(groups, wanted), len(groups) - 1)
    swapped = groups[found] == wanted

    replaced = np.full(plan.shape[0], -1)
    replaced[targets[swapped]] = target_key[swapped]
    own = np.flatnonzero(np.isin(window.seq, targets) & (window.day != replaced[window.seq]))

    lengths = sizes[found[swapped]]
    donated = (np.repeat(starts[found[swapped]] - np.r_[0, np.cumsum(lengths)[:-1]], lengths)
               + np.arange(lengths.sum()))
    moved = window.take(donated)
    receiver = np.repeat(targets[swapped], lengths)
    moved.seq, moved.agent = receiver, receiver
    moved.day = np.repeat(target_key[swapped], lengths)
    return Visits.concat(window.take(own), moved)

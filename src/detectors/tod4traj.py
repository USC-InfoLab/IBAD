"""TOD4Traj baseline: contrastive day embeddings vs per-weekday prototypes.

1. Each day is the sequence of its stays' activities — no times, no location.
2. A transformer (``src/models/tod4traj.py``) embeds every day in the context of
   the agent's other days, trained contrastively so same-weekday days of an
   agent sit together.
3. Each agent has one prototype per weekday: the normalised mean embedding of
   its reference days on that weekday (of all of them when it has none).
4. A day scores ``1 − cos(day embedding, prototype for its weekday)``, and 0 —
   "no score" — when the agent has no reference day left to judge it by.

The reference days are the train and validation days, each agent's packed in
date order. They train the model, build the prototypes and set the per-agent
baseline, each reference day judged against its prototype with its own
embedding removed. A scored day is embedded among all of the agent's days.

Prototypes and scores are computed in float32. With ``inputs.utc_days`` (NUMOSIM) days are cut at UTC midnight rather than at
local midnight, every window then spanning one UTC day more than its local
days. The benchmark's day indices apply to those UTC days as they are, and a
day's label is judged against the score of the UTC day after it
(``labels_offset = 0``).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.detectors.base import Detector
from src.evaluation.benchmark import Benchmark
from src.evaluation.scoring import Window
from src.features.visits import VisitSequences
from src.models.tod4traj import TOD4TrajEncoder, contrastive_loss

REFERENCE, ALL = 1, 2       # the windows: train + validation days, every day


@dataclass
class Days:
    """Days laid out per agent: ``(N, T, S)`` activity ids (0 pads), ``(N, T)`` weekdays (-1 pads)."""

    tokens: np.ndarray
    dow: np.ndarray


def _local_days(sequences: VisitSequences, split: int) -> Days:
    """Every recorded day up to the end of ``split``, packed per agent.

    Splits are chronological within an agent, so this is the agent's sequence
    cut at the end of ``split``.
    """
    in_window = (sequences.split >= 0) & (sequences.split <= split) & (sequences.template >= 0)
    rank = np.cumsum(in_window, axis=1) - 1
    rows, cols = np.nonzero(in_window)
    width = int(in_window.sum(axis=1).max()) if in_window.any() else 0
    n_agents, _, n_slots = sequences.bundle_id.shape
    tokens = np.zeros((n_agents, width, n_slots), dtype=np.int64)
    dow = np.full((n_agents, width), -1, dtype=np.int64)
    tokens[rows, rank[rows, cols]] = np.where(sequences.visit_mask[rows, cols],
                                              sequences.bundle_id[rows, cols], 0)
    dow[rows, rank[rows, cols]] = sequences.dow[rows, cols]
    return Days(tokens, dow)


def _utc_days(sequences: VisitSequences, utc_offset_hours: float) -> Days:
    """Every stay regrouped into the UTC day it starts on, in time order.

    Day *k* is the window's *k*-th UTC date; a day keeps as many stays as a
    local day holds, and takes the weekday of its first stay.
    """
    r, d, v = np.nonzero(sequences.visit_mask)
    local_day = sequences.day_ts[r, d].astype(np.float64)
    utc = local_day + (sequences.arrival[r, d, v].astype(np.float64) - utc_offset_hours) / 24.0
    order = np.lexsort((utc, r))
    r, d, v, utc, local_day = r[order], d[order], v[order], utc[order], local_day[order]
    day = np.floor(utc).astype(np.int64)
    weekday = (sequences.dow[r, d].astype(np.int64) + day - local_day.astype(np.int64)) % 7
    day -= day.min()

    first = np.r_[True, (r[1:] != r[:-1]) | (day[1:] != day[:-1])]
    start = np.maximum.accumulate(np.where(first, np.arange(len(r)), 0))
    rank = np.arange(len(r)) - start
    n_slots = sequences.bundle_id.shape[2]
    keep = rank < n_slots
    tokens = np.zeros((sequences.n_agents, int(day.max()) + 1, n_slots), dtype=np.int64)
    dow = np.full(tokens.shape[:2], -1, dtype=np.int64)
    tokens[r[keep], day[keep], rank[keep]] = sequences.bundle_id[r, d, v][keep]
    dow[r[first], day[first]] = weekday[first]
    return Days(tokens, dow)


@dataclass
class Reference:
    """The reference days, their embeddings, per-weekday float32 sums and day
    counts, and the prototypes they make."""

    days: Days
    emb: np.ndarray
    sums: np.ndarray
    counts: np.ndarray
    prototypes: np.ndarray


class TOD4TrajDetector(Detector):
    name = "tod4traj"
    display_name = "TOD4Traj"
    config_path = "configs/baselines/tod4traj.yaml"
    checkpoint_ext = ".pt"
    calibration_splits = (0, 1)

    def __init__(self, cfg, cohort, seed, device):
        super().__init__(cfg, cohort, seed, device)
        # Activity ids start at 1 and 0 pads, the encoder's own convention.
        self.model = TOD4TrajEncoder(cohort.sequences.n_bundles - 1, **cfg["model"]).to(device)
        self._reference = None      # Reference, built on first use

    # ---------------------------------------------------------------- days

    def _days(self, sequences: VisitSequences, split: int) -> Days:
        if self.cfg["inputs"]["utc_days"]:
            return _utc_days(sequences, self.cohort.utc_offset_hours)
        return _local_days(sequences, split)

    @torch.no_grad()
    def _embed(self, tokens: np.ndarray) -> np.ndarray:
        self.model.eval()
        batch = self.cfg["scoring"]["batch_size"]
        return np.concatenate([
            self.model(torch.from_numpy(tokens[s:s + batch]).to(self.device)).cpu().numpy()
            for s in range(0, len(tokens), batch)
        ]).astype(np.float32)

    # ---------------------------------------------------------------- fitting

    def fit(self) -> dict:
        params = self.cfg["training"]
        reference = self._days(self.cohort.sequences, REFERENCE)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(reference.tokens), torch.from_numpy(reference.dow)),
            batch_size=min(params["batch_size"], len(reference.tokens)), shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"])
        schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params["lr_step"],
                                                   gamma=params["lr_gamma"])
        for epoch in range(params["epochs"]):
            self.model.train()
            for tokens_b, dow_b in loader:
                loss = contrastive_loss(self.model(tokens_b.to(self.device)), dow_b.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            schedule.step()
            if (epoch + 1) % 20 == 0:
                print(f"    TOD4Traj epoch {epoch + 1:3d}/{params['epochs']}  loss={loss.item():.4f}")
        self._reference = None
        return {"final_loss": float(loss.item())}

    # ---------------------------------------------------------------- scoring

    def _reference_days(self) -> Reference:
        if self._reference is None:
            days = self._days(self.cohort.sequences, REFERENCE)
            emb = self._embed(days.tokens)
            n, _, dim = emb.shape
            sums = np.zeros((n, 7, dim), dtype=np.float32)
            counts = np.zeros((n, 7), dtype=np.int32)
            batch = self.cfg["scoring"]["batch_size"]
            for s in range(0, n, batch):
                for d in range(7):
                    w = (days.dow[s:s + batch] == d).astype(np.float32)
                    counts[s:s + batch, d] += w.sum(axis=1).astype(np.int32)
                    sums[s:s + batch, d] += np.einsum("bt,btd->bd", w, emb[s:s + batch],
                                                      optimize=True)
            proto = sums / (counts[:, :, None].astype(np.float32) + 1e-8)
            # A weekday the agent has no reference day on falls back to all its days.
            overall = sums.sum(axis=1, keepdims=True) / (
                counts.sum(axis=1, keepdims=True)[:, :, None].astype(np.float32) + 1e-8)
            for d in range(7):
                empty = np.flatnonzero(counts[:, d] == 0)
                proto[empty, d] = overall[empty, 0]
            proto /= np.linalg.norm(proto, axis=-1, keepdims=True) + 1e-8
            self._reference = Reference(days, emb, sums, counts, proto)
        return self._reference

    @staticmethod
    def _distances(emb: np.ndarray, dow: np.ndarray, proto: np.ndarray) -> np.ndarray:
        """``(N, T - 1)`` cosine distance to the weekday prototype; 0 on padding."""
        out = np.zeros((emb.shape[0], emb.shape[1] - 1), dtype=np.float32)
        rows = np.arange(emb.shape[0])
        for t in range(1, emb.shape[1]):
            valid = dow[:, t] >= 0
            score = 1.0 - (emb[:, t] * proto[rows, np.where(valid, dow[:, t], 0)]).sum(axis=1)
            score[~valid] = 0.0
            out[:, t - 1] = score.astype(np.float32)
        return out

    def _calibration(self) -> np.ndarray:
        """Reference days against their prototypes, each day removed from its own first."""
        reference = self._reference_days()
        emb, sums, counts = reference.emb, reference.sums, reference.counts
        out = np.zeros((emb.shape[0], emb.shape[1] - 1), dtype=np.float32)
        rows = np.arange(emb.shape[0])
        total_sums, total_counts = sums.sum(axis=1), counts.sum(axis=1)
        for t in range(1, emb.shape[1]):
            valid = reference.days.dow[:, t] >= 0
            d = np.where(valid, reference.days.dow[:, t], 0)
            e_t = emb[:, t]
            num = sums[rows, d] - e_t
            den = (counts[rows, d] - 1).astype(np.float32)
            # The agent's only day on that weekday: fall back to all its days.
            thin = den <= 0
            num[thin] = total_sums[thin] - e_t[thin]
            den[thin] = (total_counts[thin] - 1).astype(np.float32)
            ref = num / np.maximum(den, 1.0)[:, None]
            ref /= np.linalg.norm(ref, axis=-1, keepdims=True) + 1e-8
            score = 1.0 - (e_t * ref).sum(axis=1)
            score[~(valid & (den > 0))] = 0.0
            out[:, t - 1] = score.astype(np.float32)
        return out

    def score(self, sequences: VisitSequences) -> np.ndarray:
        days = self._days(sequences, ALL)
        return self._distances(self._embed(days.tokens), days.dow, self._reference_days().prototypes)

    def scored_window(self, cohort) -> Window:
        standalone = cohort.test_window is not None
        sequences = cohort.test_window if standalone else cohort.sequences
        days = self._days(sequences, ALL)
        clean = self._distances(self._embed(days.tokens), days.dow, self._reference_days().prototypes)
        return Window(sequences, clean, self._calibration(), standalone,
                      labels_offset=0 if self.cfg["inputs"]["utc_days"] else 1, state=days)

    def perturbed_scores(self, window: Window, bench: Benchmark) -> np.ndarray:
        """Each spliced agent's days re-embedded with the donor day in place."""
        days, proto = window.state, self._reference_days().prototypes
        perturbed = window.clean.copy()
        rows = np.flatnonzero(bench.plan[:, 0] >= 0)
        if len(rows):
            target, donor, donor_day = bench.plan[rows].T
            tokens = days.tokens[rows].copy()
            tokens[np.arange(len(rows)), target] = days.tokens[donor, donor_day]
            perturbed[rows] = self._distances(self._embed(tokens), days.dow[rows], proto[rows])
        return perturbed

    def save(self, path: Path) -> None:
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model"])
        self._reference = None

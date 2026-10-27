"""BeSTAD baseline: learned day templates, scored against the agent's normal profile.

1. Every day becomes a sequence of its stays, each described by when it starts
   (one of six coarse periods plus a sin/cos clock), how long it lasts, and its
   activity — no location.
2. An SJCAE clustering autoencoder (``src/models/bestad.py``) is fitted on the
   train days and assigns every day one of ``n_clusters`` learned templates.
3. Each agent's *normal profile* is the distribution of templates over its own
   train days.
4. A day scores by BeSTAD's behaviour-change score against that profile:

       score = w1 · JS(profile ‖ day's template) / ln 2
             + w2 · [the day's template never occurred in the agent's train days]
             + w3 · [the day's template is not the agent's dominant one]

A train day helped build the profile it is judged against, which would make
its own template trivially "seen". Every day is therefore judged against the
profile with its own occurrence removed — a no-op for days outside train —
so the train days that calibrate the per-agent baseline are measured the way a
held-out day is.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.detectors.base import Detector
from src.features.visits import VisitSequences
from src.models.bestad import SJCAE, sjcae_loss

TEMPORAL_DIM = 9                    # period one-hot (6) + sin + cos + duration
# Period of day by start hour, BeSTAD's convention; hours outside these are night.
_PERIODS = ((5, 9), (9, 12), (12, 14), (14, 18), (18, 22))
# BeSTAD's weights for the three components defined for a single day (the
# reference's transition, entropy and frequency terms need several days),
# renormalised to sum to one.
_WEIGHTS = np.array([0.20, 0.20, 0.15]) / 0.55
_CHUNK = 16384


class BeSTADDetector(Detector):
    name = "bestad"
    display_name = "BeSTAD"
    config_path = "configs/baselines/bestad.yaml"
    checkpoint_ext = ".pt"

    def __init__(self, cfg, cohort, seed, device):
        super().__init__(cfg, cohort, seed, device)
        # Activity ids start at 1; 0 is the visit tensors' padding.
        self.n_activities = cohort.sequences.n_bundles - 1
        self.n_clusters = cfg["model"]["n_clusters"]
        self.model = SJCAE(TEMPORAL_DIM, self.n_activities, **cfg["model"]).to(device)

    # ----------------------------------------------------------------- features

    def _features(self, sequences: VisitSequences, rows: np.ndarray, days: np.ndarray) -> torch.Tensor:
        """``(n, V, 9 + n_activities)`` stay features for the given agent-days.

        """
        n_slots = self.cfg["features"]["n_slots"]
        mask = sequences.visit_mask[rows, days]
        slot = np.clip(np.floor(sequences.arrival[rows, days] * n_slots / 24.0), 0, n_slots - 1)
        hour = slot * (24.0 / n_slots)
        period = np.full(slot.shape, 5, dtype=np.int64)
        for p, (lo, hi) in enumerate(_PERIODS):
            period[(hour >= lo) & (hour < hi)] = p
        angle = 2.0 * np.pi * slot / n_slots
        # Duration from whole seconds (subtracting the float32 clock times would
        # add rounding noise), held in float32 hours, then as a share of the day.
        seconds = (np.rint(sequences.departure[rows, days].astype(np.float64) * 3600.0)
                   - np.rint(sequences.arrival[rows, days].astype(np.float64) * 3600.0))
        hours = (seconds / 3600.0).astype(np.float32)
        duration = np.clip(hours / np.float32(24.0), 0.0, 1.0)

        x = np.zeros((*mask.shape, TEMPORAL_DIM + self.n_activities), dtype=np.float32)
        doc, stay = np.nonzero(mask)
        x[doc, stay, period[doc, stay]] = 1.0
        x[doc, stay, 6] = np.sin(angle[doc, stay])
        x[doc, stay, 7] = np.cos(angle[doc, stay])
        x[doc, stay, 8] = duration[doc, stay]
        x[doc, stay, TEMPORAL_DIM + sequences.bundle_id[rows, days][doc, stay] - 1] = 1.0
        return torch.from_numpy(x)

    @torch.no_grad()
    def _clusters(self, sequences: VisitSequences) -> np.ndarray:
        """``(N, T)`` template of every day with stays of its own, -1 elsewhere.

        A day spent entirely inside an overnight stay has no stays to embed, so
        it gets no template: it is not scored and not in any profile.
        """
        self.model.eval()
        grid = np.full(sequences.template.shape, -1, dtype=np.int64)
        rows, days = np.nonzero(sequences.visit_mask.any(axis=2))
        for start in range(0, len(rows), _CHUNK):
            r, d = rows[start:start + _CHUNK], days[start:start + _CHUNK]
            y = self.model.assign(self._features(sequences, r, d).to(self.device))
            grid[r, d] = y.argmax(dim=-1).cpu().numpy()
        return grid

    # ---------------------------------------------------------------- fitting

    def fit(self) -> dict:
        params = self.cfg["training"]
        sequences = self.cohort.sequences
        rows, days = np.nonzero((sequences.split == 0) & sequences.visit_mask.any(axis=2))
        if params["max_days"] and len(rows) > params["max_days"]:
            keep = np.sort(np.random.default_rng(self.seed).choice(
                len(rows), size=params["max_days"], replace=False))
            rows, days = rows[keep], days[keep]

        loader = DataLoader(TensorDataset(self._features(sequences, rows, days)),
                            batch_size=params["batch_size"], shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=params["lr"],
                                      weight_decay=params["weight_decay"])
        for epoch in range(params["epochs"]):
            self.model.train()
            losses = []
            for (batch,) in loader:
                batch = batch.to(self.device)
                loss = sjcae_loss(batch, self.model(batch), TEMPORAL_DIM)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                losses.append(loss.item())
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"    SJCAE epoch {epoch + 1:3d}/{params['epochs']}  loss={np.mean(losses):.4f}")
        self._build_profile()
        return {"fit_days": int(len(rows)), "final_loss": float(np.mean(losses))}

    def _build_profile(self) -> None:
        """Templates of the cohort's train days, and their per-agent counts."""
        sequences = self.cohort.sequences
        clusters = self._clusters(sequences)
        self.train_clusters = np.where(sequences.split == 0, clusters, -1)
        rows, days = np.nonzero(self.train_clusters >= 0)
        self.counts = np.zeros((sequences.n_agents, self.n_clusters))
        np.add.at(self.counts, (rows, self.train_clusters[rows, days]), 1.0)

    # ---------------------------------------------------------------- scoring

    def score(self, sequences: VisitSequences) -> np.ndarray:
        clusters = self._clusters(sequences)
        valid = clusters >= 0
        k, alpha = self.n_clusters, self.cfg["profile"]["alpha"]

        # Each train day's own occurrence leaves its profile (see module
        # docstring); train days exist only in the fitted cohort's layout.
        counts = np.broadcast_to(self.counts[:, None, :], (*clusters.shape, k)).copy()
        rows, days = np.nonzero(sequences.split == 0)
        own = self.train_clusters[rows, days]
        counts[rows[own >= 0], days[own >= 0], own[own >= 0]] -= 1.0
        profile = (counts + alpha) / (counts.sum(axis=-1, keepdims=True) + alpha * k)

        c = np.where(valid, clusters, 0)
        # JS divergence between the profile and the one-hot template c, in nats.
        mixture = 0.5 * profile
        np.put_along_axis(mixture, c[..., None],
                          np.take_along_axis(mixture, c[..., None], axis=-1) + 0.5, axis=-1)
        m_day = np.take_along_axis(mixture, c[..., None], axis=-1)[..., 0]
        js = 0.5 * (profile * np.log(profile / mixture)).sum(axis=-1) - 0.5 * np.log(m_day)

        unseen = np.take_along_axis(counts, c[..., None], axis=-1)[..., 0] <= 0
        off_dominant = c != profile.argmax(axis=-1)
        score = _WEIGHTS[0] * js / np.log(2.0) + _WEIGHTS[1] * unseen + _WEIGHTS[2] * off_dominant
        return np.where(valid, score, np.nan)[:, 1:].astype(np.float32)

    def save(self, path: Path) -> None:
        torch.save({"model": self.model.state_dict(),
                    "counts": torch.from_numpy(self.counts),
                    "train_clusters": torch.from_numpy(self.train_clusters)}, path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model"])
        self.counts = state["counts"].numpy()
        self.train_clusters = state["train_clusters"].numpy()

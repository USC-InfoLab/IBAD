"""θ-space baselines: generic detectors that see only each day's template mixture.

Each detector is fed the K-dimensional mixture θ that module 1's templates give
a day, and nothing else — no visits, no times, no places:

| Detector | Score for day *t* |
|---|---|
| HistMean | KL(θ_t ‖ mean of the agent's earlier mixtures) |
| GHMM | −log p(θ_t \\| θ_<t) under a Gaussian hidden Markov model of mixtures |
| IF | Isolation Forest outlierness of θ_t |
| OCSVM | one-class SVM outlierness of θ_t |
| DeepSVDD | distance of θ_t's learned embedding from the normal-data centre |

HistMean and GHMM read the agent's own history; the other three judge each day
on its own against the population's normal days. Together they measure how much
of the anomaly signal is already in the mixtures, before any model of the
visits themselves.
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from src.detectors.base import Detector
from src.features.visits import VisitSequences

_EPS = 1e-12


def _recorded(sequences: VisitSequences) -> np.ndarray:
    return sequences.template >= 0


class ThetaDetector(Detector):
    """Shared plumbing: which days a θ-space detector is fitted on."""

    config_path = "configs/baselines/theta.yaml"

    def train_mixtures(self) -> list[np.ndarray]:
        """Each fitted agent's train-split mixtures, in day order.

        Agents with fewer than two train days are skipped. When the config caps
        the agent count, whole agents are sampled so every agent's days stay
        contiguous.
        """
        sequences = self.cohort.sequences
        train = (sequences.split == 0) & _recorded(sequences)
        agents = np.flatnonzero(train.sum(axis=1) >= 2)
        max_agents = self.cfg["fit"]["max_agents"]
        if max_agents and len(agents) > max_agents:
            rng = np.random.default_rng(self.seed)
            agents = np.sort(rng.choice(agents, size=max_agents, replace=False))
        return [sequences.theta[a, train[a]].astype(np.float64) for a in agents]


class PointwiseThetaDetector(ThetaDetector):
    """A detector that scores every day independently of its history."""

    _last = None    # (mixtures, scores) of the previous call; see score()

    def fit(self) -> dict:
        days = np.vstack(self.train_mixtures()).astype(np.float32)
        self.fit_days(days)
        self._last = None
        return {"fit_days": len(days)}

    def fit_days(self, days: np.ndarray) -> None:
        raise NotImplementedError

    def score_days(self, days: np.ndarray) -> np.ndarray:
        """``(n, K)`` mixtures -> ``(n,)`` scores, higher = more anomalous."""
        raise NotImplementedError

    def score(self, sequences: VisitSequences) -> np.ndarray:
        # Every day is scored, a day without stays on its all-zero mixture.
        theta = np.where(_recorded(sequences)[..., None], sequences.theta, 0.0)[:, 1:]
        theta = theta.astype(np.float32)
        # A day's score depends on its mixture alone, so when the previous call
        # scored the same cohort only the days whose mixture changed (the
        # injected ones) need scoring again; the OCSVM makes this worth it.
        cached = self._last
        if cached is not None and cached[0].shape == theta.shape:
            grid = cached[1].copy()
            redo = (theta != cached[0]).any(axis=-1)
        else:
            grid = np.empty(theta.shape[:2], dtype=np.float32)
            redo = np.ones(theta.shape[:2], dtype=bool)
        if redo.any():
            grid[redo] = self.score_days(theta[redo])
        self._last = (theta.copy(), grid.copy())
        return grid

    def calibration_scores(self, sequences: VisitSequences, days: np.ndarray) -> np.ndarray:
        # A window scored on its own (NUMOSIM) calibrates on every day of the fit
        # window, those without stays included.
        if self.cohort.test_window is not None:
            days = np.ones_like(days)
        return super().calibration_scores(sequences, days)


# ---------------------------------------------------------------------------
# History-aware
# ---------------------------------------------------------------------------

class HistMean(ThetaDetector):
    """KL divergence of a day's mixture from the agent's running average.

    The history is expanding and strictly causal, so a spliced day enters the
    history of the days after it exactly as it enters IBAD's. Fits nothing.
    """

    name = "histmean"
    display_name = "HistMean"

    def fit(self) -> dict:
        return {}

    def score(self, sequences: VisitSequences) -> np.ndarray:
        theta = sequences.theta.astype(np.float64)
        recorded = _recorded(sequences)
        totals = np.cumsum(np.where(recorded[..., None], theta, 0.0), axis=1)
        counts = np.cumsum(recorded, axis=1)

        prior = totals[:, :-1] / np.maximum(counts[:, :-1], 1)[..., None]
        prior /= np.clip(prior.sum(axis=-1, keepdims=True), _EPS, None)
        observed = theta[:, 1:]
        kl = (observed * (np.log(observed + _EPS) - np.log(prior + _EPS))).sum(axis=-1)

        # A day without stays or without an earlier one scores 0, "no score".
        scored = recorded[:, 1:] & (counts[:, :-1] > 0)
        return np.where(scored, np.maximum(kl, 0.0), 0.0).astype(np.float32)

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


class GHMM(ThetaDetector):
    """One-step-ahead negative log-likelihood under a Gaussian HMM.

    The HMM has one hidden state per template and diagonal Gaussian emissions
    over mixtures, fitted on each agent's train days as separate sequences so
    no transition crosses an agent boundary. A day scores by
    −log Σ_k p(state k | θ_<t) · e_k(θ_t), where each day's emission densities
    e_k are rescaled so the best-fitting state has density 1: the score says
    how unexpected the day's best state is, not how well the day fits any state.
    """

    name = "ghmm"
    display_name = "GHMM"

    def fit(self) -> dict:
        from hmmlearn.hmm import GaussianHMM

        sequences = [s.astype(np.float32) for s in self.train_mixtures()]
        self.hmm = GaussianHMM(
            n_components=self.cohort.sequences.n_templates, covariance_type="diag",
            n_iter=self.cfg["ghmm"]["n_iter"], tol=self.cfg["ghmm"]["tol"],
            random_state=self.seed, implementation="scaling",
        )
        self.hmm.fit(np.vstack(sequences), [len(s) for s in sequences])
        return {"fit_days": int(sum(len(s) for s in sequences)),
                "converged": bool(self.hmm.monitor_.converged)}

    def _emission(self, x: np.ndarray) -> np.ndarray:
        """``(n, D)`` mixtures -> ``(n, K)`` Gaussian densities, each row scaled to a maximum of 1."""
        variances = self.hmm.covars_
        if variances.ndim == 3:
            variances = np.diagonal(variances, axis1=1, axis2=2)
        diff = x[:, None, :] - self.hmm.means_[None]
        log_density = -0.5 * (
            x.shape[1] * np.log(2 * np.pi)
            + np.log(variances + 1e-300).sum(axis=1)[None]
            + (diff ** 2 / (variances[None] + 1e-300)).sum(axis=2)
        )
        return np.exp(log_density - log_density.max(axis=1, keepdims=True))

    def score(self, sequences: VisitSequences) -> np.ndarray:
        theta = sequences.theta.astype(np.float64)
        recorded = _recorded(sequences)
        n_agents, n_days = recorded.shape
        startprob, transmat = self.hmm.startprob_, self.hmm.transmat_

        # Forward filtering; alpha is the filtered state posterior after each
        # day, and a missing day carries the prediction forward.
        alpha = np.where(recorded[:, :1], startprob[None] * self._emission(theta[:, 0]),
                         startprob[None])
        alpha /= alpha.sum(axis=1, keepdims=True) + 1e-300
        grid = np.zeros((n_agents, n_days - 1), dtype=np.float32)   # 0: no score
        for t in range(1, n_days):
            pred = alpha @ transmat
            joint = pred * self._emission(theta[:, t])
            here = recorded[:, t]
            grid[here, t - 1] = -np.log(joint.sum(axis=1)[here] + 1e-300)
            alpha = np.where(here[:, None],
                             joint / (joint.sum(axis=1, keepdims=True) + 1e-300),
                             pred / (pred.sum(axis=1, keepdims=True) + 1e-300))
        return grid

    def save(self, path: Path) -> None:
        joblib.dump(self.hmm, path)

    def load(self, path: Path) -> None:
        self.hmm = joblib.load(path)


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------

class IsolationForestDetector(PointwiseThetaDetector):
    """Isolation Forest over the population's train-day mixtures."""

    name = "iforest"
    display_name = "IF"

    def fit_days(self, days: np.ndarray) -> None:
        from sklearn.ensemble import IsolationForest

        self.forest = IsolationForest(
            n_estimators=self.cfg["iforest"]["n_estimators"],
            random_state=self.seed, n_jobs=-1,
        ).fit(days)

    def score_days(self, days: np.ndarray) -> np.ndarray:
        return -self.forest.decision_function(days)

    def save(self, path: Path) -> None:
        joblib.dump(self.forest, path)

    def load(self, path: Path) -> None:
        self.forest, self._last = joblib.load(path), None


class OCSVMDetector(PointwiseThetaDetector):
    """One-class SVM (RBF kernel) over standardised train-day mixtures."""

    name = "ocsvm"
    display_name = "OCSVM"

    def fit_days(self, days: np.ndarray) -> None:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM

        self.svm = make_pipeline(
            StandardScaler(),
            OneClassSVM(kernel="rbf", nu=self.cfg["ocsvm"]["nu"], gamma="scale"),
        ).fit(days)

    def score_days(self, days: np.ndarray) -> np.ndarray:
        # The SVM scores one row at a time on one core; NUMOSIM has millions.
        chunks = np.array_split(days, max(1, len(days) // 100_000))
        scores = joblib.Parallel(n_jobs=self.cfg["ocsvm"]["n_jobs"])(
            joblib.delayed(self.svm.decision_function)(chunk) for chunk in chunks)
        return -np.concatenate(scores)

    def save(self, path: Path) -> None:
        joblib.dump(self.svm, path)

    def load(self, path: Path) -> None:
        self.svm, self._last = joblib.load(path), None


class _SVDDNet(nn.Module):
    """Two-layer MLP encoder."""

    def __init__(self, input_dim: int, hidden: int, repr_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, repr_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSVDDDetector(PointwiseThetaDetector):
    """One-class Deep SVDD (Ruff et al., 2018) over train-day mixtures.

    The network is trained to pull normal days towards a fixed centre ``c``,
    the mean initial embedding of (up to) the first 10,000 train days; a day's
    score is its squared distance from ``c``.
    """

    name = "deepsvdd"
    display_name = "DeepSVDD"
    checkpoint_ext = ".pt"

    def __init__(self, cfg, cohort, seed, device):
        super().__init__(cfg, cohort, seed, device)
        params = cfg["deepsvdd"]
        self.net = _SVDDNet(cohort.sequences.n_templates, params["hidden"],
                            params["repr_dim"]).to(device)
        self.centre = torch.zeros(params["repr_dim"], device=device)

    @torch.no_grad()
    def _embed(self, days: np.ndarray) -> torch.Tensor:
        self.net.eval()
        x = torch.from_numpy(days).float()
        return torch.cat([
            self.net(x[s:s + 65536].to(self.device)) for s in range(0, len(x), 65536)
        ])

    def fit_days(self, days: np.ndarray) -> None:
        params = self.cfg["deepsvdd"]
        # Re-seeded here so the network's initial weights depend on the seed
        # alone, whatever was drawn before.
        torch.manual_seed(self.seed)
        self.net = _SVDDNet(days.shape[1], params["hidden"], params["repr_dim"]).to(self.device)
        self.centre = self._embed(days[:10000]).mean(dim=0)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=params["lr"])
        x = torch.from_numpy(days).float()
        for _ in range(params["epochs"]):
            self.net.train()
            for idx in torch.randperm(len(x)).split(params["batch_size"]):
                batch = x[idx].to(self.device)
                loss = ((self.net(batch) - self.centre) ** 2).sum(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def score_days(self, days: np.ndarray) -> np.ndarray:
        return ((self._embed(days) - self.centre) ** 2).sum(dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        torch.save({"net": self.net.state_dict(), "centre": self.centre.cpu()}, path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state["net"])
        self.centre, self._last = state["centre"].to(self.device), None

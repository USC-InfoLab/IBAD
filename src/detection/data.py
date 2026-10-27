"""Assembling each dataset's visit sequences and template mixtures.

Module 1 fits the templates; this module re-encodes every day through that same
fitted model — including the val and test days module 1 never saw — and packs
the result into per-agent sequences.

Each adapter returns a :class:`Cohort`. Adding a dataset means adding one
adapter and one config entry.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import numosim as numosim_data
from src.features import geolife as geolife_features
from src.features import numosim as numosim_features
from src.features.visits import (
    VisitSequences, assign_chronological_split, build_sequences, calendar_window, visit_hours,
)
from src.templates import artifacts
from src.utils.config import ROOT


@dataclass
class Cohort:
    """Everything an evaluation run needs for one dataset."""

    dataset: str
    sequences: VisitSequences          # what detectors are fitted on
    activities: list[str]              # name of visit activity id k is activities[k - 1]
    utc_offset_hours: float            # the timestamps' offset from UTC
    # The test window as a sequence of its own, when the dataset is scored that
    # way (NUMOSIM); None scores the test split of ``sequences``.
    test_window: VisitSequences | None = None


def _theta_for(documents, lda) -> np.ndarray:
    """Encode documents through module 1's fitted model, rows on the simplex."""
    theta = lda.transform(documents.X).astype(np.float32)
    return theta / theta.sum(axis=1, keepdims=True).clip(min=1e-12)


def _slots(saved: dict) -> tuple[int, int]:
    """Slot grid the templates were fitted on; documents must be encoded the same way."""
    n_slots = saved["vocab"]["n_slots"]
    return n_slots, 24 * 60 // n_slots


def load_geolife(dataset_cfg: dict, cfg: dict, templates: str = "geolife") -> Cohort:
    """All three module-0 splits, with template mixtures from ``templates``.

    The visits always use Geolife's own activity vocabulary. ``templates`` names
    the module-1 artifact the mixtures come from: Geolife's own by default, or
    another dataset's (the transferability experiment uses NHTS's), in which
    case each day is encoded against that artifact's vocabulary first.
    """
    native = artifacts.load(ROOT / cfg["templates_dir"], "geolife")
    labels = native["vocab"]["labels"]
    source = native if templates == "geolife" else artifacts.load(
        ROOT / cfg["templates_dir"], templates)
    n_slots, slot_minutes = _slots(source)

    stays = geolife_features.load_stays(dataset_cfg, which=("train", "val", "test"))
    documents, index = geolife_features.documents_from_stays(
        stays, source["vocab"]["labels"], n_slots, slot_minutes
    )
    theta = _theta_for(documents, source["lda"])

    index = assign_chronological_split(
        index, dataset_cfg["train_frac"], dataset_cfg["val_frac"]
    )
    # assign_chronological_split reorders rows; realign theta to the new order.
    order = index.index.to_numpy()
    theta = theta[order]
    index = index.reset_index(drop=True)

    activity_index = {label: i for i, label in enumerate(labels)}
    arrival, departure = visit_hours(stays["ArrivingTime"], stays["LeavingTime"])
    stay_table = pd.DataFrame({
        "agent_id": stays["AgentID"],
        "date": stays["ArrivingTime"].dt.date,
        "arrival_hour": arrival,
        "departure_hour": departure,
        # 0 is the embedding's padding row, so activity ids start at 1. An
        # activity the templates never saw counts as Home, as in the documents.
        "bundle_idx": stays["nhts_category"].map(activity_index)
        .fillna(activity_index["Home"]).astype(int) + 1,
    })

    sequences = build_sequences(
        stay_table, index, theta,
        n_bundles=len(labels) + 1,
        visit_percentile=cfg["visit_percentile"],
    )
    return Cohort(dataset="geolife", sequences=sequences, activities=list(labels),
                  utc_offset_hours=dataset_cfg["utc_offset_hours"])


def _numosim_stay_table(stays: pd.DataFrame, poi_to_bundle: dict) -> pd.DataFrame:
    """NUMOSIM stays in the shared stay-table schema."""
    start = pd.to_datetime(stays["start_datetime"])
    end = pd.to_datetime(stays["end_datetime"])
    arrival, departure = visit_hours(start, end)
    table = pd.DataFrame({
        "agent_id": stays["agent_id"].to_numpy(),
        "date": start.dt.date,
        "arrival_hour": arrival,
        "departure_hour": departure,
        # 0 is the embedding's padding row, so bundle ids start at 1.
        "bundle_idx": stays["poi_id"].astype(int).map(poi_to_bundle) + 1,
    })
    return table.dropna(subset=["bundle_idx"]).astype({"bundle_idx": int})


def load_numosim(dataset_cfg: dict, cfg: dict) -> Cohort:
    """NUMOSIM's native train/test split.

    Each window is built from its own stay file and laid out on its calendar:
    every agent's sequence holds each of the window's dates, a date with no
    stays being an empty day. Detectors are fitted on the train window (its
    last ``val_frac`` held out as validation, none by default). The
    test window is a sequence of its own: a test day is forecast from the test
    days before it, and each agent's standardisation comes from its
    train-window scores.
    """
    raw_dir = ROOT / dataset_cfg["raw_dir"]
    numosim_data.download(
        raw_dir, dataset_cfg["osf_project"],
        files=numosim_data.TRAIN_FILES + numosim_data.TEST_FILES,
    )

    saved = artifacts.load(ROOT / cfg["templates_dir"], "numosim")
    lda, bundle_labels = saved["lda"], saved["vocab"]["labels"]
    n_slots, slot_minutes = _slots(saved)

    pois = pd.read_parquet(raw_dir / "poi.parquet")
    labels, poi_to_bundle = numosim_features.build_bundle_vocabulary(pois)
    if labels != bundle_labels:
        raise ValueError(
            "NUMOSIM bundle vocabulary differs from the one the templates were "
            "fitted on; re-run template discovery."
        )

    train = pd.read_parquet(raw_dir / "stay_points_train.parquet")
    test = pd.read_parquet(raw_dir / "stay_points_test_truth.parquet")

    keep = _sample_agents(train, test, dataset_cfg["n_agents"], cfg["seed"])
    train = train[train["agent_id"].isin(keep)]
    test = test[test["agent_id"].isin(keep)]

    windows = [_numosim_window(stays, poi_to_bundle, len(labels), lda, n_slots, slot_minutes)
               for stays in (train, test)]
    train_dates, test_dates = (np.sort(index["date"].unique()) for index, _ in windows)

    # Validation, when there is any, is carved from the tail of the train
    # window so the test window is never read for tuning.
    n_val = int(round(len(train_dates) * dataset_cfg["val_frac"]))
    val_from = train_dates[-n_val] if n_val else test_dates[0]
    index = pd.concat([index for index, _ in windows], ignore_index=True)
    index["split"] = np.select([index["date"] >= test_dates[0], index["date"] >= val_from],
                               ["test", "val"], default="train")

    calendar = np.concatenate([train_dates, test_dates]).astype("datetime64[D]")
    sequences = build_sequences(
        _numosim_stay_table(pd.concat([train, test], ignore_index=True), poi_to_bundle),
        index, np.concatenate([theta for _, theta in windows]),
        n_bundles=len(labels) + 1, visit_percentile=cfg["visit_percentile"], calendar=calendar,
    )
    n_train = len(train_dates)
    return Cohort(dataset="numosim", sequences=calendar_window(sequences, 0, n_train),
                  activities=list(labels), utc_offset_hours=dataset_cfg["utc_offset_hours"],
                  test_window=calendar_window(sequences, n_train, len(calendar)))


def _sample_agents(train: pd.DataFrame, test: pd.DataFrame, n_agents: int | None,
                   seed: int) -> np.ndarray:
    """A uniform random sample of the agents seen in both windows (all when ``None``)."""
    both = np.intersect1d(train["agent_id"].unique(), test["agent_id"].unique())
    if n_agents is None or n_agents >= len(both):
        return both
    print(f"  numosim: {n_agents:,} of {len(both):,} agents, sampled with seed {seed}")
    return np.random.default_rng(seed).choice(both, size=n_agents, replace=False)


def _numosim_window(
    stays: pd.DataFrame,
    poi_to_bundle: dict,
    n_bundles: int,
    lda,
    n_slots: int,
    slot_minutes: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """One window's agent-day index and template mixtures, from its own stays.

    The index comes from the same document builder module 1 used, so the
    mixtures and the sequences cannot disagree about which day is which. A
    stay running past the window's last day would open documents beyond it;
    those belong to no window and are dropped.
    """
    documents_X, index = numosim_features.build_matrix(
        stays, poi_to_bundle, n_bundles, n_slots, slot_minutes
    )
    last = pd.to_datetime(stays["start_datetime"]).dt.date.max()
    inside = (index["date"] <= last).to_numpy()
    theta = lda.transform(documents_X[inside]).astype(np.float32)
    theta /= theta.sum(axis=1, keepdims=True).clip(min=1e-12)
    return index[inside].reset_index(drop=True), theta


LOADERS = {"geolife": load_geolife, "numosim": load_numosim}

# Cohorts are seed-independent by design: which agents and days make up the
# benchmark is fixed by the config, so replicates vary only the model. That also
# means a run of several seeds can build each cohort once.
_CACHE: dict[tuple[str, str | None], Cohort] = {}


def load_cohort(dataset: str, cfg: dict, templates: str | None = None) -> Cohort:
    """The cohort for ``dataset`` under the benchmark config ``cfg``.

    ``templates`` swaps in another dataset's module-1 templates for the
    mixtures (Geolife only); ``None`` keeps the dataset's own.
    """
    if dataset not in LOADERS:
        raise KeyError(f"Unknown dataset {dataset!r}; expected one of {list(LOADERS)}")
    key = (dataset, templates)
    if key not in _CACHE:
        kwargs = {} if templates is None else {"templates": templates}
        _CACHE[key] = LOADERS[dataset](cfg["datasets"][dataset], cfg, **kwargs)
    return _CACHE[key]

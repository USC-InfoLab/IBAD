"""NUMOSIM stay-points -> one-hot activity-bundle documents.

A NUMOSIM POI carries a set of activity types rather than a single one, so the
vocabulary is the set of distinct POI activity *bundles*.
"""

from datetime import date

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data import numosim
from src.features.documents import Documents
from src.utils.config import ROOT


def _parse_codes(value) -> list[int]:
    """Parse a POI act_types cell, which may be an array or its string form."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [int(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip().strip("[]").replace(" ", "")
        if not stripped:
            return []
        try:
            return [int(part) for part in stripped.split(",") if part]
        except ValueError:
            return []
    return []


def build_bundle_vocabulary(pois: pd.DataFrame) -> tuple[list[str], dict[int, int]]:
    """Return ('+'-joined bundle labels, poi_id -> bundle index).

    Bundles are ordered by (size, sorted names) so the vocabulary is stable
    across runs.
    """
    bundles = pois["act_types"].apply(
        lambda cell: frozenset(
            numosim.ACTIVITY_CODES.get(code, "SomethingElse")
            for code in _parse_codes(cell)
        )
    )
    ordered = sorted(bundles.unique(), key=lambda b: (len(b), sorted(b)))
    bundle_to_idx = {bundle: i for i, bundle in enumerate(ordered)}

    labels = ["+".join(sorted(bundle)) if bundle else "∅" for bundle in ordered]
    poi_to_bundle = dict(
        zip(pois["poi_id"].astype(int), bundles.map(bundle_to_idx).astype(int))
    )
    return labels, poi_to_bundle


def _expand_stays(stays: pd.DataFrame, n_slots: int, slot_minutes: int) -> pd.DataFrame:
    """One row per (stay, calendar date), splitting stays that cross midnight."""
    start = pd.to_datetime(stays["start_datetime"])
    end = pd.to_datetime(stays["end_datetime"])

    start_slot = ((start.dt.hour * 60 + start.dt.minute) // slot_minutes)
    end_slot = ((end.dt.hour * 60 + end.dt.minute) // slot_minutes)
    start_slot = start_slot.clip(0, n_slots - 1).to_numpy(np.int32)
    end_slot = end_slot.clip(0, n_slots - 1).to_numpy(np.int32)

    start_ord = np.array([d.toordinal() for d in start.dt.date], dtype=np.int32)
    end_ord = np.array([d.toordinal() for d in end.dt.date], dtype=np.int32)

    spans = (end_ord - start_ord + 1).astype(np.int32)
    source = np.repeat(np.arange(len(stays), dtype=np.int32), spans)
    day_offset = np.arange(spans.sum(), dtype=np.int32) - np.repeat(
        np.concatenate([[0], np.cumsum(spans[:-1])]), spans
    )

    expanded_start = start_slot[source].copy()
    expanded_end = end_slot[source].copy()
    # Middle and trailing days of a multi-day stay run to the day boundary.
    expanded_start[day_offset != 0] = 0
    expanded_end[day_offset != (spans[source] - 1)] = n_slots - 1

    ordinals = start_ord[source] + day_offset
    to_date = {o: date.fromordinal(int(o)) for o in np.unique(ordinals)}

    return pd.DataFrame({
        "agent_id": stays["agent_id"].to_numpy()[source],
        "poi_id": stays["poi_id"].to_numpy()[source].astype(np.int64),
        "date": [to_date[o] for o in ordinals],
        "start_slot": expanded_start,
        "end_slot": expanded_end,
    })


def build_matrix(
    stays: pd.DataFrame,
    poi_to_bundle: dict[int, int],
    n_bundles: int,
    n_slots: int,
    slot_minutes: int,
) -> tuple[csr_matrix, pd.DataFrame]:
    """Build the slot-major one-hot matrix and its agent-day index."""
    expanded = _expand_stays(stays, n_slots, slot_minutes)

    index = (
        expanded[["agent_id", "date"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(row_idx=lambda d: np.arange(len(d), dtype=np.int32))
    )
    expanded = expanded.merge(index, on=["agent_id", "date"], how="left")
    expanded["bundle_idx"] = expanded["poi_id"].map(poi_to_bundle)
    expanded = expanded.dropna(subset=["bundle_idx", "row_idx"])

    # Fan each stay out over the slots it occupies.
    widths = (
        (expanded["end_slot"] - expanded["start_slot"] + 1)
        .clip(lower=0)
        .to_numpy(np.int32)
    )
    source = np.repeat(np.arange(len(expanded), dtype=np.int32), widths)
    offsets = np.arange(widths.sum(), dtype=np.int32) - np.repeat(
        np.concatenate([[0], np.cumsum(widths[:-1])]), widths
    )

    slots = expanded["start_slot"].to_numpy(np.int32)[source] + offsets
    rows = expanded["row_idx"].to_numpy(np.int32)[source]
    bundles = expanded["bundle_idx"].to_numpy(np.int32)[source]

    X = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, slots * n_bundles + bundles)),
        shape=(len(index), n_slots * n_bundles),
        dtype=np.float32,
    )
    # Overlapping stays can hit the same cell twice; keep the matrix binary.
    X.sum_duplicates()
    X.data[:] = 1.0

    return X, index.drop(columns="row_idx")


def build_documents(dataset_cfg: dict, n_slots: int, slot_minutes: int) -> Documents:
    """Download if needed, then encode every NUMOSIM agent-day as a document."""
    raw_dir = ROOT / dataset_cfg["raw_dir"]
    numosim.download(raw_dir, dataset_cfg["osf_project"])
    stays, pois = numosim.load_stays(
        raw_dir,
        train_start=dataset_cfg["train_start"],
        train_end=dataset_cfg["train_end"],
    )

    labels, poi_to_bundle = build_bundle_vocabulary(pois)
    X, index = build_matrix(
        stays, poi_to_bundle, len(labels), n_slots, slot_minutes
    )
    return Documents(
        name="numosim", X=X, labels=labels, n_slots=n_slots, index=index
    )

"""Geolife stay-points -> one-hot activity documents."""

import numpy as np
import pandas as pd

from src.data.geolife import splits
from src.features.documents import Documents, timelines_to_matrix
from src.utils.config import ROOT


def _to_slot(ts: pd.Timestamp, slot_minutes: int, n_slots: int) -> int:
    return min((ts.hour * 60 + ts.minute) // slot_minutes, n_slots - 1)


def _expand_stays(stays: pd.DataFrame, slot_minutes: int, n_slots: int) -> pd.DataFrame:
    """One row per (stay, calendar date), splitting stays that cross midnight."""
    segments = []
    for stay in stays.itertuples(index=False):
        arrive, leave = stay.ArrivingTime, stay.LeavingTime
        if pd.isna(arrive) or pd.isna(leave) or leave <= arrive:
            continue

        cursor = arrive
        while cursor.date() < leave.date():
            segments.append({
                "agent_id": stay.AgentID,
                "date": cursor.date(),
                "category": stay.nhts_category,
                "start_slot": _to_slot(cursor, slot_minutes, n_slots),
                "end_slot": n_slots,
            })
            cursor = pd.Timestamp(cursor.date()) + pd.Timedelta(days=1)

        start_slot = _to_slot(cursor, slot_minutes, n_slots)
        end_slot = _to_slot(leave, slot_minutes, n_slots)
        # A stay shorter than one slot would otherwise fill nothing.
        if end_slot <= start_slot:
            end_slot = min(start_slot + 1, n_slots)
        segments.append({
            "agent_id": stay.AgentID,
            "date": cursor.date(),
            "category": stay.nhts_category,
            "start_slot": start_slot,
            "end_slot": end_slot,
        })

    if not segments:
        raise ValueError("No usable Geolife stays (all had invalid timestamps).")
    return pd.DataFrame(segments)


def build_timelines(
    stays: pd.DataFrame,
    act_to_idx: dict[str, int],
    n_slots: int,
    slot_minutes: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build per-(agent, date) ordinal timelines. Uncovered slots stay Home."""
    home_idx = act_to_idx["Home"]
    segments = _expand_stays(stays, slot_minutes, n_slots)
    segments["activity_idx"] = (
        segments["category"].map(act_to_idx).fillna(home_idx).astype(np.int16)
    )

    # Index days by ARRIVING date only: the extra rows that midnight-crossing
    # stays contribute have no stay starting on them and are not days of their own.
    index = (
        stays.assign(agent_id=stays["AgentID"], date=stays["ArrivingTime"].dt.date)
        .groupby(["agent_id", "date"], sort=True)
        .size()
        .rename("n_staypoints")
        .reset_index()
    )
    row_of = {
        (agent, day): row
        for row, (agent, day) in enumerate(zip(index["agent_id"], index["date"]))
    }

    timelines = np.full((len(index), n_slots), home_idx, dtype=np.int8)
    for segment in segments.itertuples(index=False):
        row = row_of.get((segment.agent_id, segment.date))
        if row is None or segment.end_slot <= segment.start_slot:
            continue
        timelines[row, segment.start_slot:segment.end_slot] = segment.activity_idx

    return timelines, index


def load_stays(dataset_cfg: dict, which: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Read module-0 splits and concatenate them.

    Defaults to the single split named in the config (module 1 fits templates on
    train alone); module 2 asks for every split so it can score val and test.
    """
    names = which if which is not None else (dataset_cfg["split"],)
    processed_dir = ROOT / dataset_cfg["processed_dir"]
    return pd.concat(
        [splits.load_split(processed_dir, name) for name in names], ignore_index=True
    )


def documents_from_stays(
    stays: pd.DataFrame,
    labels: list[str],
    n_slots: int,
    slot_minutes: int,
) -> tuple[Documents, pd.DataFrame]:
    """Encode a stay table against a fixed vocabulary.

    Module 2 passes the vocabulary saved with the fitted LDA so the columns line
    up with what the model was trained on. Activities outside that vocabulary
    keep their slots at Home rather than shifting every later column.
    """
    act_to_idx = {label: i for i, label in enumerate(labels)}
    unknown = set(stays["nhts_category"].unique()) - set(act_to_idx)
    if unknown:
        print(f"  geolife: {sorted(unknown)} absent from the fitted vocabulary; "
              "those stays fall back to Home")

    timelines, index = build_timelines(stays, act_to_idx, n_slots, slot_minutes)
    documents = Documents(
        name="geolife",
        X=timelines_to_matrix(timelines, len(labels)),
        labels=labels,
        n_slots=n_slots,
        index=index,
    )
    return documents, index


def build_documents(dataset_cfg: dict, n_slots: int, slot_minutes: int) -> Documents:
    """Load one module-0 split and encode every agent-day as a document."""
    stays = load_stays(dataset_cfg)
    labels = splits.activity_vocabulary(stays)
    documents, _ = documents_from_stays(stays, labels, n_slots, slot_minutes)
    return documents

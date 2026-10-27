"""NHTS person-days -> one-hot activity documents."""

import math

import numpy as np
import pandas as pd

from src.data import nhts
from src.features.documents import Documents, timelines_to_matrix
from src.utils.config import ROOT


def _to_slot(hhmm, slot_minutes: int, n_slots: int) -> int:
    try:
        value = int(hhmm)
    except (ValueError, TypeError):
        return 0
    return min((value // 100 * 60 + value % 100) // slot_minutes, n_slots - 1)


def build_timelines(
    trips_df: pd.DataFrame,
    person_days: list[str],
    act_to_idx: dict[str, int],
    n_slots: int,
    slot_minutes: int,
) -> np.ndarray:
    """Build ``(n_person_days, n_slots)`` ordinal activity timelines.

    Slots are filled per trip, in travel order:
      ``[depart, arrive)``      -> origin activity (travel shown as where the
                                   traveller came from, not as Home)
      ``[arrive, arrive+dwell)``-> destination activity
      everything after         -> Home

    Later trips overwrite earlier ones, so uncovered slots stay Home.
    """
    home_idx = act_to_idx["Home"]
    row_of = {pid: i for i, pid in enumerate(person_days)}
    timelines = np.full((len(person_days), n_slots), home_idx, dtype=np.int8)

    ordered = trips_df.sort_values(["PERSON_DAY_ID", "TDTRPNUM"])
    dwell_minutes = pd.to_numeric(ordered["DWELTIME"], errors="coerce")

    for trip, dwell in zip(ordered.itertuples(index=False), dwell_minutes):
        row = row_of.get(trip.PERSON_DAY_ID)
        if row is None:
            continue

        depart = _to_slot(trip.STRTTIME, slot_minutes, n_slots)
        arrive = _to_slot(trip.ENDTIME, slot_minutes, n_slots)
        origin = act_to_idx.get(trip.ORIGIN_CATEGORY, home_idx)
        dest = act_to_idx.get(trip.DEST_CATEGORY, home_idx)

        dwell_slots = (
            max(0, math.ceil(dwell / slot_minutes))
            if pd.notna(dwell) and dwell > 0
            else 0
        )
        end = min(arrive + dwell_slots, n_slots)

        timelines[row, depart:arrive] = origin
        timelines[row, arrive:end] = dest
        if end < n_slots:
            timelines[row, end:] = home_idx

    return timelines


def build_documents(dataset_cfg: dict, n_slots: int, slot_minutes: int) -> Documents:
    """Load NHTS trips and encode every person-day as a document."""
    trips_raw = nhts.load_trips(ROOT / dataset_cfg["raw_dir"])
    trips_df = nhts.preprocess_trips(trips_raw, keep_other=dataset_cfg["keep_other"])

    labels = nhts.activity_vocabulary(trips_df)
    act_to_idx = {label: i for i, label in enumerate(labels)}
    person_days = nhts.person_day_ids(trips_df)

    timelines = build_timelines(
        trips_df, person_days, act_to_idx, n_slots, slot_minutes
    )
    return Documents(
        name="nhts",
        X=timelines_to_matrix(timelines, len(labels)),
        labels=labels,
        n_slots=n_slots,
        index=pd.DataFrame({"person_day_id": person_days}),
    )

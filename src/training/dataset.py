"""Batching for IBAD: one item per agent, its whole day sequence.

Agents differ in how many days they have, and days in how many visits, so a
batch is padded to its own maxima rather than the corpus maximum — on Geolife
the longest agent has hundreds of days while most have far fewer.

Times reach the model in IBAD's input encoding (see :func:`encode_times`).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.visits import VisitSequences

_EPOCH = np.datetime64("1970-01-01", "D")


def _midnight(day, zone: ZoneInfo) -> float:
    """Unix seconds of ``day``'s midnight in ``zone``."""
    return datetime.fromisoformat(str(day)).replace(tzinfo=zone).timestamp()


def encode_times(sequences: VisitSequences, day_zone: str,
                 stamp_zone: str) -> dict[str, np.ndarray]:
    """Visit and day times as IBAD reads them.

    * Visits: arrival and departure as Unix seconds in float32 — the visit
      date's midnight in ``stamp_zone`` (the zone the dataset's timestamps are
      read in) plus the visit's hours.
    * Days: days since the corpus's first date (recovered from ``day_ts``, so a
      window cut from a longer corpus keeps its origin), measured between local
      midnights in ``day_zone``, so a day after a daylight-saving change sits
      an hour off a whole number.

    A day without a date (padding, or an empty calendar day) gets zeros.
    """
    recorded = ~np.isnat(sequences.date)
    dates = np.where(recorded, sequences.date, _EPOCH)
    unique, inverse = np.unique(dates[recorded], return_inverse=True)

    def midnights(zone: str) -> np.ndarray:
        tz = ZoneInfo(zone)
        return np.array([_midnight(day, tz) for day in unique])

    midnight = np.zeros(dates.shape, dtype=np.int64)
    midnight[recorded] = midnights(stamp_zone).astype(np.int64)[inverse]

    def seconds(hours: np.ndarray) -> np.ndarray:
        exact = midnight[..., None] + np.rint(hours.astype(np.float64) * 3600.0).astype(np.int64)
        return np.where(sequences.visit_mask, exact, 0).astype(np.float32)

    origin = (dates[recorded]
              - sequences.day_ts[recorded].astype(np.int64).astype("timedelta64[D]")).min()
    first = _midnight(origin, ZoneInfo(day_zone))
    day_ts = np.zeros(dates.shape, dtype=np.float32)
    day_ts[recorded] = ((midnights(day_zone)[inverse] - first) / 86400.0).astype(np.float32)
    return {"arrival": seconds(sequences.arrival), "departure": seconds(sequences.departure),
            "day_ts": day_ts}


def _item(sequences: VisitSequences, times: dict, row: int, length: int) -> dict:
    """One agent's arrays, trimmed to its own days and its widest day."""
    mask = sequences.visit_mask[row][:length]
    width = max(int(mask.sum(axis=1).max()), 1) if mask.any() else 1
    return {
        "arrival": torch.from_numpy(times["arrival"][row][:length, :width]),
        "departure": torch.from_numpy(times["departure"][row][:length, :width]),
        "bundle_id": torch.from_numpy(sequences.bundle_id[row][:length, :width]).long(),
        "visit_mask": torch.from_numpy(mask[:, :width]).bool(),
        "day_ts": torch.from_numpy(times["day_ts"][row][:length]),
        "theta": torch.from_numpy(sequences.theta[row][:length]).float(),
        "split": torch.from_numpy(sequences.split[row][:length]).long(),
        "dow": torch.from_numpy(sequences.dow[row][:length].astype(np.int64)),
        "agent_idx": row,
        "seq_len": length,
    }


def collate(items: list[dict]) -> dict:
    """Pad a list of agent items to the batch's own (days, visits) maxima."""
    batch_size = len(items)
    n_days = max(item["seq_len"] for item in items)
    n_visits = max(item["arrival"].shape[1] for item in items)
    n_templates = items[0]["theta"].shape[1]

    out = {
        "arrival": torch.zeros(batch_size, n_days, n_visits),
        "departure": torch.zeros(batch_size, n_days, n_visits),
        "bundle_id": torch.zeros(batch_size, n_days, n_visits, dtype=torch.long),
        "visit_mask": torch.zeros(batch_size, n_days, n_visits, dtype=torch.bool),
        "day_ts": torch.zeros(batch_size, n_days),
        "theta": torch.zeros(batch_size, n_days, n_templates),
        "split": torch.full((batch_size, n_days), -1, dtype=torch.long),
        "dow": torch.full((batch_size, n_days), -1, dtype=torch.long),
        "day_mask": torch.zeros(batch_size, n_days, dtype=torch.bool),
        "agent_idx": torch.empty(batch_size, dtype=torch.long),
    }
    for i, item in enumerate(items):
        length, width = item["seq_len"], item["arrival"].shape[1]
        for name in ("arrival", "departure", "bundle_id", "visit_mask"):
            out[name][i, :length, :width] = item[name]
        for name in ("day_ts", "theta", "split", "dow"):
            out[name][i, :length] = item[name]
        out["day_mask"][i, :length] = True
        out["agent_idx"][i] = item["agent_idx"]
    return out


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class AgentSequenceDataset(Dataset):
    """One item per agent; ``inputs`` holds the time zones of :func:`encode_times`."""

    def __init__(self, sequences: VisitSequences, inputs: dict):
        self.sequences = sequences
        self.times = encode_times(sequences, inputs["day_zone"], inputs["stamp_zone"])

    def __len__(self) -> int:
        return self.sequences.n_agents

    def __getitem__(self, row: int) -> dict:
        return _item(self.sequences, self.times, row, int(self.sequences.seq_lens[row]))

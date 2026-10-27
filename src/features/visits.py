"""Per-agent visit sequences — the input the forecaster consumes.

Module 1 collapses a day to a bag of `(slot, activity)` tokens. Module 2 needs
the day itself: an ordered list of visits with real arrival and departure times,
grouped by agent and by day.

Both datasets reach this module as the same **stay table** —
``agent_id, date, arrival_hour, departure_hour, bundle_idx`` — so one builder
serves both and adding a dataset means producing that table (see
:func:`visit_hours` for the time convention).

Days are packed compactly by default: only observed days are stored, and their
real timestamps carry the irregularity, so no calendar padding is needed. A
fixed-window dataset can instead lay its days out on the window's calendar
(NUMOSIM), where a date without a document is an
empty day in the sequence.

Each visit also records the gap since the agent's previous stay ended, measured
on the real data. It is part of the visit, so when a benchmark splices a donor
day into another agent-day the gap travels with the donor's visits: the spliced
day starts the way the donor day really did, instead of after an artificial gap
left by cutting the day out of one calendar and pasting it into another.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class VisitSequences:
    """Per-agent day sequences, padded to the longest agent."""

    arrival: np.ndarray      # (N, T, V) hours since the day's local midnight
    departure: np.ndarray    # (N, T, V) same clock; > 24 when the stay runs past midnight
    gap: np.ndarray          # (N, T, V) hours since the agent's previous stay ended; 0 for its first
    bundle_id: np.ndarray    # (N, T, V) int, 0 = padding
    visit_mask: np.ndarray   # (N, T, V) bool
    day_ts: np.ndarray       # (N, T) days since the first day in the corpus
    date: np.ndarray         # (N, T) calendar date, datetime64[D]; NaT = padding
    template: np.ndarray     # (N, T) argmax template, -1 = padding
    theta: np.ndarray        # (N, T, K) template mixture, all-zero = padding
    split: np.ndarray        # (N, T) 0/1/2, -1 = padding
    dow: np.ndarray          # (N, T) 0-6, -1 = padding
    seq_lens: np.ndarray     # (N,)
    agent_ids: list
    n_bundles: int

    @property
    def n_agents(self) -> int:
        return self.arrival.shape[0]

    @property
    def n_days(self) -> int:
        return self.arrival.shape[1]

    @property
    def n_templates(self) -> int:
        return self.theta.shape[2]


def visit_hours(start: pd.Series, end: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Arrival and departure as hours on the local clock of the arrival day.

    A visit is described by when in its day it happens, which is what makes two
    days comparable: the day's own position in time is carried separately by
    ``day_ts``. Arrival is the wall-clock hour in ``[0, 24)``; departure adds
    the stay's elapsed duration, so it exceeds 24 for a stay that runs past
    midnight. Computed from datetime fields, never from integer epochs, so the
    result does not depend on the timestamps' storage resolution.
    """
    arrival = start.dt.hour + start.dt.minute / 60.0 + start.dt.second / 3600.0
    departure = arrival + (end - start) / pd.Timedelta(hours=1)
    return arrival.to_numpy(np.float32), departure.to_numpy(np.float32)


def assign_chronological_split(
    index: pd.DataFrame, train_frac: float, val_frac: float
) -> pd.DataFrame:
    """Label each agent-day train/val/test by date order within its own agent.

    Splitting per agent rather than globally keeps every agent present in every
    split, which the per-agent embedding and the per-agent score baseline both
    need. Every agent is guaranteed at least one test day.
    """
    out = index.sort_values(["agent_id", "date"]).copy()

    labels: list[str] = []
    for _, agent_days in out.groupby("agent_id", sort=False):
        n = len(agent_days)
        n_train = max(0, round(n * train_frac))
        n_val = max(0, round(n * val_frac))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            shortage = 1 - (n - n_train - n_val)
            from_val = min(n_val, shortage)
            n_val -= from_val
            n_train -= shortage - from_val
        labels.extend(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)

    out["split"] = labels
    return out


def build_sequences(
    stays: pd.DataFrame,
    index: pd.DataFrame,
    theta_doc: np.ndarray,
    n_bundles: int,
    visit_percentile: float,
    calendar: np.ndarray | None = None,
) -> VisitSequences:
    """Pack a stay table into per-agent day sequences.

    ``index`` holds one row per agent-day with ``agent_id``, ``date`` and
    ``split``, in the same row order as ``theta_doc``. Days are placed by their
    chronological position within the agent, or, given ``calendar`` (every date
    of the corpus, in order), at their date's position in it, so that every
    agent spans the whole calendar. Each day keeps up to a percentile-derived
    number of visits.
    """
    days = index.copy()
    days["date"] = pd.to_datetime(days["date"]).dt.date
    days["theta_row"] = np.arange(len(days), dtype=np.int64)
    days = days.sort_values(["agent_id", "date"]).reset_index(drop=True)

    # Whole calendar days, counted on the dates themselves: going through epoch
    # seconds would pick up the machine's daylight-saving shifts.
    dates = days["date"].to_numpy().astype("datetime64[D]")
    first = dates.min() if calendar is None else calendar[0]
    days["day_ts"] = (dates - first).astype(np.float32)
    days["dow"] = pd.to_datetime(days["date"]).dt.dayofweek.astype(np.int8)
    days["split_id"] = days["split"].map(SPLIT_IDS).astype(np.int8)

    agent_ids = sorted(days["agent_id"].unique())
    agent_row = {agent: i for i, agent in enumerate(agent_ids)}
    days["agent_row"] = days["agent_id"].map(agent_row).astype(np.int32)
    if calendar is None:
        days["day_pos"] = days.groupby("agent_id").cumcount().astype(np.int32)
    else:
        days["day_pos"] = np.searchsorted(calendar, dates).astype(np.int32)

    n_agents = len(agent_ids)
    n_days = int(days["day_pos"].max()) + 1 if calendar is None else len(calendar)
    k = theta_doc.shape[1]

    day_ts = np.zeros((n_agents, n_days), dtype=np.float32)
    date = np.full((n_agents, n_days), np.datetime64("NaT"), dtype="datetime64[D]")
    template = np.full((n_agents, n_days), -1, dtype=np.int32)
    theta = np.zeros((n_agents, n_days, k), dtype=np.float32)
    split = np.full((n_agents, n_days), -1, dtype=np.int8)
    dow = np.full((n_agents, n_days), -1, dtype=np.int8)

    rows, positions = days["agent_row"].to_numpy(), days["day_pos"].to_numpy()
    day_ts[rows, positions] = days["day_ts"].to_numpy()
    date[rows, positions] = days["date"].to_numpy().astype("datetime64[D]")
    theta[rows, positions] = theta_doc[days["theta_row"].to_numpy()]
    template[rows, positions] = theta_doc[days["theta_row"].to_numpy()].argmax(axis=1)
    split[rows, positions] = days["split_id"].to_numpy()
    dow[rows, positions] = days["dow"].to_numpy()

    # Place each stay at (agent, day, visit) by arrival order within its day.
    located = stays.merge(
        days[["agent_id", "date", "agent_row", "day_pos", "day_ts"]],
        on=["agent_id", "date"], how="inner",
    ).sort_values(["agent_row", "day_pos", "arrival_hour"]).reset_index(drop=True)
    located["visit_pos"] = (
        located.groupby(["agent_row", "day_pos"]).cumcount().astype(np.int32)
    )

    # Gap since the previous stay, on the full stay table so a capped-away visit
    # still counts as the previous stay.
    day_start = located["day_ts"].to_numpy(np.float64) * 24.0
    start = day_start + located["arrival_hour"].to_numpy(np.float64)
    end = day_start + located["departure_hour"].to_numpy(np.float64)
    agent_rows = located["agent_row"].to_numpy()
    follows = np.r_[False, agent_rows[1:] == agent_rows[:-1]]
    located["gap_hours"] = np.where(follows, start - np.r_[0.0, end[:-1]], 0.0)

    per_day = located.groupby(["agent_row", "day_pos"]).size().to_numpy()
    n_visits = int(np.percentile(per_day, visit_percentile)) + 1
    located = located[located["visit_pos"] < n_visits]

    shape = (n_agents, n_days, n_visits)
    arrival = np.zeros(shape, dtype=np.float32)
    departure = np.zeros(shape, dtype=np.float32)
    gap = np.zeros(shape, dtype=np.float32)
    bundle_id = np.zeros(shape, dtype=np.int64)
    visit_mask = np.zeros(shape, dtype=bool)

    at = (
        located["agent_row"].to_numpy(),
        located["day_pos"].to_numpy(),
        located["visit_pos"].to_numpy(),
    )
    arrival[at] = located["arrival_hour"].to_numpy()
    departure[at] = located["departure_hour"].to_numpy()
    gap[at] = located["gap_hours"].to_numpy()
    bundle_id[at] = located["bundle_idx"].to_numpy()
    visit_mask[at] = True

    if calendar is None:
        seq_lens = days.groupby("agent_row").size().sort_index().to_numpy().astype(np.int32)
    else:
        seq_lens = np.full(n_agents, n_days, dtype=np.int32)
    print(
        f"  sequences: {n_agents} agents x {n_days} days x {n_visits} visits "
        f"(p{visit_percentile:g}); {int(visit_mask.sum()):,} visits placed"
    )
    return VisitSequences(
        arrival=arrival, departure=departure, gap=gap, bundle_id=bundle_id,
        visit_mask=visit_mask, day_ts=day_ts, date=date, template=template, theta=theta,
        split=split, dow=dow, seq_lens=seq_lens, agent_ids=agent_ids,
        n_bundles=n_bundles,
    )


def calendar_window(sequences: VisitSequences, start: int, stop: int) -> VisitSequences:
    """Calendar days ``start:stop`` of a calendar-laid-out cohort, as a cohort of their own.

    Agents and their order are unchanged, so an agent's row still indexes the
    same per-agent parameters; each day keeps its date and ``day_ts``.
    """
    def cut(values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(values[:, start:stop])

    return VisitSequences(
        arrival=cut(sequences.arrival), departure=cut(sequences.departure),
        gap=cut(sequences.gap), bundle_id=cut(sequences.bundle_id),
        visit_mask=cut(sequences.visit_mask), day_ts=cut(sequences.day_ts),
        date=cut(sequences.date), template=cut(sequences.template), theta=cut(sequences.theta),
        split=cut(sequences.split), dow=cut(sequences.dow),
        seq_lens=np.full(sequences.n_agents, stop - start, dtype=np.int32),
        agent_ids=sequences.agent_ids, n_bundles=sequences.n_bundles,
    )

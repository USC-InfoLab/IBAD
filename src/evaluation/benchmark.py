"""The splice benchmark: a cohort with behavioural anomalies in place, and where they are.

Every detector is evaluated the same way. It scores each agent's real days
(the *clean* version of the cohort) and the same days with anomalies present
(the *perturbed* version); the per-day labels say which days changed.

An anomaly is a *splice*: one of the agent's days is overwritten with the
behaviour of another day. Three policies decide where that donor day comes
from: ``intra`` (the same agent's opposite week-part), ``inter`` (a different
agent), and ``combined`` (a coin flip per agent). Both datasets are evaluated
this way.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.features.visits import VisitSequences

POLICIES = ("inter", "intra", "combined")
# Where donor days may come from: everything outside the test window, or the
# window being spliced itself.
DONORS = ("outside_test", "same_window")


@dataclass(frozen=True)
class Benchmark:
    """One evaluation scenario over a cohort."""

    name: str                    # splice policy
    split: int                   # split the anomalies sit in: 1 = val, 2 = test
    perturbed: VisitSequences    # the cohort with the anomalies in place
    labels: np.ndarray           # (N, T) 1 = anomalous day
    plan: np.ndarray             # (N, 3) target day, donor agent, donor day; -1 = clean


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """JS divergence in nats between ``p`` and each row of ``q`` (row-wise if both are 2-D).

    Computed in float64 and clamped at zero: identical days must pass an ε = 0
    guard, and float32 rounding can otherwise leave them a hair below it.
    """
    p = np.atleast_2d(p).astype(np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * (np.log(p) - np.log(m)), 0.0).sum(-1)
        kl_q = np.where(q > 0, q * (np.log(q) - np.log(m)), 0.0).sum(-1)
    return np.maximum(0.0, 0.5 * (kl_p + kl_q))


def _eligible_donor(
    rng: np.random.Generator,
    target_theta: np.ndarray,
    candidate_theta: np.ndarray,
    min_shift: float,
) -> int:
    """Index of a random donor whose mixture is far enough from the target's.

    The guard is what makes a splice a genuine anomaly: overwriting a Tuesday
    with a near-identical Wednesday would be labelled anomalous while looking
    exactly like normal behaviour.
    """
    if len(candidate_theta) == 0:
        return -1
    eligible = np.flatnonzero(jensen_shannon(target_theta, candidate_theta) >= min_shift)
    return int(rng.choice(eligible)) if eligible.size else -1


def _intra_injection(
    rng: np.random.Generator,
    sequences: VisitSequences,
    agent: int,
    target_ok: np.ndarray,
    donor_ok: np.ndarray,
    min_shift: float,
) -> tuple[int, int, int] | None:
    """A weekday day overwritten by one of this agent's weekend days, or vice versa."""
    dow = sequences.dow[agent]
    weekday_targets = np.flatnonzero(target_ok & (dow >= 0) & (dow < 5))
    weekend_targets = np.flatnonzero(target_ok & (dow >= 5))
    weekday_donors = np.flatnonzero(donor_ok & (dow >= 0) & (dow < 5))
    weekend_donors = np.flatnonzero(donor_ok & (dow >= 5))

    # A coin of 1 tries weekday targets first.
    directions = [(weekday_targets, weekend_donors), (weekend_targets, weekday_donors)]
    if not rng.integers(2):
        directions.reverse()

    for targets, donors in directions:
        if len(targets) == 0 or len(donors) == 0:
            continue
        for target in rng.permutation(targets):
            pick = _eligible_donor(
                rng, sequences.theta[agent, target],
                sequences.theta[agent, donors], min_shift,
            )
            if pick >= 0:
                return int(target), agent, int(donors[pick])
    return None


def _inter_injection(
    rng: np.random.Generator,
    sequences: VisitSequences,
    agent: int,
    target_ok: np.ndarray,
    pool: dict,
    min_shift: float,
) -> tuple[int, int, int] | None:
    """A day overwritten by some other agent's day, drawn from a global pool."""
    targets = np.flatnonzero(target_ok)
    if len(targets) == 0:
        return None
    target = int(rng.choice(targets))

    def donor_among(candidates: np.ndarray) -> int:
        candidates = candidates[pool["agents"][candidates] != agent]
        pick = _eligible_donor(
            rng, sequences.theta[agent, target], pool["theta"][candidates], min_shift
        )
        return int(candidates[pick]) if pick >= 0 else -1

    # Screen a random subset of a large pool first; the whole pool only when it misses.
    n_pool, subset = len(pool["agents"]), 256
    chosen = donor_among(rng.choice(n_pool, size=subset, replace=False)) if n_pool > subset else -1
    if chosen < 0:
        chosen = donor_among(np.arange(n_pool))
    if chosen < 0:
        return None
    return target, int(pool["agents"][chosen]), int(pool["days"][chosen])


def splice_plan(
    sequences: VisitSequences,
    policy: str,
    target_split: int,
    seed: int,
    min_shift: float,
    agent_frac: float = 1.0,
    donors: str = "outside_test",
) -> np.ndarray:
    """Choose, per agent, which day to overwrite and with which donor day.

    Returns an ``(N, 3)`` array of ``(target_day, donor_agent, donor_day)``;
    ``-1`` rows are agents left clean. Target days are restricted to
    ``target_split`` so tuning on validation never reads test. Donor content
    comes from outside the test window (Geolife, whose evaluated windows are a
    handful of days per agent, too few to feed the divergence guard) or from
    the spliced window itself (NUMOSIM's four-week test window); see ``DONORS``.

    The plan depends on the cohort only through its day layout and its template
    mixtures, so it can be built on one version of a cohort and applied to
    another with the same layout — the transferability experiment does exactly
    that to keep its benchmark fixed while the templates change.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if donors not in DONORS:
        raise ValueError(f"donors must be one of {DONORS}, got {donors!r}")

    rng = np.random.default_rng(seed)
    n_agents = sequences.n_agents
    recorded = sequences.template >= 0

    # A day can only be a target if it has a day before it to forecast from.
    can_target = recorded & (sequences.split == target_split)
    can_target[:, 0] = False
    can_donate = recorded & ((sequences.split != 2) if donors == "outside_test"
                             else (sequences.split == target_split))

    donor_rows, donor_cols = np.nonzero(can_donate)
    pool = {
        "agents": donor_rows,
        "days": donor_cols,
        "theta": sequences.theta[donor_rows, donor_cols],
    }

    plan = np.full((n_agents, 3), -1, dtype=np.int64)
    wanted = max(1, int(n_agents * agent_frac))
    n_spliced = 0
    for agent in rng.permutation(n_agents):
        if n_spliced >= wanted:
            break
        agent = int(agent)
        if not can_target[agent].any():
            continue

        if policy == "intra":
            attempts = [_intra_injection]
        elif policy == "inter":
            attempts = [_inter_injection]
        else:
            # One injection per agent, either kind; the other is the fallback so
            # the coin flip never costs yield.
            attempts = [_intra_injection, _inter_injection]
            if rng.integers(2):
                attempts.reverse()

        for attempt in attempts:
            if attempt is _intra_injection:
                found = _intra_injection(
                    rng, sequences, agent, can_target[agent], can_donate[agent], min_shift
                )
            else:
                found = _inter_injection(
                    rng, sequences, agent, can_target[agent], pool, min_shift
                )
            if found is not None:
                plan[agent] = found
                n_spliced += 1
                break
    return plan


def apply_splices(sequences: VisitSequences, plan: np.ndarray) -> VisitSequences:
    """The cohort with each planned target day overwritten by its donor day.

    Only the behaviour travels, never the calendar: the target day keeps its
    own date and day of week, and takes the donor's visits (whose times are
    hours on their own day's clock, each with the gap that preceded it on the
    donor's real day) and its template mixture.
    """
    fields = {
        name: getattr(sequences, name).copy()
        for name in ("arrival", "departure", "gap", "bundle_id", "visit_mask", "theta",
                     "template")
    }
    spliced = np.flatnonzero(plan[:, 0] >= 0)
    targets, donor_agents, donor_days = plan[spliced, 0], plan[spliced, 1], plan[spliced, 2]
    for name, values in fields.items():
        values[spliced, targets] = getattr(sequences, name)[donor_agents, donor_days]
    return replace(sequences, **fields)


def plan_labels(plan: np.ndarray, n_days: int) -> np.ndarray:
    """``(N, T)`` 0/1 grid marking each planned target day."""
    labels = np.zeros((len(plan), n_days), dtype=np.int32)
    spliced = np.flatnonzero(plan[:, 0] >= 0)
    labels[spliced, plan[spliced, 0]] = 1
    return labels


def spliced(
    sequences: VisitSequences,
    policy: str,
    split: int,
    seed: int,
    min_shift: float,
    agent_frac: float = 1.0,
    donors: str = "outside_test",
    plan: np.ndarray | None = None,
) -> Benchmark:
    """A splice benchmark over ``sequences``.

    Pass ``plan`` to reuse a plan built elsewhere instead of drawing one.
    """
    if plan is None:
        plan = splice_plan(sequences, policy, split, seed, min_shift, agent_frac, donors)
    return Benchmark(
        name=policy,
        split=split,
        perturbed=apply_splices(sequences, plan),
        labels=plan_labels(plan, sequences.n_days),
        plan=plan,
    )

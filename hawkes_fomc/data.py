from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit

from .config import PROCESSED_DIR, Session

UNDEF_PRICE = np.iinfo(np.int64).max
PRICE_SCALE = 1e-9


def load_mbp10(path: Path, cache_dir: Path = PROCESSED_DIR) -> np.ndarray:
    """Decode a Databento MBP-10 DBN file to a structured array, caching the result as .npy."""
    cache = cache_dir / (path.name.split(".")[0] + ".mbp10.npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache)
    import databento as db  # heavy import, only needed on a cache miss

    arr = db.DBNStore.from_file(path).to_ndarray()
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def seconds_et(ts_event_ns: np.ndarray, session: Session) -> np.ndarray:
    return (ts_event_ns.astype(np.int64) - session.midnight_ns) / 1e9


@dataclass(frozen=True)
class EventStream:
    name: str
    times: np.ndarray  # float64 seconds since ET midnight, strictly increasing
    marks: dict[str, np.ndarray] = field(default_factory=dict)
    session: tuple[float, float] | None = None  # (open, close) in the same units, for edge flags

    def __post_init__(self) -> None:
        if len(self.times) and not np.all(np.diff(self.times) > 0):
            raise ValueError(f"{self.name}: event times must be strictly increasing")

    def __len__(self) -> int:
        return len(self.times)

    def between(self, t0: float, t1: float) -> np.ndarray:
        return self.times[(self.times >= t0) & (self.times < t1)]

    def history(self, t0: float) -> np.ndarray:
        return self.times[self.times < t0]

    def marks_between(self, t0: float, t1: float) -> dict[str, np.ndarray]:
        m = (self.times >= t0) & (self.times < t1)
        return {k: v[m] for k, v in self.marks.items()}


@njit(cache=True)
def _group_executions(t: np.ndarray, side: np.ndarray, merge_within: float) -> np.ndarray:
    """Group consecutive executions into aggressive orders.

    An execution joins the open group when it is within `merge_within` of the previous one (exact
    timestamp ties always qualify) and its side is compatible with the group's aggressor side, meaning
    equal or unsided. Opposite aggressors are never merged, even at an identical timestamp, and an
    unsided fill can never bridge them: two aggressors that meet the book in the same nanosecond are
    two orders, and collapsing them would report one event with no side.
    """
    group = np.empty(len(t), dtype=np.int64)
    current, current_side = -1, np.int8(0)
    for i in range(len(t)):
        new = True
        if i > 0:
            dt = t[i] - t[i - 1]
            compatible = side[i] == 0 or current_side == 0 or side[i] == current_side
            new = not (compatible and (dt == 0.0 or dt <= merge_within))
        if new:
            current += 1
            current_side = side[i]
        elif current_side == 0:
            current_side = side[i]
        group[i] = current
    return group


TIMESTAMP_RESOLUTION = 1e-9  # exchange timestamps are integer nanoseconds


def _separate_ties(times: np.ndarray, resolution: float = TIMESTAMP_RESOLUTION) -> np.ndarray:
    """Push tied event times apart by one timestamp unit, preserving the exchange's own order.

    Two opposite-side aggressors can in principle meet the book in the same nanosecond; they are
    two events, but a Hawkes likelihood admits no ties. Separating them by one resolution unit in
    sequence order asserts no more precision than the timestamps already carry. On the 2021-01-28
    SPY data this never fires (phase0_audit.json reports zero such groups).
    """
    if len(times) < 2:
        return times
    out = times.copy()
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + resolution
    return out


def trade_events(
    arr: np.ndarray,
    session: Session,
    merge_within: float = 0.0,
    include_unsided: bool = True,
) -> EventStream:
    """One event per aggressive order.

    ITCH reports an order that sweeps several resting orders as several executions sharing one
    ts_event. Counting each execution as an event manufactures zero-lag self-excitation, so
    executions with identical timestamps are always merged; `merge_within` (seconds) additionally
    merges same-aggressor-side executions closer than that, for robustness checks.
    """
    is_trade = arr["action"] == b"T"
    t = seconds_et(arr["ts_event"][is_trade], session)
    raw_side = arr["side"][is_trade]
    side = np.where(raw_side == b"B", 1, np.where(raw_side == b"A", -1, 0)).astype(np.int8)
    size = arr["size"][is_trade].astype(np.int64)
    if not include_unsided:
        keep = side != 0
        t, side, size = t[keep], side[keep], size[keep]

    group = _group_executions(t, side, merge_within)
    n_groups = int(group[-1]) + 1 if len(group) else 0
    first = np.ones(len(t), dtype=bool)
    first[1:] = group[1:] != group[:-1]

    times = _separate_ties(t[first])
    marks = {
        "side": np.sign(np.bincount(group, weights=side, minlength=n_groups)).astype(np.int8),
        "size": np.bincount(group, weights=size, minlength=n_groups).astype(np.int64),
        "n_fills": np.bincount(group, minlength=n_groups).astype(np.int64),
    }
    return EventStream(name="trades", times=times, marks=marks, session=(session.open_s, session.close_s))


def midprice_change_events(arr: np.ndarray, session: Session) -> EventStream:
    """Mid-quote changes, one event per nanosecond at which the end-of-nanosecond mid differs from before."""
    bid = arr["bid_px_00"].astype(np.int64)
    ask = arr["ask_px_00"].astype(np.int64)
    valid = (bid > 0) & (ask > 0) & (bid != UNDEF_PRICE) & (ask != UNDEF_PRICE)
    t = seconds_et(arr["ts_event"], session)
    last_of_ts = np.ones(len(t), dtype=bool)
    last_of_ts[:-1] = np.diff(t) > 0
    keep = last_of_ts & valid
    t, mid = t[keep], (bid[keep] + ask[keep]) * (0.5 * PRICE_SCALE)
    change = np.diff(mid) != 0
    times = t[1:][change]
    direction = np.sign(np.diff(mid)[change]).astype(np.int8)
    return EventStream(name="midprice", times=times, marks={"direction": direction}, session=(session.open_s, session.close_s))


def coarsen_times(times: np.ndarray, resolution: float) -> np.ndarray:
    """Round timestamps down to a grid, reproducing the rounding that Filimonov & Sornette showed biases n.

    times/resolution is not exact at session-scale magnitudes, so a value already sitting on the grid
    can floor a whole bin low; the relative nudge is far smaller than a bin and fixes that.
    """
    return np.floor(times / resolution * (1 + 8 * np.finfo(float).eps)) * resolution


def jitter_times(times: np.ndarray, resolution: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform jitter within one resolution unit, the standard fix for tied/rounded timestamps."""
    jittered = np.sort(times + rng.uniform(0.0, resolution, size=len(times)))
    if not np.all(np.diff(jittered) > 0):
        raise ValueError("jitter did not break all ties; increase resolution")
    return jittered


def per_minute_counts(times: np.ndarray, t_start: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(t_start, t_end + 60.0, 60.0)
    counts, _ = np.histogram(times, bins=edges)
    return edges[:-1], counts

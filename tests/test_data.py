import numpy as np
import pytest

from hawkes_fomc.config import Session
from hawkes_fomc.data import UNDEF_PRICE, coarsen_times, jitter_times, midprice_change_events, trade_events

SESSION = Session(date="2021-01-28")
MIDNIGHT = SESSION.midnight_ns


def _records(rows):
    """rows: (seconds since ET midnight as int ns offset, action, side, size, bid, ask)."""
    dtype = np.dtype([
        ("ts_event", "u8"), ("action", "S1"), ("side", "S1"), ("size", "u4"), ("bid_px_00", "i8"), ("ask_px_00", "i8"),
    ])
    arr = np.empty(len(rows), dtype=dtype)
    for i, (ns, action, side, size, bid, ask) in enumerate(rows):
        arr[i] = (MIDNIGHT + ns, action, side, size, bid, ask)
    return arr


BASE_NS = 10 * 3600 * 10**9  # 10:00:00 ET
PX = 376_000_000_000  # 376.00 in 1e-9 units


def test_sweep_executions_merge_into_one_event_with_summed_size():
    rows = [
        (BASE_NS, b"T", b"B", 100, PX, PX + 10**7),
        (BASE_NS, b"T", b"B", 200, PX, PX + 10**7),
        (BASE_NS, b"T", b"B", 50, PX, PX + 10**7),
        (BASE_NS + 5_000_000, b"T", b"A", 100, PX, PX + 10**7),
    ]
    stream = trade_events(_records(rows), SESSION)
    assert len(stream) == 2
    assert stream.marks["size"].tolist() == [350, 100]
    assert stream.marks["n_fills"].tolist() == [3, 1]
    assert stream.marks["side"].tolist() == [1, -1]
    assert np.isclose(stream.times[0], 36000.0)
    assert np.isclose(stream.times[1] - stream.times[0], 5e-3)


def test_merge_within_joins_same_side_but_not_opposite_side():
    rows = [
        (BASE_NS, b"T", b"B", 100, PX, PX + 10**7),
        (BASE_NS + 3_000, b"T", b"B", 100, PX, PX + 10**7),  # 3 us later, same side
        (BASE_NS + 6_000, b"T", b"A", 100, PX, PX + 10**7),  # 3 us later, opposite side
        (BASE_NS + 200_000, b"T", b"B", 100, PX, PX + 10**7),  # 200 us later
    ]
    arr = _records(rows)
    assert len(trade_events(arr, SESSION)) == 4
    merged = trade_events(arr, SESSION, merge_within=1e-5)
    assert len(merged) == 3
    assert merged.marks["size"].tolist() == [200, 100, 100]
    assert merged.marks["side"].tolist() == [1, -1, 1]


def test_unsided_executions_join_a_sweep_but_cannot_bridge_opposite_aggressors():
    rows = [
        (BASE_NS, b"T", b"A", 100, PX, PX + 10**7),
        (BASE_NS, b"T", b"N", 100, PX, PX + 10**7),  # same nanosecond: always merged
        (BASE_NS + 1_000, b"T", b"N", 100, PX, PX + 10**7),  # unsided within the horizon: compatible, merged
        (BASE_NS + 2_000, b"T", b"B", 100, PX, PX + 10**7),  # opposite aggressor: never merged, even via the N
    ]
    stream = trade_events(_records(rows), SESSION, merge_within=1e-5)
    assert np.all(np.diff(stream.times) > 0)
    assert len(stream) == 2
    assert stream.marks["side"].tolist() == [-1, 1]
    assert stream.marks["size"].tolist() == [300, 100]
    unmerged = trade_events(_records(rows), SESSION)
    assert len(unmerged) == 3 and unmerged.marks["side"].tolist() == [-1, 0, 1]
    sided = trade_events(_records(rows), SESSION, include_unsided=False)
    assert len(sided) == 2 and sided.marks["size"].tolist() == [100, 100]


def test_midprice_changes_use_end_of_nanosecond_state_and_skip_undefined_quotes():
    rows = [
        (BASE_NS, b"A", b"B", 100, PX, PX + 10**7),
        (BASE_NS + 1000, b"A", b"A", 100, PX, PX + 2 * 10**7),  # mid moves up
        (BASE_NS + 1000, b"C", b"A", 100, PX, PX + 10**7),  # and back within the same nanosecond: no event
        (BASE_NS + 2000, b"C", b"B", 100, UNDEF_PRICE, PX + 10**7),  # one side empty: ignored
        (BASE_NS + 3000, b"A", b"B", 100, PX + 10**7, PX + 2 * 10**7),  # mid up by one cent
    ]
    stream = midprice_change_events(_records(rows), SESSION)
    assert len(stream) == 1
    assert np.isclose(stream.times[0], 36000.0 + 3e-6)
    assert stream.marks["direction"].tolist() == [1]


def test_opposite_aggressors_in_one_nanosecond_stay_two_events():
    rows = [
        (BASE_NS, b"T", b"A", 100, PX, PX + 10**7),
        (BASE_NS, b"T", b"B", 300, PX, PX + 10**7),  # the other aggressor, same nanosecond
        (BASE_NS + 10**6, b"T", b"A", 100, PX, PX + 10**7),
    ]
    stream = trade_events(_records(rows), SESSION)
    assert len(stream) == 3
    assert stream.marks["side"].tolist() == [-1, 1, -1]
    assert stream.marks["size"].tolist() == [100, 300, 100]
    assert np.all(np.diff(stream.times) > 0)
    assert np.isclose(stream.times[1] - stream.times[0], 1e-9)  # separated by one timestamp unit


def test_coarsen_then_jitter_restores_strict_order():
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(36000.0, 36600.0, 5000))
    rounded = coarsen_times(times, 1.0)
    assert np.all(np.diff(rounded) >= 0) and len(np.unique(rounded)) <= 601
    jittered = jitter_times(rounded, 1.0, rng)
    assert np.all(np.diff(jittered) > 0)
    assert np.all((jittered >= rounded.min()) & (jittered < rounded.max() + 1.0))


@pytest.mark.parametrize("date,hhmm,expected", [("2021-01-28", "14:00", 14 * 3600), ("2021-03-14", "14:00", 13 * 3600)])
def test_session_clock_is_dst_safe(date, hhmm, expected):
    # 2021-03-14 is the spring-forward day: 14:00 ET is only 13 h of elapsed time after midnight ET
    session = Session(date=date)
    assert session.seconds(hhmm) == expected
    assert session.clock(session.seconds(hhmm)) == f"{hhmm}:00"

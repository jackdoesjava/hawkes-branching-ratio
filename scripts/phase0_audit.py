"""Phase 0: audit the raw tick file before any modelling.

Writes results/phase0_audit.json and figures/phase0_counts_per_minute.png and prints the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hawkes_fomc import figures
from hawkes_fomc.config import DEFAULT_SESSION, ET, RAW_DIR, RESULTS_DIR, Session
from hawkes_fomc.data import load_mbp10, midprice_change_events, per_minute_counts, seconds_et, trade_events

# Scheduled FOMC statement days (14:00 ET), from federalreserve.gov meeting calendars.
FOMC_STATEMENT_DATES = {
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
}


def _clock(session: Session, ns: int) -> str:
    return pd.Timestamp(int(ns), unit="ns", tz="UTC").tz_convert(ET).strftime("%Y-%m-%d %H:%M:%S.%f")


def audit(arr: np.ndarray, session: Session) -> dict:
    ts = arr["ts_event"].astype(np.int64)
    recv = arr["ts_recv"].astype(np.int64)
    sod = seconds_et(ts, session)
    in_session = (sod >= session.open_s) & (sod < session.close_s)
    is_trade = arr["action"] == b"T"
    report: dict = {
        "fields": list(arr.dtype.names),
        "n_records": int(len(arr)),
        "publisher_id": np.unique(arr["publisher_id"]).tolist(),
        "instrument_id": np.unique(arr["instrument_id"]).tolist(),
        "ts_event_nondecreasing": bool(np.all(np.diff(ts) >= 0)),
        "ts_event_strictly_increasing": bool(np.all(np.diff(ts) > 0)),
        "ts_event_fraction_on_1us_grid": float(np.mean(ts % 1_000 == 0)),
        "ts_event_fraction_on_1ms_grid": float(np.mean(ts % 1_000_000 == 0)),
        "ts_event_distinct_sub_us_digits": int(len(np.unique(ts % 1_000))),
        "ts_recv_minus_ts_event_ns": {"median": float(np.median(recv - ts)), "p99": float(np.percentile(recv - ts, 99)), "max": int((recv - ts).max())},
        "first_record_et": _clock(session, ts[0]),
        "last_record_et": _clock(session, ts[-1]),
        "utc_offset_hours": pd.Timestamp(f"{session.date} 14:00", tz=ET).utcoffset().total_seconds() / 3600,
        "marks_et": {name: {"clock": hhmm, "seconds_since_et_midnight": session.seconds(hhmm)} for hhmm, name in session.marks},
        "regular_session": {
            "open": session.open, "close": session.close, "n_records": int(in_session.sum()),
            "first_record_et": _clock(session, ts[in_session][0]), "last_record_et": _clock(session, ts[in_session][-1]),
        },
        "actions": {a.decode(): int(c) for a, c in zip(*np.unique(arr["action"], return_counts=True))},
        "fomc_statement_day": session.date in FOMC_STATEMENT_DATES,
    }

    # Trades: raw executions versus one event per aggressive order.
    t_trades = ts[is_trade & in_session]
    side = arr["side"][is_trade & in_session]
    _, multiplicity = np.unique(t_trades, return_counts=True)
    sided = side != b"N"
    groups = pd.DataFrame({"ts": t_trades[sided], "side": side[sided]}).groupby("ts")["side"].nunique()
    report["trades_regular_session"] = {
        "executions": int(len(t_trades)),
        "unique_timestamps": int(len(multiplicity)),
        "max_executions_per_timestamp": int(multiplicity.max()),
        "multiplicity_histogram_1_to_10plus": np.bincount(np.minimum(multiplicity, 10), minlength=11)[1:].tolist(),
        "side_counts": {s.decode(): int(c) for s, c in zip(*np.unique(side, return_counts=True))},
        "same_timestamp_groups_with_both_aggressor_sides": int((groups > 1).sum()),
        "total_shares": int(arr["size"][is_trade & in_session].sum()),
    }
    for merge_within in (0.0, 1e-5):
        stream = trade_events(arr, session, merge_within=merge_within)
        times = stream.between(session.open_s, session.close_s)
        dt = np.diff(times)
        report[f"trade_events_merge_within_{merge_within:g}s"] = {
            "n_events": int(len(times)),
            "interarrival_s": {"min": float(dt.min()), "p1": float(np.percentile(dt, 1)), "median": float(np.median(dt)), "p99": float(np.percentile(dt, 99)), "max": float(dt.max())},
            "gaps_over_5s": int((dt > 5).sum()),
        }
    mid = midprice_change_events(arr, session)
    report["midprice_changes_regular_session"] = int(len(mid.between(session.open_s, session.close_s)))
    all_dt = np.diff(sod[in_session])
    report["book_update_gaps_regular_session"] = {"max_s": float(all_dt.max()), "at_et": _clock(session, ts[in_session][np.argmax(all_dt) + 1]), "gaps_over_1s": int((all_dt > 1).sum())}

    trades = trade_events(arr, session)
    windows = {"09:30-09:40": ("09:30", "09:40"), "13:55-14:05": ("13:55", "14:05"), "14:25-14:35": ("14:25", "14:35"), "15:50-16:00": ("15:50", "16:00")}
    report["trade_events_in_10min_windows"] = {k: int(len(trades.between(session.seconds(a), session.seconds(b)))) for k, (a, b) in windows.items()}
    return report


def main(path: Path, session: Session = DEFAULT_SESSION) -> None:
    arr = load_mbp10(path)
    meta_path = path.parent / "metadata.json"
    job = json.loads(meta_path.read_text())["query"] if meta_path.exists() else {}
    report = {"file": path.name, "size_bytes": path.stat().st_size, "databento_query": job, "session_date": session.date}
    report.update(audit(arr, session))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "phase0_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    t_start, t_end = session.seconds("04:00"), session.seconds("19:00")
    trades = trade_events(arr, session)
    mid = midprice_change_events(arr, session)
    book = seconds_et(arr["ts_event"], session)
    panels = [
        ("trade events", *per_minute_counts(trades.times, t_start, t_end)),
        ("mid-price changes", *per_minute_counts(mid.times, t_start, t_end)),
        ("book updates", *per_minute_counts(book, t_start, t_end)),
    ]
    fig = figures.per_minute_figure(panels, session, f"SPY on XNAS.ITCH, {session.date}: events per minute, 04:00-19:00 ET")
    for ax in fig.axes:
        ax.axvspan(t_start, session.open_s, color=figures.GRID, alpha=0.5, linewidth=0)
        ax.axvspan(session.close_s, t_end, color=figures.GRID, alpha=0.5, linewidth=0)
    print("figure:", figures.save(fig, "phase0_counts_per_minute"))


if __name__ == "__main__":
    files = sorted(RAW_DIR.glob("*.dbn.zst"))
    if len(files) != 1:
        raise SystemExit(f"expected exactly one .dbn.zst file in {RAW_DIR}, found {len(files)}")
    main(files[0])

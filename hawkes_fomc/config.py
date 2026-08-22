from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

ET = "America/New_York"
SEED = 20210128

# Kernel decay grid, as timescales in seconds: 10 us to 10 s at half-decade spacing.
# Decade spacing costs about +0.06 in n when the true timescale sits between grid points; the fast
# end has to reach 10 us or the residuals fail KS in the lower tail (see phase1_recovery).
# Components slower than about a tenth of the baseline knot spacing are not separable from mu(t) --
# a true 10 s component of mass 0.30 comes back as 0.22 with a constant baseline and 0.00 with 30 s
# knots -- so n always gets reported under both baselines rather than pretending one is right.
DEFAULT_TIMESCALES = tuple(float(x) for x in (10.0 ** (0.5 * k - 5) for k in range(13)))


@dataclass(frozen=True)
class Session:
    """One trading day. All event times in the project are float seconds since ET midnight of `date`."""

    date: str
    open: str = "09:30"
    close: str = "16:00"
    marks: tuple[tuple[str, str], ...] = (("14:00", "statement"), ("14:30", "press conference"))

    @property
    def midnight_ns(self) -> int:
        return int(pd.Timestamp(self.date, tz=ET).value)

    def seconds(self, hhmm: str) -> float:
        """Clock time on this date -> seconds since ET midnight (DST-safe, unlike h*3600+m*60)."""
        ts = pd.Timestamp(f"{self.date} {hhmm}", tz=ET)
        return (ts.value - self.midnight_ns) / 1e9

    @property
    def open_s(self) -> float:
        return self.seconds(self.open)

    @property
    def close_s(self) -> float:
        return self.seconds(self.close)

    def clock(self, seconds: float) -> str:
        ts = pd.Timestamp(self.midnight_ns + int(round(seconds * 1e9)), tz="UTC").tz_convert(ET)
        return ts.strftime("%H:%M:%S")


DEFAULT_SESSION = Session(date="2021-01-28")

"""Reproduce every result and figure from the raw DBN file: `python run_all.py`.

Phases run in order; each writes to results/ and figures/. Seeds are fixed in hawkes_fomc.config.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    [sys.executable, "-m", "pytest", "tests", "-q"],
    [sys.executable, "scripts/phase0_audit.py"],
    [sys.executable, "scripts/phase1_recovery.py"],
    [sys.executable, "scripts/phase1_stress.py"],
    [sys.executable, "scripts/phase1_rolling.py", "--boot", "100"],
    [sys.executable, "scripts/phase1_robustness.py"],
    [sys.executable, "scripts/phase2_neural.py", "--boot", "20", "--seeds", "10"],
    [sys.executable, "scripts/phase3_compare.py"],
]

if __name__ == "__main__":
    for step in STEPS:
        print("\n$", " ".join(step[1:]) if step[0] == sys.executable else " ".join(step), flush=True)
        subprocess.run(step, cwd=ROOT, check=True)

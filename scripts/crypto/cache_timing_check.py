#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Simple cache-timing harness for native kernel entrypoints.
Exits non-zero if timing variance exceeds threshold.
"""
import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path


def time_cmd(cmd):
    t0 = time.perf_counter()
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.05, help="relative stddev threshold")
    args = p.parse_args()

    # Use a lightweight Python entry that calls into native kernel if present
    script = Path(__file__).resolve().parents[2] / "algorithms" / "rft" / "kernels" / "python_bindings" / "advanced_scientific_tests.py"
    if not script.exists():
        print("no native test entry; skipping")
        sys.exit(0)

    samples = []
    for _ in range(args.runs):
        samples.append(time_cmd([sys.executable, str(script)]))
    mean = statistics.fmean(samples)
    stdev = statistics.pstdev(samples)
    rel = 0.0 if mean == 0 else stdev / mean
    print(f"runs={args.runs} mean={mean:.6f}s stdev={stdev:.6f}s rel={rel:.3%}")
    if rel > args.threshold:
        print("Timing variance above threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()

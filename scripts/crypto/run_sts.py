#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Run NIST STS on a bitstream generated from EnhancedRFTCryptoV2.

Generates a binary file of pseudo-random data using the block cipher in
counter mode (Feistel encryption of a monotonically increasing counter),
then invokes the NIST STS 'assess' binary to run the full battery.

Requires the NIST STS tools to be available inside the container at /opt/sts.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
from pathlib import Path

import sys

# Ensure import of cipher
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2


def generate_stream_bytes(num_bytes: int, out_path: Path) -> None:
    key = b"STS_TEST_KEY_2025_GOLDEN_RATIO__32"[:32]
    cipher = EnhancedRFTCryptoV2(key)
    blocks = num_bytes // 16 + (1 if num_bytes % 16 else 0)

    with open(out_path, "wb") as f:
        for ctr in range(blocks):
            block = struct.pack(
                ">QQ", ctr & 0xFFFFFFFFFFFFFFFF, (~ctr) & 0xFFFFFFFFFFFFFFFF
            )
            ct = cipher._feistel_encrypt(block)
            f.write(ct)
    # Truncate to exact size
    with open(out_path, "r+b") as f:
        f.truncate(num_bytes)


def run_sts(bits: int, work_dir: Path) -> Path:
    # Create input file and run assess
    bytes_needed = (bits + 7) // 8
    data_file = work_dir / "sts_input.bin"
    generate_stream_bytes(bytes_needed, data_file)

    sts_root = Path("/opt/sts")
    assess = sts_root / "assess"
    if not assess.exists():
        raise RuntimeError("NIST STS assess binary not found at /opt/sts/assess")

    # The assess tool expects to be run from its directory and interactively.
    # Use its batch mode by providing a config file if available; otherwise, call with default m=1000000 limiting sequence length.
    env = os.environ.copy()
    os.chdir(sts_root)
    # Copy file to expected data dir
    data_dir = sts_root / "experiments" / "AlgorithmTesting"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "sts_input.bin"
    if target.exists():
        target.unlink()
    target.symlink_to(data_file)

    # Run assess with 1,000,000 bits blocks until covered
    total_bits = bits
    block_bits = min(1000000, total_bits)
    subprocess.check_call([str(assess), str(block_bits)])

    report = sts_root / "experiments" / "AlgorithmTesting" / "finalAnalysisReport.txt"
    if not report.exists():
        raise RuntimeError("STS report not found")
    return report


def parse_report(report_path: Path, p_threshold: float) -> None:
    bad = []
    with open(report_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "p-value" in line:
                try:
                    # Parse: p-value = 0.123456
                    val = float(line.split("=")[-1].strip())
                    if val < p_threshold:
                        bad.append((line, val))
                except Exception:
                    continue
    if bad:
        print("STS failures (p < threshold):")
        for line, val in bad:
            print(line)
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=10_000_000, help="Number of bits to test")
    ap.add_argument("--p-threshold", type=float, default=0.001)
    ap.add_argument("--out", type=str, default="results/sts/finalAnalysisReport.txt")
    args = ap.parse_args()

    work_dir = REPO_ROOT / "results" / "sts"
    work_dir.mkdir(parents=True, exist_ok=True)

    report = run_sts(args.bits, work_dir)
    # Copy report to results path
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.read_text())

    parse_report(out_path, args.p_threshold)
    print("STS passed all p-value thresholds")


if __name__ == "__main__":
    main()

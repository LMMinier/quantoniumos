#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
run_formal_verification.py — Execute formal proof engine and produce reports
============================================================================

Usage:
    python run_formal_verification.py [--json] [--sizes 8,16,32,64,128]

This script:
  1. Runs the FormalProofEngine on all RFT theorems
  2. Constructs deductive proof chains from axioms
  3. Verifies each step with numerical certificates
  4. Generates human-readable and machine-readable reports
  5. Saves results to data/experiments/formal_proofs/
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algorithms.rft.theory.proof_engine import FormalProofEngine, ProofStatus


def main():
    parser = argparse.ArgumentParser(description="Formal RFT proof verification")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--sizes", type=str, default="8,16,32,64,128",
                        help="Comma-separated list of matrix sizes to test")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    print("=" * 78)
    print("  QUANTONIUMOS — FORMAL PROOF VERIFICATION")
    print("  Transitioning from empirical to deductive proofs")
    print("=" * 78)
    print(f"\n  Matrix sizes: {sizes}")
    print(f"  Starting verification...\n")

    t0 = time.time()
    engine = FormalProofEngine(sizes=sizes)
    theorems = engine.prove_all()
    elapsed = time.time() - t0

    # Print human-readable report
    report = engine.generate_report()
    print(report)
    print(f"\n  Total verification time: {elapsed:.2f}s\n")

    # Save reports
    out_dir = os.path.join(os.path.dirname(__file__), "data", "experiments", "formal_proofs")
    os.makedirs(out_dir, exist_ok=True)

    report_path = os.path.join(out_dir, "formal_proof_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
        f.write(f"\n\nVerification time: {elapsed:.2f}s\n")
    print(f"  Report saved to: {report_path}")

    json_path = os.path.join(out_dir, "formal_proofs.json")
    with open(json_path, "w") as f:
        f.write(engine.export_json())
    print(f"  JSON saved to:   {json_path}")

    # Summary table
    print("\n" + "─" * 78)
    print(f"  {'Theorem':<12} {'Status':<15} {'Verified':<10} {'Steps':<8} {'Type'}")
    print("─" * 78)
    for tid, rec in theorems.items():
        v = "✓ YES" if rec.proof.verified else "✗ NO"
        print(f"  {tid:<12} {rec.name:<15} {v:<10} {len(rec.proof.steps):<8} {rec.proof.status.value}")
    print("─" * 78)

    n_verified = sum(1 for r in theorems.values() if r.proof.verified)
    n_total = len(theorems)
    n_formal = sum(1 for r in theorems.values()
                   if r.proof.status in (ProofStatus.CONSTRUCTIVE, ProofStatus.DEDUCTIVE))

    print(f"\n  RESULTS: {n_verified}/{n_total} theorems verified")
    print(f"  FORMAL (constructive/deductive): {n_formal}/{n_total}")
    print(f"  EMPIRICAL (conjectures): {n_total - n_formal}/{n_total}")

    if args.json:
        print("\n" + engine.export_json())

    return 0 if n_verified == n_total else 1


if __name__ == "__main__":
    sys.exit(main())

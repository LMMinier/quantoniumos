#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Real-Data Validation CLI for QuantoniumOS RFT

⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE  ⚠️

This script tests RFT transforms on real biomedical datasets.
Data must be fetched first using the scripts in data/.

Usage:
    python scripts/test_real_data.py                  # Run all available tests
    python scripts/test_real_data.py --ecg            # MIT-BIH ECG only
    python scripts/test_real_data.py --eeg            # Sleep-EDF EEG only
    python scripts/test_real_data.py --genomics       # Lambda phage + PDB only
    python scripts/test_real_data.py --mri            # FastMRI (if available)
    python scripts/test_real_data.py --verbose        # Detailed output
    python scripts/test_real_data.py --json           # JSON output for CI

Environment:
    USE_REAL_DATA=1 must be set to confirm intentional use of real data.

License Notes:
    - MIT-BIH: PhysioNet DUA (research only)
    - Sleep-EDF: PhysioNet terms (research only)
    - FastMRI: CC BY-NC 4.0 (non-commercial)
    - Lambda Phage: Public domain (NCBI RefSeq)
    - PDB 1CRN: CC BY 4.0
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List

import numpy as np

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass
class TestResult:
    dataset: str
    passed: bool
    samples: int
    roundtrip_error: float
    compression_prd: Optional[float] = None
    details: Optional[str] = None


def check_env():
    if os.environ.get("USE_REAL_DATA") != "1":
        print("ERROR: USE_REAL_DATA=1 not set. This is required to confirm intentional real-data use.")
        print("Run with: USE_REAL_DATA=1 python scripts/test_real_data.py")
        sys.exit(1)


def test_ecg(verbose: bool = False) -> List[TestResult]:
    """Test RFT on MIT-BIH Arrhythmia Database."""
    results = []
    mitbih_dir = DATA / "physionet" / "mitbih"
    
    if not mitbih_dir.exists():
        if verbose:
            print("⏭ MIT-BIH not found. Run: USE_REAL_DATA=1 python data/physionet_mitbih_fetch.py")
        return results
    
    try:
        import wfdb
    except ImportError:
        print("⚠ wfdb not installed. Run: pip install wfdb")
        return results
    
    from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
    
    for rec_id in ["100", "101"]:
        rec_path = mitbih_dir / rec_id
        if not (mitbih_dir / f"{rec_id}.dat").exists():
            continue
        
        record = wfdb.rdrecord(str(mitbih_dir / rec_id))
        signal = record.p_signal[:, 0].astype(np.float64)
        
        # Test roundtrip
        window = signal[1000:2024]
        coeffs = rft_forward(window)
        recon = rft_inverse(coeffs)
        err = float(np.linalg.norm(recon - window) / np.linalg.norm(window))
        
        # Compression test (30% coefficients)
        sorted_idx = np.argsort(np.abs(coeffs))[::-1]
        keep = int(0.3 * len(coeffs))
        sparse = np.zeros_like(coeffs)
        sparse[sorted_idx[:keep]] = coeffs[sorted_idx[:keep]]
        compressed_recon = rft_inverse(sparse)
        prd = float(100 * np.linalg.norm(window - compressed_recon.real) / np.linalg.norm(window))
        
        passed = err < 1e-10
        results.append(TestResult(
            dataset=f"MIT-BIH ECG {rec_id}",
            passed=passed,
            samples=len(signal),
            roundtrip_error=err,
            compression_prd=prd,
            details=f"fs={record.fs}Hz, duration={len(signal)/record.fs:.0f}s"
        ))
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"  {status} Record {rec_id}: {len(signal)} samples, err={err:.2e}, PRD@30%={prd:.2f}%")
    
    return results


def test_eeg(verbose: bool = False) -> List[TestResult]:
    """Test RFT on Sleep-EDF Database."""
    results = []
    sleepedf_dir = DATA / "physionet" / "sleepedf"
    
    if not sleepedf_dir.exists():
        if verbose:
            print("⏭ Sleep-EDF not found. Run: USE_REAL_DATA=1 python data/physionet_sleepedf_fetch.py")
        return results
    
    try:
        import pyedflib
    except ImportError:
        print("⚠ pyedflib not installed. Run: pip install pyedflib")
        return results
    
    from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
    
    for edf_file in sleepedf_dir.glob("*-PSG.edf"):
        f = pyedflib.EdfReader(str(edf_file))
        labels = [f.getLabel(i) for i in range(f.signals_in_file)]
        
        eeg_idx = labels.index("EEG Fpz-Cz") if "EEG Fpz-Cz" in labels else 0
        eeg = f.readSignal(eeg_idx).astype(np.float64)
        fs = f.getSampleFrequency(eeg_idx)
        f.close()
        
        # Test roundtrip
        window = eeg[10000:11024]
        coeffs = rft_forward(window)
        recon = rft_inverse(coeffs)
        err = float(np.linalg.norm(recon - window) / np.linalg.norm(window))
        
        # Compression
        sorted_idx = np.argsort(np.abs(coeffs))[::-1]
        keep = int(0.3 * len(coeffs))
        sparse = np.zeros_like(coeffs)
        sparse[sorted_idx[:keep]] = coeffs[sorted_idx[:keep]]
        compressed_recon = rft_inverse(sparse)
        prd = float(100 * np.linalg.norm(window - compressed_recon.real) / np.linalg.norm(window))
        
        passed = err < 1e-10
        results.append(TestResult(
            dataset=f"Sleep-EDF {edf_file.stem}",
            passed=passed,
            samples=len(eeg),
            roundtrip_error=err,
            compression_prd=prd,
            details=f"fs={fs}Hz, duration={len(eeg)/fs/3600:.1f}h"
        ))
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"  {status} {edf_file.stem}: {len(eeg)} samples, err={err:.2e}, PRD@30%={prd:.2f}%")
    
    return results


def test_genomics(verbose: bool = False) -> List[TestResult]:
    """Test RFT on genomics data (Lambda phage + PDB)."""
    results = []
    genomics_dir = DATA / "genomics"
    
    from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
    
    # Lambda phage
    fasta = genomics_dir / "lambda_phage.fasta"
    if fasta.exists():
        with open(fasta) as f:
            lines = f.readlines()
        seq = "".join(l.strip() for l in lines if not l.startswith(">"))
        
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
        numeric = np.array([mapping.get(c.upper(), 0) for c in seq[:1024]], dtype=np.float64)
        
        coeffs = rft_forward(numeric)
        recon = rft_inverse(coeffs)
        err = float(np.linalg.norm(recon - numeric) / np.linalg.norm(numeric))
        
        passed = err < 1e-10
        results.append(TestResult(
            dataset="Lambda Phage Genome",
            passed=passed,
            samples=len(seq),
            roundtrip_error=err,
            details=f"{len(seq)} bp"
        ))
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"  {status} Lambda phage: {len(seq)} bp, err={err:.2e}")
    elif verbose:
        print("⏭ Lambda phage not found. Run: USE_REAL_DATA=1 python data/genomics_fetch.py")
    
    # PDB 1CRN
    pdb = genomics_dir / "1crn.pdb"
    if pdb.exists():
        with open(pdb) as f:
            atoms = [l for l in f if l.startswith("ATOM")]
        
        ca_coords = []
        for line in atoms:
            if line[12:16].strip() == "CA":
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                ca_coords.append([x, y, z])
        ca = np.array(ca_coords)
        
        flat = ca.flatten()
        n = 1 << (len(flat) - 1).bit_length()
        padded = np.zeros(n)
        padded[:len(flat)] = flat
        
        coeffs = rft_forward(padded)
        recon = rft_inverse(coeffs)
        err = float(np.linalg.norm(recon[:len(flat)] - flat) / np.linalg.norm(flat))
        
        passed = err < 1e-10
        results.append(TestResult(
            dataset="PDB 1CRN (Crambin)",
            passed=passed,
            samples=len(atoms),
            roundtrip_error=err,
            details=f"{len(ca)} CA atoms"
        ))
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"  {status} PDB 1CRN: {len(ca)} CA atoms, err={err:.2e}")
    elif verbose:
        print("⏭ PDB 1CRN not found. Run: USE_REAL_DATA=1 python data/genomics_fetch.py")
    
    return results


def test_mri(verbose: bool = False) -> List[TestResult]:
    """Test RFT on FastMRI data (if available)."""
    results = []
    fastmri_dir = DATA / "fastmri" / "knee_singlecoil_subset"
    
    if not fastmri_dir.exists():
        if verbose:
            print("⏭ FastMRI not found. Set FASTMRI_KNEE_URL and run: USE_REAL_DATA=1 bash data/fastmri_fetch.sh")
        return results
    
    # TODO: Implement when FastMRI is available
    if verbose:
        print("  ℹ FastMRI directory found but test not yet implemented")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Real-Data Validation CLI for QuantoniumOS RFT",
        epilog="⚠️ RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE"
    )
    parser.add_argument("--ecg", action="store_true", help="Test MIT-BIH ECG only")
    parser.add_argument("--eeg", action="store_true", help="Test Sleep-EDF EEG only")
    parser.add_argument("--genomics", action="store_true", help="Test genomics data only")
    parser.add_argument("--mri", action="store_true", help="Test FastMRI only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="JSON output for CI")
    args = parser.parse_args()
    
    check_env()
    
    # Default: run all if no specific flag
    run_all = not (args.ecg or args.eeg or args.genomics or args.mri)
    
    all_results: List[TestResult] = []
    
    if args.verbose and not args.json:
        print("=" * 60)
        print("REAL DATA VALIDATION — QuantoniumOS RFT")
        print("⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL USE")
        print("=" * 60)
    
    if run_all or args.ecg:
        if args.verbose and not args.json:
            print("\n[MIT-BIH ECG]")
        all_results.extend(test_ecg(args.verbose and not args.json))
    
    if run_all or args.eeg:
        if args.verbose and not args.json:
            print("\n[Sleep-EDF EEG]")
        all_results.extend(test_eeg(args.verbose and not args.json))
    
    if run_all or args.genomics:
        if args.verbose and not args.json:
            print("\n[Genomics]")
        all_results.extend(test_genomics(args.verbose and not args.json))
    
    if run_all or args.mri:
        if args.verbose and not args.json:
            print("\n[FastMRI]")
        all_results.extend(test_mri(args.verbose and not args.json))
    
    # Output
    if args.json:
        output = {
            "passed": all(r.passed for r in all_results),
            "total": len(all_results),
            "results": [asdict(r) for r in all_results]
        }
        print(json.dumps(output, indent=2))
    else:
        passed = sum(1 for r in all_results if r.passed)
        total = len(all_results)
        
        print()
        print("=" * 60)
        if total == 0:
            print("No real data found. Fetch datasets first:")
            print("  USE_REAL_DATA=1 python data/physionet_mitbih_fetch.py")
            print("  USE_REAL_DATA=1 python data/physionet_sleepedf_fetch.py")
            print("  USE_REAL_DATA=1 python data/genomics_fetch.py")
            sys.exit(1)
        
        print(f"Results: {passed}/{total} passed")
        for r in all_results:
            status = "✓" if r.passed else "✗"
            prd_str = f", PRD@30%={r.compression_prd:.2f}%" if r.compression_prd else ""
            print(f"  {status} {r.dataset}: err={r.roundtrip_error:.2e}{prd_str}")
        
        if passed == total:
            print("\n✓ All real-data tests passed")
            sys.exit(0)
        else:
            print(f"\n✗ {total - passed} test(s) failed")
            sys.exit(1)


if __name__ == "__main__":
    main()

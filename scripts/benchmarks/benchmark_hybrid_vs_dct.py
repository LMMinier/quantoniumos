#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Hybrid RFT/DCT vs Pure DCT Benchmark on Real Biomedical Data
=============================================================

Tests the existing hybrid cascade implementations:
- H3 Hierarchical Cascade (0.673 BPP claimed)
- FH5 Entropy Guided (0.406 BPP on edges claimed)
- H6 Dictionary Learning (best PSNR claimed)

Against pure DCT baseline on real MIT-BIH ECG and Sleep-EDF EEG data.

FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS

Usage:
    USE_REAL_DATA=1 python scripts/benchmark_hybrid_vs_dct.py
"""

import os
import sys
import numpy as np
from scipy.fftpack import dct, idct
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def require_real_data():
    """Check for real data opt-in."""
    if os.environ.get("USE_REAL_DATA") != "1":
        print("⚠️  Real data validation requires explicit opt-in.")
        print("   Set USE_REAL_DATA=1 to run.")
        print("   FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
        sys.exit(1)


def load_ecg_signal(record: str = "100") -> np.ndarray:
    """Load MIT-BIH ECG record."""
    try:
        import wfdb
    except ImportError:
        print("Install wfdb: pip install wfdb")
        sys.exit(1)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mitbih")
    record_path = os.path.join(data_dir, record)
    
    if not os.path.exists(record_path + ".dat"):
        print(f"Downloading MIT-BIH record {record}...")
        os.makedirs(data_dir, exist_ok=True)
        wfdb.dl_database('mitdb', data_dir, records=[record])
    
    record_obj = wfdb.rdrecord(record_path)
    signal = record_obj.p_signal[:, 0]  # First channel
    return signal.astype(np.float64)


def load_eeg_signal() -> np.ndarray:
    """Load Sleep-EDF EEG data (first available)."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sleep_edf")
    
    try:
        import pyedflib
    except ImportError:
        print("Install pyedflib: pip install pyedflib")
        sys.exit(1)
    
    # Find first PSG file
    for f in os.listdir(data_dir) if os.path.exists(data_dir) else []:
        if f.endswith("PSG.edf"):
            edf_path = os.path.join(data_dir, f)
            edf = pyedflib.EdfReader(edf_path)
            # Get first EEG channel
            n = edf.signals_in_file
            for i in range(n):
                label = edf.getLabel(i)
                if 'EEG' in label.upper():
                    signal = edf.readSignal(i)
                    edf.close()
                    return signal.astype(np.float64)
            edf.close()
    
    print("No Sleep-EDF data found. Using synthetic EEG-like signal.")
    t = np.linspace(0, 10, 10000)
    alpha = np.sin(2 * np.pi * 10 * t)  # Alpha rhythm
    delta = 0.5 * np.sin(2 * np.pi * 2 * t)  # Delta rhythm
    return alpha + delta + 0.1 * np.random.randn(len(t))


def pure_dct_codec(signal: np.ndarray, keep_ratio: float) -> Tuple[np.ndarray, float]:
    """
    Pure DCT compression baseline.
    
    Args:
        signal: Input signal
        keep_ratio: Fraction of coefficients to keep (0-1)
        
    Returns:
        (reconstructed_signal, prd_percent)
    """
    C = dct(signal, type=2, norm='ortho')
    k = max(1, int(len(C) * keep_ratio))
    
    # Keep top-k by magnitude
    indices = np.argsort(np.abs(C))[::-1][:k]
    C_sparse = np.zeros_like(C)
    C_sparse[indices] = C[indices]
    
    reconstructed = idct(C_sparse, type=2, norm='ortho')
    
    # PRD
    diff = signal - reconstructed
    prd = 100 * np.sqrt(np.sum(diff**2) / np.sum(signal**2))
    
    return reconstructed, prd


def run_hybrid_benchmark(signal: np.ndarray, sparsity: float, signal_name: str) -> Dict[str, Any]:
    """
    Run all hybrid variants on a signal.
    
    Args:
        signal: Input signal (will be trimmed to 10000 samples)
        sparsity: Target sparsity (e.g., 0.9 = 10% kept)
        signal_name: Name for reporting
        
    Returns:
        Dict with results for each method
    """
    # Import hybrids
    from algorithms.rft.hybrids.cascade_hybrids import (
        H3HierarchicalCascade,
        FH5EntropyGuided,
        H6DictionaryLearning
    )
    
    # Trim/pad to consistent length
    n = 10000
    if len(signal) > n:
        signal = signal[:n]
    elif len(signal) < n:
        signal = np.pad(signal, (0, n - len(signal)), mode='constant')
    
    # Normalize for fair comparison
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    
    results = {}
    keep_ratio = 1.0 - sparsity
    
    # 1. Pure DCT baseline
    _, prd_dct = pure_dct_codec(signal, keep_ratio)
    results["Pure_DCT"] = {
        "prd": prd_dct,
        "time_ms": 0.0,
        "bpp": keep_ratio * 16,  # Rough BPP estimate
    }
    
    # 2. H3 Hierarchical Cascade
    try:
        h3 = H3HierarchicalCascade()
        h3_result = h3.encode(signal, sparsity=sparsity)
        h3_recon = h3.decode(h3_result.coefficients, len(signal))
        diff = signal - h3_recon
        prd_h3 = 100 * np.sqrt(np.sum(diff**2) / np.sum(signal**2))
        results["H3_Cascade"] = {
            "prd": prd_h3,
            "time_ms": h3_result.time_ms,
            "bpp": h3_result.bpp,
            "psnr": h3_result.psnr,
        }
    except Exception as e:
        results["H3_Cascade"] = {"error": str(e)}
    
    # 3. FH5 Entropy Guided
    try:
        fh5 = FH5EntropyGuided(window_size=32, entropy_threshold=2.0)
        fh5_result = fh5.encode(signal, sparsity=sparsity)
        fh5_recon = fh5.decode(fh5_result.coefficients, len(signal))
        diff = signal - fh5_recon
        prd_fh5 = 100 * np.sqrt(np.sum(diff**2) / np.sum(signal**2))
        results["FH5_Entropy"] = {
            "prd": prd_fh5,
            "time_ms": fh5_result.time_ms,
            "bpp": fh5_result.bpp,
            "psnr": fh5_result.psnr,
        }
    except Exception as e:
        results["FH5_Entropy"] = {"error": str(e)}
    
    # 4. H6 Dictionary Learning
    try:
        h6 = H6DictionaryLearning(n_atoms=32)
        h6_result = h6.encode(signal, sparsity=sparsity)
        h6_recon = h6.decode(h6_result.coefficients, len(signal))
        diff = signal - h6_recon
        prd_h6 = 100 * np.sqrt(np.sum(diff**2) / np.sum(signal**2))
        results["H6_DictLearn"] = {
            "prd": prd_h6,
            "time_ms": h6_result.time_ms,
            "bpp": h6_result.bpp,
            "psnr": h6_result.psnr,
        }
    except Exception as e:
        results["H6_DictLearn"] = {"error": str(e)}
    
    return results


def find_winner(results: Dict[str, Any]) -> Tuple[str, float]:
    """Find method with lowest PRD."""
    best_method = None
    best_prd = float('inf')
    
    for method, data in results.items():
        if "error" not in data and data["prd"] < best_prd:
            best_prd = data["prd"]
            best_method = method
    
    return best_method, best_prd


def main():
    require_real_data()
    
    print("=" * 70)
    print("Hybrid RFT/DCT vs Pure DCT Benchmark on Real Biomedical Data")
    print("=" * 70)
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS\n")
    
    # Test conditions
    ecg_records = ["100", "101", "200", "207"]  # Different arrhythmia types
    sparsity_levels = [0.9, 0.8, 0.7]  # 10%, 20%, 30% kept
    
    all_results = []
    
    # ECG tests
    print("\n📊 ECG Compression (MIT-BIH Arrhythmia Database)")
    print("-" * 70)
    
    for record in ecg_records:
        print(f"\n🫀 Record {record}:")
        try:
            signal = load_ecg_signal(record)
        except Exception as e:
            print(f"   ⚠️  Could not load record {record}: {e}")
            continue
        
        for sparsity in sparsity_levels:
            keep_pct = int((1 - sparsity) * 100)
            results = run_hybrid_benchmark(signal, sparsity, f"ECG_{record}")
            winner, best_prd = find_winner(results)
            
            print(f"   {keep_pct:2d}% kept | ", end="")
            for method, data in results.items():
                if "error" not in data:
                    marker = "🏆" if method == winner else "  "
                    print(f"{marker}{method}: {data['prd']:.2f}% | ", end="")
            print(f"Winner: {winner}")
            
            all_results.append({
                "signal": f"ECG_{record}",
                "keep": keep_pct,
                "results": results,
                "winner": winner,
            })
    
    # EEG tests
    print("\n\n📊 EEG Compression (Sleep-EDF)")
    print("-" * 70)
    
    try:
        eeg_signal = load_eeg_signal()
        
        for sparsity in sparsity_levels:
            keep_pct = int((1 - sparsity) * 100)
            results = run_hybrid_benchmark(eeg_signal, sparsity, "EEG")
            winner, best_prd = find_winner(results)
            
            print(f"   {keep_pct:2d}% kept | ", end="")
            for method, data in results.items():
                if "error" not in data:
                    marker = "🏆" if method == winner else "  "
                    print(f"{marker}{method}: {data['prd']:.2f}% | ", end="")
            print(f"Winner: {winner}")
            
            all_results.append({
                "signal": "EEG",
                "keep": keep_pct,
                "results": results,
                "winner": winner,
            })
    except Exception as e:
        print(f"   ⚠️  Could not load EEG: {e}")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY: Which Method Wins?")
    print("=" * 70)
    
    win_counts = {}
    for r in all_results:
        w = r["winner"]
        if w:
            win_counts[w] = win_counts.get(w, 0) + 1
    
    total = len(all_results)
    print(f"\nOut of {total} test cases:")
    for method, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = "█" * int(pct / 5)
        print(f"  {method:15s}: {count:2d} wins ({pct:5.1f}%) {bar}")
    
    # Check if any hybrid beats DCT
    dct_wins = win_counts.get("Pure_DCT", 0)
    hybrid_wins = total - dct_wins
    
    print("\n" + "-" * 70)
    if hybrid_wins > dct_wins:
        print("✅ HYBRIDS WIN: Your RFT/DCT hybrids beat pure DCT on real data!")
        print(f"   Hybrid wins: {hybrid_wins}, Pure DCT wins: {dct_wins}")
    elif hybrid_wins == dct_wins:
        print("🤝 TIE: Hybrids and pure DCT are competitive.")
    else:
        # Find cases where hybrids came close
        close_calls = 0
        for r in all_results:
            if r["winner"] == "Pure_DCT":
                dct_prd = r["results"]["Pure_DCT"]["prd"]
                for method, data in r["results"].items():
                    if method != "Pure_DCT" and "error" not in data:
                        if data["prd"] < dct_prd * 1.1:  # Within 10%
                            close_calls += 1
                            break
        
        print("⚠️  DCT WINS: But checking for close calls...")
        print(f"   Pure DCT wins: {dct_wins}, Hybrid wins: {hybrid_wins}")
        print(f"   Close calls (within 10%): {close_calls}")
    
    print("\n" + "=" * 70)
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
    print("=" * 70)


if __name__ == "__main__":
    main()

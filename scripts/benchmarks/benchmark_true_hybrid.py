#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
TRUE Hybrid DCT+RFT vs Pure DCT Benchmark on Real Biomedical Data
==================================================================

Uses the ACTUAL working hybrid implementation from hybrid_benchmark.py:
- sparse_approx_hybrid: DCT for structure + DFT/RFT for texture residual

This is the proper "split budget" approach where:
- k1 coefficients go to DCT (structure)  
- k2 coefficients go to RFT (texture residual)
- Total budget = k1 + k2

FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS

Usage:
    USE_REAL_DATA=1 python scripts/benchmark_true_hybrid.py
"""

import os
import sys
import numpy as np
from scipy.fftpack import dct, idct
from numpy.fft import fft, ifft
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
        os.makedirs(data_dir, exist_ok=True)
        wfdb.dl_database('mitdb', data_dir, records=[record])
    
    record_obj = wfdb.rdrecord(record_path)
    signal = record_obj.p_signal[:, 0]
    return signal.astype(np.float64)


# ============================================================================
# Transform Implementations
# ============================================================================

def sparse_approx_dct(x: np.ndarray, k: int) -> np.ndarray:
    """Keep top-k DCT coefficients."""
    c = dct(x, norm='ortho')
    idx = np.argsort(np.abs(c))[::-1]
    c_sparse = np.zeros_like(c)
    c_sparse[idx[:k]] = c[idx[:k]]
    return idct(c_sparse, norm='ortho')


def sparse_approx_hybrid(x: np.ndarray, k1: int, k2: int) -> np.ndarray:
    """
    Hybrid DCT + RFT approximation.
    
    1. Keep top-k1 DCT coefficients for structure
    2. Compute residual
    3. Keep top-k2 DFT/RFT coefficients of residual for texture
    
    Note: Since |RFT| = |DFT| (same magnitudes), we use DFT for efficiency.
    The phase difference only matters for encoding, not approximation quality.
    """
    n = len(x)
    
    # Step 1: DCT approximation for structure
    c_dct = dct(x, norm='ortho')
    idx_dct = np.argsort(np.abs(c_dct))[::-1]
    c_dct_sparse = np.zeros_like(c_dct)
    c_dct_sparse[idx_dct[:k1]] = c_dct[idx_dct[:k1]]
    x_struct = idct(c_dct_sparse, norm='ortho')
    
    # Step 2: Residual
    r = x - x_struct
    
    # Step 3: DFT/RFT approximation of residual
    c_dft = fft(r) / np.sqrt(n)
    idx_dft = np.argsort(np.abs(c_dft))[::-1]
    c_dft_sparse = np.zeros_like(c_dft)
    c_dft_sparse[idx_dft[:k2]] = c_dft[idx_dft[:k2]]
    x_texture = np.real(ifft(c_dft_sparse * np.sqrt(n)))
    
    return x_struct + x_texture


def compute_prd(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Percent Root-mean-square Difference."""
    diff = original - reconstructed
    return 100 * np.sqrt(np.sum(diff**2) / np.sum(original**2))


# ============================================================================
# Benchmark
# ============================================================================

def run_comparison(signal: np.ndarray, total_k: int) -> Dict[str, float]:
    """
    Compare pure DCT vs hybrid for same total coefficient budget.
    
    For hybrid, we test different splits of the budget:
    - 100% DCT / 0% RFT (pure DCT)
    - 70% DCT / 30% RFT (typical)
    - 50% DCT / 50% RFT
    - 30% DCT / 70% RFT (texture-heavy)
    
    Returns dict of {method: prd}
    """
    results = {}
    
    # Pure DCT
    recon_dct = sparse_approx_dct(signal, total_k)
    results["Pure_DCT"] = compute_prd(signal, recon_dct)
    
    # Hybrid splits
    splits = [
        (0.9, 0.1, "H90/10"),  # 90% DCT, 10% RFT
        (0.7, 0.3, "H70/30"),  # 70% DCT, 30% RFT
        (0.5, 0.5, "H50/50"),  # 50/50 split
        (0.3, 0.7, "H30/70"),  # 30% DCT, 70% RFT
    ]
    
    for dct_frac, rft_frac, name in splits:
        k1 = max(1, int(total_k * dct_frac))
        k2 = max(1, int(total_k * rft_frac))
        recon_hybrid = sparse_approx_hybrid(signal, k1, k2)
        results[name] = compute_prd(signal, recon_hybrid)
    
    return results


def main():
    require_real_data()
    
    print("=" * 80)
    print("TRUE Hybrid DCT+RFT vs Pure DCT Benchmark (Same Total Budget)")
    print("=" * 80)
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS\n")
    print("Hybrid = DCT(structure) + DFT/RFT(residual) with split coefficient budget")
    print("Testing different DCT/RFT splits to find optimal configuration\n")
    
    # Test conditions
    ecg_records = ["100", "101", "200", "207", "208", "217"]
    compression_levels = [0.1, 0.2, 0.3]  # Keep 10%, 20%, 30%
    
    all_results = []
    win_counts = {}
    
    print("📊 ECG Compression Results (MIT-BIH)")
    print("-" * 80)
    
    for record in ecg_records:
        print(f"\n🫀 Record {record}:")
        try:
            signal = load_ecg_signal(record)
            # Use first 10000 samples, normalize
            signal = signal[:10000]
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        except Exception as e:
            print(f"   ⚠️  Could not load: {e}")
            continue
        
        for keep_frac in compression_levels:
            total_k = max(1, int(len(signal) * keep_frac))
            results = run_comparison(signal, total_k)
            
            # Find winner
            winner = min(results, key=results.get)
            win_counts[winner] = win_counts.get(winner, 0) + 1
            
            # Print
            keep_pct = int(keep_frac * 100)
            print(f"   {keep_pct:2d}% ({total_k:4d} coeffs) | ", end="")
            for method in ["Pure_DCT", "H70/30", "H50/50"]:
                if method in results:
                    prd = results[method]
                    marker = "🏆" if method == winner else "  "
                    print(f"{marker}{method}: {prd:5.2f}% | ", end="")
            print(f"Best: {winner} ({results[winner]:.2f}%)")
            
            all_results.append({
                "record": record,
                "keep_pct": keep_pct,
                "results": results,
                "winner": winner,
            })
    
    # Detailed analysis: find where hybrid wins
    print("\n\n" + "=" * 80)
    print("ANALYSIS: When Does Hybrid Beat Pure DCT?")
    print("=" * 80)
    
    hybrid_wins = []
    dct_wins = []
    
    for r in all_results:
        dct_prd = r["results"]["Pure_DCT"]
        best_hybrid_prd = min(r["results"][k] for k in r["results"] if k != "Pure_DCT")
        best_hybrid = min([k for k in r["results"] if k != "Pure_DCT"], 
                         key=lambda k: r["results"][k])
        
        improvement = 100 * (dct_prd - best_hybrid_prd) / dct_prd
        
        if best_hybrid_prd < dct_prd:
            hybrid_wins.append({
                "record": r["record"],
                "keep": r["keep_pct"],
                "dct_prd": dct_prd,
                "hybrid_prd": best_hybrid_prd,
                "hybrid_config": best_hybrid,
                "improvement": improvement,
            })
        else:
            dct_wins.append(r)
    
    if hybrid_wins:
        print(f"\n✅ HYBRID WINS in {len(hybrid_wins)} cases:")
        for w in sorted(hybrid_wins, key=lambda x: -x["improvement"]):
            print(f"   Record {w['record']} @ {w['keep']}%: "
                  f"{w['hybrid_config']} beats DCT by {w['improvement']:.1f}% "
                  f"({w['hybrid_prd']:.2f}% vs {w['dct_prd']:.2f}%)")
    else:
        print("\n⚠️  NO CASES where hybrid beats pure DCT")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total = len(all_results)
    print(f"\nOut of {total} test cases:")
    for method, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = "█" * int(pct / 5)
        print(f"  {method:10s}: {count:2d} wins ({pct:5.1f}%) {bar}")
    
    dct_win_count = win_counts.get("Pure_DCT", 0)
    hybrid_win_count = total - dct_win_count
    
    print("\n" + "-" * 80)
    if hybrid_win_count > dct_win_count:
        print("✅ HYBRIDS WIN: DCT+RFT hybrid beats pure DCT on real ECG data!")
    elif hybrid_win_count > 0:
        print(f"🔸 MIXED: Hybrid wins {hybrid_win_count} cases, Pure DCT wins {dct_win_count}")
    else:
        print("⚠️  PURE DCT WINS: The hybrid approach doesn't help on ECG")
        print("   Possible reasons:")
        print("   1. ECG is mostly smooth (DCT-friendly), limited high-freq texture")
        print("   2. The 'residual' after DCT is mostly noise, not coherent texture")
        print("   3. Total budget constraint means splitting hurts both domains")
    
    print("\n" + "=" * 80)
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()

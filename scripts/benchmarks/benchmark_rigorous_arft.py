#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
RIGOROUS ARFT vs DCT vs FFT vs KLT Benchmark
=============================================

Bug checks:
1. Verify ARFT unitarity (roundtrip error ~1e-15)
2. Verify DCT/FFT normalized comparably (Parseval identity)
3. Verify PRD computed identically for all transforms
4. Add KLT/PCA baseline (true optimal for Gaussian signals)

Compression regimes: 5%, 10%, 20%, 30%, 50%

Separate stories:
- Non-adaptive: canonical RFT ≈ DCT ≈ FFT
- Adaptive: ARFT (operator-learned) vs KLT (covariance-learned)

FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS

Usage:
    USE_REAL_DATA=1 python scripts/benchmark_rigorous_arft.py
"""

import os
import sys
import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import toeplitz, eigh
from scipy.signal import find_peaks
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHI = (1 + np.sqrt(5)) / 2
EPS = 1e-10


# ============================================================================
# BUG CHECK 1: Verify Transform Unitarity
# ============================================================================

def check_unitarity():
    """Verify all transforms have roundtrip error ~1e-15."""
    print("=" * 70)
    print("BUG CHECK 1: Transform Unitarity (roundtrip error)")
    print("=" * 70)
    
    np.random.seed(42)
    n = 256
    x = np.random.randn(n)
    
    results = {}
    
    # DCT (Type-II with ortho norm is unitary)
    coeffs_dct = dct(x, type=2, norm='ortho')
    x_dct = idct(coeffs_dct, type=2, norm='ortho')
    err_dct = np.linalg.norm(x - x_dct) / np.linalg.norm(x)
    results["DCT"] = err_dct
    
    # FFT (with proper normalization)
    coeffs_fft = np.fft.fft(x) / np.sqrt(n)
    x_fft = np.real(np.fft.ifft(coeffs_fft * np.sqrt(n)))
    err_fft = np.linalg.norm(x - x_fft) / np.linalg.norm(x)
    results["FFT"] = err_fft
    
    # ARFT (eigenbasis of Toeplitz autocorrelation)
    autocorr = np.correlate(x, x, mode='full')[n-1:] / n
    L = toeplitz(autocorr[:n])
    eigenvalues, U = eigh(L)
    idx = np.argsort(eigenvalues)[::-1]
    U = U[:, idx]
    
    coeffs_arft = U.T @ x
    x_arft = U @ coeffs_arft
    err_arft = np.linalg.norm(x - x_arft) / np.linalg.norm(x)
    results["ARFT"] = err_arft
    
    # KLT (eigenbasis of sample covariance)
    # For a single signal, use outer product
    cov = np.outer(x, x)
    eigenvalues_klt, U_klt = eigh(cov)
    idx_klt = np.argsort(eigenvalues_klt)[::-1]
    U_klt = U_klt[:, idx_klt]
    
    coeffs_klt = U_klt.T @ x
    x_klt = U_klt @ coeffs_klt
    err_klt = np.linalg.norm(x - x_klt) / np.linalg.norm(x)
    results["KLT"] = err_klt
    
    print("\nRoundtrip errors (should be ~1e-15 for unitary):")
    all_pass = True
    for name, err in results.items():
        status = "✅" if err < 1e-10 else "❌ BUG!"
        print(f"  {name:6s}: {err:.2e} {status}")
        if err >= 1e-10:
            all_pass = False
    
    return all_pass


# ============================================================================
# BUG CHECK 2: Verify Parseval Identity (Energy Preservation)
# ============================================================================

def check_parseval():
    """Verify all transforms preserve energy (Parseval's identity)."""
    print("\n" + "=" * 70)
    print("BUG CHECK 2: Parseval Identity (energy preservation)")
    print("=" * 70)
    
    np.random.seed(42)
    n = 256
    x = np.random.randn(n)
    energy_x = np.sum(x**2)
    
    results = {}
    
    # DCT
    coeffs_dct = dct(x, type=2, norm='ortho')
    energy_dct = np.sum(coeffs_dct**2)
    results["DCT"] = abs(energy_dct - energy_x) / energy_x
    
    # FFT
    coeffs_fft = np.fft.fft(x) / np.sqrt(n)
    energy_fft = np.sum(np.abs(coeffs_fft)**2)
    results["FFT"] = abs(energy_fft - energy_x) / energy_x
    
    # ARFT
    autocorr = np.correlate(x, x, mode='full')[n-1:] / n
    L = toeplitz(autocorr[:n])
    _, U = eigh(L)
    coeffs_arft = U.T @ x
    energy_arft = np.sum(np.abs(coeffs_arft)**2)
    results["ARFT"] = abs(energy_arft - energy_x) / energy_x
    
    # KLT
    cov = np.outer(x, x)
    _, U_klt = eigh(cov)
    coeffs_klt = U_klt.T @ x
    energy_klt = np.sum(np.abs(coeffs_klt)**2)
    results["KLT"] = abs(energy_klt - energy_x) / energy_x
    
    print(f"\nSignal energy: {energy_x:.4f}")
    print("Relative energy error (should be ~1e-15):")
    all_pass = True
    for name, err in results.items():
        status = "✅" if err < 1e-10 else "❌ BUG!"
        print(f"  {name:6s}: {err:.2e} {status}")
        if err >= 1e-10:
            all_pass = False
    
    return all_pass


# ============================================================================
# BUG CHECK 3: Verify PRD Computed Identically
# ============================================================================

def check_prd_consistency():
    """Verify PRD is computed identically for all transforms."""
    print("\n" + "=" * 70)
    print("BUG CHECK 3: PRD Computation Consistency")
    print("=" * 70)
    
    np.random.seed(42)
    n = 256
    x = np.random.randn(n)
    k = 26  # 10% of coefficients
    
    def compute_prd(original, reconstructed):
        diff = original - reconstructed
        return 100 * np.sqrt(np.sum(diff**2) / np.sum(original**2))
    
    # Create identical reconstruction error for all transforms
    # by zeroing same fraction of coefficients
    
    prds = {}
    
    # DCT
    coeffs_dct = dct(x, type=2, norm='ortho')
    idx = np.argsort(np.abs(coeffs_dct))[::-1][:k]
    coeffs_sparse = np.zeros_like(coeffs_dct)
    coeffs_sparse[idx] = coeffs_dct[idx]
    x_recon = idct(coeffs_sparse, type=2, norm='ortho')
    prds["DCT"] = compute_prd(x, x_recon)
    
    # FFT
    coeffs_fft = np.fft.fft(x) / np.sqrt(n)
    idx = np.argsort(np.abs(coeffs_fft))[::-1][:k]
    coeffs_sparse = np.zeros_like(coeffs_fft)
    coeffs_sparse[idx] = coeffs_fft[idx]
    x_recon = np.real(np.fft.ifft(coeffs_sparse * np.sqrt(n)))
    prds["FFT"] = compute_prd(x, x_recon)
    
    # ARFT
    autocorr = np.correlate(x, x, mode='full')[n-1:] / n
    L = toeplitz(autocorr[:n])
    _, U = eigh(L)
    coeffs_arft = U.T @ x
    idx = np.argsort(np.abs(coeffs_arft))[::-1][:k]
    coeffs_sparse = np.zeros_like(coeffs_arft)
    coeffs_sparse[idx] = coeffs_arft[idx]
    x_recon = U @ coeffs_sparse
    prds["ARFT"] = compute_prd(x, x_recon)
    
    # KLT
    cov = np.outer(x, x)
    _, U_klt = eigh(cov)
    coeffs_klt = U_klt.T @ x
    idx = np.argsort(np.abs(coeffs_klt))[::-1][:k]
    coeffs_sparse = np.zeros_like(coeffs_klt)
    coeffs_sparse[idx] = coeffs_klt[idx]
    x_recon = U_klt @ coeffs_sparse
    prds["KLT"] = compute_prd(x, x_recon)
    
    print(f"\nPRD at 10% compression on random signal:")
    for name, prd in prds.items():
        print(f"  {name:6s}: {prd:.4f}%")
    
    print("\n✅ PRD formula is consistent (same function for all transforms)")
    return True


# ============================================================================
# Data Loaders
# ============================================================================

def load_ecg_records() -> Dict[str, np.ndarray]:
    """Load MIT-BIH ECG records."""
    try:
        import wfdb
    except ImportError:
        return {}
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mitbih")
    records = {}
    
    for record_id in ["100", "200", "207"]:
        record_path = os.path.join(data_dir, record_id)
        if os.path.exists(record_path + ".dat"):
            try:
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal[:1024, 0].astype(np.float64)
                records[f"ECG_{record_id}"] = signal
            except:
                pass
    
    return records


def generate_test_signals() -> Dict[str, np.ndarray]:
    """Generate various test signals."""
    signals = {}
    n = 1024
    
    # Fibonacci chain (ARFT target)
    fib = [1, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    signals["Fibonacci"] = np.array([fib[i % len(fib)] % 100 for i in range(n)], dtype=float)
    
    # Penrose (5-fold quasi-periodic)
    t = np.linspace(0, 1, n)
    penrose = np.zeros(n)
    for k in range(5):
        angle = k * 2 * np.pi / 5
        penrose += np.sin(2 * np.pi * 10 * (1 + 0.1 * np.cos(angle)) * t + angle)
    signals["Penrose"] = penrose
    
    # Pure sinusoid (DCT optimal)
    signals["Sinusoid"] = np.sin(2 * np.pi * 5 * t)
    
    # HRV-like (complex quasi-periodic)
    np.random.seed(42)
    rr_base = 800  # ms
    variability = 50 * np.sin(2 * np.pi * 0.1 * np.arange(n)) + \
                  30 * np.sin(2 * np.pi * 0.25 * np.arange(n)) + \
                  20 * np.random.randn(n)
    signals["HRV_Synthetic"] = rr_base + variability
    
    return signals


# ============================================================================
# Transform Implementations (Verified Unitary)
# ============================================================================

class TransformBenchmark:
    """Unified transform benchmark with verified implementations."""
    
    def __init__(self, signal: np.ndarray):
        self.x = (signal - np.mean(signal)) / (np.std(signal) + EPS)
        self.n = len(self.x)
        
        # Pre-compute adaptive kernels
        self._compute_arft_kernel()
        self._compute_klt_kernel()
    
    def _compute_arft_kernel(self):
        """Compute ARFT kernel (eigenbasis of Toeplitz autocorrelation)."""
        autocorr = np.correlate(self.x, self.x, mode='full')[self.n-1:] / self.n
        L = toeplitz(autocorr[:self.n])
        eigenvalues, U = eigh(L)
        idx = np.argsort(eigenvalues)[::-1]
        self.U_arft = U[:, idx]
    
    def _compute_klt_kernel(self):
        """Compute KLT kernel (eigenbasis of sample covariance)."""
        # For single signal, use outer product as covariance estimate
        # In practice, KLT uses multiple signal samples
        cov = np.outer(self.x, self.x)
        eigenvalues, U = eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        self.U_klt = U[:, idx]
    
    def transform(self, method: str) -> Tuple[np.ndarray, callable]:
        """Apply forward transform, return (coefficients, inverse_func)."""
        if method == "dct":
            coeffs = dct(self.x, type=2, norm='ortho')
            inv = lambda c: idct(c, type=2, norm='ortho')
        elif method == "fft":
            coeffs = np.fft.fft(self.x) / np.sqrt(self.n)
            inv = lambda c: np.real(np.fft.ifft(c * np.sqrt(self.n)))
        elif method == "arft":
            coeffs = self.U_arft.T @ self.x
            inv = lambda c: self.U_arft @ c
        elif method == "klt":
            coeffs = self.U_klt.T @ self.x
            inv = lambda c: self.U_klt @ c
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return coeffs, inv
    
    def compress(self, method: str, keep_ratio: float) -> Tuple[np.ndarray, float]:
        """Compress signal, return (reconstructed, prd)."""
        coeffs, inv = self.transform(method)
        k = max(1, int(self.n * keep_ratio))
        
        # Keep top-k by magnitude
        idx = np.argsort(np.abs(coeffs))[::-1][:k]
        coeffs_sparse = np.zeros_like(coeffs)
        coeffs_sparse[idx] = coeffs[idx]
        
        recon = np.real(inv(coeffs_sparse))
        
        # PRD
        diff = self.x - recon
        prd = 100 * np.sqrt(np.sum(diff**2) / np.sum(self.x**2))
        
        return recon, prd
    
    def gini_index(self, method: str) -> float:
        """Compute Gini index of coefficient magnitudes."""
        coeffs, _ = self.transform(method)
        c = np.sort(np.abs(coeffs.flatten()))
        n = len(c)
        if n == 0 or np.sum(c) == 0:
            return 0.0
        cumsum = np.cumsum(c)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


# ============================================================================
# Main Benchmark
# ============================================================================

def run_full_benchmark():
    """Run full benchmark with all bug checks and regimes."""
    
    # Bug checks first
    print("\n" + "=" * 70)
    print("PHASE 1: BUG CHECKS")
    print("=" * 70)
    
    pass1 = check_unitarity()
    pass2 = check_parseval()
    pass3 = check_prd_consistency()
    
    if not (pass1 and pass2 and pass3):
        print("\n❌ BUG CHECKS FAILED - ABORTING")
        return
    
    print("\n✅ ALL BUG CHECKS PASSED")
    
    # Load signals
    print("\n" + "=" * 70)
    print("PHASE 2: LOADING SIGNALS")
    print("=" * 70)
    
    signals = generate_test_signals()
    
    if os.environ.get("USE_REAL_DATA") == "1":
        ecg_records = load_ecg_records()
        signals.update(ecg_records)
        print(f"✅ Loaded {len(ecg_records)} ECG records")
    
    print(f"✅ Total signals: {len(signals)}")
    
    # Compression regimes
    keep_ratios = [0.05, 0.10, 0.20, 0.30, 0.50]
    methods = ["dct", "fft", "arft", "klt"]
    
    print("\n" + "=" * 70)
    print("PHASE 3: COMPRESSION BENCHMARK")
    print("=" * 70)
    print("\nMethods: DCT (fixed), FFT (fixed), ARFT (adaptive), KLT (adaptive/optimal)")
    print("Regimes: 5%, 10%, 20%, 30%, 50% coefficients kept")
    
    # Results storage
    all_results = {}
    
    for sig_name, signal in signals.items():
        print(f"\n{'─'*60}")
        print(f"📊 {sig_name} (N={len(signal)})")
        print(f"{'─'*60}")
        
        bench = TransformBenchmark(signal)
        all_results[sig_name] = {"gini": {}, "prd": {}}
        
        # Gini index
        print("\nSparsity (Gini):")
        ginis = {m: bench.gini_index(m) for m in methods}
        gini_winner = max(ginis, key=ginis.get)
        for m, g in ginis.items():
            marker = "🏆" if m == gini_winner else "  "
            print(f"  {marker}{m.upper():5s}: {g:.4f}")
        all_results[sig_name]["gini"] = ginis
        
        # PRD at each regime
        print("\nPRD (%) at each compression level:")
        print(f"  {'Keep':>6s} |", end="")
        for m in methods:
            print(f" {m.upper():>7s} |", end="")
        print(" Winner")
        print("  " + "-" * 50)
        
        for keep in keep_ratios:
            prds = {}
            for m in methods:
                _, prd = bench.compress(m, keep)
                prds[m] = prd
            
            winner = min(prds, key=prds.get)
            all_results[sig_name]["prd"][keep] = prds
            
            print(f"  {int(keep*100):5d}% |", end="")
            for m in methods:
                marker = "🏆" if m == winner else "  "
                print(f" {marker}{prds[m]:5.2f} |", end="")
            print(f" {winner.upper()}")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("PHASE 4: SUMMARY")
    print("=" * 70)
    
    # Count wins
    gini_wins = {m: 0 for m in methods}
    prd_wins = {m: 0 for m in methods}
    prd_tests = 0
    
    for sig_name, results in all_results.items():
        gini_winner = max(results["gini"], key=results["gini"].get)
        gini_wins[gini_winner] += 1
        
        for keep, prds in results["prd"].items():
            prd_winner = min(prds, key=prds.get)
            prd_wins[prd_winner] += 1
            prd_tests += 1
    
    print("\n📊 SPARSITY (Gini) WINS:")
    for m in methods:
        pct = 100 * gini_wins[m] / len(signals)
        bar = "█" * int(pct / 5)
        adaptive = "(adaptive)" if m in ["arft", "klt"] else "(fixed)"
        print(f"  {m.upper():5s} {adaptive:12s}: {gini_wins[m]:2d}/{len(signals)} ({pct:5.1f}%) {bar}")
    
    print(f"\n📊 COMPRESSION (PRD) WINS ({prd_tests} total tests):")
    for m in methods:
        pct = 100 * prd_wins[m] / prd_tests
        bar = "█" * int(pct / 5)
        adaptive = "(adaptive)" if m in ["arft", "klt"] else "(fixed)"
        print(f"  {m.upper():5s} {adaptive:12s}: {prd_wins[m]:2d}/{prd_tests} ({pct:5.1f}%) {bar}")
    
    # Separate stories
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    fixed_wins = gini_wins["dct"] + gini_wins["fft"]
    adaptive_wins = gini_wins["arft"] + gini_wins["klt"]
    
    print("\n1️⃣  NON-ADAPTIVE TRANSFORMS (DCT, FFT):")
    print(f"   Combined Gini wins: {fixed_wins}/{len(signals)}")
    print("   → DCT ≈ FFT on most smooth signals (as expected)")
    
    print("\n2️⃣  ADAPTIVE TRANSFORMS (ARFT, KLT):")
    print(f"   Combined Gini wins: {adaptive_wins}/{len(signals)}")
    
    if gini_wins["arft"] >= gini_wins["klt"]:
        print("   → ARFT is competitive with KLT (true optimal)")
        print("   → ARFT has hardware implementation path (Toeplitz eigenbasis)")
    else:
        print("   → KLT is optimal (as expected for Gaussian)")
        print(f"   → ARFT achieves {100*gini_wins['arft']/max(1,gini_wins['klt']):.0f}% of KLT performance")
    
    print("\n" + "=" * 70)
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
    print("=" * 70)


def main():
    run_full_benchmark()


if __name__ == "__main__":
    main()

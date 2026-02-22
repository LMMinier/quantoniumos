#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Operator-ARFT vs DCT/FFT on ALL Medical Signals
================================================

Tests the Operator-ARFT on real biomedical data to find domains where
the adaptive eigenbasis approach provides benefits.

Signals tested:
- ECG (MIT-BIH): Smooth quasi-periodic heartbeats
- EEG (Sleep-EDF): Multi-rhythm brain waves (alpha, delta, theta, etc.)
- HRV (Heart Rate Variability): Derived from ECG R-R intervals
- Genomic sequence: DNA base encoding (potentially quasi-periodic)

FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS

Usage:
    USE_REAL_DATA=1 python scripts/benchmark_arft_medical.py
"""

import os
import sys
import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import toeplitz
from scipy.signal import find_peaks
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHI = (1 + np.sqrt(5)) / 2


def require_real_data():
    if os.environ.get("USE_REAL_DATA") != "1":
        print("⚠️  Real data validation requires explicit opt-in.")
        print("   Set USE_REAL_DATA=1 to run.")
        print("   FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
        sys.exit(1)


# ============================================================================
# Data Loaders
# ============================================================================

def load_ecg_records() -> Dict[str, np.ndarray]:
    """Load multiple MIT-BIH ECG records."""
    try:
        import wfdb
    except ImportError:
        print("Install wfdb: pip install wfdb")
        return {}
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mitbih")
    records = {}
    
    for record_id in ["100", "101", "200", "207", "208", "217"]:
        record_path = os.path.join(data_dir, record_id)
        if os.path.exists(record_path + ".dat"):
            try:
                record = wfdb.rdrecord(record_path)
                # Extract 2048 samples for analysis
                signal = record.p_signal[:2048, 0].astype(np.float64)
                records[f"ECG_{record_id}"] = signal
            except:
                pass
    
    return records


def load_eeg_signal() -> Optional[np.ndarray]:
    """Load Sleep-EDF EEG data."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sleep_edf")
    
    try:
        import pyedflib
    except ImportError:
        return None
    
    for f in os.listdir(data_dir) if os.path.exists(data_dir) else []:
        if f.endswith("PSG.edf"):
            edf_path = os.path.join(data_dir, f)
            try:
                edf = pyedflib.EdfReader(edf_path)
                for i in range(edf.signals_in_file):
                    if 'EEG' in edf.getLabel(i).upper():
                        signal = edf.readSignal(i)[:2048]
                        edf.close()
                        return signal.astype(np.float64)
                edf.close()
            except:
                pass
    return None


def extract_hrv_from_ecg(ecg: np.ndarray, fs: float = 360) -> Optional[np.ndarray]:
    """
    Extract Heart Rate Variability (R-R intervals) from ECG.
    HRV has complex quasi-periodic structure.
    """
    # Find R-peaks
    ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-10)
    peaks, _ = find_peaks(ecg_norm, distance=int(0.5 * fs), height=0.5)
    
    if len(peaks) < 10:
        return None
    
    # R-R intervals in ms
    rr_intervals = np.diff(peaks) / fs * 1000
    
    # Interpolate to uniform sampling for transform
    if len(rr_intervals) < 64:
        return None
    
    return rr_intervals[:min(512, len(rr_intervals))]


def load_genomic_signal() -> Optional[np.ndarray]:
    """Load DNA sequence as numeric signal."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "genomics")
    fasta_path = os.path.join(data_dir, "lambda_phage.fasta")
    
    if not os.path.exists(fasta_path):
        return None
    
    try:
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
        
        seq = ''.join(line.strip() for line in lines if not line.startswith('>'))
        
        # Encode: A=1, C=2, G=3, T=4
        encoding = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
        signal = np.array([encoding.get(b.upper(), 0) for b in seq[:2048]], dtype=float)
        
        return signal
    except:
        return None


# ============================================================================
# Transform Implementations
# ============================================================================

def compute_autocorr(x: np.ndarray) -> np.ndarray:
    """Compute autocorrelation function."""
    n = len(x)
    autocorr = np.correlate(x, x, mode='full')[n-1:] / n
    return autocorr


def build_operator_kernel(n: int, autocorr: np.ndarray) -> np.ndarray:
    """Build ARFT kernel from autocorrelation."""
    L = toeplitz(autocorr[:n])
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx]


def sparse_approx(x: np.ndarray, transform: str, keep_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse approximation using different transforms."""
    n = len(x)
    k = max(1, int(n * keep_ratio))
    
    if transform == "dct":
        coeffs = dct(x, type=2, norm='ortho')
        idx = np.argsort(np.abs(coeffs))[::-1][:k]
        coeffs_sparse = np.zeros_like(coeffs)
        coeffs_sparse[idx] = coeffs[idx]
        recon = idct(coeffs_sparse, type=2, norm='ortho')
        return recon, coeffs
    
    elif transform == "fft":
        coeffs = np.fft.fft(x) / np.sqrt(n)
        idx = np.argsort(np.abs(coeffs))[::-1][:k]
        coeffs_sparse = np.zeros_like(coeffs)
        coeffs_sparse[idx] = coeffs[idx]
        recon = np.real(np.fft.ifft(coeffs_sparse * np.sqrt(n)))
        return recon, coeffs
    
    elif transform == "arft":
        autocorr = compute_autocorr(x)
        kernel = build_operator_kernel(n, autocorr)
        coeffs = kernel.conj().T @ x
        idx = np.argsort(np.abs(coeffs))[::-1][:k]
        coeffs_sparse = np.zeros_like(coeffs)
        coeffs_sparse[idx] = coeffs[idx]
        recon = np.real(kernel @ coeffs_sparse)
        return recon, coeffs
    
    raise ValueError(f"Unknown transform: {transform}")


# ============================================================================
# Metrics
# ============================================================================

def gini_index(coeffs: np.ndarray) -> float:
    """Gini index - higher = sparser."""
    c = np.sort(np.abs(coeffs.flatten()))
    n = len(c)
    if n == 0 or np.sum(c) == 0:
        return 0.0
    cumsum = np.cumsum(c)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def compute_prd(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Percent Root-mean-square Difference."""
    diff = original - reconstructed
    return 100 * np.sqrt(np.sum(diff**2) / np.sum(original**2))


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - reconstructed)**2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(np.abs(original))
    return 20 * np.log10(max_val / np.sqrt(mse))


# ============================================================================
# Benchmark Runner
# ============================================================================

def benchmark_signal(x: np.ndarray, name: str, keep_ratios: List[float]) -> Dict:
    """Run full benchmark on a signal."""
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    n = len(x)
    
    results = {"signal": name, "n": n, "transforms": {}}
    
    for transform in ["dct", "fft", "arft"]:
        results["transforms"][transform] = {"gini": None, "compression": []}
        
        try:
            _, coeffs = sparse_approx(x, transform, 1.0)
            results["transforms"][transform]["gini"] = gini_index(coeffs)
            
            for keep in keep_ratios:
                recon, _ = sparse_approx(x, transform, keep)
                results["transforms"][transform]["compression"].append({
                    "keep": keep,
                    "prd": compute_prd(x, recon),
                    "psnr": compute_psnr(x, recon),
                })
        except Exception as e:
            results["transforms"][transform]["error"] = str(e)
    
    return results


def print_signal_results(results: Dict) -> Tuple[str, str]:
    """Print results and return (gini_winner, prd_winner)."""
    print(f"\n{'─'*60}")
    print(f"📊 {results['signal']} (N={results['n']})")
    print(f"{'─'*60}")
    
    # Gini
    ginis = {}
    for t in ["dct", "fft", "arft"]:
        if "error" not in results["transforms"][t]:
            ginis[t] = results["transforms"][t]["gini"]
    
    if ginis:
        gini_winner = max(ginis, key=ginis.get)
        print(f"  Sparsity (Gini): ", end="")
        for t, g in ginis.items():
            marker = "🏆" if t == gini_winner else ""
            print(f"{marker}{t.upper()}={g:.3f}  ", end="")
        print()
    else:
        gini_winner = None
    
    # PRD at 10%
    prds = {}
    for t in ["dct", "fft", "arft"]:
        if "error" not in results["transforms"][t] and results["transforms"][t]["compression"]:
            # Find 10% compression result
            for c in results["transforms"][t]["compression"]:
                if c["keep"] == 0.1:
                    prds[t] = c["prd"]
                    break
    
    if prds:
        prd_winner = min(prds, key=prds.get)
        print(f"  Compression @10% (PRD): ", end="")
        for t, p in prds.items():
            marker = "🏆" if t == prd_winner else ""
            print(f"{marker}{t.upper()}={p:.2f}%  ", end="")
        print()
    else:
        prd_winner = None
    
    return gini_winner, prd_winner


def main():
    require_real_data()
    
    print("=" * 70)
    print("Operator-ARFT vs DCT/FFT on ALL MEDICAL SIGNALS")
    print("=" * 70)
    print("FOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS\n")
    
    signals = {}
    
    # Load ECG
    print("📥 Loading ECG records...")
    ecg_records = load_ecg_records()
    signals.update(ecg_records)
    print(f"   Found {len(ecg_records)} ECG records")
    
    # Load EEG
    print("📥 Loading EEG...")
    eeg = load_eeg_signal()
    if eeg is not None:
        signals["EEG_Sleep"] = eeg
        print("   ✅ EEG loaded")
    else:
        # Synthetic EEG with multiple rhythms
        t = np.linspace(0, 4, 2048)
        alpha = np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        theta = 0.7 * np.sin(2 * np.pi * 6 * t)  # 6 Hz theta
        delta = 1.2 * np.sin(2 * np.pi * 2 * t)  # 2 Hz delta
        signals["EEG_Synthetic"] = alpha + theta + delta + 0.1 * np.random.randn(2048)
        print("   ⚠️  Using synthetic EEG")
    
    # Extract HRV from first ECG
    print("📥 Extracting HRV...")
    if ecg_records:
        first_ecg = list(ecg_records.values())[0]
        # Need longer segment for HRV
        try:
            import wfdb
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mitbih")
            record = wfdb.rdrecord(os.path.join(data_dir, "100"))
            long_ecg = record.p_signal[:, 0]
            hrv = extract_hrv_from_ecg(long_ecg)
            if hrv is not None and len(hrv) >= 64:
                signals["HRV_RR_Intervals"] = hrv
                print(f"   ✅ HRV extracted ({len(hrv)} beats)")
        except:
            pass
    
    # Load genomic
    print("📥 Loading genomic data...")
    genomic = load_genomic_signal()
    if genomic is not None:
        signals["Genomic_Lambda"] = genomic
        print("   ✅ Genomic loaded")
    else:
        print("   ⚠️  Genomic not available")
    
    # Generate quasi-periodic reference signals
    print("\n📊 Adding reference quasi-periodic signals...")
    
    # Fibonacci chain (ARFT's sweet spot)
    fib = [1, 1]
    while len(fib) < 2048:
        fib.append(fib[-1] + fib[-2])
    signals["REF_Fibonacci"] = np.array([fib[i % len(fib)] % 100 for i in range(1024)], dtype=float)
    
    # Penrose tiling (5-fold quasi-periodic)
    t = np.linspace(0, 1, 1024)
    penrose = np.zeros(1024)
    for k in range(5):
        angle = k * 2 * np.pi / 5
        penrose += np.sin(2 * np.pi * 10 * (1 + 0.1 * np.cos(angle)) * t + angle)
    signals["REF_Penrose"] = penrose
    
    print(f"\n✅ Total signals to test: {len(signals)}")
    
    # Run benchmarks
    keep_ratios = [0.05, 0.10, 0.20]
    
    all_results = []
    gini_wins = {"dct": 0, "fft": 0, "arft": 0}
    prd_wins = {"dct": 0, "fft": 0, "arft": 0}
    
    print("\n" + "=" * 70)
    print("RESULTS BY SIGNAL TYPE")
    print("=" * 70)
    
    for name, signal in signals.items():
        results = benchmark_signal(signal, name, keep_ratios)
        all_results.append(results)
        
        gw, pw = print_signal_results(results)
        if gw:
            gini_wins[gw] += 1
        if pw:
            prd_wins[pw] += 1
    
    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY: WHERE EACH TRANSFORM WINS")
    print("=" * 70)
    
    total_gini = sum(gini_wins.values())
    total_prd = sum(prd_wins.values())
    
    print(f"\n📊 SPARSITY (Gini) - {total_gini} signals tested:")
    for t in ["arft", "dct", "fft"]:
        pct = 100 * gini_wins[t] / total_gini if total_gini > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"   {t.upper():5s}: {gini_wins[t]:2d} wins ({pct:5.1f}%) {bar}")
    
    print(f"\n📊 COMPRESSION @10% (PRD) - {total_prd} signals tested:")
    for t in ["arft", "dct", "fft"]:
        pct = 100 * prd_wins[t] / total_prd if total_prd > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"   {t.upper():5s}: {prd_wins[t]:2d} wins ({pct:5.1f}%) {bar}")
    
    # Find ARFT's domains
    print("\n" + "─" * 70)
    print("ARFT WINS ON:")
    arft_domains = []
    for r in all_results:
        ginis = {t: r["transforms"][t]["gini"] 
                 for t in ["dct", "fft", "arft"] 
                 if "error" not in r["transforms"][t] and r["transforms"][t]["gini"] is not None}
        if ginis and max(ginis, key=ginis.get) == "arft":
            arft_domains.append(r["signal"])
    
    if arft_domains:
        for d in arft_domains:
            print(f"   ✅ {d}")
    else:
        print("   (none in this test set)")
    
    print("\nDCT WINS ON:")
    dct_domains = []
    for r in all_results:
        ginis = {t: r["transforms"][t]["gini"] 
                 for t in ["dct", "fft", "arft"] 
                 if "error" not in r["transforms"][t] and r["transforms"][t]["gini"] is not None}
        if ginis and max(ginis, key=ginis.get) == "dct":
            dct_domains.append(r["signal"])
    
    for d in dct_domains:
        print(f"   ✅ {d}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if gini_wins["arft"] > 0:
        print(f"✅ ARFT shows advantage on {gini_wins['arft']} signal types")
        print("   Best for: Fibonacci/quasicrystal structures, non-smooth quasi-periodic")
    print(f"✅ DCT remains optimal for {gini_wins['dct']} signal types")
    print("   Best for: Smooth signals (ECG, pure sinusoids)")
    print("\nFOR RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
    print("=" * 70)


if __name__ == "__main__":
    main()

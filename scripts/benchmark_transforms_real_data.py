#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Empirical Benchmark: FFT vs DCT vs φ-RFT vs Canonical RFT on Real Biomedical Data

⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE  ⚠️

Compares transform performance on:
- MIT-BIH ECG (PhysioNet)
- Sleep-EDF EEG (PhysioNet)
- Lambda Phage Genome (NCBI)
- PDB Protein Structure

Metrics:
- PRD (Percent Root-mean-square Difference) - lower is better
- PSNR (Peak Signal-to-Noise Ratio) - higher is better
- SNR (Signal-to-Noise Ratio) - higher is better
- Compression ratio at fixed quality

Usage:
    USE_REAL_DATA=1 python scripts/benchmark_transforms_real_data.py
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.fftpack import dct, idct

# Ensure imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as phi_rft_forward, rft_inverse_square as phi_rft_inverse
from algorithms.rft.variants.operator_variants import rft_forward as canonical_rft_forward, rft_inverse as canonical_rft_inverse


@dataclass
class BenchmarkResult:
    transform: str
    dataset: str
    keep_ratio: float
    prd: float  # Percent Root-mean-square Difference
    psnr: float  # Peak SNR (dB)
    snr: float  # Signal-to-Noise Ratio (dB)
    

def check_env():
    if os.environ.get("USE_REAL_DATA") != "1":
        print("ERROR: USE_REAL_DATA=1 required")
        sys.exit(1)


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Tuple[float, float, float]:
    """Compute PRD, PSNR, SNR between original and reconstructed signals."""
    original = np.real(original)
    reconstructed = np.real(reconstructed)
    
    error = original - reconstructed
    mse = np.mean(error ** 2)
    
    # PRD: Percent Root-mean-square Difference
    prd = 100.0 * np.sqrt(np.sum(error ** 2) / np.sum(original ** 2))
    
    # PSNR: Peak Signal-to-Noise Ratio
    peak = np.max(np.abs(original))
    if mse > 0:
        psnr = 10 * np.log10(peak ** 2 / mse)
    else:
        psnr = float('inf')
    
    # SNR: Signal-to-Noise Ratio
    signal_power = np.mean(original ** 2)
    if mse > 0:
        snr = 10 * np.log10(signal_power / mse)
    else:
        snr = float('inf')
    
    return prd, psnr, snr


def fft_compress(signal: np.ndarray, keep_ratio: float) -> np.ndarray:
    """FFT-based compression: keep top coefficients by magnitude."""
    coeffs = np.fft.fft(signal, norm='ortho')
    n_keep = int(len(coeffs) * keep_ratio)
    
    # Keep top coefficients
    idx = np.argsort(np.abs(coeffs))[::-1]
    sparse = np.zeros_like(coeffs)
    sparse[idx[:n_keep]] = coeffs[idx[:n_keep]]
    
    return np.fft.ifft(sparse, norm='ortho')


def dct_compress(signal: np.ndarray, keep_ratio: float) -> np.ndarray:
    """DCT-based compression: keep top coefficients by magnitude."""
    coeffs = dct(signal, type=2, norm='ortho')
    n_keep = int(len(coeffs) * keep_ratio)
    
    # Keep top coefficients
    idx = np.argsort(np.abs(coeffs))[::-1]
    sparse = np.zeros_like(coeffs)
    sparse[idx[:n_keep]] = coeffs[idx[:n_keep]]
    
    return idct(sparse, type=2, norm='ortho')


def phi_rft_compress(signal: np.ndarray, keep_ratio: float) -> np.ndarray:
    """φ-RFT compression: keep top coefficients by magnitude."""
    coeffs = phi_rft_forward(signal)
    n_keep = int(len(coeffs) * keep_ratio)
    
    # Keep top coefficients
    idx = np.argsort(np.abs(coeffs))[::-1]
    sparse = np.zeros_like(coeffs)
    sparse[idx[:n_keep]] = coeffs[idx[:n_keep]]
    
    return phi_rft_inverse(sparse)


def canonical_rft_compress(signal: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Canonical operator-based RFT compression."""
    coeffs, Phi = canonical_rft_forward(signal.astype(np.float64))
    n_keep = int(len(coeffs) * keep_ratio)
    
    # Keep top coefficients
    idx = np.argsort(np.abs(coeffs))[::-1]
    sparse = np.zeros_like(coeffs)
    sparse[idx[:n_keep]] = coeffs[idx[:n_keep]]
    
    return canonical_rft_inverse(sparse, Phi)


def benchmark_signal(name: str, signal: np.ndarray, keep_ratios: List[float]) -> List[BenchmarkResult]:
    """Benchmark all transforms on a signal."""
    results = []
    signal = signal.astype(np.float64)
    
    transforms = [
        ("FFT", fft_compress),
        ("DCT", dct_compress),
        ("φ-RFT", phi_rft_compress),
        ("Canonical RFT", canonical_rft_compress),
    ]
    
    for keep_ratio in keep_ratios:
        for tname, tfunc in transforms:
            try:
                recon = tfunc(signal, keep_ratio)
                prd, psnr, snr = compute_metrics(signal, recon)
                results.append(BenchmarkResult(
                    transform=tname,
                    dataset=name,
                    keep_ratio=keep_ratio,
                    prd=prd,
                    psnr=psnr,
                    snr=snr,
                ))
            except Exception as e:
                print(f"  Warning: {tname} failed on {name}: {e}")
    
    return results


def load_ecg() -> List[Tuple[str, np.ndarray]]:
    """Load MIT-BIH ECG signals."""
    signals = []
    mitbih_dir = ROOT / "data" / "physionet" / "mitbih"
    
    if not mitbih_dir.exists():
        return signals
    
    try:
        import wfdb
    except ImportError:
        print("⚠ wfdb not installed")
        return signals
    
    for rec_id in ["100", "101"]:
        if (mitbih_dir / f"{rec_id}.dat").exists():
            record = wfdb.rdrecord(str(mitbih_dir / rec_id))
            # Take 1024-sample windows at different points
            sig = record.p_signal[:, 0]
            for i, start in enumerate([1000, 10000, 50000]):
                if start + 1024 <= len(sig):
                    signals.append((f"ECG-{rec_id}-w{i}", sig[start:start+1024]))
    
    return signals


def load_eeg() -> List[Tuple[str, np.ndarray]]:
    """Load Sleep-EDF EEG signals."""
    signals = []
    sleepedf_dir = ROOT / "data" / "physionet" / "sleepedf"
    
    if not sleepedf_dir.exists():
        return signals
    
    try:
        import pyedflib
    except ImportError:
        print("⚠ pyedflib not installed")
        return signals
    
    for edf_file in list(sleepedf_dir.glob("*-PSG.edf"))[:2]:
        f = pyedflib.EdfReader(str(edf_file))
        labels = [f.getLabel(i) for i in range(f.signals_in_file)]
        eeg_idx = labels.index("EEG Fpz-Cz") if "EEG Fpz-Cz" in labels else 0
        eeg = f.readSignal(eeg_idx)
        f.close()
        
        # Take windows
        for i, start in enumerate([10000, 50000, 100000]):
            if start + 1024 <= len(eeg):
                signals.append((f"EEG-{edf_file.stem[:7]}-w{i}", eeg[start:start+1024]))
    
    return signals


def load_genomics() -> List[Tuple[str, np.ndarray]]:
    """Load genomics data."""
    signals = []
    genomics_dir = ROOT / "data" / "genomics"
    
    # Lambda phage
    fasta = genomics_dir / "lambda_phage.fasta"
    if fasta.exists():
        with open(fasta) as f:
            lines = f.readlines()
        seq = "".join(l.strip() for l in lines if not l.startswith(">"))
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
        numeric = np.array([mapping.get(c.upper(), 0) for c in seq[:1024]], dtype=np.float64)
        signals.append(("Lambda-Phage", numeric))
    
    # PDB
    pdb = genomics_dir / "1crn.pdb"
    if pdb.exists():
        with open(pdb) as f:
            atoms = [l for l in f if l.startswith("ATOM")]
        ca_coords = []
        for line in atoms:
            if line[12:16].strip() == "CA":
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                ca_coords.extend([x, y, z])
        # Pad to power of 2
        n = 1 << (len(ca_coords) - 1).bit_length()
        padded = np.zeros(n)
        padded[:len(ca_coords)] = ca_coords
        signals.append(("PDB-1CRN", padded))
    
    return signals


def print_results(results: List[BenchmarkResult]):
    """Print results as formatted tables."""
    if not results:
        print("No results to display")
        return
    
    # Group by dataset and keep_ratio
    datasets = sorted(set(r.dataset for r in results))
    keep_ratios = sorted(set(r.keep_ratio for r in results))
    transforms = ["FFT", "DCT", "φ-RFT", "Canonical RFT"]
    
    print("\n" + "=" * 90)
    print("EMPIRICAL BENCHMARK: Transform Comparison on Real Biomedical Data")
    print("=" * 90)
    
    for keep_ratio in keep_ratios:
        print(f"\n### Compression: Keep {keep_ratio*100:.0f}% of coefficients\n")
        print(f"{'Dataset':<20} {'Transform':<16} {'PRD (%)':<10} {'PSNR (dB)':<12} {'SNR (dB)':<10}")
        print("-" * 70)
        
        for dataset in datasets:
            for transform in transforms:
                matching = [r for r in results 
                           if r.dataset == dataset 
                           and r.transform == transform 
                           and r.keep_ratio == keep_ratio]
                if matching:
                    r = matching[0]
                    print(f"{r.dataset:<20} {r.transform:<16} {r.prd:<10.2f} {r.psnr:<12.2f} {r.snr:<10.2f}")
            print()
    
    # Summary: Average gain of canonical RFT over others
    print("\n" + "=" * 90)
    print("SUMMARY: Average Metrics by Transform (lower PRD = better, higher PSNR/SNR = better)")
    print("=" * 90)
    
    for transform in transforms:
        t_results = [r for r in results if r.transform == transform]
        if t_results:
            avg_prd = np.mean([r.prd for r in t_results])
            avg_psnr = np.mean([r.psnr for r in t_results])
            avg_snr = np.mean([r.snr for r in t_results])
            print(f"{transform:<16}: PRD={avg_prd:.2f}%  PSNR={avg_psnr:.2f} dB  SNR={avg_snr:.2f} dB")
    
    # Compute relative gains
    print("\n" + "-" * 70)
    print("Relative Performance (vs FFT baseline):")
    print("-" * 70)
    
    fft_results = {(r.dataset, r.keep_ratio): r for r in results if r.transform == "FFT"}
    
    for transform in ["DCT", "φ-RFT", "Canonical RFT"]:
        prd_gains = []
        psnr_gains = []
        snr_gains = []
        
        for r in results:
            if r.transform == transform:
                key = (r.dataset, r.keep_ratio)
                if key in fft_results:
                    fft_r = fft_results[key]
                    # PRD: lower is better, so gain = (fft - this) / fft * 100
                    if fft_r.prd > 0:
                        prd_gains.append((fft_r.prd - r.prd) / fft_r.prd * 100)
                    # PSNR/SNR: higher is better
                    psnr_gains.append(r.psnr - fft_r.psnr)
                    snr_gains.append(r.snr - fft_r.snr)
        
        if prd_gains:
            avg_prd_gain = np.mean(prd_gains)
            avg_psnr_gain = np.mean(psnr_gains)
            avg_snr_gain = np.mean(snr_gains)
            prd_sign = "+" if avg_prd_gain > 0 else ""
            psnr_sign = "+" if avg_psnr_gain > 0 else ""
            snr_sign = "+" if avg_snr_gain > 0 else ""
            print(f"{transform:<16}: PRD {prd_sign}{avg_prd_gain:.1f}%  PSNR {psnr_sign}{avg_psnr_gain:.2f} dB  SNR {snr_sign}{avg_snr_gain:.2f} dB")


def main():
    check_env()
    
    print("=" * 70)
    print("Loading real biomedical datasets...")
    print("⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL USE")
    print("=" * 70)
    
    all_signals = []
    
    # Load ECG
    ecg_signals = load_ecg()
    print(f"Loaded {len(ecg_signals)} ECG windows")
    all_signals.extend(ecg_signals)
    
    # Load EEG
    eeg_signals = load_eeg()
    print(f"Loaded {len(eeg_signals)} EEG windows")
    all_signals.extend(eeg_signals)
    
    # Load Genomics
    genomics_signals = load_genomics()
    print(f"Loaded {len(genomics_signals)} genomics signals")
    all_signals.extend(genomics_signals)
    
    if not all_signals:
        print("No data found. Run fetch scripts first.")
        sys.exit(1)
    
    print(f"\nTotal: {len(all_signals)} signals")
    
    # Benchmark at different compression levels
    keep_ratios = [0.10, 0.20, 0.30, 0.50]
    
    all_results = []
    
    print("\nRunning benchmarks...")
    for name, signal in all_signals:
        print(f"  Benchmarking {name}...")
        results = benchmark_signal(name, signal, keep_ratios)
        all_results.extend(results)
    
    print_results(all_results)
    
    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

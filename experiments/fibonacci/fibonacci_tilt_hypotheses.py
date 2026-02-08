#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Fibonacci-Tilt Φ-RFT: 10 Testable Hypotheses Validation
========================================================

Author: Luis M. Minier
Date: November 27, 2025

Tests 10 concrete hypotheses about the Fibonacci-tilt variant of Φ-RFT:
H1: Fibonacci sequences are asymptotically 1-sparse in Fib-Φ-RFT
H2: Fib-Φ-RFT is the "natural" diagonalizer of golden rotations
H3: Golden-structured text segments are detectable by Fib sparsity
H4: Hybrid DCT + Fib-Φ-RFT dominates on mixed quasi-periodic data
H5: Fib-Φ-RFT yields deterministic "good" sensing matrices
H6: Fib-Φ-RFT exposes quasicrystal-like diffraction peaks
H7: Fib-tilt improves anomaly detection in "almost golden" streams
H8: Golden-aligned hash mixing layer improves avalanche
H9: Fib-Φ-RFT reduces quantization sensitivity for golden signals
H10: Fib-tilt yields a low-rank model of golden Markov chains

Output: experiments/FIBONACCI_TILT_RESULTS.md
"""

import numpy as np
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import time
from scipy.fft import dct, idct, fft, ifft
from scipy.linalg import svd
from collections import Counter
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import RFT core
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI

# ============================================================================
# FIBONACCI-TILT RFT IMPLEMENTATION
# ============================================================================

def fibonacci_tilt_rft(signal: np.ndarray) -> np.ndarray:
    """
    Fibonacci-Tilt Φ-RFT: Uses Fibonacci numbers for phase modulation
    instead of continuous golden ratio.
    
    Instead of: exp(2πi·φ·{k/φ})
    Uses:       exp(2πi·F_k / n) where F_k is k-th Fibonacci number
    """
    n = len(signal)
    
    # Generate Fibonacci sequence up to n
    fib = [1, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    fib = np.array(fib[:n], dtype=np.float64)
    
    # Fibonacci-tilt phase modulation
    k = np.arange(n)
    fib_phase = np.exp(2j * np.pi * fib.astype(np.float64) / n)
    
    # Apply phase tilt then FFT
    tilted = signal * fib_phase
    coeffs = fft(tilted)
    
    return coeffs


def fibonacci_tilt_inverse(coeffs: np.ndarray) -> np.ndarray:
    """Inverse Fibonacci-tilt transform."""
    n = len(coeffs)
    
    # Generate Fibonacci sequence
    fib = [1, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    fib = np.array(fib[:n], dtype=np.float64)
    
    # Inverse FFT then remove phase tilt
    signal = ifft(coeffs)
    
    k = np.arange(n)
    fib_phase = np.exp(-2j * np.pi * fib / n)
    
    return np.real(signal * fib_phase)


# ============================================================================
# SIGNAL GENERATORS
# ============================================================================

def generate_fibonacci_sequence(n: int) -> np.ndarray:
    """Generate Fibonacci word substitution sequence (Sturmian)."""
    # Fibonacci word: 0 -> 01, 1 -> 0
    word = "0"
    while len(word) < n:
        word = word.replace("1", "X").replace("0", "01").replace("X", "0")
    
    signal = np.array([float(c) for c in word[:n]])
    return signal


def generate_golden_rotation(n: int, noise: float = 0.01) -> np.ndarray:
    """Generate sequence from golden rotation + noise."""
    angles = np.mod(np.arange(n) * PHI, 1.0)
    signal = np.sin(2 * np.pi * angles)
    signal += noise * np.random.randn(n)
    return signal


def generate_quasicrystal_1d(n: int, dim: int = 5) -> np.ndarray:
    """1D cut through Fibonacci quasicrystal."""
    # Cut-and-project from higher dimension
    k = np.arange(n)
    signal = np.zeros(n)
    
    for d in range(dim):
        phase = PHI ** d
        signal += np.cos(2 * np.pi * phase * k / n)
    
    return signal / dim


def generate_golden_text(length: int) -> str:
    """Generate text with golden-ratio run-length patterns."""
    # Fibonacci run lengths: switch characters at Fibonacci intervals
    fib = [1, 1]
    while sum(fib) < length:
        fib.append(fib[-1] + fib[-2])
    
    text = []
    chars = ['a', 'b']
    char_idx = 0
    
    for run_len in fib:
        if len(text) >= length:
            break
        text.extend([chars[char_idx]] * min(run_len, length - len(text)))
        char_idx = 1 - char_idx
    
    return ''.join(text[:length])


def text_to_signal(text: str) -> np.ndarray:
    """Convert text to normalized signal."""
    signal = np.array([ord(c) for c in text], dtype=float)
    signal = (signal - signal.mean()) / (signal.std() + 1e-10)
    return signal


# ============================================================================
# METRICS
# ============================================================================

def compute_sparsity(coeffs: np.ndarray, threshold: float = 1e-3) -> float:
    """Fraction of near-zero coefficients."""
    return np.sum(np.abs(coeffs) < threshold) / len(coeffs)


def compute_concentration_ratio(coeffs: np.ndarray, k: int) -> float:
    """Energy in top-k coefficients / total energy."""
    energy = np.abs(coeffs) ** 2
    total_energy = np.sum(energy)
    if total_energy < 1e-15:
        return 1.0
    top_k_energy = np.sum(np.sort(energy)[-k:])
    return top_k_energy / total_energy


def compute_off_diagonal_norm(matrix: np.ndarray) -> float:
    """Frobenius norm of off-diagonal elements."""
    diag_mask = np.eye(matrix.shape[0], dtype=bool)
    off_diag = matrix.copy()
    off_diag[diag_mask] = 0
    return np.linalg.norm(off_diag, 'fro')


def compute_mutual_coherence(matrix: np.ndarray) -> float:
    """Max absolute inner product between distinct normalized columns."""
    # Normalize columns
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    norms[norms < 1e-10] = 1.0
    normalized = matrix / norms
    
    # Gram matrix
    gram = np.abs(normalized.T @ normalized)
    
    # Zero out diagonal
    np.fill_diagonal(gram, 0)
    
    return np.max(gram)


def compute_rip_constant(matrix: np.ndarray, k: int = 10, trials: int = 100) -> float:
    """Approximate RIP constant via random k-sparse vectors."""
    m, n = matrix.shape
    deltas = []
    
    for _ in range(trials):
        # Random k-sparse vector
        x = np.zeros(n)
        indices = np.random.choice(n, size=min(k, n), replace=False)
        x[indices] = np.random.randn(len(indices))
        x = x / np.linalg.norm(x)
        
        # Measure ||Φx||²
        y = matrix @ x
        ratio = np.linalg.norm(y) ** 2
        
        # δ = max |ratio - 1|
        deltas.append(np.abs(ratio - 1))
    
    return np.max(deltas)


# ============================================================================
# RESULT STORAGE
# ============================================================================

@dataclass
class HypothesisResult:
    hypothesis_id: str
    hypothesis_text: str
    test_performed: str
    metric_name: str
    fib_rft_value: float
    dct_value: float
    dft_value: float
    verdict: str  # "CONFIRMED", "REJECTED", "INCONCLUSIVE"
    confidence: float  # 0-1
    details: str = ""


class ExperimentReport:
    def __init__(self):
        self.results: List[HypothesisResult] = []
        self.start_time = time.time()
    
    def add(self, result: HypothesisResult):
        self.results.append(result)
    
    def save_json(self, filepath: str):
        data = []
        for r in self.results:
            d = asdict(r)
            # Convert numpy types to Python native types for JSON serialization
            for key, value in d.items():
                if hasattr(value, 'item'):  # numpy scalar
                    d[key] = value.item()
                elif isinstance(value, (np.floating, np.integer)):
                    d[key] = float(value) if isinstance(value, np.floating) else int(value)
            data.append(d)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_markdown(self, filepath: str):
        elapsed = time.time() - self.start_time
        confirmed = sum(1 for r in self.results if r.verdict == "CONFIRMED")
        
        with open(filepath, 'w') as f:
            f.write("# Fibonacci-Tilt Φ-RFT: Hypothesis Validation Results\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Time:** {elapsed:.2f}s\n")
            f.write(f"**Hypotheses Confirmed:** {confirmed}/{len(self.results)}\n\n")
            f.write("---\n\n")
            
            for r in self.results:
                status_emoji = {
                    "CONFIRMED": "✅",
                    "REJECTED": "❌",
                    "INCONCLUSIVE": "⚠️"
                }[r.verdict]
                
                f.write(f"## {r.hypothesis_id}: {status_emoji} {r.verdict}\n\n")
                f.write(f"**Hypothesis:** {r.hypothesis_text}\n\n")
                f.write(f"**Test:** {r.test_performed}\n\n")
                f.write(f"**Metric:** {r.metric_name}\n\n")
                f.write("| Transform | Value |\n")
                f.write("|-----------|-------|\n")
                f.write(f"| Fib-Φ-RFT | **{r.fib_rft_value:.6f}** |\n")
                f.write(f"| DCT       | {r.dct_value:.6f} |\n")
                f.write(f"| DFT       | {r.dft_value:.6f} |\n\n")
                f.write(f"**Confidence:** {r.confidence*100:.1f}%\n\n")
                if r.details:
                    f.write(f"**Details:** {r.details}\n\n")
                f.write("---\n\n")


# ============================================================================
# HYPOTHESIS TESTS
# ============================================================================

def test_h1_asymptotic_sparsity(report: ExperimentReport):
    """
    H1: Fibonacci sequences are asymptotically 1-sparse in Fib-Φ-RFT.
    Test at multiple scales N, measure concentration in top-K vs K.
    """
    print("\n[H1] Testing asymptotic sparsity of Fibonacci sequences...")
    
    scales = [128, 256, 512, 1024]
    k_values = [1, 2, 5, 10]
    
    concentrations = {
        'fib_rft': [],
        'dct': [],
        'dft': []
    }
    
    for n in scales:
        signal = generate_fibonacci_sequence(n)
        
        # Transforms
        fib_coeffs = fibonacci_tilt_rft(signal)
        dct_coeffs = dct(signal, norm='ortho')
        dft_coeffs = fft(signal)
        
        # Concentration in top-5
        concentrations['fib_rft'].append(compute_concentration_ratio(fib_coeffs, 5))
        concentrations['dct'].append(compute_concentration_ratio(dct_coeffs, 5))
        concentrations['dft'].append(compute_concentration_ratio(dft_coeffs, 5))
    
    # Average concentration
    fib_conc = np.mean(concentrations['fib_rft'])
    dct_conc = np.mean(concentrations['dct'])
    dft_conc = np.mean(concentrations['dft'])
    
    # Verdict: Fib-RFT should have significantly higher concentration
    confirmed = fib_conc > dct_conc * 1.2 and fib_conc > dft_conc * 1.2
    confidence = min(1.0, (fib_conc - max(dct_conc, dft_conc)) / 0.3)
    
    report.add(HypothesisResult(
        hypothesis_id="H1",
        hypothesis_text="Fibonacci sequences are asymptotically 1-sparse in Fib-Φ-RFT",
        test_performed=f"Measured top-5 energy concentration across scales {scales}",
        metric_name="Average energy concentration in top-5 coefficients",
        fib_rft_value=fib_conc,
        dct_value=dct_conc,
        dft_value=dft_conc,
        verdict="CONFIRMED" if confirmed else "REJECTED",
        confidence=max(0.0, confidence),
        details=f"Concentrations at N=1024: Fib={concentrations['fib_rft'][-1]:.3f}, DCT={concentrations['dct'][-1]:.3f}, DFT={concentrations['dft'][-1]:.3f}"
    ))


def test_h2_golden_rotation_diagonalizer(report: ExperimentReport):
    """
    H2: Fib-Φ-RFT approximately diagonalizes golden rotation operators.
    Build rotation operator, transform to each basis, measure off-diagonal energy.
    """
    print("\n[H2] Testing golden rotation diagonalization...")
    
    n = 256
    
    # Build golden rotation operator: R[i,j] = δ(j - floor(i·φ) mod n)
    R = np.zeros((n, n))
    for i in range(n):
        j = int(np.floor(i * PHI)) % n
        R[i, j] = 1.0
    
    # Add small noise
    R += 0.01 * np.random.randn(n, n)
    
    # Transform operator to each basis
    # Φ-RFT basis
    fib_basis = np.zeros((n, n), dtype=complex)
    for k in range(n):
        e_k = np.zeros(n)
        e_k[k] = 1.0
        fib_basis[:, k] = fibonacci_tilt_rft(e_k)
    
    R_fib = fib_basis.conj().T @ R @ fib_basis
    
    # DCT basis
    dct_basis = np.zeros((n, n))
    for k in range(n):
        e_k = np.zeros(n)
        e_k[k] = 1.0
        dct_basis[:, k] = dct(e_k, norm='ortho')
    
    R_dct = dct_basis.T @ R @ dct_basis
    
    # DFT basis
    dft_basis = np.fft.fft(np.eye(n), axis=0) / np.sqrt(n)
    R_dft = dft_basis.conj().T @ R @ dft_basis
    
    # Off-diagonal norms
    fib_off = compute_off_diagonal_norm(R_fib)
    dct_off = compute_off_diagonal_norm(R_dct)
    dft_off = compute_off_diagonal_norm(R_dft)
    
    confirmed = fib_off < dct_off * 0.8 and fib_off < dft_off * 0.8
    confidence = 1.0 - (fib_off / min(dct_off, dft_off))
    
    report.add(HypothesisResult(
        hypothesis_id="H2",
        hypothesis_text="Fib-Φ-RFT is the natural diagonalizer of golden rotations",
        test_performed=f"Built golden rotation operator (n={n}), measured off-diagonal Frobenius norm",
        metric_name="Off-diagonal Frobenius norm (lower is more diagonal)",
        fib_rft_value=fib_off,
        dct_value=dct_off,
        dft_value=dft_off,
        verdict="CONFIRMED" if confirmed else "REJECTED",
        confidence=max(0.0, confidence),
        details=f"Operator: R[i,j] = δ(j - ⌊i·φ⌋ mod n) + 1% noise"
    ))


def test_h3_golden_text_detection(report: ExperimentReport):
    """
    H3: Golden-structured text segments show higher sparsity in Fib-Φ-RFT.
    Generate golden vs random text, compare sparsity.
    """
    print("\n[H3] Testing golden text structure detection...")
    
    n = 512
    
    # Golden text
    golden_text = generate_golden_text(n)
    golden_signal = text_to_signal(golden_text)
    
    # Random text
    random_text = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=n))
    random_signal = text_to_signal(random_text)
    
    # Sparsity on golden text
    fib_golden = fibonacci_tilt_rft(golden_signal)
    dct_golden = dct(golden_signal, norm='ortho')
    dft_golden = fft(golden_signal)
    
    sparsity_fib_golden = compute_sparsity(fib_golden, threshold=0.1)
    sparsity_dct_golden = compute_sparsity(dct_golden, threshold=0.1)
    sparsity_dft_golden = compute_sparsity(dft_golden, threshold=0.1)
    
    # Sparsity on random text
    fib_random = fibonacci_tilt_rft(random_signal)
    dct_random = dct(random_signal, norm='ortho')
    
    sparsity_fib_random = compute_sparsity(fib_random, threshold=0.1)
    sparsity_dct_random = compute_sparsity(dct_random, threshold=0.1)
    
    # Relative sparsity gain for Fib-RFT on golden vs random
    fib_gain = sparsity_fib_golden / (sparsity_fib_random + 1e-10)
    dct_gain = sparsity_dct_golden / (sparsity_dct_random + 1e-10)
    
    confirmed = fib_gain > dct_gain * 1.15
    confidence = min(1.0, (fib_gain - dct_gain) / 0.5)
    
    report.add(HypothesisResult(
        hypothesis_id="H3",
        hypothesis_text="Golden-structured text segments are detectable by Fib sparsity",
        test_performed=f"Generated golden-run-length text vs random text (n={n}), compared sparsity ratios",
        metric_name="Sparsity gain ratio: golden/random (higher means better detection)",
        fib_rft_value=fib_gain,
        dct_value=dct_gain,
        dft_value=1.0,  # Not tested
        verdict="CONFIRMED" if confirmed else "REJECTED",
        confidence=max(0.0, confidence),
        details=f"Fib-RFT: golden={sparsity_fib_golden:.3f}, random={sparsity_fib_random:.3f}. DCT: golden={sparsity_dct_golden:.3f}, random={sparsity_dct_random:.3f}"
    ))


def test_h4_hybrid_dominance(report: ExperimentReport):
    """
    H4: Hybrid DCT + Fib-Φ-RFT dominates on mixed quasi-periodic data.
    Build mixture, test cascade vs single-basis.
    """
    print("\n[H4] Testing hybrid cascade dominance on mixed data...")
    
    n = 512
    
    # Mixed signal: 60% smooth, 40% Fibonacci
    smooth = np.sin(2 * np.pi * np.arange(n) / 32)
    fib_component = generate_fibonacci_sequence(n)
    signal = 0.6 * smooth + 0.4 * fib_component
    
    # Pure methods
    dct_coeffs = dct(signal, norm='ortho')
    fib_coeffs = fibonacci_tilt_rft(signal)
    
    # Cascade: DCT on smooth part, Fib-RFT on texture
    # Split by low/high frequency
    freq_coeffs = fft(signal)
    cutoff = n // 4
    low_freq = np.zeros(n, dtype=complex)
    high_freq = np.zeros(n, dtype=complex)
    low_freq[:cutoff] = freq_coeffs[:cutoff]
    low_freq[-cutoff:] = freq_coeffs[-cutoff:]
    high_freq[cutoff:-cutoff] = freq_coeffs[cutoff:-cutoff]
    
    smooth_part = np.real(ifft(low_freq))
    texture_part = np.real(ifft(high_freq))
    
    dct_smooth = dct(smooth_part, norm='ortho')
    fib_texture = fibonacci_tilt_rft(texture_part)
    
    # Sparsify (keep top 10%)
    k = int(0.1 * n)
    
    # Pure methods
    dct_sparse = np.zeros_like(dct_coeffs)
    top_dct = np.argsort(np.abs(dct_coeffs))[-k:]
    dct_sparse[top_dct] = dct_coeffs[top_dct]
    
    fib_sparse = np.zeros_like(fib_coeffs)
    top_fib = np.argsort(np.abs(fib_coeffs))[-k:]
    fib_sparse[top_fib] = fib_coeffs[top_fib]
    
    # Cascade (use k/2 from each)
    dct_smooth_sparse = np.zeros_like(dct_smooth)
    top_dct_smooth = np.argsort(np.abs(dct_smooth))[-k//2:]
    dct_smooth_sparse[top_dct_smooth] = dct_smooth[top_dct_smooth]
    
    fib_texture_sparse = np.zeros_like(fib_texture)
    top_fib_texture = np.argsort(np.abs(fib_texture))[-k//2:]
    fib_texture_sparse[top_fib_texture] = fib_texture[top_fib_texture]
    
    # Reconstruct and measure error
    recon_dct = idct(dct_sparse, norm='ortho')
    recon_fib = np.real(fibonacci_tilt_inverse(fib_sparse))
    recon_cascade = idct(dct_smooth_sparse, norm='ortho') + np.real(fibonacci_tilt_inverse(fib_texture_sparse))
    
    mse_dct = np.mean((signal - recon_dct) ** 2)
    mse_fib = np.mean((signal - recon_fib) ** 2)
    mse_cascade = np.mean((signal - recon_cascade) ** 2)
    
    confirmed = mse_cascade < mse_dct and mse_cascade < mse_fib
    confidence = 1.0 - (mse_cascade / min(mse_dct, mse_fib))
    
    report.add(HypothesisResult(
        hypothesis_id="H4",
        hypothesis_text="Hybrid DCT + Fib-Φ-RFT dominates on mixed quasi-periodic data",
        test_performed=f"Mixed signal (60% smooth + 40% Fibonacci), 10% sparsity, measured reconstruction MSE",
        metric_name="Reconstruction MSE (lower is better)",
        fib_rft_value=mse_fib,
        dct_value=mse_dct,
        dft_value=mse_cascade,  # Abuse field for cascade result
        verdict="CONFIRMED" if confirmed else "REJECTED",
        confidence=max(0.0, confidence),
        details=f"Pure DCT: {mse_dct:.6f}, Pure Fib: {mse_fib:.6f}, Cascade: {mse_cascade:.6f}"
    ))


def test_h5_sensing_matrices(report: ExperimentReport):
    """
    H5: Fib-Φ-RFT yields better sensing matrices (lower mutual coherence).
    Build random sensing matrices, measure coherence.
    """
    print("\n[H5] Testing sensing matrix quality...")
    
    n = 128
    m = 64  # Measurement count
    
    # Build sensing matrices (random rows from each transform basis)
    # Fib-RFT
    fib_basis = np.zeros((n, n), dtype=complex)
    for k in range(n):
        e_k = np.zeros(n)
        e_k[k] = 1.0
        fib_basis[:, k] = fibonacci_tilt_rft(e_k)
    
    rows = np.random.choice(n, size=m, replace=False)
    Phi_fib = fib_basis[rows, :].T  # m x n
    
    # DCT
    dct_basis = np.zeros((n, n))
    for k in range(n):
        e_k = np.zeros(n)
        e_k[k] = 1.0
        dct_basis[:, k] = dct(e_k, norm='ortho')
    
    Phi_dct = dct_basis[rows, :].T
    
    # DFT
    dft_basis = np.fft.fft(np.eye(n), axis=0) / np.sqrt(n)
    Phi_dft = dft_basis[rows, :].T
    
    # Mutual coherence
    mu_fib = compute_mutual_coherence(np.real(Phi_fib))  # Take real part for coherence
    mu_dct = compute_mutual_coherence(Phi_dct)
    mu_dft = compute_mutual_coherence(np.real(Phi_dft))
    
    confirmed = mu_fib < mu_dct * 0.9 and mu_fib < mu_dft * 0.9
    confidence = 1.0 - (mu_fib / min(mu_dct, mu_dft))
    
    report.add(HypothesisResult(
        hypothesis_id="H5",
        hypothesis_text="Fib-Φ-RFT yields deterministic 'good' sensing matrices",
        test_performed=f"Built {m}x{n} sensing matrices from random rows, measured mutual coherence",
        metric_name="Mutual coherence (lower is better for compressed sensing)",
        fib_rft_value=mu_fib,
        dct_value=mu_dct,
        dft_value=mu_dft,
        verdict="CONFIRMED" if confirmed else "REJECTED",
        confidence=max(0.0, confidence),
        details=f"Lower coherence enables better sparse recovery via ℓ₁ minimization"
    ))


def test_h6_quasicrystal_peaks(report: ExperimentReport):
    """
    H6: Fib-Φ-RFT exposes sharper diffraction peaks in quasicrystals.
    Generate 1D quasicrystal, measure peak sharpness.
    """
    print("\n[H6] Testing quasicrystal diffraction peak detection...")
    
    n = 512
    signal = generate_quasicrystal_1d(n, dim=5)
    
    # Add noise
    snr_db = 20
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    signal += np.sqrt(noise_power) * np.random.randn(n)
    
    # Transforms
    fib_coeffs = np.abs(fibonacci_tilt_rft(signal))
    dct_coeffs = np.abs(dct(signal, norm='ortho'))
    dft_coeffs = np.abs(fft(signal))
    
    # Peak detection: count coefficients above 2σ
    fib_threshold = np.mean(fib_coeffs) + 2 * np.std(fib_coeffs)
    dct_threshold = np.mean(dct_coeffs) + 2 * np.std(dct_coeffs)
    dft_threshold = np.mean(dft_coeffs) + 2 * np.std(dft_coeffs)
    
    fib_peaks = np.sum(fib_coeffs > fib_threshold)
    dct_peaks = np.sum(dct_coeffs > dct_threshold)
    dft_peaks = np.sum(dft_coeffs > dft_threshold)
    
    # Sharpness: kurtosis of coefficient distribution
    fib_kurtosis = np.mean((fib_coeffs - np.mean(fib_coeffs)) ** 4) / (np.std(fib_coeffs) ** 4)
    dct_kurtosis = np.mean((dct_coeffs - np.mean(dct_coeffs)) ** 4) / (np.std(dct_coeffs) ** 4)
    dft_kurtosis = np.mean((dft_coeffs - np.mean(dft_coeffs)) ** 4) / (np.std(dft_coeffs) ** 4)
    
    # Higher kurtosis = sharper peaks
    confirmed = fib_kurtosis > dct_kurtosis * 1.1 and fib_kurtosis > dft_kurtosis * 1.1
    confidence = min(1.0, (fib_kurtosis - max(dct_kurtosis, dft_kurtosis)) / max(dct_kurtosis, dft_kurtosis))
    
    report.add(HypothesisResult(
        hypothesis_id="H6",
        hypothesis_text="Fib-Φ-RFT exposes quasicrystal-like diffraction peaks more sharply",
        test_performed=f"Generated 1D quasicrystal (dim=5) with SNR={snr_db}dB, measured coefficient kurtosis",
        metric_name="Coefficient distribution kurtosis (higher = sharper peaks)",
        fib_rft_value=fib_kurtosis,
        dct_value=dct_kurtosis,
        dft_value=dft_kurtosis,
        verdict="CONFIRMED" if confirmed else "REJECTED",
        confidence=max(0.0, confidence),
        details=f"Peaks detected (>2σ): Fib={fib_peaks}, DCT={dct_peaks}, DFT={dft_peaks}"
    ))


def test_h7_anomaly_detection(report: ExperimentReport):
    """
    H7: Fib-tilt improves anomaly detection in almost-golden streams.
    Inject anomalies, measure detection sensitivity.
    """
    print("\n[H7] Testing anomaly detection in golden streams...")
    
    n = 512
    num_anomalies = 5
    
    # Golden rotation stream
    signal = generate_golden_rotation(n, noise=0.02)
    
    # Inject anomalies (random spikes)
    anomaly_indices = np.random.choice(n, size=num_anomalies, replace=False)
    true_labels = np.zeros(n)
    true_labels[anomaly_indices] = 1
    
    for idx in anomaly_indices:
        signal[idx] += 3.0 * np.random.randn()
    
    # Transform and compute local energy bursts
    fib_coeffs = np.abs(fibonacci_tilt_rft(signal))
    dct_coeffs = np.abs(dct(signal, norm='ortho'))
    dft_coeffs = np.abs(fft(signal))
    
    # Detection: threshold at mean + 3σ
    fib_threshold = np.mean(fib_coeffs) + 3 * np.std(fib_coeffs)
    dct_threshold = np.mean(dct_coeffs) + 3 * np.std(dct_coeffs)
    dft_threshold = np.mean(dft_coeffs) + 3 * np.std(dft_coeffs)
    
    fib_detections = np.sum(fib_coeffs > fib_threshold)
    dct_detections = np.sum(dct_coeffs > dct_threshold)
    dft_detections = np.sum(dft_coeffs > dft_threshold)
    
    # Score: closer to num_anomalies is better
    fib_error = np.abs(fib_detections - num_anomalies)
    dct_error = np.abs(dct_detections - num_anomalies)
    dft_error = np.abs(dft_detections - num_anomalies)
    
    confirmed = fib_error <= dct_error and fib_error <= dft_error
    confidence = 1.0 - (fib_error / (num_anomalies + 1))
    
    report.add(HypothesisResult(
        hypothesis_id="H7",
        hypothesis_text="Fib-tilt improves anomaly detection in 'almost golden' streams",
        test_performed=f"Injected {num_anomalies} anomalies into golden rotation stream, counted detections at 3σ threshold",
        metric_name="Detection error: |detected - actual| (lower is better)",
        fib_rft_value=fib_error,
        dct_value=dct_error,
        dft_value=dft_error,
        verdict="CONFIRMED" if confirmed else "REJECTED",
        confidence=max(0.0, confidence),
        details=f"Detections: Fib={fib_detections}, DCT={dct_detections}, DFT={dft_detections}, Actual={num_anomalies}"
    ))


def test_h8_hash_avalanche(report: ExperimentReport):
    """
    H8: Golden-aligned hash mixing layer improves avalanche on structured inputs.
    Compare avalanche with Fibonacci phase vs standard phase.
    """
    print("\n[H8] Testing hash mixing avalanche quality...")
    
    n = 128
    num_trials = 100
    
    # Standard RFT hash: random phase
    def hash_standard(x):
        phase = np.random.rand(len(x))
        modulated = x * np.exp(2j * np.pi * phase)
        return np.abs(fft(modulated))
    
    # Fibonacci phase hash
    def hash_fibonacci(x):
        fib = [1, 1]
        while len(fib) < len(x):
            fib.append(fib[-1] + fib[-2])
        fib = np.array(fib[:len(x)], dtype=np.float64)
        phase = fib.astype(np.float64) / len(x)
        modulated = x * np.exp(2j * np.pi * phase)
        return np.abs(fft(modulated))
    
    # Test on structured inputs (repeated patterns)
    avalanche_fib = []
    avalanche_std = []
    
    for _ in range(num_trials):
        # Structured input: repeated pattern
        pattern = np.random.randn(n // 8)
        x = np.tile(pattern, 8)
        
        # Flip one bit
        x_flipped = x.copy()
        flip_idx = np.random.randint(n)
        x_flipped[flip_idx] += 0.1  # Small perturbation
        
        # Hash both
        h1_fib = hash_fibonacci(x)
        h2_fib = hash_fibonacci(x_flipped)
        
        h1_std = hash_standard(x)
        h2_std = hash_standard(x_flipped)
        
        # Avalanche: fraction of output bits that flipped
        avalanche_fib.append(np.mean(np.abs(h1_fib - h2_fib) > 0.1))
        avalanche_std.append(np.mean(np.abs(h1_std - h2_std) > 0.1))
    
    fib_avalanche = np.mean(avalanche_fib)
    std_avalanche = np.mean(avalanche_std)
    
    # Ideal avalanche is 0.5 (50% of bits flip)
    fib_distance_from_ideal = np.abs(fib_avalanche - 0.5)
    std_distance_from_ideal = np.abs(std_avalanche - 0.5)
    
    confirmed = fib_distance_from_ideal < std_distance_from_ideal
    confidence = 1.0 - (fib_distance_from_ideal / 0.5)
    
    report.add(HypothesisResult(
        hypothesis_id="H8",
        hypothesis_text="Golden-aligned hash mixing layer improves avalanche on structured inputs",
        test_performed=f"Hashed {num_trials} structured inputs with 1-bit perturbations, measured avalanche effect",
        metric_name="Distance from ideal avalanche (0.5), lower is better",
        fib_rft_value=fib_distance_from_ideal,
        dct_value=std_distance_from_ideal,
        dft_value=0.0,  # Not tested
        verdict="CONFIRMED" if confirmed else "REJECTED",
        confidence=max(0.0, confidence),
        details=f"Fibonacci avalanche: {fib_avalanche:.3f}, Standard avalanche: {std_avalanche:.3f}, Ideal: 0.500"
    ))


def test_h9_quantization_sensitivity(report: ExperimentReport):
    """
    H9: Fib-Φ-RFT reduces quantization sensitivity for golden signals.
    Quantize coefficients, measure reconstruction MSE.
    """
    print("\n[H9] Testing quantization sensitivity...")
    
    n = 256
    signal = generate_fibonacci_sequence(n).astype(float)
    
    # Transforms
    fib_coeffs = fibonacci_tilt_rft(signal)
    dct_coeffs = dct(signal, norm='ortho')
    dft_coeffs = fft(signal)
    
    # Quantize to 8 bits
    def quantize(coeffs, bits=8):
        max_val = np.max(np.abs(coeffs))
        levels = 2 ** bits
        quantized = np.round(coeffs / max_val * (levels / 2)) * (max_val / (levels / 2))
        return quantized
    
    fib_quant = quantize(fib_coeffs)
    dct_quant = quantize(dct_coeffs)
    dft_quant = quantize(dft_coeffs)
    
    # Reconstruct
    recon_fib = np.real(fibonacci_tilt_inverse(fib_quant))
    recon_dct = idct(dct_quant, norm='ortho')
    recon_dft = np.real(ifft(dft_quant))
    
    # MSE
    mse_fib = np.mean((signal - recon_fib) ** 2)
    mse_dct = np.mean((signal - recon_dct) ** 2)
    mse_dft = np.mean((signal - recon_dft) ** 2)
    
    confirmed = mse_fib < mse_dct and mse_fib < mse_dft
    confidence = 1.0 - (mse_fib / min(mse_dct, mse_dft))
    
    report.add(HypothesisResult(
        hypothesis_id="H9",
        hypothesis_text="Fib-Φ-RFT reduces quantization sensitivity for golden signals",
        test_performed=f"Fibonacci sequence (n={n}), 8-bit quantization, measured reconstruction MSE",
        metric_name="Reconstruction MSE after quantization (lower is better)",
        fib_rft_value=mse_fib,
        dct_value=mse_dct,
        dft_value=mse_dft,
        verdict="CONFIRMED" if confirmed else "REJECTED",
        confidence=max(0.0, confidence),
        details=f"Quantization: 8 bits uniform scalar quantization"
    ))


def test_h10_markov_low_rank(report: ExperimentReport):
    """
    H10: Fib-tilt yields low-rank model of golden Markov chains.
    Build Markov transition matrix from Fibonacci process, measure rank in each basis.
    """
    print("\n[H10] Testing Markov chain low-rank representation...")
    
    n = 64  # State space size
    
    # Generate golden Markov chain: transition probabilities follow Fibonacci weights
    fib = [1, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    fib = np.array(fib[:n], dtype=float)
    fib = fib / np.sum(fib)  # Normalize to probability
    
    # Transition matrix: P[i,j] ∝ fib[|i-j|]
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = abs(i - j)
            if dist < len(fib):
                P[i, j] = fib[dist]
    
    # Normalize rows
    P = P / np.sum(P, axis=1, keepdims=True)
    
    # Transform to each basis (use 1D transforms on rows)
    # Fib-RFT
    P_fib = np.zeros((n, n), dtype=complex)
    for i in range(n):
        P_fib[i, :] = fibonacci_tilt_rft(P[i, :])
    
    # DCT
    P_dct = np.zeros((n, n))
    for i in range(n):
        P_dct[i, :] = dct(P[i, :], norm='ortho')
    
    # SVD and effective rank (singular values > 0.1 * max)
    def effective_rank(M, threshold=0.1):
        s = svd(M, compute_uv=False)
        return np.sum(s > threshold * s[0])
    
    rank_fib = effective_rank(np.real(P_fib))
    rank_dct = effective_rank(P_dct)
    rank_orig = effective_rank(P)
    
    confirmed = rank_fib < rank_dct and rank_fib < rank_orig
    confidence = 1.0 - (rank_fib / min(rank_dct, rank_orig))
    
    report.add(HypothesisResult(
        hypothesis_id="H10",
        hypothesis_text="Fib-tilt yields a low-rank model of golden Markov chains",
        test_performed=f"Built Fibonacci-weighted Markov transition matrix (n={n}), measured effective rank via SVD",
        metric_name="Effective rank (singular values > 0.1·σ_max), lower is better",
        fib_rft_value=rank_fib,
        dct_value=rank_dct,
        dft_value=rank_orig,
        verdict="CONFIRMED" if confirmed else "REJECTED",
        confidence=max(0.0, confidence),
        details=f"Ranks: Fib-RFT={rank_fib}, DCT={rank_dct}, Original={rank_orig}"
    ))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("FIBONACCI-TILT Φ-RFT: 10 HYPOTHESIS VALIDATION")
    print("=" * 80)
    print("\nAuthor: Luis M. Minier")
    print("Date: November 27, 2025")
    print("\nTesting 10 concrete, falsifiable hypotheses about Fibonacci-tilt RFT...")
    print("=" * 80)
    
    report = ExperimentReport()
    
    # Run all hypothesis tests
    test_h1_asymptotic_sparsity(report)
    test_h2_golden_rotation_diagonalizer(report)
    test_h3_golden_text_detection(report)
    test_h4_hybrid_dominance(report)
    test_h5_sensing_matrices(report)
    test_h6_quasicrystal_peaks(report)
    test_h7_anomaly_detection(report)
    test_h8_hash_avalanche(report)
    test_h9_quantization_sensitivity(report)
    test_h10_markov_low_rank(report)
    
    # Save results
    output_dir = Path(__file__).parent
    report.save_json(str(output_dir / "fibonacci_tilt_results.json"))
    report.save_markdown(str(output_dir / "FIBONACCI_TILT_RESULTS.md"))
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    confirmed = sum(1 for r in report.results if r.verdict == "CONFIRMED")
    rejected = sum(1 for r in report.results if r.verdict == "REJECTED")
    
    print(f"\nHypotheses Confirmed: {confirmed}/10")
    print(f"Hypotheses Rejected:  {rejected}/10")
    print(f"\nTotal Time: {time.time() - report.start_time:.2f}s")
    print(f"\nResults saved:")
    print(f"  - JSON: experiments/fibonacci_tilt_results.json")
    print(f"  - Markdown: experiments/FIBONACCI_TILT_RESULTS.md")
    print("\n" + "=" * 80)
    
    return 0 if confirmed >= 7 else 1  # Success if ≥70% confirmed


if __name__ == "__main__":
    sys.exit(main())

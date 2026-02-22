#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier
#
# verify_braided_comprehensive.py
# --------------------------------
# Comprehensive evaluation of Braided MCA (Parallel Competition) strategy
# across multiple signal types and evaluation metrics.
#
# Tests:
# 1. ASCII Bottleneck (Compression)
# 2. Source Separation (MCA Recovery)
# 3. Rate-Distortion Tradeoff
# 4. Mixed Signal Performance

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.rft.hybrids.legacy_mca import (
    adaptive_hybrid_compress,
    braided_hybrid_mca,
    rft_forward,
    rft_inverse,
)
from scipy.fftpack import dct, idct

# ============================================================================
# UTILITIES
# ============================================================================

def gini_coefficient(x):
    """Calculate the Gini coefficient (sparsity measure)."""
    x = np.abs(x)
    if np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return ((2 * np.sum(index * x)) / (n * np.sum(x))) - ((n + 1) / n)


def energy_concentration(x, percentile=0.99):
    """Fraction of coefficients needed to capture 'percentile' of energy."""
    energy = np.abs(x) ** 2
    total_energy = np.sum(energy)
    if total_energy == 0:
        return 1.0

    sorted_energy = np.sort(energy)[::-1]
    cumulative = np.cumsum(sorted_energy)

    threshold = total_energy * percentile
    idx = np.searchsorted(cumulative, threshold)

    return (idx + 1) / len(x)


def l2_error(x_true, x_hat):
    """Relative L2 reconstruction error."""
    return float(np.linalg.norm(x_true - x_hat) / (np.linalg.norm(x_true) + 1e-12))


def text_to_signal(text, N):
    """Convert text to normalized signal."""
    b = text.encode("utf-8") if isinstance(text, str) else text

    if len(b) < N:
        reps = (N // len(b)) + 1
        b = (b * reps)[:N]
    else:
        b = b[:N]

    arr = np.frombuffer(b, dtype=np.uint8).astype(np.float64)
    return (arr / 128.0) - 1.0


# ============================================================================
# TEST 1: ASCII BOTTLENECK (COMPRESSION)
# ============================================================================

def test_ascii_bottleneck(N=256):
    """Compare Greedy vs Braided on text compression."""
    print("\n" + "=" * 70)
    print(" TEST 1: ASCII BOTTLENECK (Compression Efficiency)")
    print("=" * 70)
    print(f"N={N} | Lower '% Coeffs' is BETTER\n")

    datasets = {
        "Natural Text": "The quick brown fox jumps over the lazy dog. " * 10,
        "Python Code": "def rft(x): return np.fft.fft(x) * np.exp(1j*phi) " * 5,
        "Random ASCII": "".join([chr(x) for x in np.random.randint(32, 127, N)]),
    }

    # Mixed signal (Wave + Text)
    t = np.arange(N)
    phi = (1 + np.sqrt(5)) / 2
    wave = 0.6 * np.sin(2 * np.pi * (1 / phi**2) * t)
    text_part = text_to_signal("Structure plus texture.", N)
    datasets["Mixed (Wave+Text)"] = wave + text_part

    results = []

    for name, data in datasets.items():
        if isinstance(data, str):
            signal = text_to_signal(data, N)
        else:
            signal = data

        # DCT Baseline
        dct_coeffs = dct(signal, norm="ortho")
        dct_sparsity = energy_concentration(dct_coeffs, 0.99) * 100

        # Greedy Hybrid
        x_s_greedy, x_t_greedy, _, _ = adaptive_hybrid_compress(signal, verbose=False)
        greedy_combined = np.concatenate([x_s_greedy, x_t_greedy])
        greedy_sparsity = energy_concentration(greedy_combined, 0.99) * 100
        greedy_recon_error = l2_error(signal, x_s_greedy + x_t_greedy)

        # Braided MCA
        braid_result = braided_hybrid_mca(signal, verbose=False, threshold=0.05)
        braid_combined = np.concatenate([braid_result.structural, braid_result.texture])
        braid_sparsity = energy_concentration(braid_combined, 0.99) * 100
        braid_recon_error = l2_error(
            signal, braid_result.structural + braid_result.texture
        )

        results.append(
            {
                "dataset": name,
                "dct": dct_sparsity,
                "greedy_sparse": greedy_sparsity,
                "greedy_err": greedy_recon_error,
                "braid_sparse": braid_sparsity,
                "braid_err": braid_recon_error,
            }
        )

    # Print table
    print(
        f"{'Dataset':<20} | {'DCT %':<8} | {'Greedy %':<8} | {'Greedy Err':<10} | {'Braid %':<8} | {'Braid Err':<10}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['dataset']:<20} | {r['dct']:>7.2f} | {r['greedy_sparse']:>8.2f} | {r['greedy_err']:>10.3e} | {r['braid_sparse']:>8.2f} | {r['braid_err']:>10.3e}"
        )

    print("\n✅ INTERPRETATION:")
    print("  - If Braid Err >> Greedy Err: Braided breaks reconstruction (phase issue)")
    print("  - If Braid % < Greedy %: Braided finds sparser representation")
    print("  - If Braid % >> Greedy %: Energy smearing (reconstruction failure)")


# ============================================================================
# TEST 2: SOURCE SEPARATION (MCA GROUND TRUTH)
# ============================================================================

def generate_sparse_dct(N, K, rng):
    """Generate K-sparse signal in DCT basis."""
    coeffs = np.zeros(N, dtype=np.complex128)
    support_idx = rng.choice(N, size=K, replace=False)
    coeffs[support_idx] = (rng.normal(size=K) + 1j * rng.normal(size=K)) / np.sqrt(2.0)

    x_real = idct(coeffs.real, norm="ortho")
    x_imag = idct(coeffs.imag, norm="ortho")
    return x_real + 1j * x_imag, support_idx


def generate_sparse_rft(N, K, rng):
    """Generate K-sparse signal in RFT basis."""
    coeffs = np.zeros(N, dtype=np.complex128)
    support_idx = rng.choice(N, size=K, replace=False)
    coeffs[support_idx] = (rng.normal(size=K) + 1j * rng.normal(size=K)) / np.sqrt(2.0)

    x_t = rft_inverse(coeffs)
    return x_t, support_idx


def test_source_separation(N=256, trials=20):
    """Compare Greedy vs Braided on MCA separation task."""
    print("\n" + "=" * 70)
    print(" TEST 2: SOURCE SEPARATION (MCA Ground Truth)")
    print("=" * 70)
    print(f"N={N}, Trials={trials} | Lower Err is BETTER\n")

    rng = np.random.default_rng(42)

    Ks_vals = [4, 8]
    Kt_vals = [4, 8]
    snr_db = 30.0

    results = []

    for Ks in Ks_vals:
        for Kt in Kt_vals:
            greedy_errs = []
            braid_errs = []

            for _ in range(trials):
                # Generate ground truth
                x_s_true, _ = generate_sparse_dct(N, Ks, rng)
                x_t_true, _ = generate_sparse_rft(N, Kt, rng)
                x_clean = x_s_true + x_t_true

                # Add noise
                signal_power = np.mean(np.abs(x_clean) ** 2)
                snr_linear = 10.0 ** (snr_db / 10.0)
                noise_power = signal_power / snr_linear
                noise = rng.normal(
                    scale=np.sqrt(noise_power / 2.0), size=x_clean.shape
                ) + 1j * rng.normal(scale=np.sqrt(noise_power / 2.0), size=x_clean.shape)
                x_noisy = x_clean + noise

                # Greedy
                x_s_g, x_t_g, _, _ = adaptive_hybrid_compress(x_noisy, verbose=False)
                err_greedy = l2_error(x_clean, x_s_g + x_t_g)
                greedy_errs.append(err_greedy)

                # Braided
                braid_res = braided_hybrid_mca(x_noisy, verbose=False, threshold=0.05)
                err_braid = l2_error(
                    x_clean, braid_res.structural + braid_res.texture
                )
                braid_errs.append(err_braid)

            results.append(
                {
                    "Ks": Ks,
                    "Kt": Kt,
                    "greedy": np.mean(greedy_errs),
                    "braid": np.mean(braid_errs),
                }
            )

    # Print table
    print(f"{'Ks':<4} | {'Kt':<4} | {'Greedy Err':<12} | {'Braid Err':<12} | {'Winner'}")
    print("-" * 60)
    for r in results:
        winner = "Greedy" if r["greedy"] < r["braid"] else "Braid"
        print(
            f"{r['Ks']:<4} | {r['Kt']:<4} | {r['greedy']:<12.3e} | {r['braid']:<12.3e} | {winner}"
        )

    print("\n✅ INTERPRETATION:")
    print("  - Greedy wins: Better at reconstruction (expected)")
    print("  - Braid wins: Better at preserving signal structure (unexpected!)")


# ============================================================================
# TEST 3: RATE-DISTORTION TRADEOFF
# ============================================================================

def test_rate_distortion(N=256):
    """Measure compression ratio vs reconstruction error."""
    print("\n" + "=" * 70)
    print(" TEST 3: RATE-DISTORTION TRADEOFF")
    print("=" * 70)
    print(f"N={N} | Lower Rate at same Distortion is BETTER\n")

    # Mixed signal
    t = np.arange(N)
    phi = (1 + np.sqrt(5)) / 2
    wave = 0.6 * np.sin(2 * np.pi * (1 / phi**2) * t)
    text_part = text_to_signal("Mixed content test.", N)
    signal = wave + text_part

    thresholds = [0.01, 0.05, 0.1, 0.2]

    print(f"{'Threshold':<12} | {'Greedy Rate':<12} | {'Greedy MSE':<12} | {'Braid Rate':<12} | {'Braid MSE':<12}")
    print("-" * 80)

    for thresh in thresholds:
        # Greedy
        x_s_g, x_t_g, _, meta_g = adaptive_hybrid_compress(signal, verbose=False)
        greedy_recon = x_s_g + x_t_g
        greedy_mse = np.mean(np.abs(signal - greedy_recon) ** 2)
        greedy_rate = energy_concentration(
            np.concatenate([x_s_g, x_t_g]), 0.99
        )

        # Braided
        braid_res = braided_hybrid_mca(signal, verbose=False, threshold=thresh)
        braid_recon = braid_res.structural + braid_res.texture
        braid_mse = np.mean(np.abs(signal - braid_recon) ** 2)
        braid_rate = energy_concentration(
            np.concatenate([braid_res.structural, braid_res.texture]), 0.99
        )

        print(
            f"{thresh:<12.2f} | {greedy_rate:<12.3f} | {greedy_mse:<12.3e} | {braid_rate:<12.3f} | {braid_mse:<12.3e}"
        )

    print("\n✅ INTERPRETATION:")
    print("  - Greedy should have lower MSE (better reconstruction)")
    print("  - If Braid has lower Rate: It's finding sparser representation")
    print("  - If Braid MSE >> Greedy MSE: Phase destruction confirmed")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(" COMPREHENSIVE BRAIDED MCA EVALUATION")
    print("=" * 70)
    print(" Comparing Sequential Greedy vs Parallel Braided strategies")
    print(" across Compression, Separation, and Rate-Distortion tasks.\n")

    test_ascii_bottleneck(N=256)
    test_source_separation(N=256, trials=20)
    test_rate_distortion(N=256)

    print("\n" + "=" * 70)
    print(" SUMMARY & CONCLUSIONS")
    print("=" * 70)
    print("\n🔍 KEY FINDINGS:")
    print("  1. If Braid consistently has higher reconstruction error:")
    print("     → Parallel competition breaks phase coherence")
    print("     → Greedy is the better COMPRESSOR")
    print("\n  2. If Braid has better separation metrics (from Test 2):")
    print("     → It detects true sources better")
    print("     → Confirms: Greedy steals RFT bins, Braid gives RFT a chance")
    print("\n  3. If Braid has worse rate-distortion:")
    print("     → Energy smearing due to hard thresholding in freq domain")
    print("     → Need soft thresholding or L1-minimization")
    print("\n📌 NEXT STEPS:")
    print("  → If Braid fails as compressor: Document as 'detector-only'")
    print("  → If Braid succeeds: Investigate why it works (surprising!)")
    print("  → Either way: Need L1-solver (BPDN) for true separation\n")


if __name__ == "__main__":
    main()

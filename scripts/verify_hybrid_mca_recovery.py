#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier
#
# verify_hybrid_mca_recovery.py
# ------------------------------
# Monte-Carlo "MCA-style" recovery test for Theorem 10:
# Can the hybrid DCT + Φ-RFT algorithm recover a known
# decomposition x = x_s + x_t + η where:
#   - x_s is sparse in DCT basis
#   - x_t is sparse in Φ-RFT basis
#
# This is the closest thing to an "experimental proof"
# that Theorem 10's hybrid separability actually works.

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from scipy.fftpack import dct, idct
from numpy.fft import fft, ifft

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.rft.hybrids.legacy_mca import (
    PHI,
    rft_forward,
    rft_inverse,
    adaptive_hybrid_compress,
    braided_hybrid_mca,
)


rng = np.random.default_rng(42)


def generate_sparse_signal_dct(
    N: int,
    K: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a time-domain signal that is K-sparse in the DCT basis.

    Returns:
        x_s: time-domain structural component
        support: boolean mask of active DCT coefficients
    """
    coeffs = np.zeros(N, dtype=np.complex128)
    support_idx = rng.choice(N, size=K, replace=False)
    coeffs[support_idx] = (
        rng.normal(size=K) + 1j * rng.normal(size=K)
    ) / np.sqrt(2.0)
    # Inverse DCT (handle complex properly)
    x_real = idct(coeffs.real, norm="ortho")
    x_imag = idct(coeffs.imag, norm="ortho")
    x_s = x_real + 1j * x_imag
    support = np.zeros(N, dtype=bool)
    support[support_idx] = True
    return x_s, support


def generate_sparse_signal_rft(
    N: int,
    K: int,
    beta: float = 0.83,
    sigma: float = 1.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a time-domain signal that is K-sparse in the Φ-RFT basis.

    Returns:
        x_t: time-domain texture component
        support: boolean mask of active RFT coefficients
    """
    coeffs = np.zeros(N, dtype=np.complex128)
    support_idx = rng.choice(N, size=K, replace=False)
    coeffs[support_idx] = (
        rng.normal(size=K) + 1j * rng.normal(size=K)
    ) / np.sqrt(2.0)
    x_t = rft_inverse(coeffs, beta=beta, sigma=sigma)
    support = np.zeros(N, dtype=bool)
    support[support_idx] = True
    return x_t, support


def add_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add complex Gaussian noise at a target SNR (dB).
    """
    signal_power = np.mean(np.abs(x) ** 2)
    if signal_power == 0:
        return x.copy()
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = (
        rng.normal(scale=np.sqrt(noise_power / 2.0), size=x.shape)
        + 1j * rng.normal(scale=np.sqrt(noise_power / 2.0), size=x.shape)
    )
    return x + noise


def l2_error(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.linalg.norm(x_true - x_hat) / (np.linalg.norm(x_true) + 1e-12))


def support_recovery(
    true_support: np.ndarray,
    est_support: np.ndarray,
) -> Dict[str, float]:
    """
    Compute support recovery metrics:
      - precision: |S_true ∩ S_est| / |S_est|
      - recall:    |S_true ∩ S_est| / |S_true|
      - f1:        harmonic mean
    """
    true_support = true_support.astype(bool)
    est_support = est_support.astype(bool)

    tp = np.logical_and(true_support, est_support).sum()
    fp = np.logical_and(~true_support, est_support).sum()
    fn = np.logical_and(true_support, ~est_support).sum()

    precision = 0.0 if (tp + fp) == 0 else tp / float(tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / float(tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def estimate_support_dct(x_s: np.ndarray, K: int) -> np.ndarray:
    """
    Estimate DCT support by taking top-K DCT coefficients of the estimated structural component.
    """
    coeffs = dct(x_s.real, norm="ortho")
    # Ignore imaginary part for support (structure is real in many cases)
    idx_sorted = np.argsort(np.abs(coeffs))[::-1]
    support = np.zeros_like(coeffs, dtype=bool)
    support[idx_sorted[:K]] = True
    return support


def estimate_support_rft(x_t: np.ndarray, K: int) -> np.ndarray:
    """
    Estimate RFT support by taking top-K Φ-RFT coefficients of the estimated texture component.
    """
    coeffs = rft_forward(x_t)
    idx_sorted = np.argsort(np.abs(coeffs))[::-1]
    support = np.zeros_like(coeffs, dtype=bool)
    support[idx_sorted[:K]] = True
    return support


def run_single_trial(
    N: int,
    Ks: int,
    Kt: int,
    snr_db: float,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    One Monte-Carlo trial:
      1. Generate x_s, x_t with known supports.
      2. Form x = x_s + x_t + noise.
      3. Run adaptive_hybrid_compress.
      4. Measure:
         - L2 error on x_s, x_t
         - Support recovery precision/recall/F1 for DCT + RFT.
    """
    # 1. Generate sparse components
    x_s_true, supp_s_true = generate_sparse_signal_dct(N, Ks)
    x_t_true, supp_t_true = generate_sparse_signal_rft(N, Kt)

    x_clean = x_s_true + x_t_true
    x_noisy = add_noise(x_clean, snr_db=snr_db)

    # 2. Run hybrid algorithm
    # x_struct_hat, x_texture_hat, weights, meta = adaptive_hybrid_compress(
    #     x_noisy,
    #     verbose=False,
    #     max_iter=6,
    # )
    
    # USE BRAIDED MCA (Theorem 10 Fix)
    result = braided_hybrid_mca(
        x_noisy,
        max_iter=10,
        threshold=0.05,
        dct_weight=1.0,
        rft_weight=1.0,
        verbose=False
    )
    x_struct_hat = result.structural
    x_texture_hat = result.texture
    total_energy = np.linalg.norm(x_struct_hat) + np.linalg.norm(x_texture_hat)
    if total_energy > 0:
        weights = {
            "dct": float(np.linalg.norm(x_struct_hat) / total_energy),
            "rft": float(np.linalg.norm(x_texture_hat) / total_energy),
        }
    else:
        weights = {"dct": 0.0, "rft": 0.0}

    # 3. Errors in component reconstruction
    err_s = l2_error(x_s_true, x_struct_hat)
    err_t = l2_error(x_t_true, x_texture_hat)
    err_total = l2_error(x_clean, x_struct_hat + x_texture_hat)

    # 4. Support estimation
    supp_s_est = estimate_support_dct(x_struct_hat, Ks)
    supp_t_est = estimate_support_rft(x_texture_hat, Kt)

    metrics_s = support_recovery(supp_s_true, supp_s_est)
    metrics_t = support_recovery(supp_t_true, supp_t_est)

    if verbose:
        print(f"  L2 err struct: {err_s:.3e}, texture: {err_t:.3e}, total: {err_total:.3e}")
        print(
            f"  DCT support F1={metrics_s['f1']:.2f}, "
            f"RFT support F1={metrics_t['f1']:.2f}"
        )
        print("  Weights:", weights)

    return {
        "err_s": err_s,
        "err_t": err_t,
        "err_total": err_total,
        "dct_precision": metrics_s["precision"],
        "dct_recall": metrics_s["recall"],
        "dct_f1": metrics_s["f1"],
        "rft_precision": metrics_t["precision"],
        "rft_recall": metrics_t["recall"],
        "rft_f1": metrics_t["f1"],
    }


def run_sweep():
    Ns = [64, 128, 256, 512]
    Ks_list = [2, 4, 8]       # # of DCT atoms
    Kt_list = [2, 4, 8]       # # of RFT atoms
    snrs = [40.0, 30.0, 20.0]  # dB
    trials = 50

    print("=" * 70)
    print(" HYBRID DCT + Φ-RFT MCA RECOVERY TEST (Theorem 10)")
    print("=" * 70)
    print("Each row: mean over Monte-Carlo trials")
    print("Columns: component L2 errors + support F1 for DCT/RFT\n")

    header = (
        "N   Ks  Kt  SNR(dB)  "
        "err_s    err_t    err_tot   "
        "F1_DCT   F1_RFT"
    )
    print(header)
    print("-" * len(header))

    for N in Ns:
        for Ks in Ks_list:
            for Kt in Kt_list:
                for snr_db in snrs:
                    metrics_accum = {
                        "err_s": 0.0,
                        "err_t": 0.0,
                        "err_total": 0.0,
                        "dct_f1": 0.0,
                        "rft_f1": 0.0,
                    }
                    for _ in range(trials):
                        m = run_single_trial(N, Ks, Kt, snr_db, verbose=False)
                        metrics_accum["err_s"] += m["err_s"]
                        metrics_accum["err_t"] += m["err_t"]
                        metrics_accum["err_total"] += m["err_total"]
                        metrics_accum["dct_f1"] += m["dct_f1"]
                        metrics_accum["rft_f1"] += m["rft_f1"]

                    for key in metrics_accum:
                        metrics_accum[key] /= float(trials)

                    print(
                        f"{N:<3d} {Ks:<3d} {Kt:<3d} {snr_db:<7.1f}  "
                        f"{metrics_accum['err_s']:<8.2e} "
                        f"{metrics_accum['err_t']:<8.2e} "
                        f"{metrics_accum['err_total']:<8.2e} "
                        f"{metrics_accum['dct_f1']:<8.2f} "
                        f"{metrics_accum['rft_f1']:<8.2f}"
                    )


if __name__ == "__main__":
    run_sweep()

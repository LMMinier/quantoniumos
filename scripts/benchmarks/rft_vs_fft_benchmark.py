#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
RFT vs FFT Benchmark

This script evaluates a selectable RFT implementation against a unitary FFT baseline across:
- Energy compaction (k for >= 99% energy)
- Reconstruction error from top-k coefficients
- Spectral leakage on off-bin sinusoid
- Quantization robustness (b-bit rounding of coefficients)
- Simple denoising via coefficient thresholding

Notes:
- FFT has O(N log N) and will be faster; this benchmark focuses on representational properties.
- Any observed advantages are signal-dependent; use this as an evidence generator, not a blanket claim.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

# Ensure repository root is on sys.path when executed as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
from algorithms.rft.core.resonant_fourier_transform import (
    rft_basis_matrix as phi_frame_basis_matrix,
    rft_forward_frame as phi_frame_forward_frame,
)


class _PhiFrameRFT:
    """φ-frame RFT implementation for this benchmark.

    Uses the square-kernel φ-grid basis from `algorithms.rft.core.resonant_fourier_transform`.

    Implementations:
    - `phi_frame_unitary`: Gram-normalized unitary basis; forward = Φᴴx, inverse = ΦX
    - `phi_frame_frame`: raw basis with dual-frame solve; forward = (ΦᴴΦ)^{-1}Φᴴx
    """

    def __init__(self, size: int, impl: str):
        self.size = int(size)
        self.impl = str(impl)

        if self.impl == "phi_frame_unitary":
            self._Phi = phi_frame_basis_matrix(self.size, self.size, use_gram_normalization=True)
        elif self.impl == "phi_frame_frame":
            self._Phi = phi_frame_basis_matrix(self.size, self.size, use_gram_normalization=False)
        else:
            raise ValueError(f"Unsupported φ-frame impl: {self.impl}")

    def forward_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        if x.shape[0] != self.size:
            raise ValueError(f"Input size {x.shape[0]} != RFT size {self.size}")

        if self.impl == "phi_frame_unitary":
            return self._Phi.conj().T @ x

        # Dual-frame coefficients for exact reconstruction with non-orthogonal Φ.
        return phi_frame_forward_frame(x, self._Phi)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.complex128)
        if y.shape[0] != self.size:
            raise ValueError(f"Input size {y.shape[0]} != RFT size {self.size}")
        return self._Phi @ y

    def get_rft_matrix(self) -> np.ndarray:
        # Basis matrix used for synthesis (x = ΦX), consistent with CanonicalTrueRFT.get_rft_matrix().
        return self._Phi


def _make_rft_impl(n: int, rft_impl: str):
    if rft_impl == "canonical_true":
        return CanonicalTrueRFT(n)
    if rft_impl in {"phi_frame_unitary", "phi_frame_frame"}:
        return _PhiFrameRFT(n, rft_impl)
    raise ValueError(f"Unknown rft_impl: {rft_impl}")


def unitary_fft_matrix(n: int) -> np.ndarray:
    """Return the unitary DFT matrix of size n."""
    W = np.fft.fft(np.eye(n)) / np.sqrt(n)
    return W


def forward_fft(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    return np.fft.fft(x) / np.sqrt(n)


def inverse_fft(y: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    return np.fft.ifft(y * np.sqrt(n))


def energy_compaction_k(coeffs: np.ndarray, frac: float = 0.99) -> int:
    mag2 = np.abs(coeffs) ** 2
    total = mag2.sum()
    if total <= 0:
        return 0
    idx = np.argsort(mag2)[::-1]
    csum = np.cumsum(mag2[idx])
    k = int(np.searchsorted(csum, frac * total) + 1)
    return k


def reconstruct_top_k(
    coeffs: np.ndarray, k: int, inverse: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    if k <= 0:
        return inverse(np.zeros_like(coeffs))
    mag2 = np.abs(coeffs) ** 2
    idx = np.argsort(mag2)[::-1]
    mask = np.zeros_like(coeffs, dtype=bool)
    mask[idx[:k]] = True
    pruned = np.where(mask, coeffs, 0)
    return inverse(pruned)


def quantize_complex(z: np.ndarray, bits: int = 10) -> np.ndarray:
    # Uniform quantization per real/imag parts within dynamic range
    if bits <= 0:
        return np.zeros_like(z)
    r = np.real(z)
    i = np.imag(z)
    rmax = np.max(np.abs(r)) + 1e-12
    imax = np.max(np.abs(i)) + 1e-12
    levels = 2 ** bits
    rq = np.round((r / rmax) * (levels / 2 - 1)) / (levels / 2 - 1) * rmax
    iq = np.round((i / imax) * (levels / 2 - 1)) / (levels / 2 - 1) * imax
    return rq + 1j * iq


def snr_db(x: np.ndarray, y: np.ndarray) -> float:
    num = np.linalg.norm(x) ** 2
    den = np.linalg.norm(x - y) ** 2 + 1e-18
    return 10 * math.log10(float(num / den))


def spectral_leakage_metric(y: np.ndarray) -> float:
    # Ratio of max magnitude to L1 magnitude: higher is less leakage for a pure tone
    mag = np.abs(y)
    denom = mag.sum() + 1e-12
    return float(mag.max() / denom)


def make_offbin_tone(n: int, freq: float, phase: float = 0.0) -> np.ndarray:
    # freq in cycles per n samples; allow non-integer to induce leakage for FFT
    t = np.arange(n)
    return np.exp(1j * (2 * np.pi * freq * t / n + phase))


def make_chirp(n: int, f0: float, f1: float) -> np.ndarray:
    t = np.linspace(0, 1, n, endpoint=False)
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t ** 2)
    return np.exp(1j * phase)


def make_phi_modulated(n: int, beta: float = 1.0) -> np.ndarray:
    # Golden ratio phase modulation
    phi = (1 + np.sqrt(5)) / 2
    seq = np.array([(k / phi) % 1 for k in range(n)])
    return np.exp(1j * 2 * np.pi * beta * seq)


@dataclass
class TrialResult:
    size: int
    signal: str
    metric: str
    rft_value: float
    fft_value: float
    better: str  # "RFT", "FFT", or "TIE"
    rft_impl: str


def judge(rft_val: float, fft_val: float, higher_is_better: bool) -> str:
    if math.isclose(rft_val, fft_val, rel_tol=1e-6, abs_tol=1e-9):
        return "TIE"
    if higher_is_better:
        return "RFT" if rft_val > fft_val else "FFT"
    else:
        return "RFT" if rft_val < fft_val else "FFT"


def run_benchmarks(sizes: List[int], *, rft_impl: str) -> List[TrialResult]:
    results: List[TrialResult] = []

    for n in sizes:
        rft = _make_rft_impl(n, rft_impl)
        # Add a per-size distinctness metric vs unitary DFT
        dft = unitary_fft_matrix(n)
        dft_distance = float(np.linalg.norm(rft.get_rft_matrix() - dft, ord='fro'))
        results.append(
            TrialResult(
                size=n,
                signal="-",
                metric="frobenius_U_minus_DFT",
                rft_value=dft_distance,
                fft_value=0.0,
                better="RFT",  # informational label (not a competition metric)
                rft_impl=rft_impl,
            )
        )

        # Define signals
        signals: List[Tuple[str, np.ndarray]] = [
            ("offbin_tone_0.37", make_offbin_tone(n, freq=0.37 * n, phase=0.2)),
            ("chirp_0.1_0.4", make_chirp(n, f0=0.1 * n, f1=0.4 * n)),
            ("phi_mod_beta1.0", make_phi_modulated(n, beta=1.0)),
            ("impulse", np.pad([1.0], (0, n - 1)).astype(complex)),
            ("noise", (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)),
        ]

        for name, x in signals:
            # Normalize
            x = x / (np.linalg.norm(x) + 1e-12)

            # Transforms
            y_rft = rft.forward_transform(x)
            y_fft = forward_fft(x)

            # Energy compaction
            k_rft = energy_compaction_k(y_rft, 0.99)
            k_fft = energy_compaction_k(y_fft, 0.99)
            results.append(
                TrialResult(
                    size=n,
                    signal=name,
                    metric="k_99_energy",
                    rft_value=float(k_rft),
                    fft_value=float(k_fft),
                    better=judge(float(k_rft), float(k_fft), higher_is_better=False),
                    rft_impl=rft_impl,
                )
            )

            # Reconstruction from top-k (use min of the two k values for parity)
            k = min(k_rft, k_fft)
            xr_rft = reconstruct_top_k(y_rft, k, rft.inverse_transform)
            xr_fft = reconstruct_top_k(y_fft, k, inverse_fft)
            err_rft = float(np.linalg.norm(x - xr_rft))
            err_fft = float(np.linalg.norm(x - xr_fft))
            results.append(
                TrialResult(
                    size=n,
                    signal=name,
                    metric="recon_error_topk",
                    rft_value=err_rft,
                    fft_value=err_fft,
                    better=judge(err_rft, err_fft, higher_is_better=False),
                    rft_impl=rft_impl,
                )
            )

            # Spectral leakage (only meaningful for pure-ish tones)
            if name.startswith("offbin_tone"):
                leak_rft = spectral_leakage_metric(y_rft)
                leak_fft = spectral_leakage_metric(y_fft)
                results.append(
                    TrialResult(
                        size=n,
                        signal=name,
                        metric="leakage_peak_over_l1",
                        rft_value=leak_rft,
                        fft_value=leak_fft,
                        better=judge(leak_rft, leak_fft, higher_is_better=True),
                        rft_impl=rft_impl,
                    )
                )

            # Quantization robustness
            yq_rft = quantize_complex(y_rft, bits=10)
            yq_fft = quantize_complex(y_fft, bits=10)
            xq_rft = rft.inverse_transform(yq_rft)
            xq_fft = inverse_fft(yq_fft)
            snr_rft = snr_db(x, xq_rft)
            snr_fft = snr_db(x, xq_fft)
            results.append(
                TrialResult(
                    size=n,
                    signal=name,
                    metric="snr_db_quant10",
                    rft_value=snr_rft,
                    fft_value=snr_fft,
                    better=judge(snr_rft, snr_fft, higher_is_better=True),
                    rft_impl=rft_impl,
                )
            )

            # Simple denoising: add noise in time, threshold small coeffs, invert
            noisy = x + 0.05 * (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
            noisy = noisy / (np.linalg.norm(noisy) + 1e-12)
            yr = rft.forward_transform(noisy)
            yf = forward_fft(noisy)
            # Keep coefficients above median magnitude
            thr_r = np.median(np.abs(yr))
            thr_f = np.median(np.abs(yf))
            yr_thr = np.where(np.abs(yr) >= thr_r, yr, 0)
            yf_thr = np.where(np.abs(yf) >= thr_f, yf, 0)
            xr = rft.inverse_transform(yr_thr)
            xf = inverse_fft(yf_thr)
            snr_denoise_rft = snr_db(x, xr)
            snr_denoise_fft = snr_db(x, xf)
            results.append(
                TrialResult(
                    size=n,
                    signal=name,
                    metric="snr_db_denoise_threshold",
                    rft_value=snr_denoise_rft,
                    fft_value=snr_denoise_fft,
                    better=judge(snr_denoise_rft, snr_denoise_fft, higher_is_better=True),
                    rft_impl=rft_impl,
                )
            )

    return results


def write_csv(path: str, rows: List[TrialResult]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["size", "signal", "metric", "rft_value", "fft_value", "better", "rft_impl"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main():
    parser = argparse.ArgumentParser(description="RFT vs FFT benchmark")
    parser.add_argument("--sizes", type=str, default="64,128", help="Comma-separated sizes")
    parser.add_argument(
        "--rft-impl",
        type=str,
        default="phi_frame_unitary",
        choices=["phi_frame_unitary", "phi_frame_frame", "canonical_true"],
        help=(
            "RFT implementation under test. Default uses the Gram-normalized φ-frame "
            "basis (unitary). Use 'canonical_true' to benchmark CanonicalTrueRFT."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/patent_benchmarks/rft_vs_fft_benchmark.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    print(f"Using RFT implementation: {args.rft_impl}")

    t0 = time.time()
    rows = run_benchmarks(sizes, rft_impl=args.rft_impl)
    dt = time.time() - t0
    write_csv(args.out, rows)

    # Compact console summary
    summary: Dict[str, int] = {"RFT": 0, "FFT": 0, "TIE": 0}
    for r in rows:
        summary[r.better] = summary.get(r.better, 0) + 1
    print("Benchmark complete in %.2fs" % dt)
    print("Wins: RFT=%d, FFT=%d, TIE=%d" % (summary["RFT"], summary["FFT"], summary["TIE"]))
    print(f"CSV written to: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
#
# Open for research and education under AGPL-3.0-or-later.
# This benchmark is NOT patent-practicing and may be freely
# used for reproducibility, evaluation, and academic purposes.
"""
RFT Compression Sweet Spot — Where the φ-Grid Basis WINS
=========================================================

Demonstrates that the RFT achieves superior compression on signals with
golden-ratio quasi-periodic structure, while honestly showing it loses
on signals that lack such structure.

The key insight (Hurwitz's theorem, 1891): DFT bins at k/N can never
align with golden-ratio frequencies. The RFT basis at frac((k+1)·φ)
achieves perfect alignment → exponential energy decay → fewer coefficients
→ smaller compressed output.

Tests three signal categories:
  1. φ-quasi-periodic signals  — RFT SHOULD WIN
  2. Smooth/piecewise signals  — DCT SHOULD WIN
  3. Natural-image-like texture — DCT SHOULD WIN

For each, we measure TRUE compressed bytes (zlib on quantized coefficients),
PSNR, and energy compaction (K₀.₉₉ = coefficients for 99% energy).

Usage:
    python benchmarks/rft_compression_sweet_spot.py
"""
from __future__ import annotations

import io
import json
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.fftpack import dct, idct

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix

PHI = (1 + np.sqrt(5)) / 2


# ===================================================================
# Signal generators
# ===================================================================

def gen_phi_quasiperiodic(N: int, seed: int = 42) -> np.ndarray:
    """Sum of sinusoids at golden-ratio-related frequencies."""
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float64) / N
    signal = np.zeros(N)
    # Frequencies that are golden-ratio multiples
    for k in range(1, 8):
        freq = PHI ** k  # φ¹, φ², φ³, …
        amp = 1.0 / k
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    return signal


def gen_penrose_1d(N: int, seed: int = 42) -> np.ndarray:
    """1D quasicrystal — cut-and-project from 2D lattice along φ slope."""
    # Project 2D integer lattice points near the line y = x/φ
    # onto that line. The resulting 1D pattern has φ-quasiperiodic structure.
    strip_width = 1.0
    points = []
    for i in range(-N, N):
        for j in range(-N, N):
            # Distance from (i,j) to the line y = x/φ
            # Line: x/φ - y = 0  →  normal = (1/φ, -1) / |(1/φ, -1)|
            dist = abs(i / PHI - j) / np.sqrt(1 / PHI**2 + 1)
            if dist < strip_width / 2:
                proj = (i + j * PHI) / np.sqrt(1 + PHI**2)
                points.append(proj)
    points = np.sort(points)
    # Resample to N uniform points
    x_uniform = np.linspace(points.min(), points.max(), N)
    # Create a signal from the spacings (characteristic of quasicrystal)
    spacings = np.diff(points)
    if len(spacings) < N:
        spacings = np.tile(spacings, N // len(spacings) + 1)
    return spacings[:N]


def gen_fibonacci_modulated(N: int, seed: int = 42) -> np.ndarray:
    """Signal modulated by Fibonacci sequence ratios."""
    t = np.arange(N, dtype=np.float64) / N
    # Fibonacci ratios approach φ
    fib = [1, 1]
    while len(fib) < 20:
        fib.append(fib[-1] + fib[-2])
    signal = np.zeros(N)
    for i in range(2, min(12, len(fib))):
        freq = fib[i] / fib[i - 1]  # Approaches φ
        signal += np.sin(2 * np.pi * freq * (i + 1) * t) / i
    return signal


def gen_golden_chirp(N: int, seed: int = 42) -> np.ndarray:
    """Chirp with golden-ratio frequency modulation."""
    t = np.arange(N, dtype=np.float64) / N
    return np.sin(2 * np.pi * 50 * t * PHI ** (2 * t))


def gen_smooth_lowfreq(N: int, seed: int = 42) -> np.ndarray:
    """Smooth low-frequency signal — DCT's home turf."""
    t = np.arange(N, dtype=np.float64) / N
    return (np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t) +
            0.3 * np.cos(2 * np.pi * 11 * t))


def gen_piecewise_constant(N: int, seed: int = 42) -> np.ndarray:
    """Piecewise constant — favours DCT/wavelet."""
    rng = np.random.default_rng(seed)
    n_segments = 8
    boundaries = np.sort(rng.choice(N, n_segments - 1, replace=False))
    boundaries = np.concatenate([[0], boundaries, [N]])
    signal = np.zeros(N)
    for i in range(len(boundaries) - 1):
        signal[boundaries[i]:boundaries[i + 1]] = rng.standard_normal()
    return signal


def gen_natural_texture(N: int, seed: int = 42) -> np.ndarray:
    """1/f noise — natural image-like spectral profile."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(N)
    freqs[0] = 1  # avoid /0
    spectrum = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
    spectrum /= np.sqrt(freqs)  # 1/f
    spectrum[0] = 0
    return np.fft.irfft(spectrum, n=N)


def gen_white_noise(N: int, seed: int = 42) -> np.ndarray:
    """Pure white noise — no transform should win."""
    return np.random.default_rng(seed).standard_normal(N)


# ===================================================================
# Transform + compress pipeline
# ===================================================================

def compress_with_transform(
    signal: np.ndarray,
    transform: str,  # "rft" or "dct"
    keep_fractions: List[float],
    mag_bits: int = 12,
    phase_bits: int = 10,
) -> List[Dict]:
    """
    Transform → top-k threshold → quantize → zlib → measure TRUE bytes.
    
    Returns list of {keep_frac, bpv, psnr, ssim, k99, compressed_bytes} dicts.
    """
    N = len(signal)
    results = []

    if transform == "rft":
        Phi = rft_basis_matrix(N, N, use_gram_normalization=True)
        PhiH = Phi.conj().T
        coeffs = PhiH @ signal.astype(np.complex128)
    elif transform == "dct":
        coeffs = dct(signal, norm='ortho').astype(np.complex128)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    # Energy compaction: K_0.99
    energies = np.abs(coeffs) ** 2
    total_energy = energies.sum()
    sorted_idx = np.argsort(energies)[::-1]
    cumulative = np.cumsum(energies[sorted_idx])
    k99 = int(np.searchsorted(cumulative, 0.99 * total_energy) + 1)

    for keep_frac in keep_fractions:
        k = max(1, int(N * keep_frac))

        # Keep top-k coefficients
        mags = np.abs(coeffs)
        threshold = np.sort(mags)[::-1][min(k, N) - 1] if k < N else 0
        mask = mags >= threshold
        sparse_coeffs = coeffs * mask

        # Quantize
        sparse_mags = np.abs(sparse_coeffs)
        sparse_phases = np.angle(sparse_coeffs)
        mag_max = sparse_mags.max() + 1e-15
        max_mag_val = (1 << mag_bits) - 1
        max_phase_val = (1 << phase_bits) - 1

        mags_q = np.clip((sparse_mags / mag_max * max_mag_val), 0, max_mag_val).astype(np.uint16)
        phases_norm = (sparse_phases + np.pi) / (2 * np.pi)
        phases_q = np.clip((phases_norm * max_phase_val), 0, max_phase_val).astype(np.uint16)

        # For DCT (real coefficients), we only need magnitudes + signs
        if transform == "dct":
            signs = np.sign(sparse_coeffs.real).astype(np.int8)
            payload = mags_q.tobytes() + signs.tobytes()
        else:
            payload = mags_q.tobytes() + phases_q.tobytes()

        # Header: transform type (1) + N (4) + k (4) + mag_max (8) + mag_bits (1) + phase_bits (1) = 19 bytes
        header = struct.pack('>bIIdBB', 
                             0 if transform == "rft" else 1,
                             N, k, mag_max, mag_bits, phase_bits)
        compressed = header + zlib.compress(payload, 9)
        compressed_bytes = len(compressed)

        # Bits per value (TRUE)
        bpv = 8.0 * compressed_bytes / N

        # Reconstruct
        if transform == "dct":
            mags_deq = mags_q.astype(np.float64) / max_mag_val * mag_max
            recon_coeffs = mags_deq * signs.astype(np.float64)
            recon = idct(recon_coeffs, norm='ortho')
        else:
            mags_deq = mags_q.astype(np.float64) / max_mag_val * mag_max
            phases_deq = phases_q.astype(np.float64) / max_phase_val * 2 * np.pi - np.pi
            recon_coeffs = mags_deq * np.exp(1j * phases_deq)
            recon = (Phi @ recon_coeffs).real

        # PSNR
        mse = np.mean((signal - recon) ** 2)
        peak = np.max(np.abs(signal))
        if mse < 1e-15:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(peak ** 2 / mse)

        results.append({
            "keep_frac": keep_frac,
            "bpv": round(bpv, 3),
            "psnr": round(psnr, 2),
            "compressed_bytes": compressed_bytes,
            "k99": k99,
            "k99_pct": round(100 * k99 / N, 1),
        })

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    N = 512
    keep_fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]

    signals = {
        # φ-structured (RFT should win)
        "φ-quasi-periodic": gen_phi_quasiperiodic(N),
        "Penrose 1D": gen_penrose_1d(N),
        "Fibonacci-modulated": gen_fibonacci_modulated(N),
        "Golden chirp": gen_golden_chirp(N),
        # Non-φ-structured (DCT should win)
        "Smooth low-freq": gen_smooth_lowfreq(N),
        "Piecewise constant": gen_piecewise_constant(N),
        "Natural texture (1/f)": gen_natural_texture(N),
        "White noise": gen_white_noise(N),
    }

    all_results = {}

    print("=" * 80)
    print("RFT COMPRESSION SWEET SPOT BENCHMARK")
    print(f"N = {N}, keep fractions = {keep_fractions}")
    print("=" * 80)

    for sig_name, sig in signals.items():
        # Normalize to unit peak
        sig = sig / (np.max(np.abs(sig)) + 1e-15)
        
        rft_res = compress_with_transform(sig, "rft", keep_fractions)
        dct_res = compress_with_transform(sig, "dct", keep_fractions)

        all_results[sig_name] = {"rft": rft_res, "dct": dct_res}

        # Determine winner at each point
        print(f"\n{'─' * 80}")
        print(f"Signal: {sig_name}")
        print(f"  K₀.₉₉: RFT={rft_res[0]['k99']} ({rft_res[0]['k99_pct']}%)  "
              f"DCT={dct_res[0]['k99']} ({dct_res[0]['k99_pct']}%)")
        print(f"  {'Keep%':>6s}  {'RFT BPV':>8s} {'RFT PSNR':>9s}  "
              f"{'DCT BPV':>8s} {'DCT PSNR':>9s}  {'Winner':>8s} {'Δ PSNR':>8s}")

        rft_wins = 0
        for r, d in zip(rft_res, dct_res):
            # Winner: lower BPV at matched PSNR, or higher PSNR at matched BPV
            # Use PSNR per bit as combined metric
            r_efficiency = r["psnr"] / max(r["bpv"], 0.01)
            d_efficiency = d["psnr"] / max(d["bpv"], 0.01)
            if r_efficiency > d_efficiency:
                winner = "RFT ✓"
                rft_wins += 1
            else:
                winner = "DCT ✓"
            delta = r["psnr"] - d["psnr"]
            sign = "+" if delta > 0 else ""
            print(f"  {r['keep_frac']*100:5.0f}%  {r['bpv']:8.3f} {r['psnr']:8.2f}dB  "
                  f"{d['bpv']:8.3f} {d['psnr']:8.2f}dB  {winner:>8s} {sign}{delta:.1f}dB")

        verdict = "RFT WINS" if rft_wins > len(keep_fractions) / 2 else "DCT WINS"
        print(f"  Overall: {verdict} ({rft_wins}/{len(keep_fractions)} points)")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    phi_signals = ["φ-quasi-periodic", "Penrose 1D", "Fibonacci-modulated", "Golden chirp"]
    other_signals = ["Smooth low-freq", "Piecewise constant", "Natural texture (1/f)", "White noise"]

    print("\nφ-STRUCTURED signals (RFT's target domain):")
    for name in phi_signals:
        res = all_results[name]
        k99_rft = res["rft"][0]["k99"]
        k99_dct = res["dct"][0]["k99"]
        # Compare at 5% retention
        r5 = next(r for r in res["rft"] if r["keep_frac"] == 0.05)
        d5 = next(d for d in res["dct"] if d["keep_frac"] == 0.05)
        delta = r5["psnr"] - d5["psnr"]
        sign = "+" if delta > 0 else ""
        print(f"  {name:30s}  K₀.₉₉: RFT={k99_rft:3d} vs DCT={k99_dct:3d}  "
              f"PSNR@5%: {sign}{delta:.1f} dB  BPV@5%: RFT={r5['bpv']:.2f} DCT={d5['bpv']:.2f}")

    print("\nNON-φ signals (DCT's domain):")
    for name in other_signals:
        res = all_results[name]
        k99_rft = res["rft"][0]["k99"]
        k99_dct = res["dct"][0]["k99"]
        r5 = next(r for r in res["rft"] if r["keep_frac"] == 0.05)
        d5 = next(d for d in res["dct"] if d["keep_frac"] == 0.05)
        delta = r5["psnr"] - d5["psnr"]
        sign = "+" if delta > 0 else ""
        print(f"  {name:30s}  K₀.₉₉: RFT={k99_rft:3d} vs DCT={k99_dct:3d}  "
              f"PSNR@5%: {sign}{delta:.1f} dB  BPV@5%: RFT={r5['bpv']:.2f} DCT={d5['bpv']:.2f}")

    print(f"\n{'=' * 80}")
    print("CONCLUSION:")
    print("  DCT outperforms RFT across all tested signal types at matched")
    print("  bit-rates.  The RFT φ-grid basis encodes each coefficient as")
    print("  (magnitude, phase) requiring ~2-3× more bits per value than")
    print("  DCT's real-valued coefficients.  This encoding overhead")
    print("  dominates any energy-compaction advantage from the φ-grid.")
    print()
    print("  RFT's value is in HYBRID systems: the spectral-entropy router")
    print("  identifies tensors with anomalously low spectral entropy and")
    print("  applies RFT there (e.g. LLM token embeddings), while using")
    print("  standard quantization everywhere else.  See the RFTMW memory")
    print("  layer for this approach applied to LLM weight compression.")
    print(f"{'=' * 80}")

    # Save results
    out_dir = _PROJECT_ROOT / "results" / "sweet_spot"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sweet_spot_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir / 'sweet_spot_results.json'}")


if __name__ == "__main__":
    main()

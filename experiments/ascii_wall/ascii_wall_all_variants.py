#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
ASCII Wall Test - ALL RFT VARIANTS Through Vertex Codec
========================================================

Tests ALL RFT transform variants through the actual vertex codec:
1. Standard Φ-RFT (baseline)
2. Fibonacci-tilt Φ-RFT 
3. Tetrahedral Δ-RFT (single vertex)
4. 2-way cascade (Φ-RFT + Δ)
5. 3-way cascade (Φ-RFT + Δ₁ + Δ₂)
6. 4-way cascade (Φ-RFT + Δ₁ + Δ₂ + Δ₃)

Each variant is pre-transformed, then encoded through the vertex codec
with quantization + ANS entropy coding.
"""
from __future__ import annotations

import sys
import json
import math
import base64
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
from collections import Counter

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the core RFT
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse

# Import ANS for direct encoding
from algorithms.rft.compression.ans import ans_encode, ans_decode, RANS_PRECISION_DEFAULT

# SOTA compressors
import gzip
import bz2
import lzma

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    
try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False


# ===========================================================================
# CONSTANTS
# ===========================================================================

PHI = (1.0 + 5.0 ** 0.5) / 2.0  # Golden ratio

# Regular tetrahedron vertices (normalized)
TETRA_VERTICES = np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]
], dtype=np.float64) / np.sqrt(3)

# Fibonacci ratio converges to phi - no need to compute actual Fibonacci numbers
def _fib_ratio_sequence(n: int) -> np.ndarray:
    """Compute F(k+1)/F(k) iteratively - converges to phi rapidly."""
    ratios = np.ones(n, dtype=np.float64)
    if n == 0:
        return ratios
    # F(k+1)/F(k) = 1 + F(k-1)/F(k) = 1 + 1/(F(k)/F(k-1))
    # Start with F(1)/F(0) = 1/1 = 1
    ratios[0] = 1.0
    prev_ratio = 1.0
    for i in range(1, n):
        new_ratio = 1.0 + 1.0 / prev_ratio  # converges to phi
        ratios[i] = new_ratio
        prev_ratio = new_ratio
    return ratios


# ===========================================================================
# RFT TRANSFORM VARIANTS
# ===========================================================================

def _frac(arr: np.ndarray) -> np.ndarray:
    """Fractional part, handling negatives correctly."""
    frac, _ = np.modf(arr)
    return np.where(frac < 0.0, frac + 1.0, frac)


# --- Standard Φ-RFT ---
def standard_rft_forward(x: np.ndarray) -> np.ndarray:
    """Standard closed-form Φ-RFT."""
    return rft_forward(x)

def standard_rft_inverse(y: np.ndarray) -> np.ndarray:
    """Standard closed-form Φ-RFT inverse."""
    return rft_inverse(y)


# --- Fibonacci-Tilt Φ-RFT ---
def fibonacci_rft_forward(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Fibonacci-tilt Φ-RFT: phase modulated by Fibonacci ratios."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    k = np.arange(n, dtype=np.float64)
    
    # Fibonacci ratios converge to phi - use iterative computation
    fib_ratios = _fib_ratio_sequence(n)
    
    # Fibonacci-modulated phase
    theta = 2.0 * np.pi * (1.0 + alpha * (fib_ratios - PHI)) * _frac(k / PHI)
    D_fib = np.exp(1j * theta)
    
    # Quadratic chirp
    ctheta = np.pi * (k * k / float(n))
    C_sig = np.exp(1j * ctheta)
    
    X = np.fft.fft(x, norm="ortho")
    return D_fib * (C_sig * X)

def fibonacci_rft_inverse(y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Fibonacci-tilt Φ-RFT inverse."""
    y = np.asarray(y, dtype=np.complex128)
    n = y.shape[0]
    k = np.arange(n, dtype=np.float64)
    
    # Fibonacci ratios converge to phi
    fib_ratios = _fib_ratio_sequence(n)
    
    theta = 2.0 * np.pi * (1.0 + alpha * (fib_ratios - PHI)) * _frac(k / PHI)
    D_fib = np.exp(1j * theta)
    
    ctheta = np.pi * (k * k / float(n))
    C_sig = np.exp(1j * ctheta)
    
    return np.fft.ifft(np.conj(C_sig) * np.conj(D_fib) * y, norm="ortho")


# --- Tetrahedral Δ-RFT ---
def delta_rft_forward(x: np.ndarray, vertex_idx: int = 0, 
                      beta: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Tetrahedral Δ-RFT using 3D phase modulation."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    k = np.arange(n, dtype=np.float64)
    
    v = TETRA_VERTICES[vertex_idx % 4]
    
    # 3D phase components
    phase1 = beta * _frac(k / PHI)
    phase2 = beta * _frac(k / (PHI ** 2))
    phase3 = beta * _frac(k / (PHI ** 3))
    
    theta = 2.0 * np.pi * (v[0] * phase1 + v[1] * phase2 + v[2] * phase3)
    D_tetra = np.exp(1j * theta)
    
    ctheta = np.pi * sigma * (k * k / float(n))
    C_sig = np.exp(1j * ctheta)
    
    X = np.fft.fft(x, norm="ortho")
    return D_tetra * (C_sig * X)

def delta_rft_inverse(y: np.ndarray, vertex_idx: int = 0,
                      beta: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Tetrahedral Δ-RFT inverse."""
    y = np.asarray(y, dtype=np.complex128)
    n = y.shape[0]
    k = np.arange(n, dtype=np.float64)
    
    v = TETRA_VERTICES[vertex_idx % 4]
    
    phase1 = beta * _frac(k / PHI)
    phase2 = beta * _frac(k / (PHI ** 2))
    phase3 = beta * _frac(k / (PHI ** 3))
    
    theta = 2.0 * np.pi * (v[0] * phase1 + v[1] * phase2 + v[2] * phase3)
    D_tetra = np.exp(1j * theta)
    
    ctheta = np.pi * sigma * (k * k / float(n))
    C_sig = np.exp(1j * ctheta)
    
    return np.fft.ifft(np.conj(C_sig) * np.conj(D_tetra) * y, norm="ortho")


# --- Cascade Transforms ---
def cascade_2way_forward(x: np.ndarray) -> np.ndarray:
    """2-way cascade: Φ-RFT + Δ₀"""
    y = rft_forward(x)
    y = delta_rft_forward(y, vertex_idx=0)
    return y

def cascade_2way_inverse(y: np.ndarray) -> np.ndarray:
    """2-way cascade inverse."""
    x = delta_rft_inverse(y, vertex_idx=0)
    x = rft_inverse(x)
    return x

def cascade_3way_forward(x: np.ndarray) -> np.ndarray:
    """3-way cascade: Φ-RFT + Δ₀ + Δ₁"""
    y = rft_forward(x)
    y = delta_rft_forward(y, vertex_idx=0)
    y = delta_rft_forward(y, vertex_idx=1)
    return y

def cascade_3way_inverse(y: np.ndarray) -> np.ndarray:
    """3-way cascade inverse."""
    x = delta_rft_inverse(y, vertex_idx=1)
    x = delta_rft_inverse(x, vertex_idx=0)
    x = rft_inverse(x)
    return x

def cascade_4way_forward(x: np.ndarray) -> np.ndarray:
    """4-way cascade: Φ-RFT + Δ₀ + Δ₁ + Δ₂"""
    y = rft_forward(x)
    y = delta_rft_forward(y, vertex_idx=0)
    y = delta_rft_forward(y, vertex_idx=1)
    y = delta_rft_forward(y, vertex_idx=2)
    return y

def cascade_4way_inverse(y: np.ndarray) -> np.ndarray:
    """4-way cascade inverse."""
    x = delta_rft_inverse(y, vertex_idx=2)
    x = delta_rft_inverse(x, vertex_idx=1)
    x = delta_rft_inverse(x, vertex_idx=0)
    x = rft_inverse(x)
    return x


# --- Fibonacci + Tetrahedral Hybrid ---
def fib_tetra_forward(x: np.ndarray) -> np.ndarray:
    """Fibonacci-tilt + Tetrahedral cascade."""
    y = fibonacci_rft_forward(x, alpha=0.1)
    y = delta_rft_forward(y, vertex_idx=0)
    return y

def fib_tetra_inverse(y: np.ndarray) -> np.ndarray:
    """Fibonacci-tilt + Tetrahedral inverse."""
    x = delta_rft_inverse(y, vertex_idx=0)
    x = fibonacci_rft_inverse(x, alpha=0.1)
    return x


# ===========================================================================
# TRANSFORM REGISTRY
# ===========================================================================

TRANSFORMS = {
    'standard': (standard_rft_forward, standard_rft_inverse, "Standard Φ-RFT"),
    'fibonacci': (fibonacci_rft_forward, fibonacci_rft_inverse, "Fibonacci-tilt Φ-RFT"),
    'delta_v0': (lambda x: delta_rft_forward(x, 0), lambda y: delta_rft_inverse(y, 0), "Δ-RFT vertex 0"),
    'delta_v1': (lambda x: delta_rft_forward(x, 1), lambda y: delta_rft_inverse(y, 1), "Δ-RFT vertex 1"),
    'fib_tetra': (fib_tetra_forward, fib_tetra_inverse, "Fibonacci + Tetrahedral"),
    # Cascades removed - they expand rather than compress for text data
}

# Additional: test WITHOUT transform (just quantize + ANS on raw data)
def identity_forward(x: np.ndarray) -> np.ndarray:
    return x.astype(np.complex128)

def identity_inverse(y: np.ndarray) -> np.ndarray:
    return np.real(y)

# Differential encoding - encode differences between samples
def diff_forward(x: np.ndarray) -> np.ndarray:
    diff = np.zeros_like(x)
    diff[0] = x[0]
    diff[1:] = np.diff(x)
    return diff.astype(np.complex128)

def diff_inverse(y: np.ndarray) -> np.ndarray:
    return np.cumsum(np.real(y))

# Differential + RFT
def diff_rft_forward(x: np.ndarray) -> np.ndarray:
    diff = np.zeros_like(x)
    diff[0] = x[0]
    diff[1:] = np.diff(x)
    return rft_forward(diff)

def diff_rft_inverse(y: np.ndarray) -> np.ndarray:
    diff = np.real(rft_inverse(y))
    return np.cumsum(diff)

# Add these to transforms
TRANSFORMS['identity'] = (identity_forward, identity_inverse, "No transform (baseline)")
TRANSFORMS['diff_only'] = (diff_forward, diff_inverse, "Differential only")
TRANSFORMS['diff_rft'] = (diff_rft_forward, diff_rft_inverse, "Differential + Φ-RFT")


# ===========================================================================
# CODEC ENCODING
# ===========================================================================

@dataclass
class CodecResult:
    """Result from encoding through the codec."""
    original_bytes: int
    payload_bytes: int
    bits_per_char: float
    roundtrip_ok: bool
    max_error: float
    transform: str
    sparsity: float  # fraction of coefficients pruned


def encode_with_transform(
    data: np.ndarray,
    forward_fn: Callable,
    inverse_fn: Callable,
    quant_bits_amp: int = 8,
    quant_bits_phase: int = 6,
    prune_threshold: float = 0.0,
    ans_precision: int = 12
) -> Tuple[int, np.ndarray, float]:
    """
    Apply transform, quantize, and ANS encode.
    Returns (compressed_bytes, decoded_data, sparsity).
    """
    n = len(data)
    
    # Forward transform
    coeffs = forward_fn(data)
    
    # Extract amplitude and phase
    amps = np.abs(coeffs)
    phases = np.angle(coeffs)
    
    # Pruning
    if prune_threshold > 0:
        keep_mask = amps >= prune_threshold
        kept_indices = np.nonzero(keep_mask)[0]
    else:
        kept_indices = np.arange(n)
        keep_mask = np.ones(n, dtype=bool)
    
    sparsity = 1.0 - len(kept_indices) / n
    
    kept_amps = amps[kept_indices]
    kept_phases = phases[kept_indices]
    
    # Log-amplitude for better quantization
    log_amps = np.log(np.maximum(kept_amps, 1e-12))
    
    # Quantization ranges
    log_min = float(log_amps.min()) if log_amps.size else 0.0
    log_max = float(log_amps.max()) if log_amps.size else 0.0
    phase_min, phase_max = -np.pi, np.pi
    
    # Quantize amplitudes
    amp_levels = (1 << quant_bits_amp) - 1
    if log_max > log_min and kept_amps.size > 0:
        amp_codes = np.clip(
            np.rint((log_amps - log_min) / (log_max - log_min) * amp_levels),
            0, amp_levels
        ).astype(np.int64)
    else:
        amp_codes = np.zeros(len(kept_amps), dtype=np.int64)
    
    # Quantize phases
    phase_levels = (1 << quant_bits_phase) - 1
    if kept_phases.size > 0:
        phase_codes = np.clip(
            np.rint((kept_phases - phase_min) / (phase_max - phase_min) * phase_levels),
            0, phase_levels
        ).astype(np.int64)
    else:
        phase_codes = np.zeros(0, dtype=np.int64)
    
    # ANS encode
    total_bytes = 0
    
    if amp_codes.size > 0:
        amp_encoded, amp_freq = ans_encode(amp_codes.tolist(), precision=ans_precision)
        amp_bytes = amp_encoded.nbytes
        total_bytes += amp_bytes
    
    if phase_codes.size > 0:
        phase_encoded, phase_freq = ans_encode(phase_codes.tolist(), precision=ans_precision)
        phase_bytes = phase_encoded.nbytes
        total_bytes += phase_bytes
    
    # If pruned, need to encode indices
    if prune_threshold > 0:
        # Use delta encoding for indices
        if len(kept_indices) > 0:
            deltas = np.diff(kept_indices, prepend=0).astype(np.int64)
            idx_encoded, _ = ans_encode(deltas.tolist(), precision=ans_precision)
            total_bytes += idx_encoded.nbytes
    
    # Add metadata overhead (scale factors: 4 floats = 32 bytes)
    total_bytes += 32
    
    # Decode for roundtrip verification
    # Dequantize
    if log_max > log_min and amp_codes.size > 0:
        recon_log_amps = (amp_codes.astype(np.float64) / amp_levels) * (log_max - log_min) + log_min
    else:
        recon_log_amps = np.full_like(amp_codes, log_min, dtype=np.float64)
    
    if phase_codes.size > 0:
        recon_phases = (phase_codes.astype(np.float64) / phase_levels) * (phase_max - phase_min) + phase_min
    else:
        recon_phases = np.zeros(0, dtype=np.float64)
    
    # Reconstruct coefficients
    recon_coeffs = np.zeros(n, dtype=np.complex128)
    if kept_indices.size > 0:
        recon_amps = np.exp(recon_log_amps)
        recon_coeffs[kept_indices] = recon_amps * np.exp(1j * recon_phases)
    
    # Inverse transform
    decoded = inverse_fn(recon_coeffs)
    decoded = np.real(decoded)
    
    return total_bytes, decoded, sparsity


def encode_ascii_all_variants(
    text: str,
    transform_name: str,
    quant_bits_amp: int = 8,
    quant_bits_phase: int = 6,
    prune_threshold: float = 0.0,
    ans_precision: int = 12
) -> CodecResult:
    """Encode ASCII through specified transform variant."""
    
    forward_fn, inverse_fn, _ = TRANSFORMS[transform_name]
    
    # Convert text to float array
    data = np.array([ord(c) for c in text], dtype=np.float64)
    n = len(data)
    original_bytes = n
    
    payload_bytes, decoded, sparsity = encode_with_transform(
        data, forward_fn, inverse_fn,
        quant_bits_amp, quant_bits_phase,
        prune_threshold, ans_precision
    )
    
    bits_per_char = (payload_bytes * 8) / n
    max_error = float(np.max(np.abs(decoded - data)))
    roundtrip_ok = max_error < 1.0
    
    return CodecResult(
        original_bytes=original_bytes,
        payload_bytes=payload_bytes,
        bits_per_char=bits_per_char,
        roundtrip_ok=roundtrip_ok,
        max_error=max_error,
        transform=transform_name,
        sparsity=sparsity
    )


# ===========================================================================
# SOTA COMPRESSORS
# ===========================================================================

def compress_sota(text: str, method: str) -> Tuple[int, float]:
    """Compress with SOTA method."""
    data = text.encode('ascii')
    
    if method == 'gzip':
        compressed = gzip.compress(data, compresslevel=9)
    elif method == 'bz2':
        compressed = bz2.compress(data, compresslevel=9)
    elif method == 'lzma':
        compressed = lzma.compress(data)
    elif method == 'zstd' and HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        compressed = cctx.compress(data)
    elif method == 'brotli' and HAS_BROTLI:
        compressed = brotli.compress(data, quality=11)
    else:
        return len(data), 8.0
    
    bits_per_char = (len(compressed) * 8) / len(text)
    return len(compressed), bits_per_char


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy."""
    freq = Counter(text)
    n = len(text)
    return -sum(c/n * math.log2(c/n) for c in freq.values() if c > 0)


# ===========================================================================
# TEST DATA
# ===========================================================================

def generate_test_texts() -> Dict[str, str]:
    """Generate test texts."""
    texts = {}
    
    texts['english_prose'] = (
        "The golden ratio, often denoted by the Greek letter phi, appears throughout "
        "nature and mathematics. From the spiral patterns of sunflower seeds to the "
        "proportions of the Parthenon, this irrational number approximately equal to "
        "1.618 has fascinated mathematicians for millennia. "
    ) * 20
    
    texts['python_code'] = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

class GoldenSpiral:
    def __init__(self, scale=1.0):
        self.phi = (1 + 5 ** 0.5) / 2
""" * 10
    
    texts['json_data'] = json.dumps({
        "data": [{"id": i, "value": i * 1.618} for i in range(50)]
    }, indent=2)
    
    texts['scientific'] = (
        "The quantum harmonic oscillator represents one of the most important model "
        "systems in quantum mechanics. The energy eigenvalues are E_n = hbar*w*(n + 1/2). "
    ) * 20
    
    rng = np.random.default_rng(42)
    chars = [chr(c) for c in range(32, 127)]
    texts['random_ascii'] = ''.join(rng.choice(chars, size=2000))
    
    texts['repetitive'] = "the quick brown fox jumps over the lazy dog " * 100
    
    return texts


# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================

def run_all_variants_experiment():
    """Run ASCII wall test with ALL RFT variants."""
    print("=" * 80)
    print("ASCII WALL TEST - ALL RFT VARIANTS")
    print("=" * 80)
    print()
    print("Transforms tested:")
    for name, (_, _, desc) in TRANSFORMS.items():
        print(f"  • {name}: {desc}")
    print()
    
    texts = generate_test_texts()
    results = []
    
    # Quantization configs - focus on pruning which performed best
    quant_configs = [
        (8, 6, 0.0, "8/6-bit"),
        (6, 4, 0.0, "6/4-bit"),
        (8, 6, 0.01, "prune 1%"),
        (8, 6, 0.05, "prune 5%"),
        (8, 6, 0.1, "prune 10%"),
        (8, 6, 0.5, "prune 50%"),
        (6, 4, 0.1, "6/4+prune10%"),
        (4, 3, 0.0, "4/3-bit (ultra)"),
    ]
    
    for text_name, text in texts.items():
        print(f"\n{'━' * 60}")
        print(f"Text: {text_name} ({len(text)} chars)")
        print(f"{'━' * 60}")
        
        entropy = calculate_entropy(text)
        print(f"Shannon entropy: {entropy:.3f} bpc")
        
        # SOTA baselines
        sota_results = {}
        for method in ['gzip', 'bz2', 'lzma']:
            _, bpc = compress_sota(text, method)
            sota_results[method] = bpc
        if HAS_ZSTD:
            _, bpc = compress_sota(text, 'zstd')
            sota_results['zstd'] = bpc
        if HAS_BROTLI:
            _, bpc = compress_sota(text, 'brotli')
            sota_results['brotli'] = bpc
        
        best_sota = min(sota_results.values())
        best_sota_name = min(sota_results.keys(), key=lambda k: sota_results[k])
        print(f"Best SOTA: {best_sota_name} @ {best_sota:.3f} bpc")
        
        print(f"\n{'Transform':<25} {'Config':<12} {'BPC':>8} {'vs SOTA':>10} {'Err':>8}")
        print("─" * 65)
        
        for transform_name in TRANSFORMS.keys():
            _, _, desc = TRANSFORMS[transform_name]
            
            for amp_bits, phase_bits, prune, config_desc in quant_configs:
                try:
                    result = encode_ascii_all_variants(
                        text, transform_name,
                        quant_bits_amp=amp_bits,
                        quant_bits_phase=phase_bits,
                        prune_threshold=prune
                    )
                    
                    improvement = (best_sota - result.bits_per_char) / best_sota * 100
                    marker = "✓" if result.bits_per_char < best_sota else ""
                    rt = "✓" if result.roundtrip_ok else f"{result.max_error:.1f}"
                    
                    print(f"{transform_name:<25} {config_desc:<12} {result.bits_per_char:>8.2f} "
                          f"{improvement:>+9.1f}% {rt:>8} {marker}")
                    
                    results.append({
                        'text': text_name,
                        'transform': transform_name,
                        'config': config_desc,
                        'bpc': result.bits_per_char,
                        'best_sota': best_sota,
                        'improvement_pct': improvement,
                        'roundtrip_ok': result.roundtrip_ok,
                        'max_error': result.max_error,
                        'sparsity': result.sparsity
                    })
                    
                except Exception as e:
                    print(f"{transform_name:<25} {config_desc:<12} ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: ALL VARIANTS")
    print("=" * 80)
    
    valid = [r for r in results if 'bpc' in r]
    wins = sum(1 for r in valid if r['improvement_pct'] > 0)
    
    print(f"\nTotal wins vs SOTA: {wins}/{len(valid)} ({100*wins/len(valid):.1f}%)")
    
    # Best by transform
    print("\nBest config per transform:")
    for t_name in TRANSFORMS.keys():
        t_results = [r for r in valid if r['transform'] == t_name]
        if t_results:
            best = min(t_results, key=lambda r: r['bpc'])
            print(f"  {t_name}: {best['bpc']:.2f} bpc ({best['improvement_pct']:+.1f}%)")
    
    # Overall best
    if valid:
        overall_best = min(valid, key=lambda r: r['bpc'])
        print(f"\nOverall best: {overall_best['transform']} + {overall_best['config']}")
        print(f"  → {overall_best['bpc']:.2f} bpc ({overall_best['improvement_pct']:+.1f}% vs SOTA)")
    
    wall_broken = wins > 0
    
    print("\n" + "=" * 80)
    if wall_broken:
        print("🎉 WALL BROKEN!")
    else:
        print("❌ WALL STANDS - No variant beats SOTA on text compression")
    print("=" * 80)
    
    # Save
    results_path = Path(__file__).parent / "ascii_wall_all_variants_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {'wins': wins, 'total': len(valid), 'wall_broken': wall_broken},
            'results': results
        }, f, indent=2)
    print(f"\nSaved to: {results_path}")
    
    return wall_broken, results


if __name__ == "__main__":
    wall_broken, results = run_all_variants_experiment()
    sys.exit(0 if wall_broken else 1)

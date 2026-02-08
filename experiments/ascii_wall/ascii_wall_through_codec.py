#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
ASCII Wall Test Through Full Codec Pipeline
============================================

Tests whether the tetrahedral Δ-RFT transform, wired through the full
codec pipeline (quantization + ANS entropy coding), can beat SOTA
text compressors on ASCII data.

Hypothesis: The codec pipeline with proper entropy coding might reveal
compression gains that weren't visible in transform-only tests.

The test:
1. Take ASCII text samples
2. Encode through RFT Hybrid Codec with tetrahedral transforms
3. Measure actual compressed size (bits per character)
4. Compare against brotli, zstd, and the theoretical entropy
"""
from __future__ import annotations

import sys
import json
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import compression modules
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
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
# TETRAHEDRAL Δ-RFT TRANSFORMS
# ===========================================================================

PHI = (1.0 + 5.0 ** 0.5) / 2.0

# Regular tetrahedron vertices (normalized)
TETRA_VERTICES = np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]
], dtype=np.float64) / np.sqrt(3)


def _frac(arr: np.ndarray) -> np.ndarray:
    """Fractional part, handling negatives correctly."""
    frac, _ = np.modf(arr)
    return np.where(frac < 0.0, frac + 1.0, frac)


def delta_rft_forward(x: np.ndarray, vertex_idx: int = 0, 
                      beta: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """
    Tetrahedral Δ-RFT forward transform.
    
    Uses tetrahedral vertex to create 3D phase modulation:
    Δ_v = exp(2πi * v · [β₁·frac(k/φ), β₂·frac(k/φ²), β₃·frac(k/φ³)])
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    k = np.arange(n, dtype=np.float64)
    
    # Get tetrahedral vertex
    v = TETRA_VERTICES[vertex_idx % 4]
    
    # 3D phase components
    phase1 = beta * _frac(k / PHI)
    phase2 = beta * _frac(k / (PHI ** 2))
    phase3 = beta * _frac(k / (PHI ** 3))
    
    # Tetrahedral phase modulation
    theta = 2.0 * np.pi * (v[0] * phase1 + v[1] * phase2 + v[2] * phase3)
    D_tetra = np.exp(1j * theta)
    
    # Quadratic chirp
    ctheta = np.pi * sigma * (k * k / float(n))
    C_sig = np.exp(1j * ctheta)
    
    # Transform
    X = np.fft.fft(x, norm="ortho")
    return D_tetra * (C_sig * X)


def delta_rft_inverse(y: np.ndarray, vertex_idx: int = 0,
                      beta: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Tetrahedral Δ-RFT inverse transform."""
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


def cascade_3way_forward(x: np.ndarray) -> np.ndarray:
    """3-way cascade: standard RFT + Δ₁ + Δ₂ (our best performer from T6)."""
    # Standard Φ-RFT
    y = rft_forward(x)
    # Tetrahedral Δ₁ (vertex 0)
    y = delta_rft_forward(y, vertex_idx=0)
    # Tetrahedral Δ₂ (vertex 1)
    y = delta_rft_forward(y, vertex_idx=1)
    return y


def cascade_3way_inverse(y: np.ndarray) -> np.ndarray:
    """3-way cascade inverse."""
    # Reverse order
    x = delta_rft_inverse(y, vertex_idx=1)
    x = delta_rft_inverse(x, vertex_idx=0)
    x = rft_inverse(x)
    return x


# ===========================================================================
# FULL CODEC WITH TETRAHEDRAL TRANSFORMS
# ===========================================================================

@dataclass
class CodecResult:
    """Result from encoding through the full codec pipeline."""
    original_bytes: int
    compressed_bytes: int
    bits_per_char: float
    transform_time: float
    entropy_time: float
    roundtrip_ok: bool
    max_error: float


def encode_ascii_full_codec(
    text: str,
    quant_bits_amp: int = 8,
    quant_bits_phase: int = 6,
    use_cascade: bool = True,
    prune_threshold: float = 0.0
) -> CodecResult:
    """
    Encode ASCII text through full codec pipeline:
    1. Convert to float array
    2. Apply tetrahedral cascade transform (or standard RFT)
    3. Quantize amplitudes and phases
    4. Apply ANS entropy coding
    5. Measure compressed size
    """
    # Convert text to float array
    data = np.array([ord(c) for c in text], dtype=np.float64)
    n = len(data)
    
    t0 = time.perf_counter()
    
    # Apply transform
    if use_cascade:
        coeffs = cascade_3way_forward(data)
    else:
        coeffs = rft_forward(data)
    
    transform_time = time.perf_counter() - t0
    
    # Extract amplitude and phase
    amps = np.abs(coeffs)
    phases = np.angle(coeffs)
    log_amps = np.log(np.maximum(amps, 1e-12))
    
    # Pruning (optional)
    if prune_threshold > 0:
        keep_mask = amps >= prune_threshold
        kept_indices = np.nonzero(keep_mask)[0]
    else:
        kept_indices = np.arange(n)
        keep_mask = np.ones(n, dtype=bool)
    
    kept_log_amps = log_amps[kept_indices]
    kept_phases = phases[kept_indices]
    
    # Quantization
    log_min = float(kept_log_amps.min()) if kept_log_amps.size else 0.0
    log_max = float(kept_log_amps.max()) if kept_log_amps.size else 0.0
    phase_min, phase_max = -np.pi, np.pi
    
    # Quantize amplitudes
    amp_levels = (1 << quant_bits_amp) - 1
    if log_max > log_min:
        amp_codes = np.clip(
            np.rint((kept_log_amps - log_min) / (log_max - log_min) * amp_levels),
            0, amp_levels
        ).astype(np.int64)
    else:
        amp_codes = np.zeros(len(kept_log_amps), dtype=np.int64)
    
    # Quantize phases  
    phase_levels = (1 << quant_bits_phase) - 1
    phase_codes = np.clip(
        np.rint((kept_phases - phase_min) / (phase_max - phase_min) * phase_levels),
        0, phase_levels
    ).astype(np.int64)
    
    t1 = time.perf_counter()
    
    # ANS entropy coding
    total_compressed_bytes = 0
    
    # Encode amplitudes with ANS
    if amp_codes.size > 0:
        amp_encoded, amp_freq = ans_encode(amp_codes.tolist(), precision=RANS_PRECISION_DEFAULT)
        # Size = encoded data + frequency table overhead
        amp_bytes = amp_encoded.nbytes + len(json.dumps(amp_freq))
        total_compressed_bytes += amp_bytes
    else:
        amp_bytes = 0
    
    # Encode phases with ANS
    if phase_codes.size > 0:
        phase_encoded, phase_freq = ans_encode(phase_codes.tolist(), precision=RANS_PRECISION_DEFAULT)
        phase_bytes = phase_encoded.nbytes + len(json.dumps(phase_freq))
        total_compressed_bytes += phase_bytes
    else:
        phase_bytes = 0
    
    # If pruned, need to encode indices too
    if prune_threshold > 0:
        indices_bytes = kept_indices.astype(np.uint32).nbytes
        total_compressed_bytes += indices_bytes
    
    # Add metadata overhead (scale factors, etc.)
    metadata_bytes = 8 * 4  # 4 floats for scale parameters
    total_compressed_bytes += metadata_bytes
    
    entropy_time = time.perf_counter() - t1
    
    # Verify roundtrip
    # Dequantize
    if log_max > log_min:
        recon_log_amps = (amp_codes.astype(np.float64) / amp_levels) * (log_max - log_min) + log_min
    else:
        recon_log_amps = np.full_like(amp_codes, log_min, dtype=np.float64)
    
    recon_phases = (phase_codes.astype(np.float64) / phase_levels) * (phase_max - phase_min) + phase_min
    
    # Reconstruct coefficients
    recon_coeffs = np.zeros(n, dtype=np.complex128)
    recon_amps = np.exp(recon_log_amps)
    recon_coeffs[kept_indices] = recon_amps * np.exp(1j * recon_phases)
    
    # Inverse transform
    if use_cascade:
        recon_data = cascade_3way_inverse(recon_coeffs)
    else:
        recon_data = rft_inverse(recon_coeffs)
    
    recon_data = np.real(recon_data)
    max_error = float(np.max(np.abs(recon_data - data)))
    roundtrip_ok = max_error < 1.0  # Within 1 character code
    
    original_bytes = len(text)  # 1 byte per ASCII char
    bits_per_char = (total_compressed_bytes * 8) / n
    
    return CodecResult(
        original_bytes=original_bytes,
        compressed_bytes=total_compressed_bytes,
        bits_per_char=bits_per_char,
        transform_time=transform_time,
        entropy_time=entropy_time,
        roundtrip_ok=roundtrip_ok,
        max_error=max_error
    )


def compress_sota(text: str, method: str) -> Tuple[int, float]:
    """Compress with SOTA method, return (compressed_bytes, bits_per_char)."""
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
    """Calculate Shannon entropy in bits per character."""
    freq = Counter(text)
    n = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ===========================================================================
# TEST DATA
# ===========================================================================

def generate_test_texts() -> Dict[str, str]:
    """Generate various test texts for ASCII wall testing."""
    texts = {}
    
    # English prose
    texts['english_prose'] = """
    The golden ratio, often denoted by the Greek letter phi, appears throughout
    nature and mathematics. From the spiral patterns of sunflower seeds to the
    proportions of the Parthenon, this irrational number approximately equal to
    1.618 has fascinated mathematicians for millennia. The Fibonacci sequence,
    where each number is the sum of the two preceding ones, exhibits a remarkable
    property: the ratio of consecutive Fibonacci numbers converges to phi.
    """ * 10
    
    # Code
    texts['python_code'] = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def golden_ratio_approx(n):
    fib_n = fibonacci(n)
    fib_n_minus_1 = fibonacci(n - 1)
    return fib_n / fib_n_minus_1 if fib_n_minus_1 else float('inf')

class GoldenSpiral:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.phi = (1 + 5 ** 0.5) / 2
    
    def point(self, theta):
        r = self.scale * (self.phi ** (2 * theta / 3.14159))
        return r * cos(theta), r * sin(theta)
""" * 5
    
    # JSON-like structured data
    texts['json_data'] = json.dumps({
        "experiments": [
            {"id": i, "name": f"test_{i}", "value": i * 1.618, "status": "complete"}
            for i in range(100)
        ],
        "metadata": {"version": "1.0", "format": "json", "encoding": "utf-8"}
    }, indent=2)
    
    # Scientific text (ASCII only)
    texts['scientific'] = """
    The quantum harmonic oscillator represents one of the most important model
    systems in quantum mechanics. Its Hamiltonian is given by H = p^2/2m + mw^2x^2/2,
    where p is momentum, m is mass, w is angular frequency, and x is position.
    The energy eigenvalues are E_n = hbar*w*(n + 1/2) where n = 0, 1, 2, ...
    This quantization of energy levels leads to the zero-point energy hbar*w/2,
    a purely quantum mechanical phenomenon with no classical analog.
    """ * 10
    
    # Random printable ASCII
    rng = np.random.default_rng(42)
    chars = [chr(c) for c in range(32, 127)]  # Printable ASCII
    texts['random_ascii'] = ''.join(rng.choice(chars, size=2000))
    
    # Highly repetitive
    texts['repetitive'] = "the quick brown fox jumps over the lazy dog " * 100
    
    # Base64-encoded data (simulates compressed/encrypted content)
    import base64
    random_bytes = rng.integers(0, 256, size=500, dtype=np.uint8).tobytes()
    texts['base64_data'] = base64.b64encode(random_bytes).decode('ascii')
    
    return texts


# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================

def run_ascii_wall_experiment():
    """Run the full ASCII wall experiment through codec pipeline."""
    print("=" * 80)
    print("ASCII WALL TEST THROUGH FULL CODEC PIPELINE")
    print("=" * 80)
    print()
    print("Testing whether tetrahedral Δ-RFT + ANS entropy coding can beat SOTA")
    print()
    
    texts = generate_test_texts()
    results = []
    
    # Quantization configurations to test
    quant_configs = [
        (8, 6, "8-bit amp, 6-bit phase"),
        (10, 8, "10-bit amp, 8-bit phase"),
        (12, 8, "12-bit amp, 8-bit phase (higher precision)"),
    ]
    
    for text_name, text in texts.items():
        print(f"\n{'─' * 60}")
        print(f"Text: {text_name} ({len(text)} chars)")
        print(f"{'─' * 60}")
        
        # Calculate theoretical entropy
        entropy = calculate_entropy(text)
        print(f"Shannon entropy: {entropy:.3f} bpc")
        
        # SOTA baselines
        print("\nSOTA Compressors:")
        sota_results = {}
        for method in ['gzip', 'bz2', 'lzma']:
            _, bpc = compress_sota(text, method)
            sota_results[method] = bpc
            print(f"  {method}: {bpc:.3f} bpc")
        
        if HAS_ZSTD:
            _, bpc = compress_sota(text, 'zstd')
            sota_results['zstd'] = bpc
            print(f"  zstd-22: {bpc:.3f} bpc")
        
        if HAS_BROTLI:
            _, bpc = compress_sota(text, 'brotli')
            sota_results['brotli'] = bpc
            print(f"  brotli-11: {bpc:.3f} bpc")
        
        best_sota = min(sota_results.values())
        best_sota_name = min(sota_results.keys(), key=lambda k: sota_results[k])
        
        # RFT Codec tests
        print("\nRFT Codec (with tetrahedral cascade):")
        
        for amp_bits, phase_bits, desc in quant_configs:
            # With cascade
            result = encode_ascii_full_codec(
                text, 
                quant_bits_amp=amp_bits,
                quant_bits_phase=phase_bits,
                use_cascade=True
            )
            
            improvement = (best_sota - result.bits_per_char) / best_sota * 100
            marker = "✓ BETTER" if result.bits_per_char < best_sota else "✗"
            
            print(f"  Cascade {desc}: {result.bits_per_char:.3f} bpc "
                  f"({improvement:+.1f}% vs {best_sota_name}) {marker}")
            
            if not result.roundtrip_ok:
                print(f"    ⚠ Roundtrip error: {result.max_error:.2f}")
            
            results.append({
                'text': text_name,
                'method': f'cascade_{amp_bits}_{phase_bits}',
                'bpc': result.bits_per_char,
                'best_sota': best_sota,
                'improvement_pct': improvement,
                'roundtrip_ok': result.roundtrip_ok,
                'max_error': result.max_error
            })
        
        # Without cascade (baseline standard RFT)
        result_std = encode_ascii_full_codec(
            text,
            quant_bits_amp=10,
            quant_bits_phase=8,
            use_cascade=False
        )
        
        improvement_std = (best_sota - result_std.bits_per_char) / best_sota * 100
        marker = "✓ BETTER" if result_std.bits_per_char < best_sota else "✗"
        
        print(f"\n  Standard RFT (10/8): {result_std.bits_per_char:.3f} bpc "
              f"({improvement_std:+.1f}% vs {best_sota_name}) {marker}")
        
        results.append({
            'text': text_name,
            'method': 'standard_rft_10_8',
            'bpc': result_std.bits_per_char,
            'best_sota': best_sota,
            'improvement_pct': improvement_std,
            'roundtrip_ok': result_std.roundtrip_ok,
            'max_error': result_std.max_error
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: ASCII WALL TEST RESULTS")
    print("=" * 80)
    
    wins = sum(1 for r in results if r['improvement_pct'] > 0)
    total = len(results)
    
    print(f"\nWins against best SOTA: {wins}/{total} ({100*wins/total:.1f}%)")
    
    # Best cases
    best_results = sorted(results, key=lambda r: r['improvement_pct'], reverse=True)[:5]
    print("\nBest performing cases:")
    for r in best_results:
        print(f"  {r['text']} + {r['method']}: {r['improvement_pct']:+.1f}%")
    
    # Worst cases
    worst_results = sorted(results, key=lambda r: r['improvement_pct'])[:5]
    print("\nWorst performing cases:")
    for r in worst_results:
        print(f"  {r['text']} + {r['method']}: {r['improvement_pct']:+.1f}%")
    
    # Average gap
    avg_gap = np.mean([r['improvement_pct'] for r in results])
    print(f"\nAverage gap from SOTA: {avg_gap:+.1f}%")
    
    # Did we break the wall?
    wall_broken = wins > 0 and avg_gap > 0
    
    print("\n" + "=" * 80)
    if wall_broken:
        print("🎉 WALL BROKEN: RFT codec beats SOTA on some text types!")
    else:
        print("❌ WALL STANDS: RFT codec does not beat SOTA text compressors")
    print("=" * 80)
    
    # Save results
    results_path = Path(__file__).parent / "ascii_wall_codec_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'wins': wins,
                'total': total,
                'win_rate': wins / total if total > 0 else 0,
                'avg_gap_pct': float(avg_gap),
                'wall_broken': wall_broken
            },
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return wall_broken, results


if __name__ == "__main__":
    wall_broken, results = run_ascii_wall_experiment()
    sys.exit(0 if wall_broken else 1)

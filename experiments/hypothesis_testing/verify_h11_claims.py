#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Verify H11 Claims - Round-trip, Fairness, and Reconstruction Error
===================================================================

This script verifies:
1. Round-trip integrity - Can we decompress and get identical bytes back?
2. Comparison fairness - Are we comparing against Brotli at maximum compression?
3. Pruning math - What's the actual reconstruction error at 90% prune?
"""
from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
from algorithms.rft.compression.ans import ans_encode, ans_decode

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
# H11 CODEC WITH FULL ROUND-TRIP SUPPORT
# ===========================================================================

class OrderKPredictor:
    """Order-k context model for byte prediction."""
    
    def __init__(self, order: int = 2):
        self.order = order
        self.context_counts: Dict[bytes, Counter] = {}
        
    def train(self, data: bytes) -> None:
        """Build context model from training data."""
        for i in range(self.order, len(data)):
            context = data[i - self.order:i]
            symbol = data[i]
            if context not in self.context_counts:
                self.context_counts[context] = Counter()
            self.context_counts[context][symbol] += 1
    
    def predict(self, context: bytes) -> int:
        """Predict next byte given context."""
        if context in self.context_counts:
            return self.context_counts[context].most_common(1)[0][0]
        return 0
    
    def compute_residuals(self, data: bytes) -> np.ndarray:
        """Compute prediction residuals."""
        residuals = np.zeros(len(data), dtype=np.float64)
        
        for i in range(min(self.order, len(data))):
            residuals[i] = float(data[i]) - 128.0
        
        for i in range(self.order, len(data)):
            context = data[i - self.order:i]
            pred = self.predict(context)
            res = (data[i] - pred + 128) % 256 - 128
            residuals[i] = float(res)
        
        return residuals
    
    def reconstruct_from_residuals(self, residuals: np.ndarray) -> bytes:
        """Reconstruct data from residuals."""
        result = bytearray(len(residuals))
        
        for i in range(min(self.order, len(residuals))):
            result[i] = int(np.clip(np.round(residuals[i] + 128), 0, 255))
        
        for i in range(self.order, len(residuals)):
            context = bytes(result[i - self.order:i])
            pred = self.predict(context)
            res = int(np.round(residuals[i]))
            result[i] = (res + pred + 128) % 256
        
        return bytes(result)


def h11_encode_decode_roundtrip(
    text: str,
    predictor_order: int = 3,
    block_size: int = 256,
    mag_bits: int = 8,
    phase_bits: int = 6,
    prune_ratio: float = 0.9
) -> Tuple[bytes, bytes, dict]:
    """
    Full encode-decode roundtrip for H11.
    
    Returns:
        (original_bytes, reconstructed_bytes, stats_dict)
    """
    original_data = text.encode('utf-8')
    n = len(original_data)
    
    # Train predictor
    predictor = OrderKPredictor(order=predictor_order)
    predictor.train(original_data)
    
    # Compute residuals
    residuals = predictor.compute_residuals(original_data)
    
    # Store residual stats for reconstruction
    residual_min = residuals.min()
    residual_max = residuals.max()
    residual_range = residual_max - residual_min + 1e-10
    
    # Pad to block boundary
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        padded_residuals = np.concatenate([residuals, np.zeros(pad_len)])
    else:
        padded_residuals = residuals.copy()
    
    num_blocks = len(padded_residuals) // block_size
    
    # Process blocks - store ALL info needed for reconstruction
    all_coeffs = []
    all_mags = []
    all_phases = []
    all_masks = []  # Which coefficients are kept
    mag_maxes = []
    
    for b in range(num_blocks):
        block = padded_residuals[b * block_size:(b + 1) * block_size]
        
        # RFT forward
        coeffs = rft_forward(block)
        mags = np.abs(coeffs)
        phases = np.angle(coeffs)
        
        # Prune - keep top (1-prune_ratio) coefficients
        if prune_ratio > 0:
            threshold = np.percentile(mags, prune_ratio * 100)
            mask = mags >= threshold
        else:
            mask = np.ones(len(mags), dtype=bool)
        
        # Store mask and values
        all_masks.append(mask)
        mag_maxes.append(mags.max() + 1e-10)
        
        # Quantize
        mags_norm = mags / mag_maxes[-1]
        mags_q = np.clip((mags_norm * (2**mag_bits - 1)).astype(np.int32), 0, 2**mag_bits - 1)
        phases_norm = (phases + np.pi) / (2 * np.pi)
        phases_q = np.clip((phases_norm * (2**phase_bits - 1)).astype(np.int32), 0, 2**phase_bits - 1)
        
        # Apply mask (zero out pruned)
        mags_q = mags_q * mask.astype(np.int32)
        phases_q = phases_q * mask.astype(np.int32)
        
        all_mags.append(mags_q)
        all_phases.append(phases_q)
        all_coeffs.append(coeffs)
    
    # === DECODE ===
    reconstructed_residuals = np.zeros_like(padded_residuals)
    
    for b in range(num_blocks):
        # Dequantize
        mags_q = all_mags[b]
        phases_q = all_phases[b]
        mag_max = mag_maxes[b]
        mask = all_masks[b]
        
        # Dequantize magnitudes
        mags_deq = (mags_q.astype(np.float64) / (2**mag_bits - 1)) * mag_max
        
        # Dequantize phases
        phases_deq = (phases_q.astype(np.float64) / (2**phase_bits - 1)) * 2 * np.pi - np.pi
        
        # Reconstruct complex coefficients
        coeffs_recon = mags_deq * np.exp(1j * phases_deq)
        
        # RFT inverse
        block_recon = np.real(rft_inverse(coeffs_recon))
        reconstructed_residuals[b * block_size:(b + 1) * block_size] = block_recon
    
    # Trim padding
    reconstructed_residuals = reconstructed_residuals[:n]
    
    # Reconstruct bytes from residuals
    reconstructed_data = predictor.reconstruct_from_residuals(reconstructed_residuals)
    
    # Calculate reconstruction error
    residual_mse = np.mean((residuals - reconstructed_residuals[:n]) ** 2)
    residual_max_error = np.max(np.abs(residuals - reconstructed_residuals[:n]))
    
    byte_errors = sum(1 for a, b in zip(original_data, reconstructed_data) if a != b)
    exact_match = original_data == reconstructed_data
    
    # Calculate actual compressed size
    all_mags_flat = np.concatenate(all_mags).tolist()
    all_phases_flat = np.concatenate(all_phases).tolist()
    
    try:
        mag_encoded, mag_freq = ans_encode(all_mags_flat)
        phase_encoded, phase_freq = ans_encode(all_phases_flat)
        mag_bytes = mag_encoded.tobytes()
        phase_bytes = phase_encoded.tobytes()
        
        # Header overhead (simplified)
        header_size = 20  # n, order, block_size, mag_maxes, etc.
        total_compressed = header_size + len(mag_bytes) + len(phase_bytes)
    except:
        total_compressed = len(original_data)  # Fallback
    
    stats = {
        'original_bytes': n,
        'compressed_bytes': total_compressed,
        'bpc': total_compressed * 8 / n,
        'exact_roundtrip': exact_match,
        'byte_errors': byte_errors,
        'byte_error_rate': byte_errors / n,
        'residual_mse': float(residual_mse),
        'residual_max_error': float(residual_max_error),
        'coefficients_kept': float(1 - prune_ratio),
        'predictor_order': predictor_order,
        'prune_ratio': prune_ratio,
    }
    
    return original_data, reconstructed_data, stats


def get_sota_all_levels(data: bytes) -> Dict[str, Dict]:
    """Get SOTA compressed sizes at ALL compression levels."""
    results = {}
    
    # gzip levels 1-9
    for level in [1, 6, 9]:
        size = len(gzip.compress(data, compresslevel=level))
        results[f'gzip-{level}'] = {'size': size, 'bpc': size * 8 / len(data)}
    
    # bz2 levels 1-9
    for level in [1, 5, 9]:
        size = len(bz2.compress(data, compresslevel=level))
        results[f'bz2-{level}'] = {'size': size, 'bpc': size * 8 / len(data)}
    
    # lzma presets 0-9
    for preset in [0, 6, 9]:
        size = len(lzma.compress(data, preset=preset))
        results[f'lzma-{preset}'] = {'size': size, 'bpc': size * 8 / len(data)}
    
    if HAS_ZSTD:
        for level in [1, 11, 22]:  # 22 is max
            cctx = zstd.ZstdCompressor(level=level)
            size = len(cctx.compress(data))
            results[f'zstd-{level}'] = {'size': size, 'bpc': size * 8 / len(data)}
    
    if HAS_BROTLI:
        for quality in [1, 6, 11]:  # 11 is max
            size = len(brotli.compress(data, quality=quality))
            results[f'brotli-{quality}'] = {'size': size, 'bpc': size * 8 / len(data)}
    
    return results


# ===========================================================================
# TEST TEXTS
# ===========================================================================

TEST_TEXTS = {
    'english_prose': """
The quick brown fox jumps over the lazy dog. In the beginning, there was nothing 
but darkness and void. Then came light, spreading across the cosmos like ripples 
in a vast ocean of space and time. Stars were born, lived their lives of nuclear 
fusion, and died in spectacular explosions that seeded the universe with heavy 
elements. From these elements, planets formed, and on at least one of them, life 
emerged.
""".strip(),

    'python_code': '''
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr

class DataProcessor:
    def __init__(self, data: list):
        self.data = data
        self._processed = False
    
    def process(self) -> list:
        if self._processed:
            return self.data
        result = [item * 2 if isinstance(item, (int, float)) else item.upper() 
                  for item in self.data]
        self._processed = True
        return result
'''.strip(),

    'json_data': '''{
    "name": "QuantoniumOS",
    "version": "2.0.0",
    "config": {
        "block_size": 256,
        "transforms": ["rft", "dct"]
    }
}''',

    'random_ascii': ''.join(chr(np.random.randint(32, 127)) for _ in range(2000)),

    'repetitive': 'Hello World! ' * 50 + 'The quick brown fox. ' * 30,
}


# ===========================================================================
# MAIN VERIFICATION
# ===========================================================================

def main():
    print("=" * 80)
    print("H11 CLAIMS VERIFICATION")
    print("=" * 80)
    
    results = []
    
    for text_name, text in TEST_TEXTS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {text_name} ({len(text)} chars)")
        print("=" * 60)
        
        data = text.encode('utf-8')
        
        # 1. Get SOTA at ALL levels
        print("\n1. SOTA Compression at ALL Levels:")
        sota = get_sota_all_levels(data)
        best_sota_name = min(sota, key=lambda k: sota[k]['bpc'])
        best_sota_bpc = sota[best_sota_name]['bpc']
        
        print(f"   Best SOTA: {best_sota_name} @ {best_sota_bpc:.4f} bpc")
        print(f"   All levels:")
        for name, info in sorted(sota.items(), key=lambda x: x[1]['bpc']):
            print(f"      {name}: {info['bpc']:.4f} bpc ({info['size']} bytes)")
        
        # 2. Test H11 at different prune levels
        print("\n2. H11 Round-Trip Tests:")
        
        for prune in [0.0, 0.5, 0.9, 0.95, 0.99]:
            for order in [1, 3]:
                orig, recon, stats = h11_encode_decode_roundtrip(
                    text, predictor_order=order, prune_ratio=prune
                )
                
                status = "✓ EXACT" if stats['exact_roundtrip'] else f"✗ {stats['byte_errors']} errors"
                vs_sota = (stats['bpc'] - best_sota_bpc) / best_sota_bpc * 100
                
                print(f"   order={order}, prune={prune:.0%}: {stats['bpc']:.4f} bpc "
                      f"(vs {best_sota_name}: {vs_sota:+.1f}%) | "
                      f"Roundtrip: {status} | "
                      f"Max residual error: {stats['residual_max_error']:.2f}")
                
                results.append({
                    'text': text_name,
                    'order': order,
                    'prune': prune,
                    'bpc': stats['bpc'],
                    'best_sota': best_sota_name,
                    'best_sota_bpc': best_sota_bpc,
                    'vs_sota_pct': vs_sota,
                    'exact_roundtrip': stats['exact_roundtrip'],
                    'byte_errors': stats['byte_errors'],
                    'byte_error_rate': stats['byte_error_rate'],
                    'residual_max_error': stats['residual_max_error'],
                })
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    # Count exact roundtrips
    exact_count = sum(1 for r in results if r['exact_roundtrip'])
    print(f"\n1. ROUND-TRIP INTEGRITY:")
    print(f"   Exact roundtrips: {exact_count}/{len(results)}")
    
    lossy_results = [r for r in results if not r['exact_roundtrip']]
    if lossy_results:
        print(f"   ⚠️  LOSSY configurations (pruning causes byte errors):")
        for r in lossy_results[:5]:
            print(f"      {r['text']}, order={r['order']}, prune={r['prune']:.0%}: "
                  f"{r['byte_errors']} byte errors ({r['byte_error_rate']:.1%})")
    
    # Count SOTA wins
    wins = [r for r in results if r['vs_sota_pct'] < 0]
    print(f"\n2. COMPARISON FAIRNESS (vs best SOTA at max level):")
    print(f"   Wins vs SOTA: {len(wins)}/{len(results)}")
    
    if wins:
        print(f"   Winning configurations:")
        for r in sorted(wins, key=lambda x: x['vs_sota_pct'])[:10]:
            rt = "lossless" if r['exact_roundtrip'] else f"lossy ({r['byte_errors']} errs)"
            print(f"      {r['text']}, order={r['order']}, prune={r['prune']:.0%}: "
                  f"{r['bpc']:.4f} vs {r['best_sota_bpc']:.4f} bpc ({r['vs_sota_pct']:+.1f}%) [{rt}]")
    
    # Prune analysis
    print(f"\n3. PRUNING ANALYSIS (90% prune):")
    prune_90 = [r for r in results if r['prune'] == 0.9]
    for r in prune_90:
        print(f"   {r['text']}: max_error={r['residual_max_error']:.2f}, "
              f"byte_errors={r['byte_errors']}, exact={r['exact_roundtrip']}")
    
    # Final verdict
    print("\n" + "=" * 80)
    lossless_wins = [r for r in results if r['vs_sota_pct'] < 0 and r['exact_roundtrip']]
    lossy_wins = [r for r in results if r['vs_sota_pct'] < 0 and not r['exact_roundtrip']]
    
    print("FINAL VERDICT:")
    print(f"   Lossless wins vs SOTA: {len(lossless_wins)}")
    print(f"   Lossy wins vs SOTA: {len(lossy_wins)}")
    
    if lossless_wins:
        print("\n   ✓ LEGITIMATE WINS (lossless):")
        for r in lossless_wins:
            print(f"      {r['text']}, order={r['order']}, prune={r['prune']:.0%}: "
                  f"{r['bpc']:.4f} vs {r['best_sota_bpc']:.4f} ({r['vs_sota_pct']:+.1f}%)")
    else:
        print("\n   ✗ NO LOSSLESS WINS - All improvements require lossy compression")
    
    # Save results
    output_path = Path(__file__).parent / 'verify_h11_claims_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': len(results),
                'exact_roundtrips': exact_count,
                'wins_vs_sota': len(wins),
                'lossless_wins': len(lossless_wins),
                'lossy_wins': len(lossy_wins),
            },
            'results': results
        }, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()

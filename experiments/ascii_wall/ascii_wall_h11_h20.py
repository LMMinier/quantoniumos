#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
ASCII Wall Test - H11-H20 Hypotheses
=====================================

Testing advanced RFT compression strategies:
- H11: Predictive Residual RFT (strip structure first)
- H12: Symbol Embedding RFT (continuous embedding space)
- H13: Block-Adaptive RFT (local entropy gating)
- H18: Context-Gated RFT (skip on random-like regions)
"""
from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
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


PHI = (1.0 + 5.0 ** 0.5) / 2.0


# ===========================================================================
# H11: PREDICTIVE RESIDUAL RFT
# ===========================================================================

class OrderKPredictor:
    """Simple order-k context model for byte prediction."""
    
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
        """Predict next byte given context. Returns most likely byte or 0."""
        if context in self.context_counts:
            return self.context_counts[context].most_common(1)[0][0]
        return 0  # Default prediction
    
    def compute_residuals(self, data: bytes) -> np.ndarray:
        """Compute prediction residuals for data."""
        residuals = np.zeros(len(data), dtype=np.float64)
        
        # First 'order' bytes are stored as-is (shifted to signed)
        for i in range(min(self.order, len(data))):
            residuals[i] = float(data[i]) - 128.0
        
        # Rest are residuals
        for i in range(self.order, len(data)):
            context = data[i - self.order:i]
            pred = self.predict(context)
            # Signed residual in [-128, 127]
            res = (data[i] - pred + 128) % 256 - 128
            residuals[i] = float(res)
        
        return residuals
    
    def reconstruct(self, residuals: np.ndarray) -> bytes:
        """Reconstruct data from residuals."""
        result = bytearray(len(residuals))
        
        # First 'order' bytes
        for i in range(min(self.order, len(residuals))):
            result[i] = int(np.clip(np.round(residuals[i] + 128), 0, 255))
        
        # Build context as we go
        for i in range(self.order, len(residuals)):
            context = bytes(result[i - self.order:i])
            pred = self.predict(context)
            # Reconstruct: byte = (residual + pred) mod 256
            res = int(np.round(residuals[i]))
            result[i] = (res + pred + 128) % 256
        
        return bytes(result)


def h11_predictive_residual_encode(
    text: str,
    predictor_order: int = 2,
    block_size: int = 256,
    mag_bits: int = 8,
    phase_bits: int = 6,
    prune_ratio: float = 0.0
) -> Tuple[bytes, dict]:
    """
    H11: Predictive Residual RFT Codec
    
    1. Train order-k predictor on the text
    2. Compute residuals
    3. RFT on residual blocks
    4. Quantize + ANS
    """
    data = text.encode('utf-8')
    n = len(data)
    
    # Train predictor on itself (in practice, use external corpus)
    predictor = OrderKPredictor(order=predictor_order)
    predictor.train(data)
    
    # Compute residuals
    residuals = predictor.compute_residuals(data)
    
    # Stats on residuals
    residual_entropy = _estimate_entropy(residuals)
    
    # Pad to block boundary
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        residuals = np.concatenate([residuals, np.zeros(pad_len)])
    
    num_blocks = len(residuals) // block_size
    
    # Process blocks
    all_mags_q = []
    all_phases_q = []
    
    for b in range(num_blocks):
        block = residuals[b * block_size:(b + 1) * block_size]
        
        # RFT forward
        coeffs = rft_forward(block)
        
        # Separate magnitude and phase
        mags = np.abs(coeffs)
        phases = np.angle(coeffs)
        
        # Prune small coefficients
        if prune_ratio > 0:
            threshold = np.percentile(mags, prune_ratio * 100)
            mask = mags >= threshold
            mags = mags * mask
            phases = phases * mask
        
        # Quantize
        mag_max = mags.max() + 1e-10
        mags_norm = mags / mag_max
        mags_q = np.clip((mags_norm * (2**mag_bits - 1)).astype(np.int32), 0, 2**mag_bits - 1)
        
        phases_norm = (phases + np.pi) / (2 * np.pi)
        phases_q = np.clip((phases_norm * (2**phase_bits - 1)).astype(np.int32), 0, 2**phase_bits - 1)
        
        all_mags_q.extend(mags_q.tolist())
        all_phases_q.extend(phases_q.tolist())
    
    # ANS encode
    try:
        mag_encoded, mag_freq = ans_encode(all_mags_q)
        phase_encoded, phase_freq = ans_encode(all_phases_q)
        
        # Convert numpy arrays to bytes
        mag_bytes = mag_encoded.tobytes() if hasattr(mag_encoded, 'tobytes') else bytes(mag_encoded)
        phase_bytes = phase_encoded.tobytes() if hasattr(phase_encoded, 'tobytes') else bytes(phase_encoded)
        
        # Header: original length, predictor order, block size, num_blocks
        header = n.to_bytes(4, 'big') + predictor_order.to_bytes(1, 'big')
        payload = header + mag_bytes + phase_bytes
        
        return payload, {
            'original_bytes': n,
            'payload_bytes': len(payload),
            'residual_entropy': residual_entropy,
            'num_blocks': num_blocks
        }
    except Exception as e:
        return b'', {'error': str(e)}


# ===========================================================================
# H13: BLOCK-ADAPTIVE RFT
# ===========================================================================

def _block_entropy(block: np.ndarray) -> float:
    """Estimate entropy of a block."""
    # Quantize to bins for entropy estimation
    bins = np.clip((block - block.min()) / (block.max() - block.min() + 1e-10) * 256, 0, 255).astype(int)
    counts = np.bincount(bins, minlength=256)
    probs = counts[counts > 0] / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))


def _block_variance(block: np.ndarray) -> float:
    """Compute variance of a block."""
    return float(np.var(block))


def h13_block_adaptive_encode(
    text: str,
    block_size: int = 128,
    entropy_threshold_low: float = 2.0,   # Very redundant
    entropy_threshold_high: float = 7.0,  # Near random
    mag_bits: int = 8,
    phase_bits: int = 6,
    prune_low: float = 0.5,   # Heavy prune for redundant
    prune_mid: float = 0.1    # Light prune for structured
) -> Tuple[bytes, dict]:
    """
    H13: Block-Adaptive RFT
    
    Per block:
    - Low entropy (redundant): short RFT + heavy prune
    - High entropy (random): skip RFT, raw ANS
    - Mid entropy (structured): normal RFT + light prune
    """
    data = text.encode('utf-8')
    n = len(data)
    
    # Convert to float
    signal = np.array([float(b) for b in data], dtype=np.float64)
    
    # Pad
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        signal = np.concatenate([signal, np.zeros(pad_len)])
    
    num_blocks = len(signal) // block_size
    
    # Mode counts
    mode_counts = {'redundant': 0, 'random': 0, 'structured': 0}
    
    all_symbols = []  # Will include mode flags + data
    
    for b in range(num_blocks):
        block = signal[b * block_size:(b + 1) * block_size]
        ent = _block_entropy(block)
        
        if ent < entropy_threshold_low:
            # REDUNDANT: heavy prune RFT
            mode_counts['redundant'] += 1
            mode_flag = 0
            
            coeffs = rft_forward(block)
            mags = np.abs(coeffs)
            
            # Heavy prune - keep only top 50%
            threshold = np.percentile(mags, prune_low * 100)
            mask = mags >= threshold
            
            # Store only non-zero indices and values
            nonzero_idx = np.where(mask)[0]
            nonzero_mags = mags[mask]
            nonzero_phases = np.angle(coeffs)[mask]
            
            # Quantize
            if len(nonzero_mags) > 0:
                mag_max = nonzero_mags.max() + 1e-10
                mags_q = np.clip((nonzero_mags / mag_max * (2**mag_bits - 1)).astype(int), 0, 2**mag_bits - 1)
                phases_q = np.clip(((nonzero_phases + np.pi) / (2 * np.pi) * (2**phase_bits - 1)).astype(int), 0, 2**phase_bits - 1)
                
                # Encode: mode, count, indices, mags, phases
                all_symbols.append(mode_flag)
                all_symbols.append(len(nonzero_idx))
                all_symbols.extend(nonzero_idx.tolist())
                all_symbols.extend(mags_q.tolist())
                all_symbols.extend(phases_q.tolist())
            else:
                all_symbols.append(mode_flag)
                all_symbols.append(0)
        
        elif ent > entropy_threshold_high:
            # RANDOM: skip RFT, just store raw
            mode_counts['random'] += 1
            mode_flag = 1
            
            raw_q = np.clip(block.astype(int), 0, 255)
            all_symbols.append(mode_flag)
            all_symbols.extend(raw_q.tolist())
        
        else:
            # STRUCTURED: normal RFT with light prune
            mode_counts['structured'] += 1
            mode_flag = 2
            
            coeffs = rft_forward(block)
            mags = np.abs(coeffs)
            phases = np.angle(coeffs)
            
            # Light prune
            if prune_mid > 0:
                threshold = np.percentile(mags, prune_mid * 100)
                mask = mags >= threshold
                mags = mags * mask
                phases = phases * mask
            
            # Quantize all
            mag_max = mags.max() + 1e-10
            mags_q = np.clip((mags / mag_max * (2**mag_bits - 1)).astype(int), 0, 2**mag_bits - 1)
            phases_q = np.clip(((phases + np.pi) / (2 * np.pi) * (2**phase_bits - 1)).astype(int), 0, 2**phase_bits - 1)
            
            all_symbols.append(mode_flag)
            all_symbols.extend(mags_q.tolist())
            all_symbols.extend(phases_q.tolist())
    
    # ANS encode
    try:
        # Clamp symbols to valid range
        all_symbols = [max(0, min(s, 65535)) for s in all_symbols]
        encoded, freq_data = ans_encode(all_symbols)
        
        # Convert to bytes
        encoded_bytes = encoded.tobytes() if hasattr(encoded, 'tobytes') else bytes(encoded)
        
        header = n.to_bytes(4, 'big')
        payload = header + encoded_bytes
        
        return payload, {
            'original_bytes': n,
            'payload_bytes': len(payload),
            'mode_counts': mode_counts,
            'num_blocks': num_blocks
        }
    except Exception as e:
        return b'', {'error': str(e)}


# ===========================================================================
# H18: CONTEXT-GATED RFT (Randomness Detection)
# ===========================================================================

def _spectral_flatness(coeffs: np.ndarray) -> float:
    """Compute spectral flatness (0=tonal, 1=white noise)."""
    mags = np.abs(coeffs) + 1e-10
    geo_mean = np.exp(np.mean(np.log(mags)))
    arith_mean = np.mean(mags)
    return geo_mean / arith_mean


def _energy_concentration(coeffs: np.ndarray, top_k_ratio: float = 0.1) -> float:
    """What fraction of energy is in top k% of coefficients."""
    mags = np.abs(coeffs)
    total_energy = np.sum(mags ** 2)
    if total_energy < 1e-10:
        return 0.0
    
    sorted_mags = np.sort(mags)[::-1]
    k = max(1, int(len(mags) * top_k_ratio))
    top_energy = np.sum(sorted_mags[:k] ** 2)
    
    return top_energy / total_energy


def h18_context_gated_encode(
    text: str,
    block_size: int = 128,
    flatness_threshold: float = 0.8,  # Above = random-like
    concentration_threshold: float = 0.5,  # Below = random-like
    mag_bits: int = 8,
    phase_bits: int = 6,
    prune_ratio: float = 0.1
) -> Tuple[bytes, dict]:
    """
    H18: Context-Gated RFT
    
    Use spectral analysis to detect random-like regions and skip RFT there.
    """
    data = text.encode('utf-8')
    n = len(data)
    
    signal = np.array([float(b) for b in data], dtype=np.float64)
    
    # Pad
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        signal = np.concatenate([signal, np.zeros(pad_len)])
    
    num_blocks = len(signal) // block_size
    
    rft_blocks = 0
    raw_blocks = 0
    
    all_symbols = []
    
    for b in range(num_blocks):
        block = signal[b * block_size:(b + 1) * block_size]
        
        # Quick RFT to analyze structure
        coeffs = rft_forward(block)
        flatness = _spectral_flatness(coeffs)
        concentration = _energy_concentration(coeffs)
        
        # Decide: is this random-like?
        is_random = (flatness > flatness_threshold) or (concentration < concentration_threshold)
        
        if is_random:
            # Skip RFT, raw encode
            raw_blocks += 1
            all_symbols.append(0)  # Mode: raw
            raw_q = np.clip(block.astype(int), 0, 255)
            all_symbols.extend(raw_q.tolist())
        else:
            # Use RFT
            rft_blocks += 1
            all_symbols.append(1)  # Mode: RFT
            
            mags = np.abs(coeffs)
            phases = np.angle(coeffs)
            
            # Prune
            if prune_ratio > 0:
                threshold = np.percentile(mags, prune_ratio * 100)
                mask = mags >= threshold
                mags = mags * mask
                phases = phases * mask
            
            # Quantize
            mag_max = mags.max() + 1e-10
            mags_q = np.clip((mags / mag_max * (2**mag_bits - 1)).astype(int), 0, 2**mag_bits - 1)
            phases_q = np.clip(((phases + np.pi) / (2 * np.pi) * (2**phase_bits - 1)).astype(int), 0, 2**phase_bits - 1)
            
            all_symbols.extend(mags_q.tolist())
            all_symbols.extend(phases_q.tolist())
    
    try:
        encoded, freq_data = ans_encode(all_symbols)
        encoded_bytes = encoded.tobytes() if hasattr(encoded, 'tobytes') else bytes(encoded)
        header = n.to_bytes(4, 'big')
        payload = header + encoded_bytes
        
        return payload, {
            'original_bytes': n,
            'payload_bytes': len(payload),
            'rft_blocks': rft_blocks,
            'raw_blocks': raw_blocks,
            'num_blocks': num_blocks
        }
    except Exception as e:
        return b'', {'error': str(e)}


# ===========================================================================
# H12: SYMBOL EMBEDDING RFT
# ===========================================================================

def _build_cooccurrence_embedding(data: bytes, dim: int = 16) -> np.ndarray:
    """Build embedding from byte co-occurrence statistics."""
    # Build co-occurrence matrix
    cooc = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(data) - 1):
        cooc[data[i], data[i+1]] += 1
    
    # Add smoothing
    cooc += 0.1
    
    # Normalize rows
    cooc = cooc / cooc.sum(axis=1, keepdims=True)
    
    # SVD to get embedding
    try:
        U, s, Vt = np.linalg.svd(cooc)
        embedding = U[:, :dim] * np.sqrt(s[:dim])
    except:
        # Fallback: random embedding
        embedding = np.random.randn(256, dim) * 0.1
    
    return embedding


def h12_symbol_embedding_encode(
    text: str,
    embed_dim: int = 8,
    block_size: int = 64,
    mag_bits: int = 8,
    phase_bits: int = 6,
    prune_ratio: float = 0.1
) -> Tuple[bytes, dict]:
    """
    H12: Symbol Embedding RFT
    
    Map bytes to continuous embedding, then RFT on embedding dimensions.
    """
    data = text.encode('utf-8')
    n = len(data)
    
    # Build embedding from data statistics
    embedding = _build_cooccurrence_embedding(data, dim=embed_dim)
    
    # Convert text to embedded sequence: N x embed_dim
    embedded = np.array([embedding[b] for b in data], dtype=np.float64)
    
    # Pad sequence length to block boundary
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        embedded = np.vstack([embedded, np.zeros((pad_len, embed_dim))])
    
    num_blocks = len(embedded) // block_size
    
    all_mags_q = []
    all_phases_q = []
    
    # RFT along sequence axis for each embedding dimension
    for d in range(embed_dim):
        dim_signal = embedded[:, d]
        
        for b in range(num_blocks):
            block = dim_signal[b * block_size:(b + 1) * block_size]
            
            coeffs = rft_forward(block)
            mags = np.abs(coeffs)
            phases = np.angle(coeffs)
            
            # Prune
            if prune_ratio > 0:
                threshold = np.percentile(mags, prune_ratio * 100)
                mask = mags >= threshold
                mags = mags * mask
                phases = phases * mask
            
            # Quantize
            mag_max = mags.max() + 1e-10
            mags_q = np.clip((mags / mag_max * (2**mag_bits - 1)).astype(int), 0, 2**mag_bits - 1)
            phases_q = np.clip(((phases + np.pi) / (2 * np.pi) * (2**phase_bits - 1)).astype(int), 0, 2**phase_bits - 1)
            
            all_mags_q.extend(mags_q.tolist())
            all_phases_q.extend(phases_q.tolist())
    
    try:
        mag_encoded, mag_freq = ans_encode(all_mags_q)
        phase_encoded, phase_freq = ans_encode(all_phases_q)
        
        # Convert to bytes
        mag_bytes = mag_encoded.tobytes() if hasattr(mag_encoded, 'tobytes') else bytes(mag_encoded)
        phase_bytes = phase_encoded.tobytes() if hasattr(phase_encoded, 'tobytes') else bytes(phase_encoded)
        
        # Store embedding matrix too (would need in real codec)
        # For now, just measure coefficient compression
        header = n.to_bytes(4, 'big') + embed_dim.to_bytes(1, 'big')
        payload = header + mag_bytes + phase_bytes
        
        # Note: real codec would need to store/transmit embedding
        embedding_overhead = 256 * embed_dim * 4  # float32
        
        return payload, {
            'original_bytes': n,
            'payload_bytes': len(payload),
            'embedding_overhead': embedding_overhead,
            'total_with_embedding': len(payload) + embedding_overhead,
            'num_blocks': num_blocks
        }
    except Exception as e:
        return b'', {'error': str(e)}


# ===========================================================================
# UTILITIES
# ===========================================================================

def _estimate_entropy(data: np.ndarray) -> float:
    """Estimate entropy of data array."""
    # Quantize to integer bins
    data_min, data_max = data.min(), data.max()
    if data_max - data_min < 1e-10:
        return 0.0
    bins = np.clip(((data - data_min) / (data_max - data_min) * 255).astype(int), 0, 255)
    counts = np.bincount(bins, minlength=256)
    probs = counts[counts > 0] / counts.sum()
    return -np.sum(probs * np.log2(probs))


def get_sota_compressed_sizes(text: str) -> Dict[str, int]:
    """Get compressed sizes from SOTA compressors."""
    data = text.encode('utf-8')
    results = {}
    
    results['gzip'] = len(gzip.compress(data, compresslevel=9))
    results['bz2'] = len(bz2.compress(data, compresslevel=9))
    results['lzma'] = len(lzma.compress(data, preset=9))
    
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        results['zstd'] = len(cctx.compress(data))
    
    if HAS_BROTLI:
        results['brotli'] = len(brotli.compress(data, quality=11))
    
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
emerged. Evolution, that blind watchmaker, shaped organisms over billions of years 
through the simple mechanism of natural selection. Those that survived passed on 
their traits; those that didn't became footnotes in the fossil record. And so here 
we are, conscious beings contemplating our own existence, using technology to 
compress our thoughts into ever-smaller packages of information. The irony is not 
lost on us that we spend so much effort trying to say more with less, when the 
universe itself seems content to expand forever into emptiness.
""".strip(),

    'python_code': '''
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n
    
    # Initialize the base cases
    prev, curr = 0, 1
    
    # Iterate from 2 to n
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr


class DataProcessor:
    """A class for processing and transforming data."""
    
    def __init__(self, data: list):
        self.data = data
        self._processed = False
    
    def process(self) -> list:
        """Process the data and return results."""
        if self._processed:
            return self.data
        
        result = []
        for item in self.data:
            if isinstance(item, (int, float)):
                result.append(item * 2)
            elif isinstance(item, str):
                result.append(item.upper())
        
        self._processed = True
        self.data = result
        return result
'''.strip(),

    'json_data': '''{
    "name": "QuantoniumOS",
    "version": "2.0.0",
    "description": "Advanced compression using RFT transforms",
    "dependencies": {
        "numpy": ">=1.20.0",
        "scipy": ">=1.7.0"
    },
    "config": {
        "block_size": 256,
        "quantization": {"magnitude_bits": 8, "phase_bits": 6},
        "transforms": ["rft", "dct", "wavelet"],
        "entropy_coder": "ans"
    },
    "models": [
        {"name": "llama-7b", "params": 7000000000, "bits": 4},
        {"name": "gpt2-xl", "params": 1500000000, "bits": 8}
    ]
}''',

    'scientific': """
The Riemann hypothesis states that all non-trivial zeros of the zeta function 
have real part equal to 1/2. This conjecture, formulated by Bernhard Riemann 
in 1859, remains one of the most important unsolved problems in mathematics. 
The zeta function is defined as zeta(s) = sum(1/n^s) for n=1 to infinity, 
and its analytic continuation reveals deep connections to prime number 
distribution. If proven true, the hypothesis would have profound implications 
for cryptography, as many encryption schemes rely on the difficulty of 
factoring large numbers, which is intimately connected to prime distribution.
""".strip(),

    'random_ascii': ''.join(chr(np.random.randint(32, 127)) for _ in range(2000)),

    'repetitive': ('Hello World! ' * 50 + 'The quick brown fox. ' * 50 + 
                   'Pack my box with five dozen liquor jugs. ' * 30),
}


# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================

@dataclass
class HypothesisResult:
    hypothesis: str
    text_type: str
    config: str
    original_bytes: int
    payload_bytes: int
    bpc: float
    best_sota_bpc: float
    improvement_vs_sota_pct: float
    improvement_vs_raw_rft_pct: float
    extra_info: dict = field(default_factory=dict)


def run_experiment():
    """Run all H11-H18 hypotheses on test texts."""
    
    print("=" * 80)
    print("ASCII WALL TEST - H11-H20 HYPOTHESES")
    print("=" * 80)
    print()
    
    # Baseline raw RFT results (from previous experiment)
    raw_rft_bpc = {
        'english_prose': 0.638,
        'python_code': 1.31,
        'json_data': 8.74,
        'scientific': 0.67,
        'random_ascii': 8.31,
        'repetitive': 0.167,
    }
    
    results: List[HypothesisResult] = []
    
    for text_name, text in TEST_TEXTS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {text_name} ({len(text)} chars)")
        print('='*60)
        
        # Get SOTA baseline
        sota_sizes = get_sota_compressed_sizes(text)
        best_sota = min(sota_sizes.values())
        best_sota_name = min(sota_sizes, key=sota_sizes.get)
        best_sota_bpc = best_sota * 8 / len(text)
        
        print(f"  Best SOTA: {best_sota_name} @ {best_sota_bpc:.3f} bpc")
        print(f"  Raw RFT baseline: {raw_rft_bpc.get(text_name, 999):.3f} bpc")
        print()
        
        # H11: Predictive Residual
        print("  H11 - Predictive Residual RFT:")
        for order in [1, 2, 3]:
            for prune in [0.0, 0.5, 0.9]:
                payload, info = h11_predictive_residual_encode(
                    text, predictor_order=order, prune_ratio=prune
                )
                if 'error' not in info:
                    bpc = len(payload) * 8 / len(text)
                    vs_sota = (bpc - best_sota_bpc) / best_sota_bpc * 100
                    vs_raw = (bpc - raw_rft_bpc.get(text_name, bpc)) / raw_rft_bpc.get(text_name, bpc) * 100
                    
                    config = f"order={order}, prune={prune}"
                    print(f"    {config}: {bpc:.3f} bpc (vs SOTA: {vs_sota:+.1f}%, vs raw RFT: {vs_raw:+.1f}%)")
                    
                    results.append(HypothesisResult(
                        hypothesis='H11',
                        text_type=text_name,
                        config=config,
                        original_bytes=len(text),
                        payload_bytes=len(payload),
                        bpc=bpc,
                        best_sota_bpc=best_sota_bpc,
                        improvement_vs_sota_pct=-vs_sota,
                        improvement_vs_raw_rft_pct=-vs_raw,
                        extra_info={'residual_entropy': info.get('residual_entropy', 0)}
                    ))
                else:
                    print(f"    order={order}, prune={prune}: ERROR - {info['error']}")
        
        # H13: Block-Adaptive
        print("\n  H13 - Block-Adaptive RFT:")
        for block_size in [64, 128, 256]:
            payload, info = h13_block_adaptive_encode(text, block_size=block_size)
            if 'error' not in info:
                bpc = len(payload) * 8 / len(text)
                vs_sota = (bpc - best_sota_bpc) / best_sota_bpc * 100
                vs_raw = (bpc - raw_rft_bpc.get(text_name, bpc)) / raw_rft_bpc.get(text_name, bpc) * 100
                
                modes = info.get('mode_counts', {})
                config = f"block={block_size}"
                print(f"    {config}: {bpc:.3f} bpc (vs SOTA: {vs_sota:+.1f}%) - modes: {modes}")
                
                results.append(HypothesisResult(
                    hypothesis='H13',
                    text_type=text_name,
                    config=config,
                    original_bytes=len(text),
                    payload_bytes=len(payload),
                    bpc=bpc,
                    best_sota_bpc=best_sota_bpc,
                    improvement_vs_sota_pct=-vs_sota,
                    improvement_vs_raw_rft_pct=-vs_raw,
                    extra_info=info
                ))
            else:
                print(f"    block={block_size}: ERROR - {info['error']}")
        
        # H18: Context-Gated
        print("\n  H18 - Context-Gated RFT:")
        for flatness in [0.7, 0.8, 0.9]:
            payload, info = h18_context_gated_encode(text, flatness_threshold=flatness)
            if 'error' not in info:
                bpc = len(payload) * 8 / len(text)
                vs_sota = (bpc - best_sota_bpc) / best_sota_bpc * 100
                vs_raw = (bpc - raw_rft_bpc.get(text_name, bpc)) / raw_rft_bpc.get(text_name, bpc) * 100
                
                rft_pct = info.get('rft_blocks', 0) / max(1, info.get('num_blocks', 1)) * 100
                config = f"flatness={flatness}"
                print(f"    {config}: {bpc:.3f} bpc (vs SOTA: {vs_sota:+.1f}%) - RFT blocks: {rft_pct:.0f}%")
                
                results.append(HypothesisResult(
                    hypothesis='H18',
                    text_type=text_name,
                    config=config,
                    original_bytes=len(text),
                    payload_bytes=len(payload),
                    bpc=bpc,
                    best_sota_bpc=best_sota_bpc,
                    improvement_vs_sota_pct=-vs_sota,
                    improvement_vs_raw_rft_pct=-vs_raw,
                    extra_info=info
                ))
            else:
                print(f"    flatness={flatness}: ERROR - {info['error']}")
        
        # H12: Symbol Embedding
        print("\n  H12 - Symbol Embedding RFT:")
        for embed_dim in [4, 8, 16]:
            payload, info = h12_symbol_embedding_encode(text, embed_dim=embed_dim)
            if 'error' not in info:
                # BPC without embedding overhead
                bpc_no_embed = len(payload) * 8 / len(text)
                # BPC with embedding overhead (one-time cost, amortized poorly here)
                total_bytes = info.get('total_with_embedding', len(payload))
                bpc_with_embed = total_bytes * 8 / len(text)
                
                vs_sota = (bpc_no_embed - best_sota_bpc) / best_sota_bpc * 100
                
                config = f"dim={embed_dim}"
                print(f"    {config}: {bpc_no_embed:.3f} bpc (no embed) / {bpc_with_embed:.3f} bpc (with embed)")
                
                results.append(HypothesisResult(
                    hypothesis='H12',
                    text_type=text_name,
                    config=config,
                    original_bytes=len(text),
                    payload_bytes=len(payload),
                    bpc=bpc_no_embed,
                    best_sota_bpc=best_sota_bpc,
                    improvement_vs_sota_pct=-vs_sota,
                    improvement_vs_raw_rft_pct=0,
                    extra_info=info
                ))
            else:
                print(f"    dim={embed_dim}: ERROR - {info['error']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Count wins vs SOTA
    wins_sota = sum(1 for r in results if r.improvement_vs_sota_pct > 0)
    # Count wins vs raw RFT
    wins_raw = sum(1 for r in results if r.improvement_vs_raw_rft_pct > 0)
    
    print(f"\nTotal configurations tested: {len(results)}")
    print(f"Wins vs SOTA: {wins_sota}/{len(results)} ({100*wins_sota/max(1,len(results)):.1f}%)")
    print(f"Wins vs raw RFT: {wins_raw}/{len(results)} ({100*wins_raw/max(1,len(results)):.1f}%)")
    
    # Best per hypothesis
    print("\nBest result per hypothesis (vs raw RFT):")
    for hyp in ['H11', 'H12', 'H13', 'H18']:
        hyp_results = [r for r in results if r.hypothesis == hyp]
        if hyp_results:
            best = max(hyp_results, key=lambda r: r.improvement_vs_raw_rft_pct)
            print(f"  {hyp}: {best.text_type}/{best.config} -> {best.bpc:.3f} bpc "
                  f"(vs raw RFT: {best.improvement_vs_raw_rft_pct:+.1f}%)")
    
    # Best overall
    if results:
        best_overall = min(results, key=lambda r: r.bpc)
        print(f"\nBest overall: {best_overall.hypothesis} on {best_overall.text_type}")
        print(f"  Config: {best_overall.config}")
        print(f"  BPC: {best_overall.bpc:.3f} (vs SOTA {best_overall.best_sota_bpc:.3f})")
    
    # Check hypotheses
    print("\n" + "=" * 80)
    print("HYPOTHESIS VERDICTS")
    print("=" * 80)
    
    # H11: Should reduce bpc vs raw RFT by 30-40% on structured text
    h11_english = [r for r in results if r.hypothesis == 'H11' and r.text_type == 'english_prose']
    if h11_english:
        best_h11 = max(h11_english, key=lambda r: r.improvement_vs_raw_rft_pct)
        h11_verdict = "CONFIRMED" if best_h11.improvement_vs_raw_rft_pct >= 30 else "REJECTED"
        print(f"\nH11 (Predictive Residual): {h11_verdict}")
        print(f"  Best on english_prose: {best_h11.improvement_vs_raw_rft_pct:+.1f}% vs raw RFT")
        print(f"  Required: >= 30% improvement")
    
    # H13: Should improve average bpc vs raw RFT
    h13_results = [r for r in results if r.hypothesis == 'H13']
    if h13_results:
        avg_improvement = np.mean([r.improvement_vs_raw_rft_pct for r in h13_results])
        h13_verdict = "CONFIRMED" if avg_improvement > 0 else "REJECTED"
        print(f"\nH13 (Block-Adaptive): {h13_verdict}")
        print(f"  Average improvement vs raw RFT: {avg_improvement:+.1f}%")
    
    # H18: Should never do worse than always-RFT on random
    h18_random = [r for r in results if r.hypothesis == 'H18' and r.text_type == 'random_ascii']
    if h18_random:
        # Compare to raw RFT on random (8.31 bpc)
        best_h18 = min(h18_random, key=lambda r: r.bpc)
        h18_verdict = "CONFIRMED" if best_h18.bpc <= raw_rft_bpc.get('random_ascii', 999) else "REJECTED"
        print(f"\nH18 (Context-Gated): {h18_verdict}")
        print(f"  Best on random_ascii: {best_h18.bpc:.3f} bpc (raw RFT: {raw_rft_bpc.get('random_ascii', 999):.3f})")
    
    # Wall status
    wall_broken = wins_sota > 0
    if wall_broken:
        print("\n" + "=" * 80)
        print("*** WALL BROKEN! RFT variant beats SOTA! ***")
        print("=" * 80)
        for r in results:
            if r.improvement_vs_sota_pct > 0:
                print(f"  {r.hypothesis} on {r.text_type}: {r.bpc:.3f} < {r.best_sota_bpc:.3f} bpc SOTA")
    else:
        print("\n" + "=" * 80)
        print("WALL STANDS - No variant beats SOTA on text compression")
        print("=" * 80)
    
    # Save results
    output = {
        'summary': {
            'total_configs': len(results),
            'wins_vs_sota': wins_sota,
            'wins_vs_raw_rft': wins_raw,
            'wall_broken': wall_broken
        },
        'results': [
            {
                'hypothesis': r.hypothesis,
                'text': r.text_type,
                'config': r.config,
                'bpc': r.bpc,
                'best_sota_bpc': r.best_sota_bpc,
                'improvement_vs_sota_pct': r.improvement_vs_sota_pct,
                'improvement_vs_raw_rft_pct': r.improvement_vs_raw_rft_pct,
            }
            for r in results
        ]
    }
    
    output_path = Path(__file__).parent / 'ascii_wall_h11_h20_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    return 0 if wall_broken else 1


if __name__ == '__main__':
    sys.exit(run_experiment())

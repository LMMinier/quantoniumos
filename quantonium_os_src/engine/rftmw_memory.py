#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
#
# This file practices Claims 1 & 4 of USPTO Application 19/169,399.
# Licensed under LICENSE-CLAIMS-NC.md — research / education ONLY.
# Commercial use requires a separate patent license.
# See docs/project/CLAIMS_PRACTICING_FILES.txt
"""
RFTMW Memory Abstraction Layer
===============================

The middleware between LLM models and memory/compute.  Transparently
compresses and decompresses:

  1. **Model weights** — RFT for highly-structured tensors (spectral
     entropy < 0.40 AND reconstruction error < 8%), INT8+zlib elsewhere
  2. **KV-cache** — Rolling RFT compression on cached key/value tensors,
     with configurable eviction policy
  3. **Activations** — Optional streaming compression for activations
     between layers (reduces memory bandwidth)

Architecture::

    ┌───────────────────────────────────────────────────┐
    │               HuggingFace Model                   │
    │  (or any PyTorch model with .named_parameters())  │
    └───────────────┬───────────────────────────────────┘
                    │ load / forward / generate
    ┌───────────────▼───────────────────────────────────┐
    │           RFTMW Memory Layer                      │
    │                                                   │
    │  ┌─────────────┐  ┌────────────┐  ┌───────────┐  │
    │  │ Weight Store │  │ KV-Cache   │  │ Activation│  │
    │  │ (RFT/INT8)  │  │ Compressor │  │ Buffer    │  │
    │  └─────────────┘  └────────────┘  └───────────┘  │
    │                                                   │
    │  Spectral-Entropy Router: auto-selects RFT vs     │
    │  standard per tensor based on φ-structure          │
    └───────────────────────────────────────────────────┘

Usage::

    from quantonium_os_src.engine.rftmw_memory import RFTMWMemoryLayer

    mem = RFTMWMemoryLayer()
    mem.ingest_model(model)          # compress all weights
    mem.print_report()               # show per-layer stats
    w = mem.get_weight("transformer.wte.weight")  # decompress on demand
    mem.compress_kv(keys, values)    # KV-cache compression
    k, v = mem.decompress_kv(...)    # restore for attention
"""
from __future__ import annotations

import io
import struct
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Use canonical Gram-normalized RFT (NOT the legacy FFT+phi-phase hack)
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix

PHI = (1 + np.sqrt(5)) / 2


# ===================================================================
# Enums & dataclasses
# ===================================================================

class CompressionMethod(Enum):
    RFT = auto()       # Canonical Gram-normalized RFT + quantize + zlib
    INT8_ZLIB = auto()  # Standard INT8 quantization + zlib
    NONE = auto()       # No compression (tiny tensors)


@dataclass
class TensorSlot:
    """A compressed tensor in the memory layer."""
    name: str
    original_shape: Tuple[int, ...]
    original_dtype: str
    original_bytes: int
    method: CompressionMethod
    compressed_data: bytes
    compressed_bytes: int
    spectral_entropy: float
    reconstruction_error: float  # relative Frobenius norm


@dataclass
class KVCacheSlot:
    """Compressed KV-cache for one layer."""
    layer_idx: int
    seq_len: int
    num_heads: int
    head_dim: int
    key_data: bytes
    value_data: bytes
    key_bytes: int
    value_bytes: int
    original_bytes: int
    method: CompressionMethod


@dataclass
class MemoryReport:
    """Summary of memory layer state."""
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    rft_layers: int = 0
    int8_layers: int = 0
    skip_layers: int = 0
    kv_original_bytes: int = 0
    kv_compressed_bytes: int = 0
    kv_layers: int = 0


# ===================================================================
# Core compression primitives
# ===================================================================

def _spectral_entropy(w: np.ndarray) -> float:
    """Normalized spectral entropy via FFT.  Low = structured = RFT candidate.

    Normalizes input before FFT to prevent overflow on large-valued weights.
    """
    flat = w.flatten().astype(np.float64)
    n = len(flat)
    if n < 4:
        return 1.0
    # Normalize by max |x| first to prevent overflow in sum-of-squares.
    # np.linalg.norm does sum(x²) internally which overflows float64
    # for wte.weight (50257×768 = 38.6M elements with values ~±1).
    amax = np.abs(flat).max()
    if amax < 1e-30:
        return 1.0
    flat = flat / amax
    fft = np.fft.rfft(flat)
    power = np.abs(fft) ** 2
    total = power.sum()
    if total < 1e-30:
        return 1.0
    p = power / total
    p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    Hmax = np.log2(len(fft))
    return float(H / Hmax) if Hmax > 0 else 1.0


def _compress_rft(w: np.ndarray, keep_ratio: float = 0.20,
                   mag_bits: int = 12, phase_bits: int = 10) -> Tuple[bytes, float]:
    """
    Compress via canonical Gram-normalized RFT → top-k → quantize → zlib.

    Returns (compressed_bytes, reconstruction_error).

    For large tensors we process in blocks (rows or fixed-size chunks)
    so that the O(N²) basis matrix stays manageable.
    """
    flat = w.flatten().astype(np.float64)
    N = len(flat)

    # Block size for the dense RFT matrix — keep ≤ 2048 to avoid OOM
    BLOCK = min(N, 1024)
    n_blocks = (N + BLOCK - 1) // BLOCK

    # Precompute one basis (PhiH) for the block size
    Phi = rft_basis_matrix(BLOCK, BLOCK, use_gram_normalization=True)
    PhiH = Phi.conj().T  # analysis matrix

    all_mags_q: list[np.ndarray] = []
    all_phases_q: list[np.ndarray] = []
    block_peaks: list[float] = []

    max_mag_val = (1 << mag_bits) - 1
    max_phase_val = (1 << phase_bits) - 1

    for bi in range(n_blocks):
        start = bi * BLOCK
        end = min(start + BLOCK, N)
        block = flat[start:end]
        blen = len(block)

        # Pad last block if needed
        if blen < BLOCK:
            block = np.pad(block, (0, BLOCK - blen))

        coeffs = PhiH @ block
        mags = np.abs(coeffs)
        phases = np.angle(coeffs)

        # Top-k thresholding
        k = max(1, int(BLOCK * keep_ratio))
        if k < BLOCK:
            thresh = np.sort(mags)[::-1][k - 1]
            mask = mags >= thresh
            mags = mags * mask
            phases = phases * mask

        peak = mags.max() + 1e-15
        block_peaks.append(peak)

        mq = np.clip((mags / peak * max_mag_val), 0, max_mag_val).astype(np.uint16)
        pn = (phases + np.pi) / (2 * np.pi)
        pq = np.clip((pn * max_phase_val), 0, max_phase_val).astype(np.uint16)

        all_mags_q.append(mq)
        all_phases_q.append(pq)

    # Serialize
    buf = io.BytesIO()
    # Header: N(4) + BLOCK(4) + n_blocks(4) + keep_ratio(8) + mag_bits(1) + phase_bits(1)
    buf.write(struct.pack('>IIIB B d', N, BLOCK, n_blocks, mag_bits, phase_bits, keep_ratio))
    # Block peaks (one float64 per block)
    for p in block_peaks:
        buf.write(struct.pack('>d', p))
    # Quantized payload
    payload = b''.join(mq.tobytes() + pq.tobytes()
                       for mq, pq in zip(all_mags_q, all_phases_q))
    compressed_payload = zlib.compress(payload, 9)
    buf.write(struct.pack('>I', len(compressed_payload)))
    buf.write(compressed_payload)

    blob = buf.getvalue()

    # --- compute reconstruction error ---
    recon = _decompress_rft_blob(blob)[:N]
    err = np.linalg.norm(flat - recon) / (np.linalg.norm(flat) + 1e-15)

    return blob, float(err)


def _decompress_rft_blob(blob: bytes) -> np.ndarray:
    """Decompress an RFT-compressed blob back to float64 array."""
    buf = io.BytesIO(blob)
    N, BLOCK, n_blocks, mag_bits, phase_bits = struct.unpack('>IIIB B', buf.read(14))
    keep_ratio = struct.unpack('>d', buf.read(8))[0]

    max_mag_val = (1 << mag_bits) - 1
    max_phase_val = (1 << phase_bits) - 1

    block_peaks = [struct.unpack('>d', buf.read(8))[0] for _ in range(n_blocks)]

    comp_len = struct.unpack('>I', buf.read(4))[0]
    payload = zlib.decompress(buf.read(comp_len))

    Phi = rft_basis_matrix(BLOCK, BLOCK, use_gram_normalization=True)

    offset = 0
    bytes_per_block = BLOCK * 2 * 2  # uint16 mags + uint16 phases
    out_blocks: list[np.ndarray] = []

    for bi in range(n_blocks):
        chunk = payload[offset:offset + bytes_per_block]
        offset += bytes_per_block
        mq = np.frombuffer(chunk[:BLOCK * 2], dtype=np.uint16).copy()
        pq = np.frombuffer(chunk[BLOCK * 2:], dtype=np.uint16).copy()

        peak = block_peaks[bi]
        mags = mq.astype(np.float64) / max_mag_val * peak
        phases = pq.astype(np.float64) / max_phase_val * 2 * np.pi - np.pi
        coeffs = mags * np.exp(1j * phases)

        block = (Phi @ coeffs).real
        out_blocks.append(block)

    return np.concatenate(out_blocks)[:N]


# Group size for group-wise INT8 quantization.
# Each group of GROUP_SIZE elements gets its own (min, scale) pair.
# This prevents outliers in one region from destroying precision elsewhere.
# Industry standard is 128 (GPTQ, AWQ); we match that.
_INT8_GROUP_SIZE = 128


def _compress_int8_zlib(w: np.ndarray) -> Tuple[bytes, float]:
    """Group-wise INT8 + zlib.  Returns (blob, reconstruction_error).

    Each group of 128 elements is quantized independently with its own
    (min, scale) pair.  This prevents outliers in one group from
    stretching the quantization range of other groups — the same
    technique used by GPTQ and AWQ.
    """
    flat = w.flatten().astype(np.float32)
    N = len(flat)
    G = _INT8_GROUP_SIZE
    n_groups = (N + G - 1) // G

    # Pad to multiple of G
    if N % G != 0:
        padded = np.zeros(n_groups * G, dtype=np.float32)
        padded[:N] = flat
    else:
        padded = flat

    groups = padded.reshape(n_groups, G)
    gmins = groups.min(axis=1).astype(np.float64)
    gmaxs = groups.max(axis=1).astype(np.float64)
    scales = (gmaxs - gmins) / 255.0
    scales[scales == 0] = 1.0

    # Quantize all groups at once (vectorized)
    q = np.clip(np.round((groups - gmins[:, None]) / scales[:, None]), 0, 255).astype(np.uint8)
    compressed = zlib.compress(q.tobytes(), 9)

    # Serialize: header + group params + compressed payload
    buf = io.BytesIO()
    # Format version 2: N(4) + n_groups(4) + G(4)
    buf.write(struct.pack('>I I I', N, n_groups, G))
    # Per-group min and scale (float64 each)
    for gi in range(n_groups):
        buf.write(struct.pack('>d d', gmins[gi], scales[gi]))
    buf.write(struct.pack('>I', len(compressed)))
    buf.write(compressed)
    blob = buf.getvalue()

    # Reconstruction error (measured against original, not padded)
    recon = (q.astype(np.float32) * scales[:, None].astype(np.float32)
             + gmins[:, None].astype(np.float32)).flatten()[:N]
    err = float(np.linalg.norm(flat - recon) / (np.linalg.norm(flat) + 1e-15))

    return blob, err


def _decompress_int8_blob(blob: bytes) -> np.ndarray:
    """Decompress a group-wise INT8+zlib blob back to float32 array."""
    buf = io.BytesIO(blob)
    header = buf.read(12)
    N, n_groups, G = struct.unpack('>I I I', header)

    gmins = np.empty(n_groups, dtype=np.float64)
    scales = np.empty(n_groups, dtype=np.float64)
    for gi in range(n_groups):
        gmins[gi], scales[gi] = struct.unpack('>d d', buf.read(16))

    comp_len = struct.unpack('>I', buf.read(4))[0]
    q = np.frombuffer(zlib.decompress(buf.read(comp_len)),
                      dtype=np.uint8).reshape(n_groups, G)
    recon = (q.astype(np.float32) * scales[:, None].astype(np.float32)
             + gmins[:, None].astype(np.float32))
    return recon.flatten()[:N]


# ===================================================================
# KV-Cache compression
# ===================================================================

def _compress_kv_tensor(t: np.ndarray, method: CompressionMethod,
                         keep_ratio: float = 0.30) -> Tuple[bytes, int]:
    """Compress a single K or V tensor.  Returns (blob, original_bytes)."""
    orig = t.nbytes
    if method == CompressionMethod.RFT:
        blob, _ = _compress_rft(t, keep_ratio=keep_ratio, mag_bits=10, phase_bits=8)
    else:
        blob, _ = _compress_int8_zlib(t)
    return blob, orig


def _decompress_kv_tensor(blob: bytes, method: CompressionMethod,
                           shape: Tuple[int, ...]) -> np.ndarray:
    """Decompress a K or V tensor."""
    if method == CompressionMethod.RFT:
        flat = _decompress_rft_blob(blob)
    else:
        flat = _decompress_int8_blob(blob)
    return flat[:int(np.prod(shape))].reshape(shape)


# ===================================================================
# Main memory layer
# ===================================================================

class RFTMWMemoryLayer:
    """
    Middleware memory abstraction layer for LLM inference.

    Transparently compresses model weights and KV-cache using the RFT
    where it helps (embeddings, KV-cache with temporal φ-structure)
    and standard INT8+zlib everywhere else.
    """

    # Spectral entropy threshold: RFT if entropy < this
    # (lowered from 0.87 — real LLM data showed 40-62% errors at 0.87)
    ENTROPY_THRESHOLD = 0.40
    # Minimum tensor size (elements) to bother compressing
    MIN_COMPRESS_SIZE = 512
    # RFT keep-ratio for weight compression
    WEIGHT_KEEP_RATIO = 0.30
    # RFT keep-ratio for KV-cache (more aggressive — trades quality for speed)
    KV_KEEP_RATIO = 0.30
    # Maximum RFT error before falling back to INT8
    MAX_RFT_ERROR = 0.08

    def __init__(
        self,
        entropy_threshold: float = 0.40,
        weight_keep_ratio: float = 0.30,
        kv_keep_ratio: float = 0.30,
        max_rft_error: float = 0.08,
    ):
        self.entropy_threshold = entropy_threshold
        self.weight_keep_ratio = weight_keep_ratio
        self.kv_keep_ratio = kv_keep_ratio
        self.max_rft_error = max_rft_error

        self._weights: Dict[str, TensorSlot] = {}
        self._kv_cache: Dict[int, KVCacheSlot] = {}
        self._report = MemoryReport()

    # ----- Weight management -----

    def ingest_tensor(self, name: str, w: np.ndarray,
                      force_method: Optional[CompressionMethod] = None) -> TensorSlot:
        """Compress and store a single named weight tensor."""
        original_bytes = w.nbytes
        numel = w.size

        if numel < self.MIN_COMPRESS_SIZE:
            # Tiny tensor — store raw
            slot = TensorSlot(
                name=name,
                original_shape=w.shape,
                original_dtype=str(w.dtype),
                original_bytes=original_bytes,
                method=CompressionMethod.NONE,
                compressed_data=w.tobytes(),
                compressed_bytes=original_bytes,
                spectral_entropy=1.0,
                reconstruction_error=0.0,
            )
            self._weights[name] = slot
            self._report.skip_layers += 1
            return slot

        entropy = _spectral_entropy(w)

        # Detect near-constant tensors (e.g. LayerNorm weight = all-ones).
        # These have near-zero entropy but NO useful φ-structure;
        # RFT top-k would destroy them.  INT8 gives ~0% error.
        variance = float(np.var(w))
        is_near_constant = variance < 1e-6

        if force_method is not None:
            method = force_method
        elif is_near_constant:
            method = CompressionMethod.INT8_ZLIB
        elif entropy < self.entropy_threshold:
            method = CompressionMethod.RFT
        else:
            method = CompressionMethod.INT8_ZLIB

        if method == CompressionMethod.RFT:
            blob, err = _compress_rft(w, keep_ratio=self.weight_keep_ratio)
            # Error-based fallback: if RFT error too high, try INT8
            # (only when the router chose RFT, NOT when the caller forced it)
            if force_method is None and err > self.max_rft_error:
                blob_i8, err_i8 = _compress_int8_zlib(w)
                if err_i8 < err:
                    blob, err, method = blob_i8, err_i8, CompressionMethod.INT8_ZLIB
            if method == CompressionMethod.RFT:
                self._report.rft_layers += 1
            else:
                self._report.int8_layers += 1
        else:
            blob, err = _compress_int8_zlib(w)
            self._report.int8_layers += 1

        slot = TensorSlot(
            name=name,
            original_shape=w.shape,
            original_dtype=str(w.dtype),
            original_bytes=original_bytes,
            method=method,
            compressed_data=blob,
            compressed_bytes=len(blob),
            spectral_entropy=entropy,
            reconstruction_error=err,
        )
        self._weights[name] = slot
        self._report.total_original_bytes += original_bytes
        self._report.total_compressed_bytes += len(blob)
        return slot

    def ingest_model(self, model: Any, *, layer_limit: Optional[int] = None,
                      verbose: bool = True) -> MemoryReport:
        """
        Compress all parameters of a PyTorch model.

        Args:
            model: Any object with ``.named_parameters()`` (e.g. HuggingFace).
            layer_limit: Optional cap on number of layers to process (for speed).
            verbose: Print per-layer stats.

        Returns:
            MemoryReport with compression summary.
        """
        t0 = time.perf_counter()
        count = 0

        if verbose:
            print("=" * 72)
            print("RFTMW MEMORY LAYER — Ingesting model weights")
            print("=" * 72)

        for name, param in model.named_parameters():
            if layer_limit is not None and count >= layer_limit:
                break
            w = param.detach().cpu().numpy()
            slot = self.ingest_tensor(name, w)
            count += 1

            if verbose:
                ratio = slot.original_bytes / max(slot.compressed_bytes, 1)
                tag = "★ RFT" if slot.method == CompressionMethod.RFT else (
                    "  INT8" if slot.method == CompressionMethod.INT8_ZLIB else "  skip")
                print(f"  {name[:55]:<55} {ratio:>6.2f}x  H={slot.spectral_entropy:.3f}  "
                      f"err={slot.reconstruction_error*100:.3f}%  {tag}")

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"\nIngested {count} layers in {elapsed:.1f}s")
            self.print_report()

        return self._report

    def get_weight(self, name: str) -> np.ndarray:
        """Decompress a single weight tensor on demand."""
        slot = self._weights[name]
        if slot.method == CompressionMethod.NONE:
            return np.frombuffer(slot.compressed_data,
                                 dtype=np.dtype(slot.original_dtype)).reshape(slot.original_shape)
        elif slot.method == CompressionMethod.RFT:
            flat = _decompress_rft_blob(slot.compressed_data)
            return flat[:int(np.prod(slot.original_shape))].reshape(slot.original_shape).astype(np.float32)
        else:
            flat = _decompress_int8_blob(slot.compressed_data)
            return flat[:int(np.prod(slot.original_shape))].reshape(slot.original_shape)

    def get_state_dict(self) -> Dict[str, Any]:
        """Decompress all weights back into a state_dict (for PyTorch load)."""
        import torch
        sd = {}
        for name, slot in self._weights.items():
            arr = self.get_weight(name)
            sd[name] = torch.from_numpy(arr)
        return sd

    # ----- KV-cache management -----

    def compress_kv(self, layer_idx: int,
                    keys: np.ndarray, values: np.ndarray,
                    method: Optional[CompressionMethod] = None) -> KVCacheSlot:
        """
        Compress key/value tensors for one layer of the KV-cache.

        Expected shapes:  (batch, num_heads, seq_len, head_dim)
        or flattened equivalent.
        """
        original_bytes = keys.nbytes + values.nbytes

        if method is None:
            # Use RFT for KV-cache — temporal sequences along seq_len axis
            # tend to have periodic structure from positional embeddings
            ent = _spectral_entropy(keys)
            method = CompressionMethod.RFT if ent < 0.92 else CompressionMethod.INT8_ZLIB

        k_blob, _ = _compress_kv_tensor(keys, method, keep_ratio=self.kv_keep_ratio)
        v_blob, _ = _compress_kv_tensor(values, method, keep_ratio=self.kv_keep_ratio)

        slot = KVCacheSlot(
            layer_idx=layer_idx,
            seq_len=keys.shape[-2] if keys.ndim >= 3 else keys.shape[0],
            num_heads=keys.shape[-3] if keys.ndim >= 4 else 1,
            head_dim=keys.shape[-1] if keys.ndim >= 2 else keys.shape[0],
            key_data=k_blob,
            value_data=v_blob,
            key_bytes=len(k_blob),
            value_bytes=len(v_blob),
            original_bytes=original_bytes,
            method=method,
        )
        self._kv_cache[layer_idx] = slot
        self._report.kv_original_bytes += original_bytes
        self._report.kv_compressed_bytes += len(k_blob) + len(v_blob)
        self._report.kv_layers += 1
        return slot

    def decompress_kv(self, layer_idx: int,
                      key_shape: Tuple[int, ...],
                      value_shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        """Decompress KV-cache for one layer."""
        slot = self._kv_cache[layer_idx]
        keys = _decompress_kv_tensor(slot.key_data, slot.method, key_shape)
        values = _decompress_kv_tensor(slot.value_data, slot.method, value_shape)
        return keys, values

    def evict_kv(self, layer_idx: int) -> None:
        """Free KV-cache for a layer."""
        if layer_idx in self._kv_cache:
            slot = self._kv_cache.pop(layer_idx)
            self._report.kv_compressed_bytes -= (slot.key_bytes + slot.value_bytes)
            self._report.kv_original_bytes -= slot.original_bytes
            self._report.kv_layers -= 1

    def evict_all_kv(self) -> None:
        """Free all KV-cache memory."""
        self._kv_cache.clear()
        self._report.kv_original_bytes = 0
        self._report.kv_compressed_bytes = 0
        self._report.kv_layers = 0

    # ----- Reporting -----

    def print_report(self):
        """Print memory utilization summary."""
        r = self._report
        print()
        print("─" * 72)
        print("RFTMW MEMORY REPORT")
        print("─" * 72)

        orig_mb = r.total_original_bytes / 1024 / 1024
        comp_mb = r.total_compressed_bytes / 1024 / 1024
        ratio = r.total_original_bytes / max(r.total_compressed_bytes, 1)
        savings_pct = (1 - r.total_compressed_bytes / max(r.total_original_bytes, 1)) * 100

        print(f"  Weights:")
        print(f"    Original:     {orig_mb:>8.2f} MB")
        print(f"    Compressed:   {comp_mb:>8.2f} MB  ({ratio:.2f}x, {savings_pct:.1f}% saved)")
        print(f"    RFT layers:   {r.rft_layers}")
        print(f"    INT8 layers:  {r.int8_layers}")
        print(f"    Skipped:      {r.skip_layers}")

        if r.kv_layers > 0:
            kv_orig_mb = r.kv_original_bytes / 1024 / 1024
            kv_comp_mb = r.kv_compressed_bytes / 1024 / 1024
            kv_ratio = r.kv_original_bytes / max(r.kv_compressed_bytes, 1)
            print(f"  KV-Cache:")
            print(f"    Original:     {kv_orig_mb:>8.2f} MB")
            print(f"    Compressed:   {kv_comp_mb:>8.2f} MB  ({kv_ratio:.2f}x)")
            print(f"    Layers:       {r.kv_layers}")

        total_orig = r.total_original_bytes + r.kv_original_bytes
        total_comp = r.total_compressed_bytes + r.kv_compressed_bytes
        total_ratio = total_orig / max(total_comp, 1)
        print(f"  TOTAL:          {total_orig/1024/1024:.2f} MB → {total_comp/1024/1024:.2f} MB "
              f"({total_ratio:.2f}x)")
        print("─" * 72)

    def layer_report(self) -> List[Dict[str, Any]]:
        """Return per-layer stats as list of dicts."""
        rows = []
        for name, slot in self._weights.items():
            rows.append({
                "name": name,
                "shape": slot.original_shape,
                "method": slot.method.name,
                "original_mb": slot.original_bytes / 1024 / 1024,
                "compressed_mb": slot.compressed_bytes / 1024 / 1024,
                "ratio": slot.original_bytes / max(slot.compressed_bytes, 1),
                "spectral_entropy": slot.spectral_entropy,
                "reconstruction_error": slot.reconstruction_error,
            })
        return rows

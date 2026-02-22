#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
#
# Open for research and education under AGPL-3.0-or-later.
# This test file is NOT patent-practicing and may be freely
# used for validation, benchmarking, and academic purposes.
"""
Tests for the RFTMW Memory Abstraction Layer.

Exercises weight compression (RFT vs INT8), KV-cache compression,
on-demand decompression, and the spectral-entropy router — all
without requiring a HuggingFace download.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantonium_os_src.engine.rftmw_memory import (
    RFTMWMemoryLayer,
    CompressionMethod,
    _spectral_entropy,
    _compress_rft,
    _decompress_rft_blob,
    _compress_int8_zlib,
    _decompress_int8_blob,
)

PHI = (1 + np.sqrt(5)) / 2


# ===================================================================
# Helpers
# ===================================================================

def _phi_periodic_signal(n: int) -> np.ndarray:
    """Signal with golden-ratio quasi-periodic structure (low entropy)."""
    t = np.arange(n, dtype=np.float64) / n
    sig = np.zeros(n)
    for k in range(1, 6):
        sig += np.sin(2 * np.pi * PHI ** k * t) / k
    return sig


def _random_signal(n: int, seed: int = 42) -> np.ndarray:
    """Uniform random (high entropy — should NOT be RFT-routed)."""
    return np.random.default_rng(seed).standard_normal(n)


class FakeModel:
    """Minimal duck-type to satisfy ingest_model()."""

    def __init__(self):
        rng = np.random.default_rng(0)
        # Embedding-like: quasi-periodic, low entropy
        t = np.arange(768, dtype=np.float64) / 768
        emb = np.zeros((50, 768), dtype=np.float32)
        for k in range(1, 6):
            emb += (np.sin(2 * np.pi * PHI ** k * t) / k).astype(np.float32)
        # Attention-like: random, high entropy
        attn = rng.standard_normal((768, 768)).astype(np.float32)
        # MLP-like: random
        mlp = rng.standard_normal((768, 3072)).astype(np.float32)
        # Tiny bias (should be skipped)
        bias = rng.standard_normal(768).astype(np.float32)

        self._params = {
            "transformer.wte.weight": emb,
            "transformer.h.0.attn.c_proj.weight": attn,
            "transformer.h.0.mlp.c_fc.weight": mlp,
            "transformer.h.0.ln_1.bias": bias,
        }

    def named_parameters(self):
        for name, w in self._params.items():
            yield name, _NumpyParam(w)


class _NumpyParam:
    """Minimal stand-in for torch.nn.Parameter."""
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def numel(self):
        return self._arr.size

    def element_size(self):
        return self._arr.dtype.itemsize


# ===================================================================
# Unit tests
# ===================================================================

class TestSpectralEntropy:
    def test_low_entropy_periodic(self):
        sig = _phi_periodic_signal(512)
        h = _spectral_entropy(sig)
        assert h < 0.7, f"φ-periodic signal should have low entropy, got {h:.3f}"

    def test_high_entropy_random(self):
        sig = _random_signal(512)
        h = _spectral_entropy(sig)
        assert h > 0.8, f"Random signal should have high entropy, got {h:.3f}"


class TestRFTCompress:
    def test_roundtrip_1d(self):
        sig = _phi_periodic_signal(256)
        blob, err = _compress_rft(sig, keep_ratio=0.50)
        recon = _decompress_rft_blob(blob)[:256]
        assert recon.shape == (256,)
        assert err < 0.10, f"50% keep should give <10% error, got {err*100:.2f}%"

    def test_roundtrip_2d(self):
        w = np.random.default_rng(1).standard_normal((64, 64)).astype(np.float64)
        blob, err = _compress_rft(w, keep_ratio=1.0)
        recon = _decompress_rft_blob(blob)[:64 * 64].reshape(64, 64)
        # At 100% keep, error is quantization-only
        assert err < 0.05, f"100% keep error too high: {err*100:.2f}%"

    def test_compression_ratio(self):
        sig = _phi_periodic_signal(1024)
        blob, _ = _compress_rft(sig, keep_ratio=0.10)
        # Should be smaller than raw float64
        assert len(blob) < sig.nbytes, (
            f"Compressed {len(blob)} >= original {sig.nbytes}")


class TestINT8Compress:
    def test_roundtrip(self):
        w = np.random.default_rng(2).standard_normal(4096).astype(np.float32)
        blob, err = _compress_int8_zlib(w)
        recon = _decompress_int8_blob(blob)
        assert recon.shape == (4096,)
        assert err < 0.05, f"INT8 error too high: {err*100:.2f}%"


class TestMemoryLayer:

    def test_ingest_model_routes_correctly(self):
        mem = RFTMWMemoryLayer()  # default threshold=0.40 + error fallback
        model = FakeModel()
        report = mem.ingest_model(model, verbose=False)

        # Embedding should go to RFT (low entropy)
        emb_slot = mem._weights["transformer.wte.weight"]
        assert emb_slot.method == CompressionMethod.RFT, (
            f"Embedding should be RFT, got {emb_slot.method}")

        # Attention/MLP should go to INT8
        attn_slot = mem._weights["transformer.h.0.attn.c_proj.weight"]
        assert attn_slot.method == CompressionMethod.INT8_ZLIB, (
            f"Attention should be INT8, got {attn_slot.method}")

        # Bias (768 elements > MIN_COMPRESS_SIZE=512) gets compressed, not skipped
        bias_slot = mem._weights["transformer.h.0.ln_1.bias"]
        assert bias_slot.method == CompressionMethod.INT8_ZLIB, (
            f"Bias should use INT8 (high entropy), got {bias_slot.method}")

    def test_get_weight_roundtrip(self):
        mem = RFTMWMemoryLayer()
        model = FakeModel()
        mem.ingest_model(model, verbose=False)

        for name, param in model.named_parameters():
            original = param.numpy()
            restored = mem.get_weight(name)
            assert restored.shape == original.shape, f"Shape mismatch for {name}"
            # Allow ~5% relative error for lossy compression
            err = np.linalg.norm(original.astype(np.float32) - restored) / (
                np.linalg.norm(original) + 1e-15)
            assert err < 0.15, f"Error too high for {name}: {err*100:.2f}%"

    def test_report_totals(self):
        mem = RFTMWMemoryLayer()
        model = FakeModel()
        report = mem.ingest_model(model, verbose=False)
        assert report.total_original_bytes > 0
        assert report.total_compressed_bytes > 0
        assert report.total_compressed_bytes < report.total_original_bytes

    def test_layer_limit(self):
        mem = RFTMWMemoryLayer()
        model = FakeModel()
        mem.ingest_model(model, layer_limit=2, verbose=False)
        assert len(mem._weights) == 2

    def test_constant_tensor_uses_int8(self):
        """Constant tensors (e.g. LayerNorm weight=ones) should NOT route to RFT."""
        mem = RFTMWMemoryLayer()
        ones = np.ones(1024, dtype=np.float32)
        slot = mem.ingest_tensor("ln.weight", ones)
        assert slot.method == CompressionMethod.INT8_ZLIB, (
            f"Constant tensor should use INT8, got {slot.method}")
        assert slot.reconstruction_error < 0.01, (
            f"Constant tensor error too high: {slot.reconstruction_error*100:.2f}%")

    def test_force_method(self):
        mem = RFTMWMemoryLayer()
        w = np.random.default_rng(10).standard_normal(2048).astype(np.float32)
        slot = mem.ingest_tensor("forced_rft", w,
                                 force_method=CompressionMethod.RFT)
        assert slot.method == CompressionMethod.RFT


class TestKVCache:

    def test_compress_decompress_kv(self):
        mem = RFTMWMemoryLayer()
        rng = np.random.default_rng(99)
        # Simulate KV-cache: (batch=1, heads=4, seq_len=64, head_dim=32)
        shape = (1, 4, 64, 32)
        keys = rng.standard_normal(shape).astype(np.float32)
        values = rng.standard_normal(shape).astype(np.float32)

        slot = mem.compress_kv(layer_idx=0, keys=keys, values=values)
        assert slot.key_bytes > 0
        assert slot.value_bytes > 0
        assert slot.original_bytes == keys.nbytes + values.nbytes

        k_out, v_out = mem.decompress_kv(0, shape, shape)
        assert k_out.shape == shape
        assert v_out.shape == shape

        # Allow higher error for aggressive KV compression
        k_err = np.linalg.norm(keys - k_out) / (np.linalg.norm(keys) + 1e-15)
        assert k_err < 0.30, f"KV key error too high: {k_err*100:.1f}%"

    def test_evict_kv(self):
        mem = RFTMWMemoryLayer()
        shape = (1, 2, 32, 16)
        rng = np.random.default_rng(0)
        keys = rng.standard_normal(shape).astype(np.float32)
        values = rng.standard_normal(shape).astype(np.float32)

        mem.compress_kv(0, keys, values)
        assert 0 in mem._kv_cache
        mem.evict_kv(0)
        assert 0 not in mem._kv_cache

    def test_evict_all_kv(self):
        mem = RFTMWMemoryLayer()
        shape = (1, 2, 32, 16)
        rng = np.random.default_rng(0)
        for i in range(3):
            k = rng.standard_normal(shape).astype(np.float32)
            v = rng.standard_normal(shape).astype(np.float32)
            mem.compress_kv(i, k, v)
        assert len(mem._kv_cache) == 3
        mem.evict_all_kv()
        assert len(mem._kv_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

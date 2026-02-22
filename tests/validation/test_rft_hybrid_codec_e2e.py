# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""End-to-end smoke tests for the hybrid (lossy) RFT codec."""
import numpy as np

from algorithms.rft.hybrids.rft_hybrid_codec import encode_tensor_hybrid, decode_tensor_hybrid


def test_hybrid_roundtrip_low_error() -> None:
    rng = np.random.default_rng(42)
    tensor = rng.standard_normal((128,), dtype=np.float32)

    result = encode_tensor_hybrid(
        tensor,
        prune_threshold=0.0,
        quant_amp_bits=10,
        quant_phase_bits=10,
        collect_residual_samples=False,
    )

    reconstructed = decode_tensor_hybrid(result.container)
    max_error = float(np.max(np.abs(reconstructed - tensor)))
    # Larger tolerance (codec is intentionally lossy but should stay tight)
    assert max_error < 1e-2


def test_hybrid_roundtrip_large_tensor() -> None:
    rng = np.random.default_rng(12345)
    # N=4096 is the largest practical size for O(N²) basis construction
    # (N=1M would require an 8 TiB basis matrix).
    tensor = rng.standard_normal((4096,), dtype=np.float32)

    result = encode_tensor_hybrid(
        tensor,
        prune_threshold=0.0,
        quant_amp_bits=12,
        quant_phase_bits=12,
        collect_residual_samples=False,
    )

    reconstructed = decode_tensor_hybrid(result.container)
    max_error = float(np.max(np.abs(reconstructed - tensor)))
    assert max_error < 5e-2

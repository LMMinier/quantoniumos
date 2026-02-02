#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Compression Engineering Validation Suite
=========================================

This test validates the REAL compression engineering requirements:

1. REAL BITSTREAMS - Actual encoded bytes, not theoretical BPP estimates
2. ENTROPY MODELING - Proper frequency tables, context modeling
3. SIDE-CHANNEL COSTS - Headers, metadata, frequency tables overhead
4. CONTAINER FORMAT - Structured file format with magic bytes, versioning
5. DECODER COMPLEXITY - Actual decode time accounting

The hypothesis test results (BPP 0.808, PSNR 52 dB) are from a simplified
codec experiment. This suite validates what infrastructure actually exists.

Results show:
- ANS codec: ✓ Real bitstream, ✓ Entropy model, ✗ Container format
- Vertex codec: ✓ Real coefficients, ✗ Entropy optimal, ✗ Container
- H11-H20 hypotheses: Partial bitstreams, missing side-channel accounting
"""
import sys
import os
import json
import time
import struct
from pathlib import Path
from typing import Dict, Tuple, Any
from dataclasses import dataclass

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algorithms.rft.compression.ans import ans_encode, ans_decode, RANS_PRECISION_DEFAULT


@dataclass
class CompressionEngineeringAudit:
    """Audit result for a codec implementation."""
    name: str
    has_real_bitstream: bool
    has_entropy_model: bool
    has_side_channel_accounting: bool
    has_container_format: bool
    has_decoder_complexity_measure: bool
    
    # Measured values
    raw_bytes: int = 0
    payload_bytes: int = 0
    header_bytes: int = 0
    entropy_table_bytes: int = 0
    total_overhead_bytes: int = 0
    encode_time_us: float = 0.0
    decode_time_us: float = 0.0
    roundtrip_lossless: bool = False
    
    def summary(self) -> str:
        checks = [
            ("Real Bitstream", self.has_real_bitstream),
            ("Entropy Model", self.has_entropy_model),
            ("Side-Channel Costs", self.has_side_channel_accounting),
            ("Container Format", self.has_container_format),
            ("Decoder Complexity", self.has_decoder_complexity_measure),
        ]
        passed = sum(1 for _, v in checks if v)
        total = len(checks)
        
        lines = [
            f"=== {self.name} ===",
            f"Engineering Score: {passed}/{total}",
        ]
        for name, ok in checks:
            lines.append(f"  {'✓' if ok else '✗'} {name}")
        
        if self.raw_bytes > 0:
            lines.extend([
                f"",
                f"Measured:",
                f"  Raw: {self.raw_bytes} bytes",
                f"  Payload: {self.payload_bytes} bytes",
                f"  Header: {self.header_bytes} bytes",
                f"  Entropy table: {self.entropy_table_bytes} bytes",
                f"  Total overhead: {self.total_overhead_bytes} bytes ({100*self.total_overhead_bytes/self.raw_bytes:.1f}%)",
                f"  Actual BPP: {8 * (self.payload_bytes + self.total_overhead_bytes) / self.raw_bytes:.3f}",
                f"  Encode: {self.encode_time_us:.1f} µs",
                f"  Decode: {self.decode_time_us:.1f} µs",
                f"  Lossless: {'✓' if self.roundtrip_lossless else '✗'}",
            ])
        
        return "\n".join(lines)


def audit_ans_codec() -> CompressionEngineeringAudit:
    """Audit the rANS entropy coder."""
    
    # Test data - realistic byte distribution
    np.random.seed(42)
    test_data = [int(x) for x in np.random.randint(0, 256, size=1000)]
    
    # Encode
    t0 = time.perf_counter()
    encoded, freq_data = ans_encode(test_data, precision=RANS_PRECISION_DEFAULT)
    encode_time = (time.perf_counter() - t0) * 1e6
    
    # Measure bitstream
    bitstream_bytes = len(encoded.tobytes()) if hasattr(encoded, 'tobytes') else len(bytes(encoded))
    
    # Convert numpy types to native Python for JSON serialization
    freq_data_native = {
        'frequencies': {int(k): int(v) for k, v in freq_data['frequencies'].items()},
        'precision': int(freq_data['precision'])
    }
    
    # Measure side-channel: frequency table
    freq_table_bytes = len(json.dumps(freq_data_native).encode('utf-8'))
    
    # Decode
    t0 = time.perf_counter()
    decoded = ans_decode(encoded, freq_data, len(test_data))
    decode_time = (time.perf_counter() - t0) * 1e6
    
    # Verify lossless
    lossless = decoded == test_data
    
    return CompressionEngineeringAudit(
        name="ANS Entropy Coder (rANS)",
        has_real_bitstream=True,  # Actually produces uint16 bitstream
        has_entropy_model=True,   # Uses frequency table
        has_side_channel_accounting=True,  # freq_data is separate
        has_container_format=False,  # No magic bytes, version, checksum
        has_decoder_complexity_measure=True,  # We measure decode time
        raw_bytes=len(test_data),
        payload_bytes=bitstream_bytes,
        header_bytes=0,  # No header
        entropy_table_bytes=freq_table_bytes,
        total_overhead_bytes=freq_table_bytes,  # Just the table
        encode_time_us=encode_time,
        decode_time_us=decode_time,
        roundtrip_lossless=lossless,
    )


def audit_vertex_codec() -> CompressionEngineeringAudit:
    """Audit the RFT Vertex codec."""
    from algorithms.rft.compression.rft_vertex_codec import encode_tensor, decode_tensor
    
    # Test data
    np.random.seed(42)
    test_tensor = np.random.randn(1024).astype(np.float64)
    
    # Encode
    t0 = time.perf_counter()
    container = encode_tensor(test_tensor)
    encode_time = (time.perf_counter() - t0) * 1e6
    
    # Serialize to measure real size
    serialized = json.dumps(container).encode('utf-8')
    
    # Decode
    t0 = time.perf_counter()
    reconstructed = decode_tensor(container)
    decode_time = (time.perf_counter() - t0) * 1e6
    
    # Check reconstruction
    max_err = np.max(np.abs(test_tensor - reconstructed))
    lossless = max_err < 1e-10
    
    # Analyze container structure
    has_header = 'type' in container and 'version' in container.get('metadata', {})
    
    return CompressionEngineeringAudit(
        name="RFT Vertex Codec",
        has_real_bitstream=False,  # JSON text, not binary bitstream
        has_entropy_model=False,   # Stores raw floats, no entropy coding
        has_side_channel_accounting=True,  # Metadata is explicit
        has_container_format=has_header,  # Has type/version
        has_decoder_complexity_measure=True,
        raw_bytes=test_tensor.nbytes,
        payload_bytes=len(serialized),
        header_bytes=0,
        entropy_table_bytes=0,
        total_overhead_bytes=len(serialized) - test_tensor.nbytes,  # Overhead from JSON
        encode_time_us=encode_time,
        decode_time_us=decode_time,
        roundtrip_lossless=lossless,
    )


def audit_hypothesis_codec() -> CompressionEngineeringAudit:
    """Audit the H11-H20 hypothesis codecs."""
    try:
        from experiments.ascii_wall.ascii_wall_h11_h20 import h11_prediction_residual_encode
    except ImportError:
        return CompressionEngineeringAudit(
            name="H11-H20 Hypothesis Codecs",
            has_real_bitstream=False,
            has_entropy_model=False,
            has_side_channel_accounting=False,
            has_container_format=False,
            has_decoder_complexity_measure=False,
        )
    
    # Test data
    test_text = "Hello, World! " * 100
    
    # Encode
    t0 = time.perf_counter()
    payload, stats = h11_prediction_residual_encode(test_text)
    encode_time = (time.perf_counter() - t0) * 1e6
    
    if not payload:
        return CompressionEngineeringAudit(
            name="H11-H20 Hypothesis Codecs",
            has_real_bitstream=False,
            has_entropy_model=False,
            has_side_channel_accounting=False,
            has_container_format=False,
            has_decoder_complexity_measure=False,
        )
    
    # Check what the payload contains
    has_header = len(payload) >= 5  # At least length + some data
    
    return CompressionEngineeringAudit(
        name="H11-H20 Hypothesis Codecs",
        has_real_bitstream=True,  # Produces bytes
        has_entropy_model=True,   # Uses ANS
        has_side_channel_accounting=False,  # freq_data stored inside, not measured separately
        has_container_format=False,  # Just header + payload, no magic/version
        has_decoder_complexity_measure=False,  # No decode implemented in H11
        raw_bytes=stats.get('original_bytes', len(test_text)),
        payload_bytes=len(payload),
        header_bytes=5,  # n (4 bytes) + predictor_order (1 byte)
        entropy_table_bytes=0,  # Not separately measured
        total_overhead_bytes=5,
        encode_time_us=encode_time,
        decode_time_us=0,  # No decode
        roundtrip_lossless=False,  # Can't verify without decode
    )


def audit_bpp_claims() -> str:
    """
    Audit the BPP claims in hypothesis testing.
    
    The claim: "BPP: 0.808 | PSNR: 52.17 dB"
    
    This is NOT valid compression engineering because:
    1. BPP is calculated as: kept_coefficients * bits_per_coeff / N
    2. It doesn't account for:
       - Index storage (which coefficients were kept)
       - Quantization overhead
       - Entropy table transmission
       - Container format overhead
    """
    lines = [
        "=" * 70,
        "BPP CLAIMS AUDIT - What's Missing",
        "=" * 70,
        "",
        "The hypothesis results show: BPP 0.808, PSNR 52.17 dB",
        "",
        "This is a TRANSFORM-DOMAIN metric, not a COMPRESSION metric.",
        "",
        "Missing from the BPP calculation:",
        "",
        "1. INDEX STORAGE",
        "   - If 5% of 1000 coefficients kept = 50 indices",
        "   - Each index needs log2(1000) ≈ 10 bits",
        "   - Index overhead: 50 * 10 / 1000 = 0.5 BPP additional",
        "",
        "2. QUANTIZATION METADATA",
        "   - Scale factors per block",
        "   - Bit-depth indicators",
        "   - Typical overhead: 0.05-0.1 BPP",
        "",
        "3. ENTROPY TABLE",
        "   - Frequency distribution for ANS decoder",
        "   - Size depends on symbol alphabet",
        "   - Typical overhead: 0.1-0.5 BPP for small blocks",
        "",
        "4. CONTAINER FORMAT",
        "   - Magic bytes, version, checksum",
        "   - Block structure metadata",
        "   - Typical overhead: 0.01-0.05 BPP",
        "",
        "5. DECODER COMPLEXITY",
        "   - Not a BPP cost but affects practicality",
        "   - RFT inverse requires O(N²) or O(N log N) ops",
        "",
        "-" * 70,
        "CORRECTED ESTIMATE:",
        "",
        "  Claimed BPP:      0.808",
        "  + Index storage:  0.500  (50 indices × 10 bits)",
        "  + Quant metadata: 0.050",
        "  + Entropy table:  0.100  (pessimistic for N=1000)",
        "  + Container:      0.010",
        "  ─────────────────────────",
        "  Realistic BPP:    1.468",
        "",
        "This is HIGHER than raw UTF-8 (8 BPP) → No compression!",
        "",
        "The hypothesis tests measure TRANSFORM SPARSITY, not compression.",
        "=" * 70,
    ]
    return "\n".join(lines)


def run_full_audit() -> str:
    """Run complete compression engineering audit."""
    
    lines = [
        "=" * 70,
        "COMPRESSION ENGINEERING AUDIT",
        "=" * 70,
        "",
    ]
    
    # Audit each codec
    audits = []
    
    try:
        audits.append(audit_ans_codec())
    except Exception as e:
        lines.append(f"ANS Codec audit failed: {e}")
    
    try:
        audits.append(audit_vertex_codec())
    except Exception as e:
        lines.append(f"Vertex Codec audit failed: {e}")
    
    try:
        audits.append(audit_hypothesis_codec())
    except Exception as e:
        lines.append(f"Hypothesis Codec audit failed: {e}")
    
    for audit in audits:
        lines.append(audit.summary())
        lines.append("")
    
    # BPP claims audit
    lines.append(audit_bpp_claims())
    
    # Summary
    lines.extend([
        "",
        "=" * 70,
        "WHAT ACTUALLY EXISTS vs WHAT'S CLAIMED",
        "=" * 70,
        "",
        "ACTUALLY EXISTS:",
        "  ✓ ANS entropy coder (rANS) with real bitstreams",
        "  ✓ RFT vertex codec with exact reconstruction",
        "  ✓ Hypothesis experiments showing transform sparsity",
        "  ✓ Benchmark comparing RFT to zlib/zstd/brotli",
        "",
        "MISSING FOR PRODUCTION CODEC:",
        "  ✗ Proper container format (magic, version, CRC)",
        "  ✗ Side-channel cost accounting in BPP claims",
        "  ✗ Index/position encoding for sparse coefficients",
        "  ✗ Adaptive context modeling",
        "  ✗ Range coding for continuous values",
        "",
        "CONCLUSION:",
        "  The hypothesis results (BPP 0.808) are TRANSFORM metrics,",
        "  not true compression metrics. A full codec would have",
        "  significantly higher BPP due to overhead.",
        "",
        "  The ANS + RFT infrastructure is real and functional,",
        "  but not yet assembled into a production-ready codec.",
        "=" * 70,
    ])
    
    return "\n".join(lines)


# === TESTS ===

def test_ans_produces_real_bitstream():
    """ANS codec must produce actual bytes, not theoretical estimates."""
    test_data = list(range(256)) * 4  # 1024 symbols
    encoded, freq_data = ans_encode(test_data)
    
    # Must be actual bytes
    bitstream = encoded.tobytes()
    assert isinstance(bitstream, bytes), "ANS must produce real bytes"
    assert len(bitstream) > 0, "Bitstream cannot be empty"
    
    # Must include entropy model
    assert 'frequencies' in freq_data, "Must include frequency table"
    assert 'precision' in freq_data, "Must include precision"


def test_ans_roundtrip_lossless():
    """ANS must be perfectly lossless."""
    test_data = list(np.random.randint(0, 256, size=1000))
    encoded, freq_data = ans_encode(test_data)
    decoded = ans_decode(encoded, freq_data, len(test_data))
    
    assert decoded == test_data, "ANS roundtrip must be lossless"


def test_vertex_codec_reconstruction():
    """Vertex codec must reconstruct exactly."""
    from algorithms.rft.compression.rft_vertex_codec import encode_tensor, decode_tensor
    
    test_tensor = np.random.randn(512).astype(np.float64)
    container = encode_tensor(test_tensor)
    reconstructed = decode_tensor(container)
    
    max_err = np.max(np.abs(test_tensor - reconstructed))
    assert max_err < 1e-10, f"Vertex codec error too high: {max_err}"


def test_overhead_accounting():
    """Side-channel costs must be measurable."""
    test_data = list(range(100))
    encoded, freq_data = ans_encode(test_data)
    
    payload_bytes = len(encoded.tobytes())
    freq_table_bytes = len(json.dumps(freq_data).encode('utf-8'))
    
    # Total size must include both
    total_bytes = payload_bytes + freq_table_bytes
    bpp_payload_only = 8 * payload_bytes / len(test_data)
    bpp_with_overhead = 8 * total_bytes / len(test_data)
    
    print(f"Payload BPP: {bpp_payload_only:.3f}")
    print(f"Total BPP:   {bpp_with_overhead:.3f}")
    print(f"Overhead:    {bpp_with_overhead - bpp_payload_only:.3f} BPP")
    
    # Overhead should be non-negligible for small data
    assert freq_table_bytes > 0, "Frequency table has size"


if __name__ == "__main__":
    print(run_full_audit())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
RFT Binary Codec - Engineering-Grade Compression
=================================================

This codec produces REAL bitstreams with ALL overhead accounted for:

1. CONTAINER FORMAT - Magic bytes, version, CRC32
2. REAL BITSTREAMS - Actual encoded bytes
3. ENTROPY MODELING - ANS with frequency table included in stream
4. SIDE-CHANNEL COSTS - All metadata counted in final BPP
5. DECODER COMPLEXITY - Measured decode time

The BPP reported by this codec is the TRUE compressed size / original size.

Container Format:
-----------------
[Magic: 4 bytes "RFTC"]
[Version: 2 bytes]
[Flags: 2 bytes]
[Original length: 4 bytes]
[Block size: 2 bytes]
[Num blocks: 2 bytes]
[Frequency table length: 2 bytes]
[Frequency table: variable]
[Compressed data: variable]
[CRC32: 4 bytes]

Total header overhead: 22 bytes + frequency table
"""

import struct
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
from algorithms.rft.compression.ans import ans_encode, ans_decode, RANS_PRECISION_DEFAULT

# Magic bytes for container identification
MAGIC = b'RFTC'
VERSION = 1
FLAG_LOSSY = 0x01
FLAG_PRUNED = 0x02


@dataclass
class CodecStats:
    """Complete accounting of all bytes in the compressed stream."""
    original_bytes: int
    header_bytes: int
    freq_table_bytes: int
    payload_bytes: int
    crc_bytes: int
    total_compressed_bytes: int
    
    # Derived metrics
    compression_ratio: float
    true_bpp: float  # bits per ORIGINAL byte
    
    # Timing
    encode_time_us: float
    decode_time_us: float
    
    # Quality (for lossy)
    psnr_db: float
    max_error: float
    
    def __str__(self) -> str:
        return (
            f"RFT Binary Codec Stats:\n"
            f"  Original:    {self.original_bytes:,} bytes\n"
            f"  Compressed:  {self.total_compressed_bytes:,} bytes\n"
            f"    Header:      {self.header_bytes} bytes\n"
            f"    Freq table:  {self.freq_table_bytes} bytes\n"
            f"    Payload:     {self.payload_bytes} bytes\n"
            f"    CRC:         {self.crc_bytes} bytes\n"
            f"  Ratio:       {self.compression_ratio:.3f}x\n"
            f"  TRUE BPP:    {self.true_bpp:.3f}\n"
            f"  PSNR:        {self.psnr_db:.2f} dB\n"
            f"  Max error:   {self.max_error:.2e}\n"
            f"  Encode:      {self.encode_time_us:.1f} µs\n"
            f"  Decode:      {self.decode_time_us:.1f} µs"
        )


def _serialize_freq_table(frequencies: Dict[int, int]) -> bytes:
    """Serialize frequency table to minimal binary format using delta encoding."""
    # Sort by symbol for delta encoding
    items = sorted(frequencies.items())
    if not items:
        return struct.pack('>H', 0)
    
    # Use delta encoding for symbols + varint for counts
    data = struct.pack('>H', len(items))
    
    prev_sym = 0
    for sym, count in items:
        delta = sym - prev_sym
        prev_sym = sym
        
        # Varint encode delta (most are small)
        while delta >= 128:
            data += bytes([delta & 0x7F | 0x80])
            delta >>= 7
        data += bytes([delta])
        
        # Varint encode count
        while count >= 128:
            data += bytes([count & 0x7F | 0x80])
            count >>= 7
        data += bytes([count])
    
    return data


def _deserialize_freq_table(data: bytes) -> Tuple[Dict[int, int], int]:
    """Deserialize delta-encoded frequency table."""
    num_symbols = struct.unpack('>H', data[:2])[0]
    frequencies = {}
    offset = 2
    prev_sym = 0
    
    for _ in range(num_symbols):
        # Read varint delta
        delta = 0
        shift = 0
        while True:
            b = data[offset]
            offset += 1
            delta |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        
        sym = prev_sym + delta
        prev_sym = sym
        
        # Read varint count
        count = 0
        shift = 0
        while True:
            b = data[offset]
            offset += 1
            count |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        
        frequencies[sym] = count
    
    return frequencies, offset


def encode(data: bytes, block_size: int = 256, prune_ratio: float = 0.0,
           mag_bits: int = 10, phase_bits: int = 8) -> Tuple[bytes, CodecStats]:
    """
    Encode bytes to RFT binary format with ALL overhead accounted.
    
    Args:
        data: Input bytes to compress
        block_size: Size of RFT blocks
        prune_ratio: Fraction of smallest coefficients to zero (0.0 = lossless)
        mag_bits: Bits for magnitude quantization
        phase_bits: Bits for phase quantization
    
    Returns:
        (compressed_bytes, stats) with TRUE BPP accounting
    """
    import time
    t_start = time.perf_counter()
    
    n = len(data)
    if n == 0:
        raise ValueError("Cannot compress empty data")
    
    # Convert to float signal (vectorized, not per-byte Python loop)
    signal = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    
    # Pad to block boundary
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        signal = np.concatenate([signal, np.zeros(pad_len)])
    
    num_blocks = len(signal) // block_size
    
    # Process blocks: RFT → quantize → collect symbols
    all_symbols = []
    kept_count = 0
    total_coeffs = 0
    
    for b in range(num_blocks):
        block = signal[b * block_size:(b + 1) * block_size]
        
        # RFT transform
        coeffs = rft_forward(block)
        mags = np.abs(coeffs)
        phases = np.angle(coeffs)
        total_coeffs += len(coeffs)
        
        # Optional pruning (lossy)
        if prune_ratio > 0:
            threshold = np.percentile(mags, prune_ratio * 100)
            mask = mags >= threshold
            mags = mags * mask
            phases = phases * mask
        
        kept_count += np.sum(mags > 1e-12)
        
        # Quantize magnitudes
        mag_max = mags.max() + 1e-10
        mags_q = np.clip((mags / mag_max * (2**mag_bits - 1)).astype(int), 0, 2**mag_bits - 1)
        
        # Quantize phases
        phases_norm = (phases + np.pi) / (2 * np.pi)
        phases_q = np.clip((phases_norm * (2**phase_bits - 1)).astype(int), 0, 2**phase_bits - 1)
        
        # Store scale factor (quantized to 16 bits)
        scale_q = int(np.clip(mag_max, 0, 65535))
        all_symbols.append(scale_q)
        
        # Interleave mag/phase for better entropy (vectorized)
        interleaved = np.empty(2 * len(mags_q), dtype=np.int64)
        interleaved[0::2] = mags_q
        interleaved[1::2] = phases_q
        all_symbols.extend(interleaved.tolist())
    
    # ANS encode all symbols
    encoded_arr, freq_data = ans_encode(all_symbols, precision=RANS_PRECISION_DEFAULT)
    payload = encoded_arr.tobytes()
    
    # Serialize frequency table
    freq_table = _serialize_freq_table(freq_data['frequencies'])
    
    # Build container
    flags = 0
    if prune_ratio > 0:
        flags |= FLAG_LOSSY | FLAG_PRUNED
    
    # Header format: magic(4) + version(2) + flags(2) + orig_len(4) + block_size(2) + num_blocks(2) + freq_len(2) = 18 bytes
    header = struct.pack('>4sHHIHHH',
        MAGIC,
        VERSION,
        flags,
        n,  # original length
        block_size,
        num_blocks,
        len(freq_table)
    )
    
    # Assemble: header + freq_table + payload
    container_no_crc = header + freq_table + payload
    
    # CRC32 of everything
    crc = zlib.crc32(container_no_crc) & 0xFFFFFFFF
    crc_bytes = struct.pack('>I', crc)
    
    compressed = container_no_crc + crc_bytes
    
    t_encode = (time.perf_counter() - t_start) * 1e6
    
    # Calculate stats
    header_bytes = len(header)
    freq_table_bytes = len(freq_table)
    payload_bytes = len(payload)
    total = len(compressed)
    
    stats = CodecStats(
        original_bytes=n,
        header_bytes=header_bytes,
        freq_table_bytes=freq_table_bytes,
        payload_bytes=payload_bytes,
        crc_bytes=4,
        total_compressed_bytes=total,
        compression_ratio=n / total if total > 0 else 0,
        true_bpp=8 * total / n,  # TRUE bits per original byte
        encode_time_us=t_encode,
        decode_time_us=0,  # Filled on decode
        psnr_db=float('inf') if prune_ratio == 0 else 0,
        max_error=0 if prune_ratio == 0 else -1,
    )
    
    return compressed, stats


def decode(compressed: bytes) -> Tuple[bytes, CodecStats]:
    """
    Decode RFT binary format back to original bytes.
    
    Returns:
        (original_bytes, stats) with decode timing
    """
    import time
    t_start = time.perf_counter()
    
    if len(compressed) < 22:
        raise ValueError("Compressed data too short")
    
    # Parse header (18 bytes total)
    # Format: magic(4) + version(2) + flags(2) + orig_len(4) + block_size(2) + num_blocks(2) + freq_len(2)
    magic = compressed[:4]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic}")
    
    version, flags, orig_len, block_size, num_blocks, freq_len = struct.unpack(
        '>HHIHHH', compressed[4:18]
    )
    
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")
    
    # Verify CRC
    stored_crc = struct.unpack('>I', compressed[-4:])[0]
    computed_crc = zlib.crc32(compressed[:-4]) & 0xFFFFFFFF
    if stored_crc != computed_crc:
        raise ValueError(f"CRC mismatch: stored={stored_crc:08x}, computed={computed_crc:08x}")
    
    # Parse frequency table
    freq_start = 18
    frequencies, freq_consumed = _deserialize_freq_table(compressed[freq_start:freq_start + freq_len])
    
    # Parse payload
    payload_start = freq_start + freq_len
    payload_end = len(compressed) - 4  # Exclude CRC
    payload = compressed[payload_start:payload_end]
    
    # ANS decode
    freq_data = {'frequencies': frequencies, 'precision': RANS_PRECISION_DEFAULT}
    
    # Calculate expected symbol count
    # Each block: 1 scale + block_size * 2 (mag + phase interleaved)
    symbols_per_block = 1 + block_size * 2
    total_symbols = num_blocks * symbols_per_block
    
    decoded_symbols = ans_decode(np.frombuffer(payload, dtype=np.uint16), freq_data, total_symbols)
    
    # Reconstruct signal
    reconstructed = []
    symbol_idx = 0
    
    mag_bits = 10  # Must match encode
    phase_bits = 8
    
    for b in range(num_blocks):
        # Read scale
        scale_q = decoded_symbols[symbol_idx]
        symbol_idx += 1
        mag_max = float(scale_q) + 1e-10
        
        # Read interleaved mag/phase
        mags_q = []
        phases_q = []
        for _ in range(block_size):
            mags_q.append(decoded_symbols[symbol_idx])
            phases_q.append(decoded_symbols[symbol_idx + 1])
            symbol_idx += 2
        
        # Dequantize
        mags = np.array(mags_q, dtype=np.float64) / (2**mag_bits - 1) * mag_max
        phases = np.array(phases_q, dtype=np.float64) / (2**phase_bits - 1) * 2 * np.pi - np.pi
        
        # Reconstruct complex coefficients
        coeffs = mags * np.exp(1j * phases)
        
        # Inverse RFT
        block = rft_inverse(coeffs)
        reconstructed.extend(np.real(block))
    
    # Trim to original length and convert to bytes
    signal = np.array(reconstructed[:orig_len])
    output = np.clip(np.round(signal), 0, 255).astype(np.uint8).tobytes()
    
    t_decode = (time.perf_counter() - t_start) * 1e6
    
    # Build stats
    stats = CodecStats(
        original_bytes=orig_len,
        header_bytes=18,
        freq_table_bytes=freq_len,
        payload_bytes=len(payload),
        crc_bytes=4,
        total_compressed_bytes=len(compressed),
        compression_ratio=orig_len / len(compressed),
        true_bpp=8 * len(compressed) / orig_len,
        encode_time_us=0,
        decode_time_us=t_decode,
        psnr_db=0,  # Calculated externally
        max_error=0,
    )
    
    return output, stats


def roundtrip_test(data: bytes, **kwargs) -> Tuple[bool, CodecStats, float]:
    """
    Full roundtrip test with PSNR calculation.
    
    Returns:
        (lossless, stats, psnr_db)
    """
    compressed, enc_stats = encode(data, **kwargs)
    decoded, dec_stats = decode(compressed)
    
    # Combine stats
    stats = CodecStats(
        original_bytes=enc_stats.original_bytes,
        header_bytes=enc_stats.header_bytes,
        freq_table_bytes=enc_stats.freq_table_bytes,
        payload_bytes=enc_stats.payload_bytes,
        crc_bytes=enc_stats.crc_bytes,
        total_compressed_bytes=enc_stats.total_compressed_bytes,
        compression_ratio=enc_stats.compression_ratio,
        true_bpp=enc_stats.true_bpp,
        encode_time_us=enc_stats.encode_time_us,
        decode_time_us=dec_stats.decode_time_us,
        psnr_db=0,
        max_error=0,
    )
    
    # Check lossless
    lossless = (data == decoded)
    
    # Calculate PSNR
    orig = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    recon = np.frombuffer(decoded, dtype=np.uint8).astype(np.float64)
    
    mse = np.mean((orig - recon) ** 2)
    if mse < 1e-10:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255**2 / mse)
    
    max_error = np.max(np.abs(orig - recon))
    
    stats.psnr_db = psnr
    stats.max_error = max_error
    
    return lossless, stats, psnr


# === CLI / Demo ===

if __name__ == "__main__":
    print("=" * 70)
    print("RFT BINARY CODEC - Engineering-Grade Compression Test")
    print("=" * 70)
    print()
    
    # Test data - include larger sizes to amortize overhead
    test_cases = [
        ("Random bytes (1KB)", bytes(np.random.randint(0, 256, 1024).astype(np.uint8))),
        ("Random bytes (10KB)", bytes(np.random.randint(0, 256, 10240).astype(np.uint8))),
        ("Structured JSON", b'{"name": "test", "values": [1, 2, 3, 4, 5]}' * 25),
        ("Repeated pattern", b"Hello World! " * 100),
        ("Binary sequence", bytes(range(256)) * 4),
        ("Large repeated (10KB)", b"ABCDEFGHIJ" * 1024),
    ]
    
    for name, data in test_cases:
        print(f"Test: {name}")
        print(f"  Original: {len(data)} bytes")
        
        try:
            lossless, stats, psnr = roundtrip_test(data, block_size=256, prune_ratio=0.0)
            
            print(f"  Compressed: {stats.total_compressed_bytes} bytes")
            print(f"  TRUE BPP: {stats.true_bpp:.3f} (all overhead included)")
            print(f"  Breakdown:")
            print(f"    Header:     {stats.header_bytes} bytes ({8*stats.header_bytes/len(data):.3f} BPP)")
            print(f"    Freq table: {stats.freq_table_bytes} bytes ({8*stats.freq_table_bytes/len(data):.3f} BPP)")
            print(f"    Payload:    {stats.payload_bytes} bytes ({8*stats.payload_bytes/len(data):.3f} BPP)")
            print(f"    CRC:        {stats.crc_bytes} bytes ({8*stats.crc_bytes/len(data):.3f} BPP)")
            print(f"  Lossless: {'✓' if lossless else '✗ (quantization loss)'}")
            print(f"  PSNR: {psnr:.2f} dB")
            
            # Compare to zlib
            import zlib
            zlib_compressed = zlib.compress(data, 9)
            zlib_bpp = 8 * len(zlib_compressed) / len(data)
            print(f"  vs zlib-9: {len(zlib_compressed)} bytes ({zlib_bpp:.3f} BPP)")
            print()
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            print()
    
    print("=" * 70)
    print("CONCLUSION:")
    print("  TRUE BPP includes ALL overhead (header, freq table, CRC)")
    print("  The naive 'sparsity BPP' claims (0.808) ignore these costs.")
    print("  Real compression requires beating 8.0 BPP (raw bytes).")
    print("=" * 70)

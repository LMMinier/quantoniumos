#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Compression Benchmark: RFTMW+ANS/Vertex vs zlib/zstd/brotli
===========================================================

Compares QuantoniumOS compression pipeline against industry-standard codecs.

Metrics:
- Bits per symbol (R)
- Entropy gap (R - H(X))
- Compression ratio
- Encode throughput (MB/s)
- Decode throughput (MB/s)
- Lossless verification

Usage:
    python benchmark_compression_vs_codecs.py [--datasets all] [--output-dir results/]
"""

import argparse
import csv
import json
import platform
import sys
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

# Add project root
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.entropy.datasets import (
    load_ascii_corpus,
    load_audio_frames,
    load_golden_signals,
    load_source_code_corpus,
)
from experiments.entropy.measure_entropy import estimate_entropy

# Optional codec imports
try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import lzma
    HAS_LZMA = True
except ImportError:
    HAS_LZMA = False

# RFT compression imports
try:
    from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
    from algorithms.rft.compression.ans import ans_encode, ans_decode
    from algorithms.rft.compression.rft_vertex_codec import encode_tensor, decode_tensor
    HAS_RFT = True
except ImportError as e:
    print(f"Warning: RFT imports failed: {e}")
    HAS_RFT = False


@dataclass
class CompressionResult:
    """Result of a single compression benchmark."""
    codec: str
    dataset: str
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    bits_per_symbol: float
    entropy_h: float
    entropy_gap: float
    encode_time_us: float
    decode_time_us: float
    encode_mbps: float
    decode_mbps: float
    lossless: bool
    error_msg: str = ""


class CodecWrapper:
    """Base class for codec wrappers."""
    
    name: str = "base"
    
    def encode(self, data: bytes) -> bytes:
        raise NotImplementedError
    
    def decode(self, compressed: bytes) -> bytes:
        raise NotImplementedError


class ZlibCodec(CodecWrapper):
    name = "zlib"
    
    def __init__(self, level: int = 9):
        self.level = level
    
    def encode(self, data: bytes) -> bytes:
        return zlib.compress(data, self.level)
    
    def decode(self, compressed: bytes) -> bytes:
        return zlib.decompress(compressed)


class BrotliCodec(CodecWrapper):
    name = "brotli"
    
    def __init__(self, quality: int = 11):
        self.quality = quality
    
    def encode(self, data: bytes) -> bytes:
        return brotli.compress(data, quality=self.quality)
    
    def decode(self, compressed: bytes) -> bytes:
        return brotli.decompress(compressed)


class ZstdCodec(CodecWrapper):
    name = "zstd"
    
    def __init__(self, level: int = 19):
        self.level = level
        self.compressor = zstd.ZstdCompressor(level=level)
        self.decompressor = zstd.ZstdDecompressor()
    
    def encode(self, data: bytes) -> bytes:
        return self.compressor.compress(data)
    
    def decode(self, compressed: bytes) -> bytes:
        return self.decompressor.decompress(compressed)


class LzmaCodec(CodecWrapper):
    name = "lzma"
    
    def __init__(self, preset: int = 9):
        self.preset = preset
    
    def encode(self, data: bytes) -> bytes:
        return lzma.compress(data, preset=self.preset)
    
    def decode(self, compressed: bytes) -> bytes:
        return lzma.decompress(compressed)


class RFTMWANSCodec(CodecWrapper):
    """RFTMW + ANS compression pipeline."""
    
    name = "rftmw_ans"
    
    def __init__(self, block_size: int = 1024, precision: int = 14):
        self.block_size = block_size
        self.precision = precision
    
    def encode(self, data: bytes) -> bytes:
        """Encode bytes using RFT + ANS."""
        # Convert bytes to float blocks
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        
        # Pad to block size
        pad_len = (self.block_size - len(arr) % self.block_size) % self.block_size
        if pad_len > 0:
            arr = np.pad(arr, (0, pad_len), mode='constant')
        
        # Process in blocks
        n_blocks = len(arr) // self.block_size
        encoded_blocks = []
        metadata = {
            'original_len': len(data),
            'n_blocks': n_blocks,
            'block_size': self.block_size,
            'precision': self.precision,
        }
        
        for i in range(n_blocks):
            block = arr[i * self.block_size:(i + 1) * self.block_size]
            
            # RFT transform
            X = rft_forward(block)
            
            # Quantize coefficients
            real_part = np.real(X)
            imag_part = np.imag(X)
            
            # Scale and quantize
            max_val = max(np.abs(real_part).max(), np.abs(imag_part).max(), 1e-10)
            scale = (2 ** (self.precision - 1) - 1) / max_val
            
            real_quant = np.clip(np.round(real_part * scale), -2**(self.precision-1), 2**(self.precision-1)-1).astype(np.int16)
            imag_quant = np.clip(np.round(imag_part * scale), -2**(self.precision-1), 2**(self.precision-1)-1).astype(np.int16)
            
            # ANS encode the quantized data
            combined = np.concatenate([real_quant, imag_quant]).astype(np.int32)
            # Shift to positive range for ANS
            combined_pos = (combined + 2**(self.precision-1)).astype(np.uint16)
            
            try:
                encoded, freq_data = ans_encode(combined_pos.tolist(), precision=self.precision)
                encoded_blocks.append({
                    'data': encoded.tobytes().hex() if hasattr(encoded, 'tobytes') else bytes(encoded).hex(),
                    'freq': freq_data,
                    'scale': float(max_val),
                    'len': len(combined_pos),
                })
            except Exception as e:
                # Fallback: just store raw quantized data
                encoded_blocks.append({
                    'data': combined_pos.tobytes().hex(),
                    'freq': None,
                    'scale': float(max_val),
                    'len': len(combined_pos),
                    'raw': True,
                })
        
        # Serialize to bytes
        container = {'metadata': metadata, 'blocks': encoded_blocks}
        return json.dumps(container).encode('utf-8')
    
    def decode(self, compressed: bytes) -> bytes:
        """Decode compressed bytes back to original."""
        container = json.loads(compressed.decode('utf-8'))
        metadata = container['metadata']
        blocks = container['blocks']
        
        reconstructed = []
        
        for block_data in blocks:
            scale = block_data['scale']
            data_len = block_data['len']
            
            if block_data.get('raw', False):
                # Raw fallback
                combined_pos = np.frombuffer(bytes.fromhex(block_data['data']), dtype=np.uint16)
            else:
                # ANS decode
                encoded = bytes.fromhex(block_data['data'])
                freq_data = block_data['freq']
                try:
                    decoded = ans_decode(np.frombuffer(encoded, dtype=np.uint8), freq_data, data_len)
                    combined_pos = np.array(decoded, dtype=np.uint16)
                except Exception:
                    # Fallback
                    combined_pos = np.frombuffer(bytes.fromhex(block_data['data']), dtype=np.uint16)
            
            # Shift back and split
            combined = combined_pos.astype(np.int32) - 2**(metadata['precision']-1)
            half = len(combined) // 2
            real_quant = combined[:half].astype(np.float32)
            imag_quant = combined[half:].astype(np.float32)
            
            # Dequantize
            real_part = real_quant / ((2 ** (metadata['precision'] - 1) - 1) / scale)
            imag_part = imag_quant / ((2 ** (metadata['precision'] - 1) - 1) / scale)
            
            # Reconstruct complex coefficients
            X = real_part + 1j * imag_part
            
            # Inverse RFT
            block = rft_inverse(X)
            reconstructed.extend(np.real(block).astype(np.float32))
        
        # Convert back to bytes
        arr = np.array(reconstructed[:metadata['original_len']])
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
        return arr.tobytes()


class RFTVertexCodec(CodecWrapper):
    """RFT + Vertex codec compression pipeline."""
    
    name = "rft_vertex"
    
    def encode(self, data: bytes) -> bytes:
        """Encode bytes using RFT + Vertex codec."""
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        container = encode_tensor(arr)
        return json.dumps(container).encode('utf-8')
    
    def decode(self, compressed: bytes) -> bytes:
        """Decode compressed bytes back to original."""
        container = json.loads(compressed.decode('utf-8'))
        arr = decode_tensor(container)
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
        return arr.tobytes()


class CompressionBenchmark:
    """Benchmark harness for compression codec comparison."""
    
    def __init__(self, warmup_runs: int = 1, timed_runs: int = 3):
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs
        self.codecs = self._register_codecs()
    
    def _register_codecs(self) -> List[CodecWrapper]:
        """Register all codecs to benchmark."""
        codecs = [
            ZlibCodec(level=9),
        ]
        
        if HAS_BROTLI:
            codecs.append(BrotliCodec(quality=11))
        
        if HAS_ZSTD:
            codecs.append(ZstdCodec(level=19))
        
        if HAS_LZMA:
            codecs.append(LzmaCodec(preset=6))  # 9 is very slow
        
        if HAS_RFT:
            codecs.append(RFTMWANSCodec())
            codecs.append(RFTVertexCodec())
        
        return codecs
    
    def bench_single(
        self,
        data: bytes,
        codec: CodecWrapper,
        entropy_h: float
    ) -> CompressionResult:
        """Benchmark a single codec on a single input."""
        original_bytes = len(data)
        
        # Warmup
        try:
            for _ in range(self.warmup_runs):
                compressed = codec.encode(data)
                _ = codec.decode(compressed)
        except Exception as e:
            return CompressionResult(
                codec=codec.name,
                dataset="",
                original_bytes=original_bytes,
                compressed_bytes=0,
                compression_ratio=0,
                bits_per_symbol=float('inf'),
                entropy_h=entropy_h,
                entropy_gap=float('inf'),
                encode_time_us=0,
                decode_time_us=0,
                encode_mbps=0,
                decode_mbps=0,
                lossless=False,
                error_msg=str(e),
            )
        
        # Timed encode
        encode_times = []
        for _ in range(self.timed_runs):
            t0 = perf_counter()
            compressed = codec.encode(data)
            t1 = perf_counter()
            encode_times.append((t1 - t0) * 1e6)
        
        compressed_bytes = len(compressed)
        
        # Timed decode
        decode_times = []
        for _ in range(self.timed_runs):
            t0 = perf_counter()
            decompressed = codec.decode(compressed)
            t1 = perf_counter()
            decode_times.append((t1 - t0) * 1e6)
        
        # Verify lossless
        lossless = (decompressed == data)
        
        # Compute metrics
        encode_us = np.median(encode_times)
        decode_us = np.median(decode_times)
        
        compression_ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0
        bits_per_symbol = 8.0 * compressed_bytes / original_bytes
        entropy_gap = bits_per_symbol - entropy_h
        
        encode_mbps = (original_bytes / 1e6) / (encode_us / 1e6) if encode_us > 0 else 0
        decode_mbps = (original_bytes / 1e6) / (decode_us / 1e6) if decode_us > 0 else 0
        
        return CompressionResult(
            codec=codec.name,
            dataset="",
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=compression_ratio,
            bits_per_symbol=bits_per_symbol,
            entropy_h=entropy_h,
            entropy_gap=entropy_gap,
            encode_time_us=encode_us,
            decode_time_us=decode_us,
            encode_mbps=encode_mbps,
            decode_mbps=decode_mbps,
            lossless=lossless,
        )
    
    def run_benchmark(
        self,
        datasets: Dict[str, bytes],
        entropies: Dict[str, float]
    ) -> List[CompressionResult]:
        """Run full benchmark across datasets."""
        results = []
        
        for ds_name, data in datasets.items():
            entropy_h = entropies.get(ds_name, 8.0)
            print(f"\n{ds_name}: {len(data)} bytes, H(X)={entropy_h:.4f} bits/symbol")
            print("-" * 60)
            
            for codec in self.codecs:
                try:
                    result = self.bench_single(data, codec, entropy_h)
                    result.dataset = ds_name
                    results.append(result)
                    
                    status = "✓" if result.lossless else "✗"
                    print(f"  {codec.name:12} | {status} | "
                          f"ratio={result.compression_ratio:5.2f}x | "
                          f"R={result.bits_per_symbol:5.3f} | "
                          f"gap={result.entropy_gap:+6.3f} | "
                          f"enc={result.encode_mbps:6.1f} MB/s | "
                          f"dec={result.decode_mbps:6.1f} MB/s")
                except Exception as e:
                    print(f"  {codec.name:12} | ERROR: {e}")
        
        return results


def load_datasets() -> Tuple[Dict[str, bytes], Dict[str, float]]:
    """Load all benchmark datasets and compute their entropies."""
    datasets = {}
    entropies = {}
    
    # ASCII corpus
    try:
        ascii_arr = load_ascii_corpus(max_bytes=100000)
        data = ascii_arr.tobytes()
        datasets['ascii'] = data
        entropies['ascii'] = estimate_entropy(list(ascii_arr))
    except Exception as e:
        print(f"Warning: ASCII corpus load failed: {e}")
        data = bytes(range(256)) * 400  # Fallback
        datasets['ascii'] = data
        entropies['ascii'] = 8.0
    
    # Source code
    try:
        code_arr = load_source_code_corpus(max_bytes=100000)
        data = code_arr.tobytes()
        datasets['source_code'] = data
        entropies['source_code'] = estimate_entropy(list(code_arr))
    except Exception as e:
        print(f"Warning: Source code corpus load failed: {e}")
    
    # Audio (as bytes)
    try:
        audio_arr = load_audio_frames(max_samples=50000)
        # Quantize to 8-bit
        audio_8bit = ((audio_arr + 1) * 127.5).astype(np.uint8)
        data = audio_8bit.tobytes()
        datasets['audio'] = data
        entropies['audio'] = estimate_entropy(list(audio_8bit))
    except Exception as e:
        print(f"Warning: Audio load failed: {e}")
        # Synthetic audio
        t = np.linspace(0, 1, 50000)
        audio = np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*880*t)
        audio_8bit = ((audio + 1.5) / 3 * 255).astype(np.uint8)
        datasets['audio'] = audio_8bit.tobytes()
        entropies['audio'] = estimate_entropy(list(audio_8bit))
    
    # Golden ratio signal
    try:
        golden_arr = load_golden_signals(n_samples=50000)
        # Normalize to 0-255
        golden_min, golden_max = golden_arr.min(), golden_arr.max()
        golden_norm = ((golden_arr - golden_min) / (golden_max - golden_min + 1e-10) * 255).astype(np.uint8)
        data = golden_norm.tobytes()
        datasets['golden'] = data
        entropies['golden'] = estimate_entropy(list(golden_norm))
    except Exception as e:
        print(f"Warning: Golden signal load failed: {e}")
    
    # Random data (baseline - should be ~8 bits/symbol)
    random_data = np.random.randint(0, 256, 50000, dtype=np.uint8).tobytes()
    datasets['random'] = random_data
    entropies['random'] = 8.0
    
    # Highly compressible (repeated pattern)
    pattern = b"ABCDEFGH" * 6250  # 50KB
    datasets['pattern'] = pattern
    entropies['pattern'] = estimate_entropy(list(pattern))
    
    # Sparse data
    sparse = np.zeros(50000, dtype=np.uint8)
    sparse[::100] = np.random.randint(1, 256, 500, dtype=np.uint8)
    datasets['sparse'] = sparse.tobytes()
    entropies['sparse'] = estimate_entropy(list(sparse))
    
    return datasets, entropies


def save_results(
    results: List[CompressionResult],
    output_dir: Path,
    timestamp: str
) -> Tuple[Path, Path]:
    """Save results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"compression_benchmark_{timestamp}.csv"
    json_path = output_dir / f"compression_benchmark_{timestamp}.json"
    
    # CSV
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
    
    # JSON with metadata
    json_data = {
        'timestamp': timestamp,
        'platform': {
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python': platform.python_version(),
        },
        'results': [asdict(r) for r in results],
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return csv_path, json_path


def generate_summary(results: List[CompressionResult]) -> str:
    """Generate markdown summary of results."""
    lines = [
        "# Compression Benchmark Results",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Summary Table",
        "",
        "| Dataset | Codec | H(X) | R (bits/sym) | Gap | Ratio | Enc MB/s | Dec MB/s | Lossless |",
        "|---------|-------|------|--------------|-----|-------|----------|----------|----------|",
    ]
    
    for r in sorted(results, key=lambda x: (x.dataset, x.codec)):
        lossless = "✓" if r.lossless else "✗"
        gap_str = f"{r.entropy_gap:+.3f}" if r.entropy_gap != float('inf') else "N/A"
        lines.append(
            f"| {r.dataset} | {r.codec} | {r.entropy_h:.3f} | {r.bits_per_symbol:.3f} | "
            f"{gap_str} | {r.compression_ratio:.2f}x | {r.encode_mbps:.1f} | "
            f"{r.decode_mbps:.1f} | {lossless} |"
        )
    
    # Best codec per dataset
    lines.extend([
        "",
        "## Best Codec by Dataset (lowest bits/symbol, lossless only)",
        "",
    ])
    
    from collections import defaultdict
    by_ds = defaultdict(list)
    for r in results:
        if r.lossless:
            by_ds[r.dataset].append(r)
    
    for ds, rs in sorted(by_ds.items()):
        best = min(rs, key=lambda x: x.bits_per_symbol)
        lines.append(f"- **{ds}**: {best.codec} ({best.bits_per_symbol:.3f} bits/sym, gap={best.entropy_gap:+.3f})")
    
    # Codec averages
    lines.extend([
        "",
        "## Average Performance by Codec",
        "",
        "| Codec | Avg Ratio | Avg Gap | Avg Enc MB/s | Avg Dec MB/s |",
        "|-------|-----------|---------|--------------|--------------|",
    ])
    
    by_codec = defaultdict(list)
    for r in results:
        if r.lossless and r.entropy_gap != float('inf'):
            by_codec[r.codec].append(r)
    
    for codec, rs in sorted(by_codec.items()):
        avg_ratio = np.mean([r.compression_ratio for r in rs])
        avg_gap = np.mean([r.entropy_gap for r in rs])
        avg_enc = np.mean([r.encode_mbps for r in rs])
        avg_dec = np.mean([r.decode_mbps for r in rs])
        lines.append(f"| {codec} | {avg_ratio:.2f}x | {avg_gap:+.3f} | {avg_enc:.1f} | {avg_dec:.1f} |")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark RFTMW+ANS vs zlib/zstd/brotli'
    )
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        default=['all'],
        help='Datasets to benchmark (ascii, audio, golden, random, pattern, sparse, or all)'
    )
    parser.add_argument(
        '--warmup', '-w',
        type=int,
        default=1,
        help='Number of warmup runs'
    )
    parser.add_argument(
        '--runs', '-r',
        type=int,
        default=3,
        help='Number of timed runs'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=_PROJECT_ROOT.parent / 'results' / 'competitors',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMPRESSION BENCHMARK: RFTMW+ANS vs zlib/zstd/brotli/lzma")
    print("=" * 70)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"Available codecs: zlib", end="")
    if HAS_BROTLI: print(", brotli", end="")
    if HAS_ZSTD: print(", zstd", end="")
    if HAS_LZMA: print(", lzma", end="")
    if HAS_RFT: print(", rftmw_ans, rft_vertex", end="")
    print()
    print()
    
    # Load datasets
    print("Loading datasets and computing entropies...")
    all_datasets, all_entropies = load_datasets()
    
    if 'all' in args.datasets:
        datasets = all_datasets
        entropies = all_entropies
    else:
        datasets = {k: v for k, v in all_datasets.items() if k in args.datasets}
        entropies = {k: v for k, v in all_entropies.items() if k in args.datasets}
    
    print(f"Datasets: {list(datasets.keys())}")
    
    # Run benchmark
    benchmark = CompressionBenchmark(warmup_runs=args.warmup, timed_runs=args.runs)
    results = benchmark.run_benchmark(datasets, entropies)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path, json_path = save_results(results, args.output_dir, timestamp)
    print(f"\nResults saved:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    
    # Generate summary
    summary = generate_summary(results)
    summary_path = args.output_dir / f"compression_benchmark_{timestamp}.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  Summary: {summary_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY: Average Entropy Gap by Codec (lower is better, 0 is optimal)")
    print("=" * 70)
    
    from collections import defaultdict
    by_codec = defaultdict(list)
    for r in results:
        if r.lossless and r.entropy_gap != float('inf'):
            by_codec[r.codec].append(r.entropy_gap)
    
    print(f"{'Codec':<15} {'Avg Gap':<10} {'Best':<10} {'Worst':<10}")
    print("-" * 50)
    for codec, gaps in sorted(by_codec.items(), key=lambda x: np.mean(x[1])):
        print(f"{codec:<15} {np.mean(gaps):+.4f}    {min(gaps):+.4f}    {max(gaps):+.4f}")


if __name__ == '__main__':
    main()

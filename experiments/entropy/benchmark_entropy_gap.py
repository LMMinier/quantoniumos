#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Entropy Gap Benchmark
=====================

Benchmark comparing how close different codecs get to the Shannon limit.
Core experiment for validating RFTMW's compression efficiency.

Entropy Gap = R - H(X)  where:
  R = achieved bit rate (bits per symbol)
  H(X) = Shannon entropy (theoretical minimum)

A perfect compressor has entropy gap = 0.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
import numpy as np

# Add project root to path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.entropy.datasets import load_dataset, list_datasets
from experiments.entropy.measure_entropy import estimate_entropy, entropy_per_symbol


# =============================================================================
# Codec Registry
# =============================================================================

@dataclass
class CompressionResult:
    """Result of compressing data with a codec."""
    codec_name: str
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    bits_per_symbol: float
    encode_time_ms: float
    decode_time_ms: float
    lossless: bool
    max_error: float  # 0 for lossless, reconstruction error otherwise


def _quantize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Quantize float data to uint8."""
    data = np.asarray(data)
    if data.dtype == np.uint8:
        return data
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        normalized = (data - dmin) / (dmax - dmin)
    else:
        normalized = np.zeros_like(data)
    return (normalized * 255).astype(np.uint8)


def _quantize_to_int16(data: np.ndarray) -> np.ndarray:
    """Quantize float data to int16."""
    data = np.asarray(data)
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        normalized = (data - dmin) / (dmax - dmin)
    else:
        normalized = np.zeros_like(data)
    return ((normalized - 0.5) * 65534).astype(np.int16)


# =============================================================================
# Codec Implementations
# =============================================================================

def codec_rftmw_ans(data: np.ndarray) -> CompressionResult:
    """RFTMW + ANS entropy coding."""
    from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
    from algorithms.rft.compression.rft_vertex_codec import encode_tensor, decode_tensor
    import json
    
    original = data.copy()
    n_symbols = data.size
    original_bytes = n_symbols * data.itemsize
    
    # Encode
    t0 = time.perf_counter()
    container = encode_tensor(data.ravel().astype(np.float64))
    encode_time = (time.perf_counter() - t0) * 1000
    
    # Estimate compressed size from container
    container_json = json.dumps(container, default=str)
    compressed_bytes = len(container_json)
    
    # Decode
    t0 = time.perf_counter()
    decoded = decode_tensor(container)
    decoded = decoded.reshape(data.shape)
    decode_time = (time.perf_counter() - t0) * 1000
    
    # Check reconstruction
    max_error = float(np.max(np.abs(original.astype(np.float64) - decoded)))
    lossless = max_error < 1e-5
    
    return CompressionResult(
        codec_name='rftmw_ans',
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=original_bytes / max(compressed_bytes, 1),
        bits_per_symbol=compressed_bytes * 8 / n_symbols,
        encode_time_ms=encode_time,
        decode_time_ms=decode_time,
        lossless=lossless,
        max_error=max_error,
    )


def codec_fft_ans(data: np.ndarray) -> CompressionResult:
    """FFT + quantization + ANS entropy coding."""
    from algorithms.rft.compression.ans import ans_encode, ans_decode
    
    original = data.copy().ravel()
    n_symbols = original.size
    original_bytes = n_symbols * original.itemsize
    
    # FFT transform
    t0 = time.perf_counter()
    if np.iscomplexobj(original):
        coeffs = np.fft.fft(original.real)
    else:
        coeffs = np.fft.rfft(original.astype(np.float64))
    
    # Quantize coefficients to 8-bit per component
    real_part = np.real(coeffs)
    imag_part = np.imag(coeffs)
    scale = max(np.abs(real_part).max(), np.abs(imag_part).max(), 1e-10)
    
    # Quantize to 8-bit (0-255 range)
    real_q = ((real_part / scale + 1) * 127.5).clip(0, 255).astype(np.uint8)
    imag_q = ((imag_part / scale + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Combine
    combined = np.concatenate([real_q, imag_q]).tolist()
    
    # ANS encode
    encoded, freq_data = ans_encode(combined)
    encode_time = (time.perf_counter() - t0) * 1000
    
    # Compressed size: encoded data + scale + freq_data (approximate)
    compressed_bytes = len(encoded) * 2 + 8 + 100  # overhead estimate
    
    # Decode
    t0 = time.perf_counter()
    decoded_symbols = ans_decode(encoded, freq_data, len(combined))
    
    n_coeffs = len(decoded_symbols) // 2
    real_r = (np.array(decoded_symbols[:n_coeffs], dtype=np.float64) / 127.5 - 1) * scale
    imag_r = (np.array(decoded_symbols[n_coeffs:], dtype=np.float64) / 127.5 - 1) * scale
    coeffs_r = real_r + 1j * imag_r
    
    # Inverse FFT
    if len(coeffs_r) == len(original):
        reconstructed = np.fft.ifft(coeffs_r).real
    else:
        reconstructed = np.fft.irfft(coeffs_r, n=len(original))
    decode_time = (time.perf_counter() - t0) * 1000
    
    max_error = float(np.max(np.abs(original.astype(np.float64) - reconstructed)))
    
    return CompressionResult(
        codec_name='fft_ans',
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=original_bytes / max(compressed_bytes, 1),
        bits_per_symbol=compressed_bytes * 8 / n_symbols,
        encode_time_ms=encode_time,
        decode_time_ms=decode_time,
        lossless=False,  # Quantization is lossy
        max_error=max_error,
    )


def codec_dct_ans(data: np.ndarray) -> CompressionResult:
    """DCT + quantization + ANS entropy coding."""
    from scipy.fft import dct, idct
    from algorithms.rft.compression.ans import ans_encode, ans_decode
    
    original = data.copy().ravel().astype(np.float64)
    n_symbols = original.size
    original_bytes = n_symbols * 8  # float64
    
    # DCT transform
    t0 = time.perf_counter()
    coeffs = dct(original, type=2, norm='ortho')
    
    # Quantize to 8-bit
    scale = max(np.abs(coeffs).max(), 1e-10)
    coeffs_q = ((coeffs / scale + 1) * 127.5).clip(0, 255).astype(np.uint8).tolist()
    
    # ANS encode
    encoded, freq_data = ans_encode(coeffs_q)
    encode_time = (time.perf_counter() - t0) * 1000
    
    compressed_bytes = len(encoded) * 2 + 8 + 100  # overhead estimate
    
    # Decode
    t0 = time.perf_counter()
    decoded_symbols = ans_decode(encoded, freq_data, len(coeffs_q))
    
    coeffs_r = (np.array(decoded_symbols, dtype=np.float64) / 127.5 - 1) * scale
    reconstructed = idct(coeffs_r, type=2, norm='ortho')
    decode_time = (time.perf_counter() - t0) * 1000
    
    max_error = float(np.max(np.abs(original - reconstructed)))
    
    return CompressionResult(
        codec_name='dct_ans',
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=original_bytes / max(compressed_bytes, 1),
        bits_per_symbol=compressed_bytes * 8 / n_symbols,
        encode_time_ms=encode_time,
        decode_time_ms=decode_time,
        lossless=False,
        max_error=max_error,
    )


def codec_brotli(data: np.ndarray) -> CompressionResult:
    """Brotli general-purpose compression."""
    try:
        import brotli
    except ImportError:
        raise ImportError("brotli not installed. Install with: pip install brotli")
    
    original = _quantize_to_uint8(data.ravel())
    n_symbols = original.size
    original_bytes = n_symbols
    
    t0 = time.perf_counter()
    compressed = brotli.compress(original.tobytes(), quality=11)
    encode_time = (time.perf_counter() - t0) * 1000
    compressed_bytes = len(compressed)
    
    t0 = time.perf_counter()
    decompressed = np.frombuffer(brotli.decompress(compressed), dtype=np.uint8)
    decode_time = (time.perf_counter() - t0) * 1000
    
    max_error = float(np.max(np.abs(original.astype(np.int32) - decompressed.astype(np.int32))))
    lossless = np.array_equal(original, decompressed)
    
    return CompressionResult(
        codec_name='brotli',
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=original_bytes / max(compressed_bytes, 1),
        bits_per_symbol=compressed_bytes * 8 / n_symbols,
        encode_time_ms=encode_time,
        decode_time_ms=decode_time,
        lossless=lossless,
        max_error=max_error,
    )


def codec_zstd(data: np.ndarray) -> CompressionResult:
    """Zstandard compression."""
    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError("zstandard not installed. Install with: pip install zstandard")
    
    original = _quantize_to_uint8(data.ravel())
    n_symbols = original.size
    original_bytes = n_symbols
    
    t0 = time.perf_counter()
    cctx = zstd.ZstdCompressor(level=22)
    compressed = cctx.compress(original.tobytes())
    encode_time = (time.perf_counter() - t0) * 1000
    compressed_bytes = len(compressed)
    
    t0 = time.perf_counter()
    dctx = zstd.ZstdDecompressor()
    decompressed = np.frombuffer(dctx.decompress(compressed), dtype=np.uint8)
    decode_time = (time.perf_counter() - t0) * 1000
    
    max_error = float(np.max(np.abs(original.astype(np.int32) - decompressed.astype(np.int32))))
    lossless = np.array_equal(original, decompressed)
    
    return CompressionResult(
        codec_name='zstd',
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=original_bytes / max(compressed_bytes, 1),
        bits_per_symbol=compressed_bytes * 8 / n_symbols,
        encode_time_ms=encode_time,
        decode_time_ms=decode_time,
        lossless=lossless,
        max_error=max_error,
    )


def codec_hybrid_dct_rft(data: np.ndarray) -> CompressionResult:
    """Hybrid DCT + RFT (simplified)."""
    from scipy.fft import dct, idct
    from algorithms.rft.compression.rft_vertex_codec import encode_tensor, decode_tensor
    import json
    
    original = data.copy().ravel().astype(np.float64)
    n_symbols = original.size
    original_bytes = n_symbols * 8  # float64
    
    t0 = time.perf_counter()
    # DCT transform
    dct_coeffs = dct(original, type=2, norm='ortho')
    
    # Encode with vertex codec (uses RFT internally)
    container = encode_tensor(dct_coeffs)
    encode_time = (time.perf_counter() - t0) * 1000
    
    container_json = json.dumps(container, default=str)
    compressed_bytes = len(container_json)
    
    # Decode
    t0 = time.perf_counter()
    dct_r = decode_tensor(container)
    reconstructed = idct(dct_r, type=2, norm='ortho')
    decode_time = (time.perf_counter() - t0) * 1000
    
    max_error = float(np.max(np.abs(original - reconstructed)))
    
    return CompressionResult(
        codec_name='hybrid_dct_rft',
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=original_bytes / max(compressed_bytes, 1),
        bits_per_symbol=compressed_bytes * 8 / n_symbols,
        encode_time_ms=encode_time,
        decode_time_ms=decode_time,
        lossless=False,
        max_error=max_error,
    )


# Codec registry
CODECS: Dict[str, Callable] = {
    'rftmw_ans': codec_rftmw_ans,
    'fft_ans': codec_fft_ans,
    'dct_ans': codec_dct_ans,
    'brotli': codec_brotli,
    'zstd': codec_zstd,
    'hybrid_dct_rft': codec_hybrid_dct_rft,
}


# =============================================================================
# Benchmark Runner
# =============================================================================

def benchmark_entropy_gap(
    dataset_name: str,
    codecs: Optional[List[str]] = None,
    block_size: int = 256,
    max_blocks: int = 100,
    quantization_bits: int = 8
) -> Dict[str, Any]:
    """
    Benchmark entropy gap for specified codecs on a dataset.
    
    Returns:
        Dictionary with entropy measurements and codec results
    """
    if codecs is None:
        codecs = list(CODECS.keys())
    
    # Load data
    data = load_dataset(dataset_name, block_size=block_size, max_blocks=max_blocks)
    
    # Measure source entropy
    entropy_info = entropy_per_symbol(data, quantization_bits)
    source_entropy = entropy_info['H_plugin']
    
    results = {
        'dataset': dataset_name,
        'block_size': block_size,
        'n_blocks': len(data),
        'source_entropy_bits': source_entropy,
        'theoretical_min_bytes': source_entropy * data.size / 8,
        'codecs': {}
    }
    
    for codec_name in codecs:
        if codec_name not in CODECS:
            print(f"  Warning: Unknown codec '{codec_name}', skipping")
            continue
        
        try:
            codec_fn = CODECS[codec_name]
            result = codec_fn(data)
            
            # Compute entropy gap
            entropy_gap = result.bits_per_symbol - source_entropy
            
            # SANITY CHECK: Rate must be >= entropy (Shannon limit)
            # Allow small negative gaps due to measurement noise
            rate_valid = entropy_gap >= -0.1
            
            results['codecs'][codec_name] = {
                **asdict(result),
                'entropy_gap': entropy_gap,
                'gap_percent': (entropy_gap / source_entropy * 100) if source_entropy > 0 else 0,
                'rate_valid': rate_valid,
            }
            
        except Exception as e:
            results['codecs'][codec_name] = {'error': str(e)}
    
    return results


def generate_report(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate CSV and plots from benchmark results."""
    import csv
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV output
    csv_path = output_dir / f"{results['dataset']}_entropy_gap.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'codec', 'bits_per_symbol', 'entropy_gap', 'gap_percent',
            'compression_ratio', 'encode_ms', 'decode_ms', 'lossless', 'max_error'
        ])
        
        for name, data in results['codecs'].items():
            if 'error' in data:
                continue
            writer.writerow([
                name,
                f"{data['bits_per_symbol']:.4f}",
                f"{data['entropy_gap']:.4f}",
                f"{data['gap_percent']:.2f}",
                f"{data['compression_ratio']:.3f}",
                f"{data['encode_time_ms']:.2f}",
                f"{data['decode_time_ms']:.2f}",
                data['lossless'],
                f"{data['max_error']:.6f}"
            ])
    
    print(f"  CSV saved: {csv_path}")
    
    # Try to generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        codecs = []
        gaps = []
        for name, data in sorted(results['codecs'].items()):
            if 'error' not in data:
                codecs.append(name)
                gaps.append(data['entropy_gap'])
        
        if codecs:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['green' if g <= 0.1 else 'orange' if g <= 0.5 else 'red' for g in gaps]
            bars = ax.bar(codecs, gaps, color=colors, edgecolor='black')
            
            ax.axhline(y=0, color='blue', linestyle='--', linewidth=2, label='Shannon Limit')
            ax.set_ylabel('Entropy Gap (bits/symbol)')
            ax.set_xlabel('Codec')
            ax.set_title(f"Entropy Gap: {results['dataset']} (H={results['source_entropy_bits']:.3f} bits/sym)")
            ax.legend()
            
            # Add value labels
            for bar, gap in zip(bars, gaps):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{gap:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plot_path = output_dir / f"{results['dataset']}_entropy_gap.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Plot saved: {plot_path}")
    
    except ImportError:
        print("  (matplotlib not available, skipping plot)")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark entropy gap for compression codecs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all codecs on golden signals
  python benchmark_entropy_gap.py --dataset golden

  # Benchmark specific codecs
  python benchmark_entropy_gap.py --dataset audio --codecs rftmw_ans fft_ans zstd

  # Benchmark all datasets
  python benchmark_entropy_gap.py --all

  # Save results to custom directory
  python benchmark_entropy_gap.py --dataset ascii --output-dir ./my_results
        """
    )
    
    parser.add_argument('--dataset', '-d', choices=list_datasets())
    parser.add_argument('--all', '-a', action='store_true', help='Benchmark all datasets')
    parser.add_argument('--codecs', '-c', nargs='+', choices=list(CODECS.keys()),
                       help='Codecs to benchmark (default: all)')
    parser.add_argument('--block-size', '-b', type=int, default=256)
    parser.add_argument('--max-blocks', '-n', type=int, default=100)
    parser.add_argument('--quantization-bits', '-q', type=int, default=8)
    parser.add_argument('--output-dir', '-o', type=Path, 
                       default=_PROJECT_ROOT / 'results' / 'entropy_gap')
    parser.add_argument('--json', action='store_true', help='Output JSON results')
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all:
        parser.error("Specify --dataset or --all")
    
    datasets = list_datasets() if args.all else [args.dataset]
    all_results = {}
    
    print("="*60)
    print("ENTROPY GAP BENCHMARK")
    print("="*60)
    
    for ds in datasets:
        print(f"\nDataset: {ds}")
        print("-"*40)
        
        try:
            results = benchmark_entropy_gap(
                ds,
                codecs=args.codecs,
                block_size=args.block_size,
                max_blocks=args.max_blocks,
                quantization_bits=args.quantization_bits
            )
            all_results[ds] = results
            
            # Print summary table
            print(f"Source entropy: {results['source_entropy_bits']:.4f} bits/symbol")
            print(f"{'Codec':<20} {'Rate':<10} {'Gap':<10} {'Ratio':<10} {'Lossless'}")
            print("-"*60)
            
            for name, data in sorted(results['codecs'].items(), 
                                     key=lambda x: x[1].get('entropy_gap', 999)):
                if 'error' in data:
                    print(f"{name:<20} ERROR: {data['error'][:30]}")
                else:
                    print(f"{name:<20} {data['bits_per_symbol']:<10.4f} "
                          f"{data['entropy_gap']:<10.4f} {data['compression_ratio']:<10.3f} "
                          f"{'✓' if data['lossless'] else '✗'}")
            
            # Generate report
            generate_report(results, args.output_dir / 'plots')
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Save JSON
    if args.json:
        json_path = args.output_dir / 'all_results.json'
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nJSON saved: {json_path}")
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Transform Benchmark: Φ-RFT vs FFT/DCT/Wavelets
==============================================

Compares RFT against industry-standard transforms on real workloads.

Metrics:
- Forward transform time (µs)
- Inverse transform time (µs)
- Round-trip error (relative L2 norm)
- Energy compaction (% energy in top-k coefficients)
- Sparsity (L1/L2 ratio normalized)

Usage:
    python benchmark_transforms_vs_fft.py [--datasets all] [--sizes 256,1024,4096]
"""

import argparse
import csv
import json
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.fft import dct, idct

# Add project root
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

# Use optimized RFT with fused diagonals for better performance
from algorithms.rft.core.rft_optimized import (
    rft_forward_optimized as rft_forward,
    rft_inverse_optimized as rft_inverse,
)
# Also import original for comparison
from algorithms.rft.core.resonant_fourier_transform import (
    rft_forward_square as rft_forward_orig,
    rft_inverse_square as rft_inverse_orig,
)
from experiments.entropy.datasets import (
    load_ascii_corpus,
    load_audio_frames,
    load_golden_signals,
)

# Optional: PyWavelets for wavelet comparison
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


@dataclass
class TransformResult:
    """Result of a single transform benchmark."""
    transform: str
    dataset: str
    size: int
    fwd_us: float
    inv_us: float
    total_us: float
    roundtrip_err: float
    energy_top10_pct: float
    energy_top1pct_pct: float
    sparsity_ratio: float


def energy_compaction(X: np.ndarray, top_k: int) -> float:
    """Compute percentage of energy in top-k coefficients."""
    mags = np.abs(X.ravel()) ** 2
    total_energy = np.sum(mags)
    if total_energy == 0:
        return 0.0
    sorted_mags = np.sort(mags)[::-1]
    top_energy = np.sum(sorted_mags[:top_k])
    return 100.0 * top_energy / total_energy


def sparsity_ratio(X: np.ndarray) -> float:
    """
    Compute normalized sparsity ratio (L1/L2).
    Lower = more sparse. Range: [1, sqrt(n)]
    """
    l1 = np.linalg.norm(X.ravel(), 1)
    l2 = np.linalg.norm(X.ravel(), 2)
    n = X.size
    if l2 == 0:
        return 0.0
    # Normalize by sqrt(n) so range is [1/sqrt(n), 1]
    return l1 / (np.sqrt(n) * l2)


class TransformBenchmark:
    """Benchmark harness for transform comparison."""
    
    def __init__(self, warmup_runs: int = 3, timed_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs
        self.transforms = self._register_transforms()
    
    def _register_transforms(self) -> Dict[str, Tuple[Callable, Callable]]:
        """Register all transforms to benchmark."""
        transforms = {
            'fft': (
                lambda x: np.fft.fft(x, norm='ortho'),
                lambda X: np.fft.ifft(X, norm='ortho').real
            ),
            'rft': (
                rft_forward,
                rft_inverse
            ),
            'rft_orig': (
                rft_forward_orig,
                rft_inverse_orig
            ),
            'dct': (
                lambda x: dct(x, type=2, norm='ortho'),
                lambda X: idct(X, type=2, norm='ortho')
            ),
        }
        
        if HAS_PYWT:
            # Daubechies-4 wavelet
            transforms['dwt_db4'] = (
                lambda x: np.concatenate(pywt.dwt(x, 'db4')),
                lambda X: pywt.idwt(X[:len(X)//2], X[len(X)//2:], 'db4')[:len(X)//2*2]
            )
        
        return transforms
    
    def bench_single(
        self,
        x: np.ndarray,
        fwd: Callable,
        inv: Callable,
        name: str
    ) -> Dict[str, Any]:
        """Benchmark a single transform on a single input."""
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        
        # Warmup
        for _ in range(self.warmup_runs):
            X = fwd(x)
            _ = inv(X)
        
        # Timed forward
        fwd_times = []
        for _ in range(self.timed_runs):
            t0 = perf_counter()
            X = fwd(x)
            t1 = perf_counter()
            fwd_times.append((t1 - t0) * 1e6)
        
        # Timed inverse
        inv_times = []
        for _ in range(self.timed_runs):
            t0 = perf_counter()
            x_rec = inv(X)
            t1 = perf_counter()
            inv_times.append((t1 - t0) * 1e6)
        
        # Ensure x_rec is real and same size
        x_rec = np.real(x_rec)
        if len(x_rec) != n:
            # Wavelet may change size slightly
            x_rec = x_rec[:n] if len(x_rec) > n else np.pad(x_rec, (0, n - len(x_rec)))
        
        # Compute metrics
        err_norm = np.linalg.norm(x[:len(x_rec)] - x_rec)
        x_norm = np.linalg.norm(x)
        roundtrip_err = err_norm / x_norm if x_norm > 0 else 0.0
        
        return {
            'fwd_us': np.median(fwd_times),
            'inv_us': np.median(inv_times),
            'total_us': np.median(fwd_times) + np.median(inv_times),
            'roundtrip_err': roundtrip_err,
            'energy_top10_pct': energy_compaction(X, 10),
            'energy_top1pct_pct': energy_compaction(X, max(1, n // 100)),
            'sparsity_ratio': sparsity_ratio(X),
        }
    
    def run_benchmark(
        self,
        datasets: Dict[str, np.ndarray],
        sizes: Optional[List[int]] = None
    ) -> List[TransformResult]:
        """Run full benchmark across datasets and sizes."""
        results = []
        
        if sizes is None:
            sizes = [256, 1024, 4096, 16384]
        
        for ds_name, data in datasets.items():
            for size in sizes:
                if len(data) < size:
                    print(f"  Skipping {ds_name} @ n={size} (data too small)")
                    continue
                
                x = data[:size].astype(np.float64)
                
                for tf_name, (fwd, inv) in self.transforms.items():
                    try:
                        metrics = self.bench_single(x, fwd, inv, tf_name)
                        result = TransformResult(
                            transform=tf_name,
                            dataset=ds_name,
                            size=size,
                            **metrics
                        )
                        results.append(result)
                        
                        print(f"  {tf_name:10} | {ds_name:12} | n={size:5} | "
                              f"fwd={metrics['fwd_us']:8.1f}µs | "
                              f"err={metrics['roundtrip_err']:.2e}")
                    except Exception as e:
                        print(f"  {tf_name:10} | {ds_name:12} | n={size:5} | ERROR: {e}")
        
        return results


def load_datasets() -> Dict[str, np.ndarray]:
    """Load all benchmark datasets."""
    datasets = {}
    
    # ASCII corpus
    try:
        ascii_data = load_ascii_corpus(max_bytes=50000)
        datasets['ascii'] = ascii_data.astype(np.float64)
    except Exception as e:
        print(f"Warning: Could not load ASCII corpus: {e}")
        datasets['ascii'] = np.random.randint(32, 127, 50000).astype(np.float64)
    
    # Audio frames
    try:
        audio_data = load_audio_frames(max_samples=50000)
        datasets['audio'] = audio_data.astype(np.float64)
    except Exception as e:
        print(f"Warning: Could not load audio: {e}")
        # Synthetic audio-like signal
        t = np.linspace(0, 1, 50000)
        datasets['audio'] = (np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*880*t)).astype(np.float64)
    
    # Golden ratio signals
    try:
        golden_data = load_golden_signals(n_samples=50000)
        datasets['golden'] = golden_data.astype(np.float64)
    except Exception as e:
        print(f"Warning: Could not load golden signals: {e}")
        PHI = (1 + np.sqrt(5)) / 2
        t = np.arange(50000)
        datasets['golden'] = np.cos(2*np.pi * PHI**(t/5000)).astype(np.float64)
    
    # Random noise (baseline)
    datasets['noise'] = np.random.randn(50000)
    
    # Sparse signal
    sparse = np.zeros(50000)
    sparse[::100] = np.random.randn(500)
    datasets['sparse'] = sparse
    
    return datasets


def save_results(
    results: List[TransformResult],
    output_dir: Path,
    timestamp: str
) -> Tuple[Path, Path]:
    """Save results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"transform_benchmark_{timestamp}.csv"
    json_path = output_dir / f"transform_benchmark_{timestamp}.json"
    
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


def generate_summary(results: List[TransformResult]) -> str:
    """Generate markdown summary of results."""
    lines = [
        "# Transform Benchmark Results",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Summary by Transform",
        "",
        "| Transform | Avg Fwd (µs) | Avg Inv (µs) | Avg Error | Avg Sparsity |",
        "|-----------|--------------|--------------|-----------|--------------|",
    ]
    
    # Aggregate by transform
    from collections import defaultdict
    by_tf = defaultdict(list)
    for r in results:
        by_tf[r.transform].append(r)
    
    for tf, rs in sorted(by_tf.items()):
        avg_fwd = np.mean([r.fwd_us for r in rs])
        avg_inv = np.mean([r.inv_us for r in rs])
        avg_err = np.mean([r.roundtrip_err for r in rs])
        avg_sparse = np.mean([r.sparsity_ratio for r in rs])
        lines.append(f"| {tf} | {avg_fwd:.1f} | {avg_inv:.1f} | {avg_err:.2e} | {avg_sparse:.3f} |")
    
    lines.extend([
        "",
        "## Detailed Results by Dataset",
        "",
    ])
    
    # Group by dataset
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r.dataset].append(r)
    
    for ds, rs in sorted(by_ds.items()):
        lines.extend([
            f"### {ds}",
            "",
            "| Transform | Size | Fwd (µs) | Inv (µs) | Error | Energy Top-10 |",
            "|-----------|------|----------|----------|-------|---------------|",
        ])
        for r in sorted(rs, key=lambda x: (x.size, x.transform)):
            lines.append(
                f"| {r.transform} | {r.size} | {r.fwd_us:.1f} | {r.inv_us:.1f} | "
                f"{r.roundtrip_err:.2e} | {r.energy_top10_pct:.1f}% |"
            )
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Φ-RFT vs FFT/DCT/Wavelets'
    )
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        default=['all'],
        help='Datasets to benchmark (ascii, audio, golden, noise, sparse, or all)'
    )
    parser.add_argument(
        '--sizes', '-s',
        type=str,
        default='256,1024,4096,16384',
        help='Comma-separated list of sizes to test'
    )
    parser.add_argument(
        '--warmup', '-w',
        type=int,
        default=3,
        help='Number of warmup runs'
    )
    parser.add_argument(
        '--runs', '-r',
        type=int,
        default=10,
        help='Number of timed runs'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=_PROJECT_ROOT.parent / 'results' / 'competitors',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    print("=" * 60)
    print("TRANSFORM BENCHMARK: Φ-RFT vs FFT/DCT/Wavelets")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"Sizes: {sizes}")
    print(f"Warmup: {args.warmup}, Timed runs: {args.runs}")
    print()
    
    # Load datasets
    print("Loading datasets...")
    all_datasets = load_datasets()
    
    if 'all' in args.datasets:
        datasets = all_datasets
    else:
        datasets = {k: v for k, v in all_datasets.items() if k in args.datasets}
    
    print(f"Datasets: {list(datasets.keys())}")
    print()
    
    # Run benchmark
    print("Running benchmarks...")
    print("-" * 60)
    benchmark = TransformBenchmark(warmup_runs=args.warmup, timed_runs=args.runs)
    results = benchmark.run_benchmark(datasets, sizes)
    print("-" * 60)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path, json_path = save_results(results, args.output_dir, timestamp)
    print(f"\nResults saved:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    
    # Generate summary
    summary = generate_summary(results)
    summary_path = args.output_dir / f"transform_benchmark_{timestamp}.md"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"  Summary: {summary_path}")
    
    # Print quick summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    
    from collections import defaultdict
    by_tf = defaultdict(list)
    for r in results:
        by_tf[r.transform].append(r)
    
    print(f"{'Transform':<12} {'Avg Time (µs)':<15} {'Avg Error':<12} {'Sparsity':<10}")
    print("-" * 50)
    for tf, rs in sorted(by_tf.items()):
        avg_time = np.mean([r.total_us for r in rs])
        avg_err = np.mean([r.roundtrip_err for r in rs])
        avg_sparse = np.mean([r.sparsity_ratio for r in rs])
        print(f"{tf:<12} {avg_time:<15.1f} {avg_err:<12.2e} {avg_sparse:<10.3f}")


if __name__ == '__main__':
    main()

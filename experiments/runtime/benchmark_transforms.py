#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Runtime Benchmark Suite
========================

Timing comparisons between RFT, FFT, DCT with size sweeps.
Measures actual performance characteristics.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Callable
import numpy as np
from scipy.fft import dct, idct

# Add project root to path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse


@dataclass
class TimingResult:
    """Result of timing a transform."""
    transform_name: str
    size: int
    forward_mean_us: float
    forward_std_us: float
    inverse_mean_us: float
    inverse_std_us: float
    total_mean_us: float
    n_iterations: int


def benchmark_transform(
    forward_fn: Callable,
    inverse_fn: Callable,
    sizes: List[int],
    n_iterations: int = 100,
    warmup: int = 10
) -> List[TimingResult]:
    """
    Benchmark a transform across multiple sizes.
    
    Args:
        forward_fn: Forward transform function
        inverse_fn: Inverse transform function
        sizes: List of signal sizes to test
        n_iterations: Number of timing iterations
        warmup: Number of warmup iterations (not timed)
    
    Returns:
        List of timing results
    """
    results = []
    rng = np.random.default_rng(42)
    
    for size in sizes:
        x = rng.standard_normal(size).astype(np.float64)
        
        # Warmup
        for _ in range(warmup):
            X = forward_fn(x)
            _ = inverse_fn(X)
        
        # Time forward transform
        forward_times = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            X = forward_fn(x)
            t1 = time.perf_counter()
            forward_times.append((t1 - t0) * 1e6)  # Convert to microseconds
        
        # Time inverse transform
        inverse_times = []
        X = forward_fn(x)
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            _ = inverse_fn(X)
            t1 = time.perf_counter()
            inverse_times.append((t1 - t0) * 1e6)
        
        results.append(TimingResult(
            transform_name='',  # Set by caller
            size=size,
            forward_mean_us=np.mean(forward_times),
            forward_std_us=np.std(forward_times),
            inverse_mean_us=np.mean(inverse_times),
            inverse_std_us=np.std(inverse_times),
            total_mean_us=np.mean(forward_times) + np.mean(inverse_times),
            n_iterations=n_iterations,
        ))
    
    return results


# Transform registry
TRANSFORMS = {
    'RFT': {
        'forward': rft_forward,
        'inverse': lambda X: rft_inverse(X).real,
    },
    'FFT': {
        'forward': np.fft.fft,
        'inverse': lambda X: np.fft.ifft(X).real,
    },
    'RFFT': {
        'forward': np.fft.rfft,
        'inverse': lambda X: np.fft.irfft(X),
    },
    'DCT': {
        'forward': lambda x: dct(x, type=2, norm='ortho'),
        'inverse': lambda X: idct(X, type=2, norm='ortho'),
    },
}


def run_benchmark(
    transforms: List[str],
    sizes: List[int],
    n_iterations: int = 100
) -> Dict[str, List[TimingResult]]:
    """Run benchmarks for specified transforms and sizes."""
    all_results = {}
    
    for name in transforms:
        if name not in TRANSFORMS:
            print(f"Warning: Unknown transform '{name}', skipping")
            continue
        
        print(f"Benchmarking {name}...")
        transform = TRANSFORMS[name]
        
        results = benchmark_transform(
            transform['forward'],
            transform['inverse'],
            sizes,
            n_iterations
        )
        
        # Set transform name
        for r in results:
            r.transform_name = name
        
        all_results[name] = results
    
    return all_results


def generate_report(
    results: Dict[str, List[TimingResult]],
    output_dir: Path
) -> None:
    """Generate CSV and plots from benchmark results."""
    import csv
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV output
    csv_path = output_dir / 'transform_timing.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'transform', 'size', 'forward_us', 'forward_std', 
            'inverse_us', 'inverse_std', 'total_us'
        ])
        
        for name, timings in results.items():
            for t in timings:
                writer.writerow([
                    name, t.size, 
                    f"{t.forward_mean_us:.2f}", f"{t.forward_std_us:.2f}",
                    f"{t.inverse_mean_us:.2f}", f"{t.inverse_std_us:.2f}",
                    f"{t.total_mean_us:.2f}"
                ])
    
    print(f"CSV saved: {csv_path}")
    
    # JSON output
    json_path = output_dir / 'transform_timing.json'
    json_data = {
        name: [asdict(t) for t in timings]
        for name, timings in results.items()
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"JSON saved: {json_path}")
    
    # Try to generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Forward transform timing
        ax1 = axes[0]
        for name, timings in results.items():
            sizes = [t.size for t in timings]
            times = [t.forward_mean_us for t in timings]
            ax1.loglog(sizes, times, 'o-', label=name, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Signal Size (N)')
        ax1.set_ylabel('Time (μs)')
        ax1.set_title('Forward Transform Timing')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add O(N log N) reference line
        ref_sizes = np.array(sizes)
        ref_times = ref_sizes * np.log2(ref_sizes) / 10  # Scaled
        ax1.loglog(ref_sizes, ref_times, '--', color='gray', 
                   label='O(N log N)', alpha=0.5)
        
        # Total timing
        ax2 = axes[1]
        for name, timings in results.items():
            sizes = [t.size for t in timings]
            times = [t.total_mean_us for t in timings]
            ax2.loglog(sizes, times, 'o-', label=name, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Signal Size (N)')
        ax2.set_ylabel('Time (μs)')
        ax2.set_title('Round-Trip (Forward + Inverse) Timing')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = output_dir / 'transform_timing.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved: {plot_path}")
        
    except ImportError:
        print("(matplotlib not available, skipping plot)")


def print_summary(results: Dict[str, List[TimingResult]]) -> None:
    """Print summary table to console."""
    print("\n" + "="*80)
    print("TRANSFORM TIMING SUMMARY")
    print("="*80)
    
    # Get all sizes
    sizes = sorted(set(t.size for timings in results.values() for t in timings))
    
    # Header
    print(f"{'Size':>10}", end='')
    for name in results.keys():
        print(f" | {name:>12}", end='')
    print(" | (μs)")
    print("-"*80)
    
    # Data rows (forward timing)
    print("Forward Transform:")
    for size in sizes:
        print(f"{size:>10}", end='')
        for name, timings in results.items():
            timing = next((t for t in timings if t.size == size), None)
            if timing:
                print(f" | {timing.forward_mean_us:>12.1f}", end='')
            else:
                print(f" | {'N/A':>12}", end='')
        print()
    
    print("-"*80)
    print("Round-Trip (Fwd + Inv):")
    for size in sizes:
        print(f"{size:>10}", end='')
        for name, timings in results.items():
            timing = next((t for t in timings if t.size == size), None)
            if timing:
                print(f" | {timing.total_mean_us:>12.1f}", end='')
            else:
                print(f" | {'N/A':>12}", end='')
        print()
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark transform timing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all transforms with default sizes
  python benchmark_transforms.py

  # Benchmark specific transforms
  python benchmark_transforms.py --transforms RFT FFT DCT

  # Custom size range
  python benchmark_transforms.py --sizes 64 256 1024 4096 16384

  # More iterations for accuracy
  python benchmark_transforms.py --iterations 500
        """
    )
    
    parser.add_argument(
        '--transforms', '-t',
        nargs='+',
        choices=list(TRANSFORMS.keys()),
        default=list(TRANSFORMS.keys()),
        help='Transforms to benchmark'
    )
    parser.add_argument(
        '--sizes', '-s',
        nargs='+',
        type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        help='Signal sizes to test'
    )
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=100,
        help='Number of timing iterations per size'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=_PROJECT_ROOT / 'results' / 'runtime',
        help='Output directory for results'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRANSFORM RUNTIME BENCHMARK")
    print("="*60)
    print(f"Transforms: {', '.join(args.transforms)}")
    print(f"Sizes: {args.sizes}")
    print(f"Iterations: {args.iterations}")
    print()
    
    results = run_benchmark(
        args.transforms,
        args.sizes,
        args.iterations
    )
    
    if not args.quiet:
        print_summary(results)
    
    generate_report(results, args.output_dir)
    
    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()

# Competitor Benchmarks

**Compare QuantoniumOS stack against industry-standard tools.**

This benchmark suite compares:
1. **Transform layer**: Φ-RFT vs FFT/DCT/Wavelets
2. **Compression layer**: RFTMW+ANS/Vertex vs zlib/zstd/brotli
3. **Crypto layer**: RFT cipher vs AES-GCM/ChaCha20 (throughput only!)

## Quick Start

```bash
# Install dependencies
pip install -r requirements-bench.txt

# Run all benchmarks (laptop preset, ~5 minutes)
python experiments/competitors/run_all_benchmarks.py --preset laptop

# Quick run for CI (~1 minute)
python experiments/competitors/run_all_benchmarks.py --preset quick

# Run specific benchmarks
python experiments/competitors/run_all_benchmarks.py --benchmarks transforms,compression
```

## Dependencies

```bash
# Core dependencies (in pyproject.toml)
pip install -e .

# Additional benchmark dependencies
pip install brotli zstandard cryptography pywavelets psutil

# Or install all at once
pip install -r requirements-bench.txt
```

## Individual Benchmarks

### Transform Benchmark (RFT vs FFT/DCT)

```bash
python experiments/competitors/benchmark_transforms_vs_fft.py \
    --datasets ascii audio golden \
    --sizes 256,1024,4096,16384 \
    --runs 10
```

**Metrics:**
- Forward/inverse transform time (µs)
- Round-trip reconstruction error
- Energy compaction (% in top-k coefficients)
- Sparsity ratio (L1/L2 normalized)

### Compression Benchmark (RFTMW vs zlib/zstd/brotli)

```bash
python experiments/competitors/benchmark_compression_vs_codecs.py \
    --datasets ascii audio golden random pattern \
    --runs 5
```

**Metrics:**
- Bits per symbol (R)
- Entropy gap: R - H(X) (lower is better, 0 is Shannon limit)
- Compression ratio
- Encode/decode throughput (MB/s)
- Lossless verification

### Crypto Throughput Benchmark

```bash
python experiments/competitors/benchmark_crypto_throughput.py \
    --sizes 1024,4096,16384,65536 \
    --runs 10
```

**Metrics:**
- Encrypt/decrypt throughput (MB/s)
- Avalanche effect (% bit diffusion)

⚠️ **WARNING**: This is NOT a security comparison! The RFT cipher is a research tool
and should NOT be used for production security applications.

## Output

Results are saved to `results/competitors/`:

```
results/competitors/
├── transform_benchmark_YYYYMMDD_HHMMSS.csv
├── transform_benchmark_YYYYMMDD_HHMMSS.json
├── transform_benchmark_YYYYMMDD_HHMMSS.md
├── compression_benchmark_YYYYMMDD_HHMMSS.csv
├── compression_benchmark_YYYYMMDD_HHMMSS.json
├── compression_benchmark_YYYYMMDD_HHMMSS.md
├── crypto_benchmark_YYYYMMDD_HHMMSS.csv
├── crypto_benchmark_YYYYMMDD_HHMMSS.json
├── crypto_benchmark_YYYYMMDD_HHMMSS.md
└── competitor_benchmark_report_YYYYMMDD_HHMMSS.md
```

## Presets

| Preset | Description | Time |
|--------|-------------|------|
| `quick` | Fast run for CI | ~1 min |
| `laptop` | Balanced (default) | ~5 min |
| `desktop` | Full for workstations | ~15 min |
| `full` | Comprehensive | ~30+ min |

## Interpretation Guide

### Transform Results

- **Round-trip error < 1e-10**: Transform is numerically stable
- **RFT vs FFT speed**: RFT uses FFT internally, should be similar
- **Energy compaction**: Higher % in fewer coefficients = better for compression

### Compression Results

| Entropy Gap | Rating |
|-------------|--------|
| < 0.1 | Excellent (near Shannon limit) |
| < 0.5 | Good |
| < 1.0 | Acceptable |
| > 1.0 | Poor |

### Crypto Results

- **Avalanche ~50%**: Good bit diffusion
- **AES-GCM typically fastest**: Uses hardware AES-NI
- **ChaCha20**: Optimized for software
- **RFT cipher**: Research only, not for production!

## Adding Your Own Datasets

Edit `experiments/entropy/datasets.py` to add new data sources:

```python
def load_your_dataset(max_bytes: int = 100000) -> np.ndarray:
    # Load your data here
    data = ...
    return np.array(data, dtype=np.uint8)
```

Then update the benchmark scripts to include your dataset.

## CI Integration

The benchmarks can run in GitHub Actions:

```yaml
- name: Run competitor benchmarks
  run: |
    pip install -r requirements-bench.txt
    python experiments/competitors/run_all_benchmarks.py --preset quick
```

Results are saved as artifacts for comparison across runs.

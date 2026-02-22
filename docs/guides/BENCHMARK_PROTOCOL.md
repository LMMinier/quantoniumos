# Benchmark Protocol

> **Purpose:** Standardize benchmark methodology to prevent "unfair comparison" criticism.

---

## Mandatory Checklist

Every benchmark MUST satisfy ALL of the following:

| Requirement | Description | Verification |
|-------------|-------------|--------------|
| ✅ Same input length | All transforms use identical signal length N | Assert N equality |
| ✅ Same energy/bitrate | Equal compression ratios or coefficient counts | Log target ratio |
| ✅ Same reconstruction metric | PSNR, MSE, or specified metric applied uniformly | Metric in filename |
| ✅ Same preprocessing | DC removal, normalization, windowing | Document in config |
| ✅ Same runtime environment | Python version, NumPy version, hardware | Log in results |
| ✅ Same random seed | For reproducibility | Seed in config |

---

## Standard Benchmark Configuration

```python
BENCHMARK_CONFIG = {
    # Signal parameters
    "signal_lengths": [64, 128, 256, 512, 1024, 2048],
    "coefficient_retention": [0.05, 0.10, 0.25, 0.50],  # 5%, 10%, 25%, 50%
    
    # Preprocessing
    "remove_dc": True,
    "normalize": True,
    "window": "hann",
    
    # Metrics
    "metrics": ["psnr", "mse", "sparsity", "energy_compaction"],
    
    # Reproducibility
    "random_seed": 42,
    "num_trials": 10,
    
    # Environment
    "python_version": "3.10+",
    "numpy_version": "1.24+",
}
```

---

## Transform Comparison Protocol

### Transforms Under Test

| Transform | Implementation | Notes |
|-----------|----------------|-------|
| FFT | `numpy.fft.fft` | Reference baseline |
| DCT | `scipy.fftpack.dct` | Type-II DCT |
| DWT | `pywt.wavedec` | Daubechies-4 |
| Φ-RFT (selectable impl) | `tools/benchmarking/rft_vs_fft_benchmark.py --rft-impl ...` | This work; output labels `rft_impl` |

### Fair Comparison Rules

1. **No cherrypicking:** Report ALL results, including losses
2. **Same coefficient budget:** If RFT keeps K coefficients, so do others
3. **Same metric:** Don't compare PSNR for one, MSE for another
4. **Same signals:** Identical test signals for all transforms
5. **Same implementation quality:** Production-level for all

---

## Signal Classes

### Class 1: In-Family (RFT-Favorable)

Golden-ratio quasi-periodic signals where RFT is expected to win.

```python
def generate_in_family_signal(N, seed=42):
    """Signal with golden-ratio frequency structure."""
    np.random.seed(seed)
    t = np.linspace(0, 1, N)
    phi = (1 + np.sqrt(5)) / 2
    signal = (np.sin(2*np.pi*10*t) + 
              np.sin(2*np.pi*10*phi*t) + 
              0.1*np.random.randn(N))
    return signal
```

### Class 2: Out-of-Family (RFT-Unfavorable)

Signals where RFT should NOT have advantage.

```python
def generate_out_of_family_signal(N, seed=42):
    """White noise - no structure to exploit."""
    np.random.seed(seed)
    return np.random.randn(N)
```

### Class 3: Mixed (Real-World)

Real-world signals with mixed characteristics.

```python
def generate_mixed_signal(N, seed=42):
    """Chirp + noise - partial structure."""
    np.random.seed(seed)
    t = np.linspace(0, 1, N)
    signal = np.sin(2*np.pi*(5 + 20*t)*t) + 0.3*np.random.randn(N)
    return signal
```

---

## Required Output Format

Every benchmark MUST output:

**Minimum requirement:** results must be unambiguous and traceable to an implementation choice.

This repo uses two common formats:

1) **JSON** (for multi-metric, multi-class benchmark suites)
2) **CSV** (for transform-vs-baseline harnesses). For CSV harnesses, include an explicit `rft_impl` column.

```json
{
  "benchmark_id": "rft_vs_fft_20251214_001",
  "date": "2025-12-14",
  "config": {
    "signal_length": 256,
    "coefficient_retention": 0.10,
    "preprocessing": "dc_removal+hann",
    "random_seed": 42,
    "num_trials": 10
  },
  "environment": {
    "python": "3.12.0",
    "numpy": "1.26.0",
    "scipy": "1.11.0",
    "platform": "Linux 5.15.0"
  },
  "results": {
    "in_family": {
      "rft": {"psnr_mean": 45.2, "psnr_std": 1.3, "runtime_ms": 12.4},
      "fft": {"psnr_mean": 30.1, "psnr_std": 0.8, "runtime_ms": 0.2},
      "dct": {"psnr_mean": 32.5, "psnr_std": 0.9, "runtime_ms": 0.3},
      "winner": "rft"
    },
    "out_of_family": {
      "rft": {"psnr_mean": 12.1, "psnr_std": 0.5, "runtime_ms": 12.4},
      "fft": {"psnr_mean": 12.0, "psnr_std": 0.5, "runtime_ms": 0.2},
      "dct": {"psnr_mean": 12.0, "psnr_std": 0.5, "runtime_ms": 0.3},
      "winner": "tie"
    },
    "mixed": {
      "rft": {"psnr_mean": 28.3, "psnr_std": 2.1, "runtime_ms": 12.4},
      "fft": {"psnr_mean": 25.1, "psnr_std": 1.8, "runtime_ms": 0.2},
      "dct": {"psnr_mean": 26.8, "psnr_std": 1.9, "runtime_ms": 0.3},
      "winner": "rft"
    }
  },
  "summary": {
    "rft_win_rate": 0.67,
    "rft_loss_rate": 0.0,
    "tie_rate": 0.33,
    "note": "RFT wins on structured signals, ties on noise"
  }
}
```

---

## Forbidden Practices

| Practice | Why Forbidden |
|----------|---------------|
| Reporting only wins | Cherry-picking |
| Different N for different transforms | Unfair comparison |
| Different metrics for different transforms | Apples to oranges |
| Omitting runtime | Hiding computational cost |
| Omitting failure cases | Overclaiming |
| "Optimized" RFT vs "naive" FFT | Unfair implementation quality |
| Custom preprocessing for RFT only | Data snooping |

---

## Reproducibility Requirements

### Code

```bash
# Anyone should be able to run:
python benchmarks/rft_realworld_benchmark.py --config standard
# And get identical results (within numerical precision)
```

### Data

- All test signals are generated from documented seeds
- Real-world data has documented sources
- Preprocessing is deterministic

### Environment

```bash
# Lock dependencies
pip freeze > requirements-lock.txt
```

---

## Expected Results Template

### Honest Reporting

See [VERIFIED_BENCHMARKS](docs/research/benchmarks/VERIFIED_BENCHMARKS.md) for the current reproducible metrics.

**Interpretation:** RFT excels on golden-ratio structured signals; performance on other families is mixed.

---

## Validation Tests

Run before publishing any benchmark:

```python
def validate_benchmark(results):
    """Ensure benchmark meets protocol."""
    assert "config" in results
    assert results["config"]["random_seed"] is not None
    assert "environment" in results
    assert "out_of_family" in results["results"]  # MUST include unfavorable cases
    assert results["summary"]["rft_loss_rate"] >= 0  # MUST report losses
    return True
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-14 | Initial protocol |

---

*Last updated: December 2025*

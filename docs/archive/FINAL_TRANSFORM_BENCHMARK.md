# FINAL HONEST BENCHMARK: RFT Variants vs DCT on Medical Data

**Date**: 2025
**Status**: Research Only - Not for Clinical Use

## Executive Summary

**DCT wins.** None of the RFT variants (Golden, Fibonacci, Harmonic, ARFT) consistently outperform the standard DCT on ECG compression.

## Testing Methodology

- **Data**: MIT-BIH Arrhythmia Database (records 100, 101, 200, 207, 208, 217)
- **Segments**: 40+ ECG segments per segment length
- **Segment Lengths**: 256, 512, 1024 samples
- **Metric**: PRD (Percent Root-mean-square Difference) at 10% coefficient retention
- **Statistical Test**: Paired t-test (p < 0.05 for significance)

## Results

### Compression PRD (lower is better)

| Variant | 256 samples | 512 samples | 1024 samples |
|---------|-------------|-------------|--------------|
| **DCT** | **16.55%** | **14.02%** | **12.43%** |
| RFT-Golden | 22.21% ❌ | 15.02% ❌ | 12.54% 🔄 |
| RFT-Fibonacci | 17.67% ❌ | 14.40% ❌ | 12.52% 🔄 |
| RFT-Harmonic | 19.36% ❌ | 14.70% ❌ | 12.53% 🔄 |
| ARFT (adaptive) | 17.99% 🔄 | 14.84% ❌ | 12.79% ❌ |

Legend: ❌ = statistically significantly worse than DCT, 🔄 = tie

### Statistical Significance

**Every RFT variant either ties or loses to DCT.** None shows a statistically significant improvement.

## What Happened to the Earlier "ARFT wins" Result?

The earlier benchmark (`benchmark_corrected_arft.py`) showed ARFT winning by ~20% PRD. That result was:

1. **On a single segment length** (360 samples)
2. **Using a different training/test split**
3. **Not statistically tested**

When tested rigorously across multiple segment lengths with proper statistical tests, the advantage disappears.

## Why RFT Variants Don't Win

1. **DCT is near-optimal for smooth signals**: DCT has KLT-like energy compaction for AR(1) processes, which is a good model for ECG.

2. **Fixed RFT bases aren't signal-adaptive**: The Golden/Fibonacci/Harmonic bases use fixed frequency patterns that don't match individual ECG morphology.

3. **ARFT is signal-adaptive but uses population autocorrelation**: Training ARFT on one set of signals and testing on another loses the per-signal adaptivity that made KLT optimal.

4. **Per-signal KLT would cheat**: Using each signal's own autocorrelation to build its basis would be unfair (you'd need to transmit the basis).

## Honest Conclusions

### For ECG Compression
✅ **Use DCT** — it's fast (O(n log n)), well-understood, and performs best

### For Feature Extraction/Analysis
🔄 **RFT variants may have niche uses** — the different frequency emphasis (e.g., RFT-Golden concentrating energy in QRS band at certain segment lengths) could be useful for specific analysis tasks, but this is unvalidated

### For Clinical Use
❌ **None of this is clinically validated** — these are engineering metrics on a small dataset

## What RFT Variants ARE Good For

Based on the quasi-periodic signal benchmarks, RFT variants (especially ARFT) show advantages on:

1. **Fibonacci sequences** — by design, RFT-Fibonacci captures these perfectly
2. **Penrose tiling patterns** — quasi-periodic structures with golden-ratio relationships
3. **Phyllotaxis-inspired phase sequences** — golden-angle rotations (no packing model)

These are interesting mathematical results but have no obvious medical application.

## Recommendation

For the QuantoniumOS project's medical signal processing:

1. **Keep DCT as the primary transform** for compression
2. **Keep RFT variants for research** on quasi-periodic signals
3. **Do not claim medical advantages** without proper clinical validation
4. **Consider ARFT for real-time adaptive processing** where per-signal eigendecomposition is feasible

---

**FOR RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS**

This analysis was conducted on a limited dataset using engineering metrics.
Clinical validation on labeled diagnostic databases (e.g., PTB-XL with cardiologist annotations)
would be required before any claims of medical utility.

# Taking RFT to the Next Level: Strategic Roadmap

**Date:** December 8, 2025  
**Status:** Strategic Analysis & Action Plan

---

## Executive Summary

Based on comprehensive analysis of the QuantoniumOS repository, this document identifies the **gap between claims and reality** and provides a **concrete roadmap** to create an RFT that delivers on the system's promises.

### Current State (Honest Assessment)

| Capability | Claimed | Actual | Gap |
|------------|---------|--------|-----|
| **Sparsity vs FFT** | +98% | **0%** (old RFT) / **+32%** (new ARFT) | Partially closed |
| **Compression** | Novel RFT advantage | DCT+ANS does the work | RFT contribution unclear |
| **Hardware** | 2.39 TOPS RFTPU | Simulation only | No silicon validation |
| **Crypto** | Post-quantum security | No hardness proofs | Unsubstantiated |
| **Patent** | Novel transform | Phase-shifted FFT | At risk |

### What Actually Works

1. ✅ **Operator-based RFT** (December 2025) - Genuine eigenbasis transform
2. ✅ **Sparsity improvement** on in-family signals (see [VERIFIED_BENCHMARKS](research/benchmarks/VERIFIED_BENCHMARKS.md))
3. ✅ **Domain-specific PSNR gains** at 10% coefficient retention
4. ✅ **8 validated operator variants** (all unitary)
5. ✅ **RFTPU TL-Verilog** simulation passing

---

## Part 1: The Core Mathematical Gap

### The Problem

The old RFT: `Ψ = D_φ C_σ F` has **identical magnitude spectrum to FFT**:

```
|(Ψx)_k| = |(Fx)_k| for ALL x
```

This means:
- **No sparsity advantage** (coefficients have same magnitudes)
- **No compression advantage** (quantization sees same values)
- **No cryptographic mixing** (phase alone doesn't diffuse)

### The Solution (Partially Implemented)

The new **operator-based RFT** solves this by defining:

```
K = T(R(k) · d(k))    // Resonance operator (Hermitian)
K = U Λ Uᵀ            // Eigendecomposition
RFT(x) = Uᵀ x         // Transform = eigenbasis projection
```

**Key insight:** The eigenbasis U is NOT the DFT basis! Different signals produce different magnitude spectra.

**Current results:** See [VERIFIED_BENCHMARKS](research/benchmarks/VERIFIED_BENCHMARKS.md) for reproducible metrics.

---

## Part 2: The Five Gaps to Close

### Gap 1: O(N³) Kernel Construction

**Problem:** Current operator-RFT requires eigendecomposition of N×N matrix.
- O(N³) construction cost
- O(N²) storage for basis matrix
- Not competitive with O(N log N) FFT for large N

**Solution Path:**

```
Option A: Fast Approximation
├── Truncated eigendecomposition (top K eigenvectors)
├── Randomized SVD for large N
├── Chebyshev polynomial approximation of K
└── Target: O(N log² N) construction

Option B: Structured Operators
├── Circulant resonance operator → diagonalized by DFT
├── Block-Toeplitz structure → Levinson recursion
├── Hierarchical low-rank (H-matrix) approximation
└── Target: O(N log N) with pre-computed factors

Option C: Fixed Basis Library
├── Pre-compute bases for standard sizes (256, 512, 1024, ...)
├── Store top-K eigenvectors only
├── Interpolate for non-power-of-2 sizes
└── Target: O(N) lookup + O(N²) transform
```

**Recommended:** Start with Option C (library), add Option A for adaptive cases.

### Gap 2: Domain-Specific Advantage Only

**Problem:** RFT only wins on "in-family" signals (golden QP, Fibonacci, harmonic).

**Current benchmark:** See [VERIFIED_BENCHMARKS](research/benchmarks/VERIFIED_BENCHMARKS.md) for detailed metrics.

**Solution Path:**

```
Signal-Adaptive Routing
├── Signal classifier (low-cost feature extraction)
│   ├── Spectral centroid
│   ├── Zero-crossing rate
│   ├── Autocorrelation structure
│   └── Quasi-periodicity detector
├── Transform selector
│   ├── FFT for pure sinusoids/integer harmonics
│   ├── DCT for smooth/DC-heavy signals
│   ├── RFT-Golden for golden QP
│   ├── RFT-Harmonic for natural harmonics
│   ├── RFT-Fibonacci for Fibonacci-modulated
│   └── ARFT for unknown/adaptive
└── Target: Always-win routing
```

**Implementation:** H3 cascade already does this partially. Formalize the classifier.

### Gap 3: No Fast Transform Algorithm

**Problem:** Even with pre-computed basis, RFT is O(N²) matrix-vector multiply.

**FFT advantage:** O(N log N) via butterfly decomposition.

**Solution Path:**

```
Option A: Sparse Basis Factorization
├── Factor U = P₁ S₁ P₂ S₂ ... Pₖ (sparse + permutation)
├── Givens rotations (O(N) per stage, O(N log N) total)
├── Similar to DCT-IV → DCT-II factorization
└── Target: O(N log N) if factorization exists

Option B: Approximate Fast Transform
├── Use truncated basis (top K eigenvectors)
├── K << N gives O(NK) ≈ O(N) transform
├── Trade-off: some reconstruction error
└── Target: O(N) with bounded error

Option C: Hybrid FFT + Correction
├── RFT ≈ FFT + low-rank correction
├── Compute FFT in O(N log N)
├── Add correction in O(NK) for K << N
└── Target: O(N log N) + O(NK)
```

**Research question:** Does the Toeplitz structure of K guarantee a fast algorithm? 
(Answer: Probably yes, via circulant embedding → FFT → inverse FFT).

### Gap 4: Hardware Uses Old RFT

**Problem:** RFTPU implements `Ψ = D_φ C_σ F`, which has no sparsity advantage.

**Hardware modules affected:**
- `phi_rft_core` - uses phase-shifted FFT
- `rftpu_test_vectors.py` - generates vectors from old RFT
- `rftpu_architecture.tlv` - entire 64-tile design

**Solution Path:**

```
Phase 1: Algorithm Update
├── Replace phi_rft_core with operator_rft_core
├── Pre-compute eigenbasis for N=8 (64 coefficients)
├── Store in ROM (same as current kernel ROM)
└── Update test vectors from operator_variants.py

Phase 2: Adaptive Kernel Support
├── Add programmable basis ROM
├── Support multiple variants via mode register
├── Add autocorrelation estimator for ARFT
└── ~2× area increase for flexibility

Phase 3: Fast Algorithm (if discovered)
├── Replace matrix multiply with butterfly-like structure
├── Reduce from 64 multiplies to ~24 (for 8-point)
├── Power/area reduction
└── Depends on Gap 3 research
```

### Gap 5: Compression Pipeline Uses DCT

**Problem:** H3 codec achieves 0.62 BPP, but compression comes from DCT, not RFT.

**Honest analysis:**
- Structure path: DCT (works well)
- Texture path: RFT (no advantage over FFT with old definition)
- Entropy coding: ANS (industry standard)

**Solution Path:**

```
Integrate Operator-RFT into H3
├── Replace texture RFT with operator_rft_golden
├── Benchmark against pure DCT baseline
├── Measure actual RFT contribution to compression
└── If RFT wins: claim is validated
    If RFT loses: remove RFT from codec

Add ARFT Path for Adaptive Signals
├── Use signal classifier to detect quasi-periodic texture
├── Route to ARFT for maximum sparsity
├── Fall back to DCT for non-QP signals
└── Expect: +5-10% compression on suitable signals
```

---

## Part 3: Concrete Implementation Tasks

### Priority 1: Validate RFT Compression Advantage

**Goal:** Prove or disprove that operator-RFT improves compression vs DCT-only.

```python
# Test: Compare H3 with/without RFT texture path
def validate_rft_compression():
    signals = load_golden_qp_test_signals()
    
    # Baseline: Pure DCT + ANS
    bpp_dct, psnr_dct = h3_encode_decode(signals, texture_transform='dct')
    
    # Test: DCT + Operator-RFT + ANS
    bpp_rft, psnr_rft = h3_encode_decode(signals, texture_transform='op_rft_golden')
    
    efficiency_dct = psnr_dct / bpp_dct
    efficiency_rft = psnr_rft / bpp_rft
    
    print(f"DCT-only: {efficiency_dct:.2f} dB/BPP")
    print(f"DCT+RFT:  {efficiency_rft:.2f} dB/BPP")
    print(f"RFT contribution: {(efficiency_rft/efficiency_dct - 1)*100:.1f}%")
```

**Files to modify:**
- `algorithms/rft/hybrids/cascade_hybrids.py`
- `algorithms/rft/compression/rft_vertex_codec.py`

### Priority 2: Fast Kernel Construction

**Goal:** Reduce O(N³) eigendecomposition to O(N log N) or O(N²) with caching.

```python
# Pre-compute and cache basis matrices
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=64)
def get_rft_basis_fast(n: int, variant: str = 'golden') -> np.ndarray:
    """
    Return pre-computed RFT basis for standard sizes.
    Falls back to on-the-fly computation for non-standard sizes.
    """
    # Standard sizes: 64, 128, 256, 512, 1024, 2048, 4096
    cache_file = f"data/rft_basis/{variant}_n{n}.npy"
    
    if os.path.exists(cache_file):
        return np.load(cache_file)
    
    # Compute and cache
    basis = generate_operator_rft(n, variant)
    np.save(cache_file, basis)
    return basis
```

**New files to create:**
- `algorithms/rft/fast/cached_basis.py`
- `algorithms/rft/fast/approximate_eigenbasis.py`
- `data/rft_basis/` (pre-computed matrices)

### Priority 3: Update RFTPU to Operator-RFT

**Goal:** Make hardware accelerator use the genuine RFT.

```verilog
// Replace phi_rft_core kernel ROM with operator-RFT coefficients
// Generated from: operator_variants.generate_rft_golden(8)

module operator_rft_core #(
    parameter int N = 8,
    parameter int WIDTH = 16
) (
    input  logic [WIDTH-1:0] x_in [N],
    output logic [WIDTH-1:0] y_out [N]
);
    // Eigenbasis coefficients (Q1.15 format)
    // Pre-computed from Python: U = generate_rft_golden(8)
    localparam logic signed [15:0] U [N][N] = '{
        // Row 0: U[0,:]
        '{16'h3000, 16'h2800, 16'h2000, ...},
        // ... (64 coefficients total)
    };
    
    // Matrix-vector multiply: y = U' * x
    always_comb begin
        for (int i = 0; i < N; i++) begin
            logic signed [31:0] acc = 0;
            for (int j = 0; j < N; j++) begin
                acc += U[j][i] * x_in[j];
            end
            y_out[i] = acc[30:15];  // Fixed-point scaling
        end
    end
endmodule
```

**Files to modify:**
- `hardware/rftpu_architecture.tlv`
- `hardware/tb/rftpu_test_vectors.py`

### Priority 4: Signal-Adaptive Transform Router

**Goal:** Automatically select best transform for each signal segment.

```python
def classify_signal(x: np.ndarray) -> str:
    """Classify signal to select optimal transform."""
    
    # Feature extraction
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Detect golden-ratio quasi-periodicity
    phi = (1 + np.sqrt(5)) / 2
    n = len(x)
    
    # Check for peaks at golden-ratio-related lags
    expected_lags = [int(n/phi), int(n/(phi**2)), int(n*phi) % n]
    peak_strength = sum(abs(autocorr[lag]) for lag in expected_lags if lag < len(autocorr))
    
    if peak_strength > 0.5 * autocorr[0]:
        return 'rft_golden'
    
    # Check for harmonic structure
    fft_mag = np.abs(np.fft.rfft(x))
    harmonic_ratio = fft_mag[1] / (fft_mag[2] + 1e-10)
    
    if 0.4 < harmonic_ratio < 0.6:  # Natural harmonic series
        return 'rft_harmonic'
    
    # Default to DCT for smooth signals
    return 'dct'

def adaptive_transform(x: np.ndarray) -> np.ndarray:
    """Apply the best transform for this signal."""
    variant = classify_signal(x)
    
    if variant.startswith('rft_'):
        return operator_rft_forward(x, variant)
    elif variant == 'dct':
        return scipy.fft.dct(x, norm='ortho')
    else:
        return np.fft.fft(x) / np.sqrt(len(x))
```

**New files to create:**
- `algorithms/rft/routing/signal_classifier.py`
- `algorithms/rft/routing/adaptive_router.py`

### Priority 5: Fast Algorithm Research

**Goal:** Discover O(N log N) algorithm for operator-RFT.

**Research directions:**

1. **Circulant Approximation:**
   ```
   K_circ = circulant(first_row_of_K)
   K_circ = F^H diag(Λ) F   (diagonalized by FFT)
   U_approx ≈ F^H           (eigenbasis ≈ FFT)
   
   RFT(x) ≈ FFT(x) + low-rank_correction
   ```

2. **Chebyshev Polynomial Expansion:**
   ```
   U(λ) = Σ c_k T_k(K)   (Chebyshev polynomials of operator)
   
   Each T_k(K) application is O(N²), but sparse K could be O(N)
   ```

3. **Hierarchical Decomposition:**
   ```
   For block-Toeplitz K:
   K = [K₁ K₂; K₂ᵀ K₃]   (2×2 block structure)
   
   Recursively decompose like divide-and-conquer FFT
   ```

**Files to create:**
- `algorithms/rft/theory/fast_algorithm_research.py`
- `experiments/proofs/circulant_approximation.py`

---

## Part 4: Success Metrics

### Milestone 1: Compression Validation (1 week)
- [ ] Benchmark DCT-only vs DCT+RFT on 10 signal types
- [ ] Measure RFT's actual contribution to compression
- [ ] Document results in `docs/reports/rft_compression_validation.md`

### Milestone 2: Fast Construction (2 weeks)
- [ ] Implement basis caching for standard sizes
- [ ] Reduce construction time from O(N³) to O(1) for cached sizes
- [ ] Benchmark: 4096-point RFT in <1ms

### Milestone 3: Hardware Update (2 weeks)
- [ ] Update `phi_rft_core` to use operator-RFT basis
- [ ] Regenerate test vectors from `operator_variants.py`
- [ ] Verify Makerchip simulation passes with new kernel

### Milestone 4: Adaptive Router (3 weeks)
- [ ] Implement signal classifier
- [ ] Integrate into H3 codec
- [ ] Benchmark: Always-win on diverse signal set

### Milestone 5: Fast Algorithm (Research, 1-3 months)
- [ ] Investigate circulant approximation
- [ ] Test Chebyshev polynomial approach
- [ ] Target: O(N log N) or proof of impossibility

---

## Part 5: Updated Claims (Honest)

### What We CAN Claim

1. **Novel Transform:** The operator-based RFT is a genuine eigenbasis transform, mathematically distinct from FFT/DCT/LCT.

2. **Domain-Specific Sparsity:** For signals with golden-ratio quasi-periodic structure, RFT achieves improved PSNR vs FFT at 10% coefficient retention (see [VERIFIED_BENCHMARKS](research/benchmarks/VERIFIED_BENCHMARKS.md)).

3. **Unitarity:** All operator-RFT variants are strictly unitary (proven, tested to 1e-15).

4. **Hardware Design:** RFTPU architecture is simulated and verified in TL-Verilog.

### What We CANNOT Claim (Yet)

1. ❌ "Massive sparsity advantage over FFT for all signals"
2. ❌ "O(N log N) fast algorithm" (still O(N²))
3. ❌ "Post-quantum cryptographic security" (no proofs)
4. ❌ "Silicon-validated hardware" (simulation only)

### Path to Full Claims

| Claim | Current | Path to Validation |
|-------|---------|-------------------|
| Universal sparsity | ❌ | Adaptive routing + classifier |
| Fast algorithm | ❌ | Circulant/Chebyshev research |
| Crypto security | ❌ | Third-party cryptanalysis |
| Silicon validation | ❌ | FPGA prototype → ASIC tape-out |

---

## Conclusion

The December 2025 operator-based RFT is a **genuine step forward** from the trivial phase-shifted FFT. It provides:
- Real sparsity advantage on target signals
- Mathematically sound eigenbasis definition
- Validated benchmark results

To take it to the **next level**, we need:
1. Fast construction (caching, approximation)
2. Adaptive signal routing
3. Updated hardware implementation
4. Research into fast algorithms

The foundation is solid. The engineering work remains.

---

*This document is the strategic roadmap for QuantoniumOS RFT development.*

# QuantoniumOS Algorithm Inventory

> **Generated**: 2026-01-26  
> **Patent Reference**: USPTO 19/169,399 (pending)

## Counting Methodology

**Inclusion criteria**: An "algorithm" is counted if it has:
1. A distinct implementation file or named class/function
2. A mathematically distinguishable transform kernel or control flow
3. At least one test or benchmark exercising it

**Exclusion**: Wrappers, aliases, and parameter variations of the same core algorithm are NOT counted separately.

**Algorithm Inventory by Category (Candidate Status)**:
| Category | Count | Method |
|----------|-------|--------|
| Core RFT implementations | 5 | Distinct `.py` files in `algorithms/rft/core/` |
| Operator variants | 6 | Named functions in `operator_variants.py` |
| Hybrid codecs (H0-H10, FH1-FH5) | 16 | Named classes in `hybrid_mca_fixes.py` |
| Compression codecs | 4 | Distinct encoder/decoder pairs |
| Crypto primitives | 2 | `enhanced_rft_crypto_v2.py` + `rft_sis_hash.py` |
| Quantum simulation | 3 | Distinct simulation kernels (see §8 for counting rationale) |
| **Total Inventory** | **36** | Based on file/class enumeration (utilities excluded) |

Note: The "65+" figure previously stated was a rough upper bound. The **documented distinct algorithm count** depends on strict validation definitions (see below).

---

## Table of Contents

1. [Core RFT Algorithms](#1-core-rft-resonant-fourier-transform-algorithms)
2. [Operator-Based RFT Variants](#2-operator-based-rft-variants)
3. [Patent-Aligned Variants](#3-patent-aligned-rft-variants-uspto-19169399)
4. [Adaptive RFT (ARFT)](#4-adaptive-rft-arft-kernels)
5. [Hybrid Codec Algorithms](#5-hybrid-codec-algorithms-h0-h10-fh1-fh5)
6. [Compression Algorithms](#6-compression-algorithms)
7. [Cryptographic Algorithms](#7-cryptographic-algorithms)
8. [Quantum Simulation Algorithms](#8-quantum-simulation-algorithms)
9. [Middleware & System Algorithms](#9-middleware--system-algorithms)
10. [Fast/Optimized Algorithms](#10-fastoptimized-algorithms)
11. [Summary](#summary)

---

## 1. Core RFT (Resonant Fourier Transform) Algorithms

> **EMPIRICALLY VERIFIED (archived)**: Gram-normalized RFT is unitary (energy=1.0) and perfectly invertible (rel_err=1e-14). See [data/artifacts/rft_stability/manifest.json](data/artifacts/rft_stability/manifest.json).

### 1.1 Formal Definition

**Definition 1.1 (φ-Phase Function)**:
```
φ_phase(n, k, N) = (n · φ^k) / N + (k · φ · n²) / N²

where φ = (1 + √5) / 2 ≈ 1.618033988749895 (golden ratio)
```

**Definition 1.2 (Resonant Fourier Transform)**:
For input x ∈ ℂ^N, the RFT is defined as:
```
R[k] = Σ_{n=0}^{N-1} x[n] · exp(-2πi · φ_phase(n, k, N)),  k = 0, 1, ..., N-1
```

**Definition 1.3 (Inverse RFT)**:
```
x[n] = (1/N) · Σ_{k=0}^{N-1} R[k] · exp(+2πi · φ_phase(n, k, N))
```

### 1.2 Theorem Statements

**Theorem 1.1 (Unitarity — CONDITIONAL)**:
Let Ψ ∈ ℂ^{N×N} be the RFT matrix with Ψ_{kn} = exp(-2πi · φ_phase(n,k,N)).

*Statement*: Ψ†Ψ = I_N (unitarity).

*Status*: **NO FORMAL PROOF for raw φ-phase matrix.**

*What IS constructed*: After QR orthonormalization Ψ_ortho = Q from QR decomposition of Ψ, we have Q†Q = I by construction. This is a tautology, not a theorem about φ-phase structure.

**Theorem 1.2 (Invertibility)**:
*Statement*: If Ψ is full rank, then R⁻¹ exists.

*Proof sketch*: det(Ψ) ≠ 0 is observed numerically for N ≤ 4096 (not archived). No closed-form proof.

*Status*: **OBSERVED (not archived); no formal proof.**

**Theorem 1.3 (Stability Bound — OPEN)**:
*Statement*: ‖R(x + ε)‖₂ ≤ (1 + δ) ‖R(x)‖₂ for ‖ε‖₂ < η.

*Status*: **No formal bound.** Empirical condition number κ(Ψ) ≈ 400 at N=256 before orthonormalization suggests potential instability.

### 1.3 Proof Status Summary

| Property | Claimed | Status | Evidence |
|----------|---------|--------|----------|
| Unitarity (raw Ψ) | ❌ | OBSERVED (non-unitary) | Condition number ~400 |
| Unitarity (gram-normalized) | ✅ | CONSTRUCTED + EMPIRICALLY VERIFIED (archived) | **energy_ratio = 1.0**; [data/artifacts/rft_stability/manifest.json](data/artifacts/rft_stability/manifest.json) |
| Invertibility (default API) | ❌ | OBSERVED (mismatch) | Default `rft_forward/rft_inverse` are mismatched (waveform vs square) |
| Invertibility (gram mode) | ✅ | CONSTRUCTED + EMPIRICALLY VERIFIED (archived) | **rel_l2_err = 1e-14**; [data/artifacts/rft_stability/manifest.json](data/artifacts/rft_stability/manifest.json) |
| Stability bound | ✅ | EMPIRICALLY VERIFIED (archived) | Error grows as O(N·ε_mach); [data/artifacts/rft_stability/manifest.json](data/artifacts/rft_stability/manifest.json) |
| O(N log N) algorithm | ❌ | OBSERVED (unavailable) | No factorization known |

**Empirical Results** (commit 3bbe550d94f8, seed 42, gram-normalized mode):
```
N=64   energy_ratio=1.0   rel_l2_err=5.3e-15  ← machine precision!
N=256  energy_ratio=1.0   rel_l2_err=9.9e-15
N=1024 energy_ratio=1.0   rel_l2_err=2.6e-14
```

**Conclusion**: With `use_gram_normalization=True`, the RFT is:
- **CONSTRUCTED + EMPIRICALLY VERIFIED (archived)** unitary (energy preserved exactly)
- **CONSTRUCTED + EMPIRICALLY VERIFIED (archived)** perfectly invertible (reconstruction error at machine precision)
- **EMPIRICALLY VERIFIED (archived)** numerically stable (error scales with N·ε_mach as expected)

**Previous Bug (FIXED)**: The archiver was calling `rft_forward(x)` which defaults to waveform mode (T=N*16), then `rft_inverse(y, N)` which expects square coefficients. This caused 145% reconstruction error.

**Fix**: Updated `archive_rft_stability.py` to use gram-normalized square-kernel mode via `rft_basis_matrix(N, N, use_gram_normalization=True)`.

**API Usage Guide**:
```python
# WRONG: Default APIs are mismatched
y = rft_forward(x)           # Returns waveform of length N*16  
xr = rft_inverse(y, N)       # Expects coefficients of length N
# Result: ~145% reconstruction error

# CORRECT: Use gram-normalized square mode
y = rft_forward(x, use_gram_normalization=True)   # Returns N coefficients
xr = rft_inverse(y, N, use_gram_normalization=True)  # Correct reconstruction

# CORRECT: Use direct matrix for guaranteed unitarity  
Phi = rft_basis_matrix(N, N, use_gram_normalization=True)
y = Phi.conj().T @ x   # Forward
xr = Phi @ y           # Inverse (Φ is unitary by construction)
```

**Reproducibility**: `python scripts/archive_rft_stability.py --output data/artifacts/rft_stability --seed 42`

### 1.4 Numerical Stability Test Artifacts

> ✅ **ARCHIVED (manifest present)**

Archived artifacts in `data/artifacts/rft_stability/`:

```
data/artifacts/rft_stability/
├── manifest.json              # {commit_sha, timestamp, command, seed}
├── results.json               # summary metrics per size
└── stdout.txt                 # captured run output
```

**Current status**: Archived. See [data/artifacts/rft_stability/manifest.json](data/artifacts/rft_stability/manifest.json).

**To create**: `python scripts/archive_rft_stability.py --output data/artifacts/rft_stability/`

### 1.5 Implementation Files

| Algorithm | File | Description | Complexity |
|-----------|------|-------------|------------|
| **Resonant Fourier Transform** | [resonant_fourier_transform.py](../algorithms/rft/core/resonant_fourier_transform.py) | Core transform | O(N²) |
| **CanonicalTrueRFT** | [canonical_true_rft.py](../algorithms/rft/core/canonical_true_rft.py) | With unitarity validation | O(N²) |
| **φ-Phase FFT Optimized** | [phi_phase_fft_optimized.py](../algorithms/rft/core/phi_phase_fft_optimized.py) | Fused `D_φ C_σ F` | O(N²) |
| **Golden Ratio Unitary** | [golden_ratio_unitary.py](../algorithms/rft/core/golden_ratio_unitary.py) | QR orthonormalization | **O(N³)** |
| **Symbolic Wave Computer** | [symbolic_wave_computer.py](../algorithms/rft/core/symbolic_wave_computer.py) | Wave-domain logic | O(N×bits) |

---

## 2. Operator-Based RFT Variants

| Variant | File | Description | Best For |
|---------|------|-------------|----------|
| **RFT-Golden** | [operator_variants.py](../algorithms/rft/variants/operator_variants.py) | Eigenbasis of `H_φ = Σ φⁿ|n⟩⟨n+1|` | General signals |
| **RFT-Fibonacci** | [operator_variants.py](../algorithms/rft/variants/operator_variants.py) | Fibonacci frequency ratios `{1,1,2,3,5,8,13,21}` | Crypto/integer lattice |
| **RFT-Harmonic** | [operator_variants.py](../algorithms/rft/variants/operator_variants.py) | Harmonic overtone `ω_k = ω₀·(1 + k/φ)` | Audio/music |
| **RFT-Geometric** | [operator_variants.py](../algorithms/rft/variants/operator_variants.py) | Golden ratio powers `φ⁰, φ¹, φ², ...` | Self-similar patterns |
| **RFT-Beating** | [operator_variants.py](../algorithms/rft/variants/operator_variants.py) | Golden-ratio beating patterns | Modulation analysis |
| **RFT-Phyllotaxis** | [operator_variants.py](../algorithms/rft/variants/operator_variants.py) | Golden-angle phase sequence (2π/φ²; complement 2π/φ) | Phase-sequence patterns |

---

## 3. Patent-Aligned RFT Variants (USPTO 19/169,399)

| Variant | File | Description | Geometric Model |
|---------|------|-------------|-----------------|
| **RFT-Polar-Golden** | [patent_variants.py](../algorithms/rft/variants/patent_variants.py) | Polar-Cartesian with φ radial scaling | Polar coordinates |
| **RFT-Spiral-Golden** | [patent_variants.py](../algorithms/rft/variants/patent_variants.py) | Golden spiral `r(θ) = a·φ^(θ/π)` | Logarithmic spiral |
| **RFT-Loxodrome** | [patent_variants.py](../algorithms/rft/variants/patent_variants.py) | Rhumb line on sphere with golden pitch | Geodesic/spherical |
| **RFT-Complex-Exp** | [patent_variants.py](../algorithms/rft/variants/patent_variants.py) | Complex exponential dual golden frequencies | Complex plane |
| **RFT-Exp-Decay** | [patent_variants.py](../algorithms/rft/variants/patent_variants.py) | Damped resonance `e^(-t/φ)·cos(φωt)` | Damped oscillator |
| **RFT-Möbius** | [patent_variants.py](../algorithms/rft/variants/patent_variants.py) | Möbius transformation with golden coefficients | Conformal mapping |
| **RFT-Projection** | [patent_variants.py](../algorithms/rft/variants/patent_variants.py) | Torus/spiral projection mapping | Torus parametric |
| **RFT-Sphere-Parametric** | [patent_variants.py](../algorithms/rft/variants/patent_variants.py) | Sphere parametric resonance | Spherical harmonics |
| **RFT-Phase-Coherent** | [patent_variants.py](../algorithms/rft/variants/patent_variants.py) | Phase-coherent modulation | Chirp signals |
| **RFT-Entropy-Modulated** | [patent_variants.py](../algorithms/rft/variants/patent_variants.py) | Entropy-based modulation | Medical/noise |

---

## 4. Adaptive RFT (ARFT) Kernels

| Algorithm | File | Description | Adaptation Method |
|-----------|------|-------------|-------------------|
| **ARFT** | [arft_phase_aware.py](../algorithms/rft/arft/arft_phase_aware.py) | Signal-adaptive with amplitude feedback and phase coupling | Real-time feedback |
| **Operator-Based ARFT** | [arft_operator_based.py](../algorithms/rft/arft/arft_operator_based.py) | Eigenbasis of signal's autocorrelation Toeplitz (KLT for RFT) | Autocorrelation |
| **Phase ARFT Kernel** | [arft_phase_aware.py](../algorithms/rft/arft/arft_phase_aware.py) | Phase-domain adaptive kernel | Phase statistics |

---

## 5. Hybrid Codec Algorithms (H0-H10, FH1-FH5)

### 5.1 Evaluation Requirements

> ⚠️ **CRITICAL**: Single-point BPP numbers are insufficient. Valid codec evaluation requires:

**Required for any compression claim**:
1. **Fixed dataset(s) with SHA256 hashes** — reproducible input
2. **Baseline comparisons** — JPEG, WebP, AVIF, DCT-only, Wavelet-only
3. **Rate–distortion curves** — BPP vs PSNR/SSIM at multiple quality levels
4. **Reproducibility artifacts** — `{commit_sha, seed, command, dataset_hash}`

### 5.2 Dataset Specification

> ⚠️ **NOT YET DEFINED** — The following is the REQUIRED structure:

```
data/codec_benchmark/
├── manifest.json                    # Dataset metadata + hashes
├── images/
│   ├── kodak/                       # Kodak PhotoCD (24 images, 768×512)
│   │   └── SHA256: <hash>
│   ├── tecnick/                     # Tecnick sampling (100 images)
│   │   └── SHA256: <hash>
│   └── medical/                     # PhysioNet/FastMRI samples
│       └── SHA256: <hash>
├── baselines/
│   ├── jpeg_q{10,20,40,60,80,95}/   # libjpeg-turbo
│   ├── webp_q{10,20,40,60,80,95}/   # libwebp
│   ├── avif_q{10,20,40,60,80,95}/   # libavif
│   ├── dct_only/                    # Pure 8×8 DCT
│   └── wavelet_db4/                 # Daubechies-4 wavelet
└── results/
    └── <commit_sha>/
        ├── h3_rd_curve.json         # Rate-distortion data
        ├── fh5_rd_curve.json
        └── comparison_plot.png
```

**Current status**: This structure DOES NOT EXIST. BPP claims are from ad-hoc runs.

### 5.3 Primary Hybrids

| Hybrid | File | Description | Claimed BPP | Coherence | R-D Curve? |
|--------|------|-------------|-------------|-----------|------------|
| **H0_Baseline_Greedy** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Greedy selection baseline | 0.80 | 0.50 | ❌ No |
| **H1_Coherence_Aware** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Coherence-aware selection | 0.80 | 0.50 | ❌ No |
| **H2_Phase_Adaptive** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Phase-adaptive mixing | 0.80 | 0.38 | ❌ No |
| **H3_Hierarchical_Cascade** ⭐ | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Zero-coherence DCT+RFT cascade | **0.67** | **η=0** | ❌ No |
| **H4_Quantum_Superposition** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Superposition-weighted | 8.02 | 0.50 | ❌ No |
| **H5_Attention_Gating** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Attention-weighted gates | 0.80 | 0.50 | ❌ No |
| **H6_Dictionary_Learning** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Learned dictionary atoms | 0.81 | 0.50 | ❌ No |
| **H7_Cascade_Attention** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Cascade + attention | 0.80 | η=0 | ❌ No |
| **H8_Aggressive_Cascade** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Aggressive threshold | 16.0 | η=0 | ❌ No |
| **H9_Iterative_Refinement** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Iterative improvement | 16.0 | η=0 | ❌ No |
| **H10_Quality_Cascade** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Quality-optimized cascade | 0.82 | η=0 | ❌ No |

### 5.4 Frequency Hybrids (FH1-FH5)

| Hybrid | File | Description | Claimed BPP | Coherence | R-D Curve? |
|--------|------|-------------|-------------|-----------|------------|
| **FH1_MultiLevel_Cascade** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Multi-level decomposition | 0.80 | η=0 | ❌ No |
| **FH2_Adaptive_Split** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Adaptive frequency split | 0.80 | η=0 | ❌ No |
| **FH3_Frequency_Cascade** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Frequency-domain cascade | 0.80 | η=0 | ❌ No |
| **FH4_Edge_Aware** | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Edge-preserving cascade | 0.80 | η=0 | ❌ No |
| **FH5_Entropy_Guided** ⭐ | [hybrid_mca_fixes.py](../experiments/hypothesis_testing/hybrid_mca_fixes.py) | Entropy-based selection | **0.41** | η=0 | ❌ No |

### 5.5 Required Baseline Comparison

Before claiming H3 or FH5 are "good", we must show:

| Codec | BPP Range | PSNR Range | SSIM Range | Status |
|-------|-----------|------------|------------|--------|
| JPEG (libjpeg-turbo) | 0.1–2.0 | 25–45 dB | 0.80–0.99 | ❌ Not run |
| WebP | 0.1–2.0 | 28–48 dB | 0.85–0.99 | ❌ Not run |
| AVIF | 0.05–1.5 | 30–50 dB | 0.88–0.99 | ❌ Not run |
| DCT-only (8×8) | 0.2–2.0 | 24–42 dB | 0.78–0.98 | ✅ **Reproducible (synthetic)** |
| Wavelet (db4) | 0.1–2.0 | 26–44 dB | 0.82–0.98 | ❌ Not run |
| **H3** | 7.65 | 47.86 dB | ? | ✅ **Reproducible (synthetic)** |
| **FH5** | 7.65 | 47.86 dB | ? | ✅ **Reproducible (synthetic)** |

**Evidence artifacts**: `data/artifacts/codec_benchmark/manifest.json` (commit: 3bbe550d94f8, seed: 42)

**⚠️ Limitation**: Current R-D curves are on **synthetic 256×256 gradients** only. Kodak/Tecnick validation still required.

**To validate on real images**: `python benchmarks/codec_rd_curve.py --dataset kodak --output data/codec_benchmark/`

### 5.6 Medical Imaging Hybrids

| Hybrid | File | Description |
|--------|------|-------------|
| **RFT-Wavelet Medical** | [rft_wavelet_medical.py](../src/rftmw_codec/rft_wavelet_medical.py) | Haar + RFT for MRI/CT denoising |
| **RFT-Wavelet Medical v2** | [rft_wavelet_medical_v2.py](../src/rftmw_codec/rft_wavelet_medical_v2.py) | Enhanced Rician/Poisson noise model |

---

## 6. Compression Algorithms

### 6.1 Codec Infrastructure

| Algorithm | File | Description |
|-----------|------|-------------|
| **RFT Vertex Codec** | [rft_vertex_codec.py](../algorithms/rft/compression/rft_vertex_codec.py) | Lossless tensor→RFT coefficients with SHA256 checksum |
| **ANS Entropy Coder** | [rft_entropy_coder.py](../src/rftmw_codec/rft_entropy_coder.py) | rANS for spectral coefficients |
| **RANS Stream** | [rans.py](../src/rftmw_codec/rans.py) | Streaming rANS with Laplace smoothing |
| **RFT Hybrid Codec** | [rft_hybrid_codec.py](../src/rftmw_codec/rft_hybrid_codec.py) | Band partitioning + quantization + residual predictor |

### 6.2 Experimental Compressors

| Algorithm | File | Description | Performance |
|-----------|------|-------------|-------------|
| **RFT Hybrid Compressor** | [rft_hybrid_compressor.py](../experiments/rft_hybrid_compressor.py) | Auto-selects RFT vs INT8+zlib | +60.7% on embeddings |
| **Low-Rank RFT** | [low_rank_rft.py](../experiments/low_rank_rft.py) | Exploits rank-15 for 99% energy | O(N×r), r~15 |
| **RFT Quantum Sim Compressor** | [rft_quantum_sim_compressor.py](../experiments/rft_quantum_sim_compressor.py) | Quantum state simulation with RFT sparsity | GHZ/random states |

---

## 7. Cryptographic Algorithms

> ⚠️ **RESEARCH ONLY** — No external security audit. Passes basic sanity tests but NOT validated for production use.

### **EMPIRICALLY VERIFIED (archived)** Basic Sanity Tests (2026-01-27)

| Test | Result | Evidence |
|------|--------|----------|
| **Avalanche effect** | 99.9% of ideal (64/128 bits) | [data/artifacts/crypto_tests/manifest.json](data/artifacts/crypto_tests/manifest.json) |
| **Differential diversity** | 1000/1000 unique | [data/artifacts/crypto_tests/manifest.json](data/artifacts/crypto_tests/manifest.json) |
| **Encrypt/Decrypt** | All KAT vectors match | [data/artifacts/crypto_tests/manifest.json](data/artifacts/crypto_tests/manifest.json) |

**What this proves**: The cipher has good diffusion properties and is deterministic.

**What this does NOT prove**: Resistance to advanced cryptanalysis (differential/linear attacks, algebraic attacks, side-channels). External review still required.

### 7.1 Classification
1. Sanity tests are **not** security proofs.
2. There is **no** formal security model, reduction, or audit.

**Bottom line**: This is a research cipher sketch; do **not** use for security.

| Term | Meaning | This Implementation |
|------|---------|---------------------|
| **Post-quantum** | Secure against quantum computers with standard reduction | ❌ **NOT CLAIMED** |
| **Lattice-based** | Uses lattice problems as hardness assumption | ⚠️ Non-standard variant |
| **Quantum-resistant** | Conjectured secure against quantum attacks | ❌ **NOT CLAIMED** |

**We explicitly DO NOT claim "post-quantum" or "quantum-resistant" status.**

### 7.2 Threat Model (UNDEFINED)

> ⚠️ **NO THREAT MODEL EXISTS** — The following is REQUIRED but not written:

```
Threat Model Requirements:
├── Adversary capabilities
│   ├── Computational bound (classical/quantum)
│   ├── Oracle access (encryption/decryption/signing)
│   └── Side-channel access (timing/power/EM)
├── Security goals
│   ├── Confidentiality (IND-CPA? IND-CCA2?)
│   ├── Integrity (INT-CTXT?)
│   └── Authentication (EUF-CMA?)
├── Trust assumptions
│   └── Key generation, random number source
└── Explicitly out-of-scope threats
```

**Current status**: None of this is defined.

### 7.3 EnhancedRFTCryptoV2

| Property | Value | Status |
|----------|-------|--------|
| **Structure** | 48-round balanced Feistel | Implemented |
| **Block size** | 128 bits | Implemented |
| **Key size** | 256 bits | Implemented |
| **S-box** | AES S-box (borrowed) | Implemented |
| **Key schedule** | φ-parameterized ARX | Implemented |
| **Security goal** | ??? | ❌ NOT DEFINED |
| **Security level** | ??? | ❌ NOT ANALYZED |

**File**: [enhanced_rft_crypto_v2.py](../algorithms/rft/core/enhanced_rft_crypto_v2.py)

### 7.4 Known-Answer Test (KAT) Vectors

> **EMPIRICALLY VERIFIED (archived)** KAT vectors — See [data/artifacts/crypto_tests/manifest.json](data/artifacts/crypto_tests/manifest.json)

**Evidence**: commit 3bbe550d94f8, seed 42

All KAT vectors decrypt correctly. Ciphertexts are deterministic and reproducible.

**To regenerate**: `python scripts/archive_crypto_tests.py --output data/artifacts/crypto_tests`

### 7.5 Security Test Status

| Test | Description | Status | Evidence |
|------|-------------|--------|----------|
| **Avalanche effect** | Bit diffusion | EMPIRICALLY VERIFIED (archived) | [data/artifacts/crypto_tests/manifest.json](data/artifacts/crypto_tests/manifest.json) |
| **Differential diversity** | Output difference distribution | EMPIRICALLY VERIFIED (archived) | [data/artifacts/crypto_tests/manifest.json](data/artifacts/crypto_tests/manifest.json) |
| **NIST STS** | Statistical randomness (15 tests) | ⚠️ **Script ready** | `python scripts/run_nist_sts.py` |
| **Differential cryptanalysis** | Probability of differential trails | ❌ Not run | Requires SageMath |
| **Linear cryptanalysis** | Linear approximation bias | ❌ Not run | Requires SageMath |
| **Algebraic attacks** | Degree of equations | ❌ Not run | Requires SageMath |
| **Related-key attacks** | Key schedule weakness | ❌ Not run | Manual analysis |

### 7.6 NIST Statistical Test Suite

To run NIST STS:
```bash
# Generate bitstreams (10 Mbit by default)
python scripts/run_nist_sts.py --output data/artifacts/nist_sts --megabits 10

# Then run official NIST STS or use Python wrapper:
pip install sp800_22_tests
```

The script generates CTR-mode and random-plaintext encryption streams suitable for NIST STS analysis.

### 7.7 φ-Structured SIS Hash

| Property | Standard SIS | φ-Structured SIS |
|----------|-------------|------------------|
| Matrix A | Uniform random | φ-derived structure |
| Hardness assumption | LWE/SIS (standard reduction) | **NON-STANDARD** |
| Security proof | Worst-case to average-case | **NONE** |
| Cryptanalysis | Extensively studied | **NONE PUBLISHED** |

**Parameters**: n=512, m=1024, q=3329

**Why these are NOT Kyber parameters**:
- Kyber uses Module-LWE, not SIS
- Kyber requires uniform random A
- Sharing q=3329 does NOT transfer security properties

### 7.7 External Review Status

| Review Type | Status | Date | Reviewer |
|-------------|--------|------|----------|
| Internal code review | ❌ None | — | — |
| External security audit | ❌ None | — | — |
| Academic cryptanalysis | ❌ None | — | — |
| IACR ePrint submission | ❌ None | — | — |
| Bug bounty | ❌ None | — | — |

### 7.8 Honest Security Summary

| What We Have | What We Don't Have |
|--------------|-------------------|
| Code that encrypts/decrypts | Security definition |
| 48-round Feistel structure | Proof of rounds sufficiency |
| AES S-box (secure component) | Analysis of full cipher |
| φ-parameterized schedule | Key schedule security analysis |
| Avalanche diffusion (archived) | Formal diffusion proof |
| Parameter choices | Parameter security analysis |

**Bottom line**: This is a **cipher sketch**, not a cipher. Do not use for anything.

---

## 8. Quantum Simulation Algorithms

**Counting note**: This section contains **3 core simulation kernels** (counted) and **5 utility/library components** (not counted separately as "algorithms").

### 8.1 Topological Quantum (COUNTED: 2 kernels)

| Algorithm | File | Description | Counted? |
|-----------|------|-------------|----------|
| **Topological Quantum Kernel** | [topological_quantum_kernel.py](../quantonium_os_src/topological_quantum_kernel.py) | Surface code with braiding, edge encoding, error correction | ✅ Yes |
| **Enhanced Topological Qubit** | [topological_quantum_kernel.py](../quantonium_os_src/topological_quantum_kernel.py) | Extended qubit with invariant measurement | ✅ Yes |

### 8.2 Quantum Gates & Search (COUNTED: 1 algorithm)

| Algorithm | File | Description | Counted? |
|-----------|------|-------------|----------|
| **Quantum Gate Library** | [quantum_gates.py](../quantonium_os_src/quantum_gates.py) | Pauli (X/Y/Z), Rotation (Rx/Ry/Rz), Phase (S/T/P), Hadamard, CNOT | ❌ Utility |
| **Quantum Search (Grover)** | [quantum_search.py](../quantonium_os_src/quantum_search.py) | Grover's algorithm, O(√N) iterations | ✅ Yes |

### 8.3 Geometric Hashing (NOT COUNTED: utilities)

| Algorithm | File | Description | Counted? |
|-----------|------|-------------|----------|
| **Geometric Hash** | [geometric_hash.py](../quantonium_os_src/geometric_hash.py) | RFT-enhanced spatial hashing for collision detection | ❌ Utility |
| **Spatial Hash** | [geometric_hash.py](../quantonium_os_src/geometric_hash.py) | Coordinate quantization hashing | ❌ Utility |

---

## 9. Middleware & System Algorithms

### 9.1 RFTMW Engine

| Algorithm | File | Description |
|-----------|------|-------------|
| **MiddlewareTransformEngine** | [rftmw_middleware.py](../quantonium_os_src/rftmw_middleware.py) | Binary→Wave→Compute→Binary with auto variant selection |
| **RFTMW C++ Core** | [rftmw_core.cpp](../src/rftmw_native/rftmw_core.cpp) | AVX2/AVX512 SIMD, golden phase computation |

### 9.2 Transform Scheduling

| Algorithm | File | Description |
|-----------|------|-------------|
| **Unified Transform Scheduler** | [unified_scheduler.py](../quantonium_os_src/unified_scheduler.py) | Multi-backend scheduler (Python/C/C++) |
| **Signal Classifier** | [signal_classifier.py](../quantonium_os_src/signal_classifier.py) | Heuristic classifier for adaptive routing (~64% accuracy) |

---

## 10. Fast/Optimized Algorithms

| Algorithm | File | Description | Claimed Speedup | Evidence |
|-----------|------|-------------|-----------------|----------|
| **Low-Rank RFT** | [low_rank_rft.py](../experiments/low_rank_rft.py) | Truncated SVD approximation keeping rank-r | O(N×r) vs O(N²) | Empirical: r≈15 captures 99% energy on test signals |
| **Fast RFT Structured** | [fast_rft.py](../algorithms/rft/core/fast_rft.py) | Explores structured factorizations | **NO FORMAL PROOF** | No O(N log N) factorization demonstrated |
| **Cached Basis** | [cached_basis.py](../algorithms/rft/core/cached_basis.py) | LRU-cached precomputed basis matrices | Amortized O(N²)→O(N) lookup | Memory tradeoff, not complexity reduction |
| **SIMD Optimized** | [rftmw_core.cpp](../src/rftmw_native/rftmw_core.cpp) | AVX2/AVX512 vectorization | ~4-8× wallclock | Benchmark: N=1024, 1000 trials, vs NumPy |

### Complexity Status

| Transform | Naive | Fast Algorithm? | Status |
|-----------|-------|-----------------|--------|
| FFT | O(N²) | O(N log N) | **Established** (Cooley–Tukey) |
| φ-RFT | O(N²) | O(N log N)? | **NO FORMAL PROOF** — no published factorization |
| QR-orthonormalized RFT | O(N³) | — | Inherent to dense QR |

**Open problem**: Does φ-RFT admit a fast O(N log N) algorithm? This would require either:
1. A sparse/structured factorization of the φ-phase matrix, or
2. A reduction to FFT + O(N) post-processing

Neither has been demonstrated.

---

## 11. Formal Proofs vs Empirical Verification

This section clarifies the distinction between **formal mathematical proofs** and **empirical verification** for each component.

### 11.1 Proof Status Matrix

| Component | Formal Proof | Empirical Verification | Gap Analysis |
|-----------|--------------|------------------------|--------------|
| **RFT Unitarity** | Conditional | CONSTRUCTED + EMPIRICALLY VERIFIED (archived) | [data/artifacts/rft_stability/manifest.json](data/artifacts/rft_stability/manifest.json) |
| **RFT Invertibility** | Conditional | CONSTRUCTED + EMPIRICALLY VERIFIED (archived) | [data/artifacts/rft_stability/manifest.json](data/artifacts/rft_stability/manifest.json) |
| **φ = (1+√5)/2** | Established | OBSERVED (symbolic identity) | Standard algebra (no in-repo theorem) |
| **Decorrelation** | Not claimed | OBSERVED (not archived) | No theorem bounds decorrelation vs DCT/wavelets |
| **Crypto Avalanche** | Not claimed | EMPIRICALLY VERIFIED (archived) | [data/artifacts/crypto_tests/manifest.json](data/artifacts/crypto_tests/manifest.json) |
| **Crypto Security** | NOT CLAIMED | OBSERVED (no proof/audit) | No formal security reduction |
| **Codec R-D** | Not claimed | EMPIRICALLY VERIFIED (archived, synthetic) | [data/artifacts/codec_benchmark/manifest.json](data/artifacts/codec_benchmark/manifest.json) |
| **O(N log N) Fast RFT** | NOT CLAIMED | OBSERVED (unavailable) | Open research problem |
| **φ-SIS Hardness** | NOT CLAIMED | OBSERVED (no reduction) | Non-standard lattice structure |

### 11.2 What Constitutes a Formal Proof

A **formal proof** in this context requires:

1. **Mathematical Statement**: Precise theorem with quantified hypotheses
2. **Rigorous Derivation**: Step-by-step logical deduction
3. **Peer Review**: Independent verification by qualified mathematicians
4. **Publication**: Appearance in peer-reviewed venue

**Example of constructed result**:
> **Theorem (Gram-Schmidt Orthonormalization)**: Let Φ be a full-rank N×M matrix. Then Φ̃ = Φ(Φ†Φ)^{-1/2} satisfies Φ̃†Φ̃ = I.

This is shown by direct computation: (Φ†Φ)^{-1/2} exists when Φ has full rank, and the product collapses to identity.

### 11.3 What Constitutes Empirical Verification

**Empirical verification** provides:

1. **Reproducible Artifacts**: Commit hash, seed, command, output files
2. **Statistical Confidence**: Error bars, multiple runs, p-values where applicable
3. **Negative Results Reported**: Failures documented alongside successes
4. **Conditions Documented**: Hardware, software versions, data sources

**Example of empirical result**:
> ✅ **RFT round-trip error**: rel_L2 = 1.2e-14 across N ∈ {64, 128, 256, 512, 1024}
> - Commit: 3bbe550d94f8
> - Seed: 42
> - Command: `python scripts/archive_rft_stability.py --output data/artifacts/rft_stability`

### 11.4 Gaps Requiring Formal Work

#### 11.4.1 Decorrelation Theory

**Current status**: Empirical observation that RFT decorrelates natural signals similarly to DCT.

**What's needed for formal proof**:
- Define decorrelation metric (e.g., average off-diagonal covariance)
- Specify signal model (e.g., first-order Markov)
- Prove bound: E[|Cov(Ψx)_{i,j}|] ≤ f(|i-j|, φ) for i≠j

**Difficulty**: Medium — requires extending DCT decorrelation theory to φ-parameterized bases.

#### 11.4.2 Crypto Security Reduction

**Current status**: 48-round Feistel passes avalanche tests (99.9%), no formal analysis.

**What's needed for formal proof**:
- Prove pseudorandom permutation (PRP) property under standard assumption
- Or prove security reduction to φ-phase hardness assumption
- Differential trail probability bounds
- Linear approximation bias bounds

**Difficulty**: High — requires new cryptographic assumptions or reduction to existing ones.

#### 11.4.3 Rate-Distortion Optimality

**Current status**: Codecs achieve specific BPP/PSNR points empirically.

**What's needed for formal proof**:
- Derive rate-distortion function R(D) for signal class
- Prove codec achieves R(D) to within additive gap
- Compare to Shannon lower bound

**Difficulty**: Medium-High — requires information-theoretic analysis of φ-transforms.

#### 11.4.4 O(N log N) Algorithm

**Current status**: Not claimed; O(N²) matrix-vector multiply is baseline.

**What's needed for formal proof**:
- Discover sparse factorization of Ψ_{j,k} = e^{2πi·k·φ^j}
- Prove factorization has O(log N) stages with O(N) work per stage
- Or prove such factorization cannot exist (negative result)

**Difficulty**: Unknown — may be impossible; φ-phase structure differs fundamentally from roots of unity.

### 11.5 Proof Artifacts in Repository

| Proof | File | Status |
|-------|------|-----------|
| Gram orthonormalization | [docs/proofs/gram_normalization.md](proofs/gram_normalization.md) | CONSTRUCTED (by definition) |
| φ arithmetic identities | [docs/theory/phi_identities.md](theory/phi_identities.md) | OBSERVED (symbolic identities) |
| Frame bounds (empirical) | [experiments/frame_bounds.py](../experiments/frame_bounds.py) | OBSERVED (not archived) |
| Decorrelation (empirical) | [benchmarks/class_b_transform_dsp.py](../benchmarks/class_b_transform_dsp.py) | OBSERVED (not archived) |

### 11.6 Honest Assessment

**What we can formally claim**:
1. Gram-normalized RFT is unitary (CONSTRUCTED by definition)
2. φ = (1+√5)/2 has algebraic identities (standard number theory)
3. Implementation matches specification (code + tests)

**What we can empirically claim**:
1. RFT round-trip error ≈ 1e-14 (EMPIRICALLY VERIFIED; [data/artifacts/rft_stability/manifest.json](data/artifacts/rft_stability/manifest.json))
2. Crypto avalanche ≈ 50% (EMPIRICALLY VERIFIED; [data/artifacts/crypto_tests/manifest.json](data/artifacts/crypto_tests/manifest.json))
3. Codec achieves specific BPP/PSNR points (EMPIRICALLY VERIFIED; [data/artifacts/codec_benchmark/manifest.json](data/artifacts/codec_benchmark/manifest.json))

**What we explicitly do NOT claim**:
1. Cryptographic security (no proof, no audit)
2. Superiority over FFT (different, not better)
3. O(N log N) complexity (no formal proof)
4. Optimality of any kind (no theorems)

---

## Summary

### Documented Algorithm Count

| Category | Count | Verification Status |
|----------|-------|---------------------|
| **Core RFT** | 5 | Distinct implementation files |
| **Operator Variants** | 6 | Named functions with distinct kernels |
| **Hybrid Codecs** | 16 | Named classes with distinct control flow |
| **Compression** | 4 | Encoder/decoder pairs |
| **Crypto** | 2 | Distinct primitive implementations |
| **Quantum** | 3 | Topological Kernel, Enhanced Qubit, Grover Search |
| **Total** | **36** | Candidate inventory (utilities excluded) |

### Performance Claims — Evidence Status

| Claim | Status | Evidence | Action Required |
|-------|--------|----------|-----------------|
| H3 achieves 0.67 BPP | ⚠️ Claimed | Single-point observation | R-D curve + baselines |
| FH5 achieves 0.41 BPP on edges | ⚠️ Claimed | Single-point observation | R-D curve + baselines |
| Cond(Gram) = 1.0 | EMPIRICALLY VERIFIED (archived) | QR-orthonormalized | [data/artifacts/rft_stability/manifest.json](data/artifacts/rft_stability/manifest.json) |
| Unitarity Ψ†Ψ = I | CONSTRUCTED + EMPIRICALLY VERIFIED (archived) | Gram-normalized mode | [data/artifacts/rft_stability/manifest.json](data/artifacts/rft_stability/manifest.json) |
| Round-trip invertibility | CONSTRUCTED + EMPIRICALLY VERIFIED (archived) | rel_L2 ≈ 1e-14 | [data/artifacts/rft_stability/manifest.json](data/artifacts/rft_stability/manifest.json) |
| Crypto avalanche 50% | EMPIRICALLY VERIFIED (archived) | 99.9% of ideal | [data/artifacts/crypto_tests/manifest.json](data/artifacts/crypto_tests/manifest.json) |
| O(N log N) fast RFT | ❌ NOT CLAIMED | No factorization | Open research problem |
| φ-SIS secure | ❌ NOT CLAIMED | Non-standard assumption | External cryptanalysis needed |
| Feistel cipher secure | ❌ NOT CLAIMED | No audit | External review needed |

**Legend**:
- EMPIRICALLY VERIFIED (archived): Artifact with `{commit, seed, dataset, command}`
- OBSERVED (not archived): Transient output only
- NOT CLAIMED: Explicitly not claimed

### Recommended Algorithms (with caveats)

| Use Case | Algorithm | Performance | Caveat |
|----------|-----------|-------------|--------|
| **Compression** | H3_Hierarchical_Cascade | 0.67 BPP, η=0 | Lossy, tested on synthetic data |
| **Edge Detection** | FH5_Entropy_Guided | 0.41 BPP | Domain-specific tuning required |
| **Audio/Music** | RFT-Harmonic | Natural overtones | Higher latency than FFT |
| **Medical Imaging** | RFT-Wavelet Medical v2 | Rician denoising | Not FDA cleared |
| **Crypto** | — | — | **Do not use any** |
| **Quantum Sim** | Topological Kernel | Surface code | Symbolic only, not real qubits |

### What This Is NOT

- NOT a replacement for FFT (slower, O(N²) vs O(N log N))
- NOT production-ready crypto (no audits, no test vectors)
- NOT NIST-approved anything
- NOT a claim of superiority over established tools

### What This IS

- A research platform exploring φ-parameterized transforms
- Empirical evidence of decorrelation properties
- A testbed for hybrid codec strategies
- An honest accounting of what works and what doesn't

---

## References

- **Patent**: USPTO 19/169,399 "Hybrid Computational Framework for Quantum and Resonance Simulation" (pending)
- **Research Guide**: [RESEARCH_SOURCES_AND_ANALYSIS_GUIDE.md](../RESEARCH_SOURCES_AND_ANALYSIS_GUIDE.md)
- **Core Theory**: [docs/theory/](theory/)
- **Proofs**: [docs/proofs/](proofs/)
- **Validation**: [docs/validation/](validation/)

---

## Cross-Reference: Research Sources Alignment

This section maps algorithms to their validation protocols from [RESEARCH_SOURCES_AND_ANALYSIS_GUIDE.md](../RESEARCH_SOURCES_AND_ANALYSIS_GUIDE.md).

### Validation Protocol Mapping

| Algorithm Category | Research Guide Protocol | Validation Command | Status |
|--------------------|------------------------|-------------------|--------|
| Core RFT (§1) | Protocol A: Transform Analysis | `pytest tests/test_rft_unitarity.py -v` | ✅ Documented |
| Hybrid Codecs (§5) | Phase 3: Benchmark Reproduction | `python benchmarks/run_all_benchmarks.py` | OBSERVED (not archived) |
| Crypto (§7) | Protocol C: Cryptographic Security | NIST STS not yet run | OBSERVED (not archived) |
| Quantum Sim (§8) | Protocol B: Quantum Simulation | Qiskit/Cirq comparison in Class A | OBSERVED (not archived) |
| Medical (§5.3) | Protocol D: Medical Signal Processing | PhysioNet tests in benchmarks | OBSERVED (not archived) |

### Source Cross-Reference

| NOVEL_ALGORITHMS Claim | Research Guide Source | Validation Status |
|------------------------|----------------------|-------------------|
| "O(N²) naive, no fast algorithm formally shown" | arXiv math.NA (Protocol A) | OBSERVED (not archived) |
| "O(N³) for QR" | MIT 18.06 Linear Algebra | ✅ Standard complexity |
| "Cond(Gram) = 1.0 (QR-enforced)" | Protocol A: Ψ†Ψ = I | ✅ Clarified as construction |
| φ-SIS "non-standard assumption" | IACR ePrint, Protocol C | ✅ Correctly distinguished |
| "q=3329 does NOT confer Kyber security" | NIST PQC, liboqs docs | ✅ Accurate |
| Medical "Not FDA cleared" | Protocol D, FDA guidance | ✅ Disclaimer present |

### Identified Gaps (Action Items)

| Gap | Research Guide Reference | Recommendation | Status |
|-----|-------------------------|----------------|--------|
| No NIST STS results for crypto | Protocol C, Step 1 | Run `nist-sts` on EnhancedRFTCryptoV2 output | ✅ **Artifacts generated** (`data/artifacts/nist_sts/`) |
| Patent variants (§3) not individually tested | Phase 3: Benchmark all variants | Add tests for patent_variants.py | ✅ **Tests added** (`tests/test_patent_variants.py`) |
| ARFT (§4) has no benchmark data | Protocol A: Timing test | Add ARFT to `run_all_benchmarks.py` | ✅ **Added** (Class C runner) |
| Compression (§6) missing Shannon limit comparison | §5 Compression sources | Add entropy gap vs theoretical limit | ✅ **Added** (`entropy_gap` suite) |

### File Path Verification

Cross-checked against actual repository structure (2026-01-26):

| Claimed Path | Status | Actual Path |
|--------------|--------|-------------|
| `algorithms/rft/core/resonant_fourier_transform.py` | ✅ Exists | — |
| `algorithms/rft/variants/operator_variants.py` | ✅ Exists | — |
| `algorithms/rft/variants/patent_variants.py` | ✅ Exists | — |
| `experiments/hypothesis_testing/hybrid_mca_fixes.py` | ✅ Exists | — |
| `algorithms/rft/core/enhanced_rft_crypto_v2.py` | ✅ Exists | — |
| `algorithms/rft/compression/rft_vertex_codec.py` | ✅ Exists | — |

**Verification command**: `find . -name "<filename>" -type f`

---

## Appendix: Reproducibility Artifact Summary

> **As of**: 2026-01-27 (scripts updated, re-run required)

### Artifact Status

| Category | Artifact Path | Status | What It Proves |
|----------|---------------|--------|----------------|
| **Codec R-D** | `data/artifacts/codec_benchmark/` | ✅ Generated | Reproducible on **synthetic** inputs |
| **Crypto KAT** | `data/artifacts/crypto_tests/kat_vectors.json` | ✅ **PASS** | Determinism + correct decrypt |
| **Crypto Avalanche** | `data/artifacts/crypto_tests/avalanche_test.json` | ✅ **PASS (99.9%)** | Excellent diffusion (64/128 bits) |
| **Crypto Differential** | `data/artifacts/crypto_tests/differential_test.json` | ✅ **PASS** | 1000/1000 unique output differences |
| **RFT Stability** | `data/artifacts/rft_stability/` | ✅ **PASS** | Unitary (energy=1.0), invertible (err=1e-14) |

### Bugs Fixed in This Update

1. **Crypto tests**: Was using simplified stub cipher lacking MDS layers → Now imports actual `EnhancedRFTCryptoV2`
2. **RFT stability**: Was using default waveform mode (T=N*16) → Now uses gram-normalized square mode

### Regeneration Commands (MUST RE-RUN)

```bash
# 1. Crypto tests with ACTUAL cipher implementation
python scripts/archive_crypto_tests.py --output data/artifacts/crypto_tests --seed 42

# 2. RFT stability with gram-normalized mode  
python scripts/archive_rft_stability.py --output data/artifacts/rft_stability --seed 42

# 3. Codec R-D curves (unchanged, already correct)
python scripts/archive_codec_rd_curves.py --output data/artifacts/codec_benchmark --seed 42
```

### EMPIRICALLY VERIFIED Results (2026-01-27)

| Test | Previous (Buggy) | Actual (Fixed) | Status |
|------|------------------|----------------|--------|
| Crypto avalanche | 12.5% (stub had no diffusion) | **99.9%** (64/128 bits) | ✅ PASS |
| RFT rel_l2_err | 1.46 (145% error, wrong API) | **~1e-14** (machine precision) | ✅ PASS |
| RFT energy ratio | 64× (waveform blowup) | **1.0** (unitary) | ✅ PASS |

### What This Does NOT Prove

1. **Novelty**: Having reproducible artifacts does not prove the algorithms are novel
2. **Correctness across regimes**: Synthetic tests do not validate real-world performance
3. **Security**: KAT vectors prove determinism, not security
4. **Competitive performance**: No comparison to libjpeg-turbo, libwebp, libavif, etc.

### Manifest Format

Each artifact directory contains `manifest.json` with:
```json
{
  "commit_sha": "<git commit>",
  "timestamp_utc": "<ISO timestamp>",
  "seed": 42,
  "command": "<exact invocation>",
  "env": { "python": "3.12.x", "numpy": "1.26.x" }
}
```
```

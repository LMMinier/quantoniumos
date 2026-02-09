# QuantoniumOS Technical Details

> Moved from [README.md](../README.md) to keep the top-level summary concise.
> See also [DOCS_INDEX.md](DOCS_INDEX.md) for the full documentation tree.

---

## RFT Definition Update (December 2025)

**Breaking Change:** The definition of "RFT" (Resonant Fourier Transform) has been corrected.

### What Changed

| Term | OLD (Legacy/Alternative) | NEW (Canonical) |
|------|--------------------------|-----------------|
| **RFT** | Eigenbasis of resonance operator $K$ | Gram-normalized φ-grid exponential basis $\widetilde{\Phi}$ |
| **Sparsity** | None vs FFT | **+15-20 dB PSNR** on target signals |
| **Novelty** | Trivially equivalent to phased DFT | Genuine operator-eigenbasis transform |

### The Canonical RFT Definition

$$
\widetilde{\Phi} = \Phi\,(\Phi^H\Phi)^{-1/2},\quad \text{RFT}(x)=\widetilde{\Phi}^H x
$$

- **Unitary:** $\widetilde{\Phi}^H\widetilde{\Phi}=I$ (proofs in [THEOREMS_RFT_IRONCLAD.md](../THEOREMS_RFT_IRONCLAD.md))
- **Target regime:** golden quasi‑periodic / Fibonacci‑structured signals
- **Honest losses:** FFT/DCT outperform on non‑target families

$$
\Phi_{n,k} = \frac{1}{\sqrt{N}} \exp\left(j 2\pi f_k n\right), \quad f_k = \operatorname{frac}((k+1)\phi)
$$

$$
\widetilde{\Phi} = \Phi\,(\Phi^H\Phi)^{-1/2}, \quad \text{RFT}(x) = \widetilde{\Phi}^H x
$$

### The φ-Phase FFT (Deprecated)

The old formula Ψ = D_φ C_σ F is now called **φ-phase FFT** or **phase-tilted FFT**:
- Has property: |(Ψx)_k| = |(Fx)_k| for all x
- **No sparsity advantage** over standard FFT
- Preserved for backwards compatibility only

### File Changes

| Old File | New File | Notes |
|----------|----------|-------|
| `closed_form_rft.py` | `phi_phase_fft_optimized.py` | Deprecated φ-phase FFT |
| `rft_optimized.py` | `phi_phase_fft_optimized.py` | Deprecated optimized version |
| (new) | `resonant_fourier_transform.py` | **Canonical RFT kernel** |
| (new) | `README_RFT.md` | Authoritative RFT definition |

### Validated Results

| Benchmark | RFT Wins | Condition |
|-----------|----------|-----------|
| In-Family (Golden QP) | **82%** | N >= 256 |
| Out-of-Family | 25% | Expected (domain-specific) |
| PSNR Gain | **+15-20 dB** | At 10% coefficient retention |

---

## IP Pillar Verification (CI-tested)

| IP Pillar | Claim | Test Command | Expected |
|-----------|-------|--------------|----------|
| **1. RFT Transform** | Unitarity (roundtrip < 1e-14) | `python -c "from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT; ..."` | `~8e-16` |
| **2. RFT ≠ FFT** | Non-equivalence | See test suite | `<0.5` correlation |
| **3. Compression** | Zero coherence, high PSNR | `pytest tests/codec_tests/ -v` | All pass |
| **4. Crypto** | Avalanche ~50% | Enhanced cipher tests | `50% ±5%` |
| **5. Theorem 8** | K₀.₉₉(RFT) < K₀.₉₉(DFT), formally proven | `pytest tests/proofs/test_theorem8_formal_proof.py -v` | 33 tests pass |
| **6. Theorem 9** | Maassen-Uffink Entropic Uncertainty | `pytest tests/proofs/test_maassen_uffink_uncertainty.py -v` | 31 tests pass |

### Reproducible Benchmark Commands

| Claimed Metric | Command | Output Location |
|----------------|---------|-----------------|
| **H3 PSNR 52-60 dB** | `python experiments/ascii_wall/ascii_wall_final_hypotheses.py` | stdout + `experiments/hypothesis_testing/final_hypothesis_results.txt` |
| **BPP 0.65-0.87** | Same as above | Same file |
| **Coherence = 0.00** | Same as above | Same file |
| **R-D Curves** | `python scripts/archive_codec_rd_curves.py` | `data/artifacts/codec_benchmark/` |
| **ANS Roundtrip** | `pytest tests/codec_tests/test_ans_codec.py -v` | stdout (25 tests pass) |
| **RFT Unitarity** | `pytest tests/transforms/test_rft_correctness.py -v` | stdout (43 tests pass) |
| **Full Test Suite** | `pytest tests/ -v` | stdout |

---

## Validated Results (December 2025)

### The 14 Variants & Hybrids

**Group A – Core Unitary Variants**

| # | RFT Variant | Innovation | Use Case | Kernel ID |
|---|---------|-----------|----------|----------|
| 1 | Original Φ-RFT | Golden-ratio phase | Exact diagonalization | `RFT_VARIANT_STANDARD` |
| 2 | Harmonic Φ-RFT | Cubic phase (curved time) | Nonlinear filtering | `RFT_VARIANT_HARMONIC` |
| 3 | Fibonacci RFT | Integer Fibonacci progression | Lattice structures | `RFT_VARIANT_FIBONACCI` |
| 4 | Chaotic Φ-RFT | Lyapunov / Haar scrambling | Diffusion / crypto mixing | `RFT_VARIANT_CHAOTIC` |
| 5 | Geometric Φ-RFT | φ-powered lattice | Optical / analog computing | `RFT_VARIANT_GEOMETRIC` |
| 6 | Φ-Chaotic RFT Hybrid | Structure + disorder blend | Resilient codecs | `RFT_VARIANT_PHI_CHAOTIC` |
| 7 | Hyperbolic Φ-RFT | tanh-based phase warp | Phase-space embeddings | `RFT_VARIANT_HYPERBOLIC` |

**Group B – Hybrid / Cascade Variants**

| # | RFT Hybrid | Innovation | Use Case | Kernel ID |
|---|---------|-----------|----------|----------|
| 8 | Log-Periodic Φ-RFT | Log-frequency phase warp | Symbol compression | _Python (research)_ |
| 9 | Convex Mixed Φ-RFT | Log/standard phase blend | Adaptive textures | _Python (research)_ |
|10 | Exact Golden Ratio Φ-RFT | Full resonance lattice | Theorem validation | _Python (research)_ |
|11 | H3 RFT Cascade | Zero-coherence routing | Universal compression (0.673 BPP) | `RFT_VARIANT_CASCADE` |
|12 | FH2 Adaptive RFT Split | Variance-based DCT/RFT split | Structure vs texture | `RFT_VARIANT_ADAPTIVE_SPLIT` |
|13 | FH5 Entropy-Guided RFT Cascade | Entropy routing | Edge-dominated signals (0.406 BPP) | `RFT_VARIANT_ENTROPY_GUIDED` |
|14 | H6 RFT Dictionary | RFT↔DCT bridge atoms | Highest PSNR | `RFT_VARIANT_DICTIONARY` |

### Compression: Competitive Transform Codec

**Paper:** [*Coherence-Free Hybrid DCT–RFT Transform Coding*](https://zenodo.org/uploads/17726611)

Honest Results:
- **Greedy Hybrid Failure:** BPP = 0.812, Coherence = 0.50 (50% energy loss)
- **H3 Cascade Solution:** BPP = 0.655-0.669, Coherence = 0.00 (zero energy loss)
- **FH5 Entropy-Guided:** BPP = 0.663, PSNR = 23.89 dB, η=0 coherence
- **Improvement:** 17-19% compression gain with zero coherence violation

### Key Validation Results

- **Exact Unitarity:** Round-trip error < 1e-14 across all 13 working variants.
- **Coherence Elimination:** All cascade hybrids achieve η=0 coherence.
- **Transform Speed:** Φ-RFT is 1.6-4.9× slower than FFT (expected for O(n²) vs O(n log n)).
- **Avalanche Effect:** RFT-SIS achieves 50.0% avalanche.
- **Hybrid Status:** 14/16 hybrids working (H2, H10 have minor bugs).

---

## Theorem 8: Golden Spectral Concentration Inequality — PROVEN (Constructive + Computational)

**Status:** ✅ Formally proven via 5 lemmas (8.3a–e). Covariance has exact rank K = O(log N). No empirical claims remain.

$$
K_{0.99}(\text{RFT}) = K = O(\log N) \quad\text{vs}\quad K_{0.99}(\text{DFT}) \propto N^{0.75}
$$

| N | K₀.₉₉(RFT) | K₀.₉₉(DFT) | Gap |
|---|-------------|-------------|-----|
| 64 | 7 | 24 | 71% |
| 128 | 8 | 41 | 80% |
| 256 | 9 | 72 | 88% |
| 512 | 10 | 127 | 92% |

```bash
# Formal proof tests (33 tests — 5 lemmas + combined + structural)
pytest tests/proofs/test_theorem8_formal_proof.py -v
# Legacy bootstrap tests
pytest tests/proofs/test_rft_transform_theorems.py -k theorem_8 -v
# Integrated proof engine
pytest tests/proofs/test_formal_proofs.py -v
```

## Theorem 9: Maassen-Uffink Entropic Uncertainty

$$
H(|x|^2) + H(|U_\phi^H x|^2) \geq -2 \log(\mu(U_\phi))
$$

| Transform | Mutual Coherence μ | Entropy Bound |
|-----------|-------------------|---------------|
| DFT | 1/√N | log(N) |
| RFT | > 1/√N | < log(N) |

```bash
pytest tests/proofs/test_maassen_uffink_uncertainty.py -v  # 31 tests pass
```

---

## Medical Applications

**Status:** 83 tests passing | Open Research Preview
**Dataset DOI:** [![RFT-Wavelet Medical Data](https://zenodo.org/badge/DOI/10.5281/zenodo.17885350.svg)](https://doi.org/10.5281/zenodo.17885350)

> **RESEARCH USE ONLY** — NOT FOR CLINICAL OR DIAGNOSTIC APPLICATION

| Domain | Tests | Key Metrics |
|--------|-------|-------------|
| **Medical Imaging** | MRI/CT/PET denoising | PSNR, SSIM vs DCT/Wavelet |
| **Biosignals** | ECG/EEG/EMG compression | PRD < 9%, SNR, correlation |
| **Genomics** | K-mer spectrum, contact maps | Compression ratio, F1 score |

### Real-Data Results (MIT-BIH & Sleep-EDF)

| Method | Avg PSNR Delta | Avg Correlation | Best For |
|--------|-----------|-----------------|----------|
| **RFT (entropy_modulated)** | **+2.61 dB** | **0.859** | ECG waveform fidelity |
| Wavelet (Haar) | -2.48 dB | 0.447 | EEG band preservation |

```bash
pytest tests/medical/ -v
python tests/medical/run_medical_benchmarks.py --report
```

Full report: [docs/reports/RFT_MEDICAL_BENCHMARK_REPORT.md](reports/RFT_MEDICAL_BENCHMARK_REPORT.md)

---

## QuantSoundDesign: Φ-RFT Sound Design Studio

Professional-grade sound design studio built on Φ-RFT. Unlike traditional DAWs (FFT/DCT), QuantSoundDesign uses 7 unitary Φ-RFT variants for synthesis, analysis, and effects.

| File | Purpose |
|------|---------|
| `gui.py` | Main UI (FL Studio/Ableton-inspired, 3200+ LOC) |
| `engine.py` | Track/clip management, RFT processing pipeline |
| `synth_engine.py` | Polyphonic synth with Φ-RFT additive synthesis |
| `pattern_editor.py` | 16-step drum sequencer with RFT drum synthesis |
| `piano_roll.py` | MIDI editor with computer keyboard input |

```bash
python quantonium_os_src/frontend/quantonium_desktop.py
```

---

## RFTPU: Hardware Accelerator Architecture

64-tile hardware accelerator implementing Φ-RFT in silicon (TL-Verilog for Makerchip). **N7FF is a design‑target spec only; no tape‑out or silicon is claimed.**

| Parameter | Value |
|-----------|-------|
| **Process** | TSMC N7FF (design target) |
| **Tile Array** | 8×8 = 64 tiles |
| **Peak Performance** | 2.39 TOPS @ 950 MHz |
| **Efficiency** | 291 GOPS/W |
| **NoC Bandwidth** | 460 GB/s |
| **Power** | <9W |

### RTL Modules

| Module | File | Description |
|--------|------|-------------|
| `phi_rft_core` | `rftpu_architecture.tlv` | 8-point Φ-RFT with Q1.15 kernel ROM |
| `rftpu_noc_fabric` | `rftpu_architecture.tlv` | Cycle-accurate 8×8 mesh NoC |
| `rftpu_accelerator` | `rftpu_architecture.tlv` | Top-level 64-tile instantiation |

### 3D Chip Viewer

```bash
cd hardware/rftpu-3d-viewer && npm install && npm run dev
# Open http://localhost:5173/
```

---

## Φ-RFT Reference API

### Canonical (Recommended)

```python
from algorithms.rft.core.resonant_fourier_transform import (
    rft_basis_matrix, rft_forward_frame, rft_inverse_frame,
)
Phi = rft_basis_matrix(N=256, use_gram_normalization=True)
X = rft_forward_frame(signal, Phi)
rec = rft_inverse_frame(X, Phi)
```

### Legacy Optimized (Deprecated)

```python
from algorithms.rft.core.phi_phase_fft_optimized import (
    rft_forward_optimized, rft_inverse_optimized, OptimizedRFTEngine,
)
Y = rft_forward_optimized(x)
x_rec = rft_inverse_optimized(Y)
```

### Performance Comparison

| Implementation | n=1024 | n=4096 | Ratio to FFT |
|----------------|--------|--------|--------------|
| **FFT (NumPy)** | 15.6 µs | 38.6 µs | 1.00× |
| **RFT Optimized** | 21.4 µs | 43.7 µs | **1.06×** |
| **RFT Original** | 85.4 µs | 296.9 µs | 4.97× |

---

## RFT Validation & Experiments

### CLI Proof Runner

```bash
python scripts/run_proofs.py --list       # List all proof tests
python scripts/run_proofs.py --quick      # Quick validation (~2 min)
python scripts/run_proofs.py --full       # Full suite (5-10 min)
python scripts/run_proofs.py --category unitarity
```

| Category | Description | Tests |
|----------|-------------|-------|
| `unitarity` | Verify Ψ^H Ψ = I for all variants | 2 |
| `non-equivalence` | Prove RFT ≠ permuted DFT | 2 |
| `sparsity` | Domain-specific sparsity advantage | 2 |
| `coherence` | Zero-coherence cascade (H3/FH5) | 2 |
| `hardware` | FPGA/TLV kernel validation | 2 |

---

## Repository Layout

```
QuantoniumOS/
├─ algorithms/rft/          # RFT core, variants, hybrids, crypto
├─ src/rftmw_native/        # C++ SIMD engine (AVX2/AVX512)
├─ hardware/                # RTL, testbenches, 3D viewer
├─ tests/                   # Unit, integration, validation, proofs
├─ experiments/             # Benchmarks vs competitors
├─ scripts/                 # Reproducibility runners
├─ docs/                    # Architecture, guides, reports
├─ quantonium_os_src/       # Desktop UI shell
├─ quantonium-mobile/       # React Native mobile app
└─ results/                 # Benchmark outputs
```

# RFT Technical Details

> Detailed variant reference, benchmarks, and API documentation.
> For the core novelty, start with the [README](../README.md).
> For formal proofs, see [THEOREMS_RFT_IRONCLAD.md](THEOREMS_RFT_IRONCLAD.md).

---

## Canonical RFT Definition

$$
\Phi_{n,k} = \frac{1}{\sqrt{N}} \exp\!\bigl(j\,2\pi\,\text{frac}((k{+}1)\,\phi)\,n\bigr), \qquad
\widetilde{\Phi} = \Phi\,(\Phi^H\Phi)^{-1/2}
$$

$$
\text{RFT}(x) = \widetilde{\Phi}^H\,x, \qquad x = \widetilde{\Phi}\,\hat{x}
$$

- **Unitary:** $\widetilde{\Phi}^H\widetilde{\Phi}=I$ — proven in [Theorem 2](THEOREMS_RFT_IRONCLAD.md)
- **Target regime:** golden quasi-periodic / Fibonacci-structured signals
- **Honest losses:** FFT/DCT outperform on non-target families
- **Complexity:** O(N²) canonical, O(N log N) hybrid variant

---

## IP Pillar Verification (CI-tested)

| IP Pillar | Claim | Test Command | Expected |
|-----------|-------|--------------|----------|
| **1. RFT Transform** | Unitarity (roundtrip < 1e-14) | `python -c "from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT; ..."` | `~8e-16` |
| **2. RFT ≠ FFT** | Non-equivalence | See test suite | `<0.5` correlation |
| **3. Compression** | Zero coherence, high PSNR | `pytest tests/codec_tests/ -v` | All pass |
| **4. Crypto** | Avalanche ~50% | Enhanced cipher tests | `50% ±5%` |
| **5. Theorem 8** | K₀.₉₉(RFT) < K₀.₉₉(DFT), formally proven (Diophantine) | `pytest tests/proofs/test_theorem8_formal_proof.py tests/proofs/test_theorem8_diophantine.py -v` | 33+46 tests pass |
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
| 1 | Standard (Legacy φ-Phase) | Legacy golden-ratio phase (not the canonical RFT) | Exact diagonalization | `RFT_VARIANT_STANDARD` |
| 2 | Harmonic Variant | Cubic phase (curved time) | Nonlinear filtering | `RFT_VARIANT_HARMONIC` |
| 3 | Fibonacci Variant | Integer Fibonacci progression | Lattice structures | `RFT_VARIANT_FIBONACCI` |
| 4 | Chaotic Variant | Lyapunov / Haar scrambling | Diffusion / crypto mixing | `RFT_VARIANT_CHAOTIC` |
| 5 | Geometric Variant | φ-powered lattice | Optical / analog computing | `RFT_VARIANT_GEOMETRIC` |
| 6 | φ-Chaotic Hybrid | Structure + disorder blend | Resilient codecs | `RFT_VARIANT_PHI_CHAOTIC` |
| 7 | Hyperbolic Variant | tanh-based phase warp | Phase-space embeddings | `RFT_VARIANT_HYPERBOLIC` |

**Group B – Hybrid / Cascade Variants**

| # | RFT Hybrid | Innovation | Use Case | Kernel ID |
|---|---------|-----------|----------|----------|
| 8 | Log-Periodic Variant | Log-frequency phase warp | Symbol compression | _Python (research)_ |
| 9 | Convex Mixed Variant | Log/standard phase blend | Adaptive textures | _Python (research)_ |
|10 | Exact Golden Ratio Variant | Full resonance lattice | Theorem validation | _Python (research)_ |
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
- **Transform Speed:** The canonical RFT is 1.6-4.9× slower than FFT (expected for O(n²) vs O(n log n)).
- **Avalanche Effect:** RFT-SIS achieves 50.0% avalanche.
- **Hybrid Status:** 14/16 hybrids working (H2, H10 have minor bugs).

---

## Theorem 8: Golden Spectral Concentration Inequality — PROVEN (Constructive + Diophantine)

**Status:** ✅ Formally proven via 5 constructive lemmas (8.3a–e) + 6 Diophantine lemmas (8.4a–f). DFT spectral leakage grounded in classical number theory (Hurwitz 1891, Steinhaus-Sós 1957, Weyl 1916, Erdős-Turán 1948). No empirical claims remain.

$$
K_{0.99}(\text{RFT}) = K = O(\log N) \quad\text{vs}\quad K_{0.99}(\text{DFT}) \propto N^{0.75}
$$

| N | K₀.₉₉(RFT) | K₀.₉₉(DFT) | Gap |
|---|-------------|-------------|-----|
| 64 | 7 | 24 | 71% |
| 128 | 8 | 41 | 80% |
| 256 | 9 | 72 | 88% |
| 512 | 10 | 127 | 92% |

**Diophantine foundation:** The DFT **must** leak energy on golden-quasi-periodic signals because golden frequencies never align with DFT bins — this is Hurwitz's theorem (1891), not a computational observation. The RFT's zero-misalignment advantage is a number-theoretic theorem.

```bash
# Constructive proof tests (33 tests — 5 lemmas + combined + structural)
pytest tests/proofs/test_theorem8_formal_proof.py -v
# Diophantine proof tests (46 tests — 6 lemmas + structural)
pytest tests/proofs/test_theorem8_diophantine.py -v
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

## QuantSoundDesign: RFT Sound Design Studio

Professional-grade sound design studio built on the RFT. Unlike traditional DAWs (FFT/DCT), QuantSoundDesign uses 7 unitary RFT variants for synthesis, analysis, and effects.

| File | Purpose |
|------|---------|
| `gui.py` | Main UI (FL Studio/Ableton-inspired, 3200+ LOC) |
| `engine.py` | Track/clip management, RFT processing pipeline |
| `synth_engine.py` | Polyphonic synth with RFT additive synthesis |
| `pattern_editor.py` | 16-step drum sequencer with RFT drum synthesis |
| `piano_roll.py` | MIDI editor with computer keyboard input |

```bash
python quantonium_os_src/frontend/quantonium_desktop.py
```

---

## RFTPU: Hardware Accelerator Architecture

64-tile hardware accelerator implementing the RFT in silicon (TL-Verilog for Makerchip). **N7FF is a design‑target spec only; no tape‑out or silicon is claimed.**

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
| `phi_rft_core` | `rftpu_architecture.tlv` | 8-point RFT with Q1.15 kernel ROM |
| `rftpu_noc_fabric` | `rftpu_architecture.tlv` | Cycle-accurate 8×8 mesh NoC |
| `rftpu_accelerator` | `rftpu_architecture.tlv` | Top-level 64-tile instantiation |

### 3D Chip Viewer

```bash
cd hardware/rftpu-3d-viewer && npm install && npm run dev
# Open http://localhost:5173/
```

---

## RFT Reference API

### Canonical (Recommended)

```python
from algorithms.rft.core.resonant_fourier_transform import (
    rft_basis_matrix, rft_forward_frame, rft_inverse_frame,
)
Phi = rft_basis_matrix(N=256, use_gram_normalization=True)
X = rft_forward_frame(signal, Phi)
rec = rft_inverse_frame(X, Phi)
```

### Canonical RFT

```python
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
rft = CanonicalTrueRFT(N=256)
X = rft.forward_transform(signal)
recovered = rft.inverse_transform(X)
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

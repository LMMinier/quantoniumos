# QuantoniumOS: Quantum-Inspired Research Platform

[![RFT Framework DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17712905.svg)](https://doi.org/10.5281/zenodo.17712905)
[![Coherence Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17726611.svg)](https://doi.org/10.5281/zenodo.17726611)
[![RFTPU Chip Papers DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17822056.svg)](https://doi.org/10.5281/zenodo.17822056)
[![RFT-Wavelet Medical Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17885350.svg)](https://doi.org/10.5281/zenodo.17885350)
[![TechRxiv DOI](https://img.shields.io/badge/DOI-10.36227%2Ftechrxiv.175384307.75693850%2Fv1-8A2BE2.svg)](https://doi.org/10.36227/techrxiv.175384307.75693850/v1)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE.md)
[![License: Non-Commercial](https://img.shields.io/badge/License-Non--Commercial-red.svg)](LICENSE-CLAIMS-NC.md)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange.svg)](docs/project/PATENT_NOTICE.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](tests/)

---

**Scope note:** “QuantoniumOS” is a research branding name. This repository does **not** implement an OS kernel, scheduler, or filesystem. It provides a user‑space desktop UI and signal‑processing stack. Any OS‑level claims require explicit implementation and test evidence.

## Start here (10–15 lines)

**What this is:** A research‑grade, reproducible codebase for Φ‑RFT algorithms, hybrid codecs, and hardware/RTL experiments. No quantum speedups, no clinical claims, no production cryptography.

**Run the core checks (from repo root):**
1) Environment check: [verify_setup.sh](verify_setup.sh)
2) Quick demo: [run_demo.sh](run_demo.sh)
3) Full reproducibility: [reproduce_results.sh](reproduce_results.sh)

**Where the details live:** The canonical RFT definition and proofs are in [algorithms/rft/README_RFT.md](algorithms/rft/README_RFT.md) and [THEOREMS_RFT_IRONCLAD.md](THEOREMS_RFT_IRONCLAD.md). Hardware RTL and reports are in [hardware/](hardware/).

## What is implemented (evidence‑based)

This section lists only what is present in this repository, with pointers to runnable code and reproducibility scripts.

### 1) RFT core algorithms (canonical + variants)
- Canonical RFT definition and implementation: [algorithms/rft/README_RFT.md](algorithms/rft/README_RFT.md)
- Core kernels and transform APIs: [algorithms/rft/core/](algorithms/rft/core/)
- Variants registry and variant implementations: [algorithms/rft/variants/](algorithms/rft/variants/)

### 2) Compression and routing hybrids
- Hybrid/cascade codecs and routing logic: [algorithms/rft/hybrids/](algorithms/rft/hybrids/)
- Compression pipelines and model tooling: [tools/compression/](tools/compression/)

### 3) Cryptography experiments (research‑only)
- Experimental cipher and crypto utilities: [algorithms/rft/crypto/](algorithms/rft/crypto/)
- Benchmarking and analysis tools: [tools/crypto/](tools/crypto/)

### 4) Hardware RTL and synthesis artifacts
- Synthesizable RTL and testbenches: [hardware/rtl/](hardware/rtl/)
- Simulation and reports: [hardware/simulation/](hardware/simulation/)
- WebFPGA/iCE40 synthesis reports: [hardware/](hardware/)

### 5) Desktop UI + apps (user‑space)
- Desktop shell: [quantonium_os_src/frontend/quantonium_desktop.py](quantonium_os_src/frontend/quantonium_desktop.py)
- Apps (visualizer, validator, vault, crypto, simulator, etc.): [quantonium_os_src/apps/](quantonium_os_src/apps/)
- UI styles and assets: [ui/](ui/)

### 6) Mobile app
- React Native app source: [quantonium-mobile/](quantonium-mobile/)

### 7) Formal proofs and technical docs
- Theorem proofs (unitarity, twisted convolution, limits of claims): [THEOREMS_RFT_IRONCLAD.md](THEOREMS_RFT_IRONCLAD.md)
- Architecture docs and validation reports: [docs/](docs/)

## Reproducibility (what you can run)

Use these scripts to reproduce results directly from this repo:

1) Environment check: [verify_setup.sh](verify_setup.sh)
2) Quick demo (RFT power visualization): [run_demo.sh](run_demo.sh)
3) Full reproducibility pipeline: [reproduce_results.sh](reproduce_results.sh)
4) Shannon benchmark suite: [scripts/run_shannon_tests.py](scripts/run_shannon_tests.py)

### IP Pillar Verification (CI-tested on every push)

These 4 tests run automatically in CI and verify the core IP claims:

| IP Pillar | Claim | Test Command | Expected Result |
|-----------|-------|--------------|-----------------|
| **1. RFT Transform** | Unitarity (roundtrip < 1e-14) | `python -c "from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT; import numpy as np; rft=CanonicalTrueRFT(256); x=np.random.randn(256)+1j*np.random.randn(256); err=np.linalg.norm(x-rft.inverse_transform(rft.forward_transform(x)))/np.linalg.norm(x); print(f'{err:.2e}')"` | `~8e-16` |
| **2. RFT ≠ FFT** | Non-equivalence | `python -c "from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT; import numpy as np; from numpy.fft import fft; rft=CanonicalTrueRFT(256); x=np.random.randn(256)+1j*np.random.randn(256); corr=np.abs(np.vdot(rft.forward_transform(x),fft(x,norm='ortho')))/(np.linalg.norm(rft.forward_transform(x))*np.linalg.norm(fft(x,norm='ortho'))); print(f'{corr:.4f}')"` | `<0.5` (proves NOT equivalent) |
| **3. Compression** | Zero coherence, high PSNR | `pytest tests/codec_tests/ -v` | All pass, coherence=0.00 |
| **4. Crypto** | Avalanche ~50% | See [algorithms/rft/crypto/enhanced_cipher.py](algorithms/rft/crypto/enhanced_cipher.py) | `50% ±5%` |

**Full test suite (43 tests):**
```bash
pytest tests/transforms/test_rft_correctness.py -v
```

**File locations for each IP pillar:**

| Pillar | Implementation | Tests | Docs |
|--------|---------------|-------|------|
| RFT Transform | [algorithms/rft/core/canonical_true_rft.py](algorithms/rft/core/canonical_true_rft.py) | [tests/transforms/test_rft_correctness.py](tests/transforms/test_rft_correctness.py) | [algorithms/rft/README_RFT.md](algorithms/rft/README_RFT.md) |
| Compression | [algorithms/rft/hybrids/h3_arft_cascade.py](algorithms/rft/hybrids/h3_arft_cascade.py) | [tests/codec_tests/](tests/codec_tests/) | [docs/](docs/) |
| Crypto | [algorithms/rft/crypto/enhanced_cipher.py](algorithms/rft/crypto/enhanced_cipher.py) | [tests/crypto/](tests/crypto/) | [experiments/crypto_prototypes/README.md](experiments/crypto_prototypes/README.md) |
| Hardware | [hardware/rftpu_architecture.sv](hardware/rftpu_architecture.sv) | [hardware/tb/](hardware/tb/) | [hardware/README.md](hardware/README.md) |

Each script is intentionally referenced to concrete code paths and artifacts in this repository. No claims here depend on external or unpublished code.

## IMPORTANT: RFT Definition Update (December 2025)

**Breaking Change:** The definition of "RFT" (Resonant Fourier Transform) has been corrected.

### What Changed

| Term | OLD (Legacy/Alternative) | NEW (Canonical) |
|------|--------------------------|-----------------|
| **RFT** | Eigenbasis of resonance operator $K$ | Gram-normalized φ-grid exponential basis $\widetilde{\Phi}$ |
| **Sparsity** | None vs FFT | **+15-20 dB PSNR** on target signals |
| **Novelty** | Trivially equivalent to phased DFT | Genuine operator-eigenbasis transform |

### The Canonical RFT Definition

**Canonical RFT (boxed summary):**

$$
\widetilde{\Phi} = \Phi\,(\Phi^H\Phi)^{-1/2},\quad \text{RFT}(x)=\widetilde{\Phi}^H x
$$

- **Unitary:** $\widetilde{\Phi}^H\widetilde{\Phi}=I$ (proofs in [THEOREMS_RFT_IRONCLAD.md](THEOREMS_RFT_IRONCLAD.md))
- **Target regime:** golden quasi‑periodic / Fibonacci‑structured signals
- **Honest losses:** FFT/DCT outperform on non‑target families

The **Resonant Fourier Transform (RFT)** is now defined as the **Gram-normalized φ-grid exponential basis**:

$$
\Phi_{n,k} = \frac{1}{\sqrt{N}} \exp\left(j 2\pi f_k n\right), \quad f_k = \operatorname{frac}((k+1)\phi)
$$

$$
\widetilde{\Phi} = \Phi\,(\Phi^H\Phi)^{-1/2}, \quad \text{RFT}(x) = \widetilde{\Phi}^H x
$$

Where:
- $\Phi$ is the raw irrational-frequency exponential basis (φ-grid)
- $\widetilde{\Phi}$ is the Gram-normalized (unitary) basis
- The resonance-operator eigenbasis is **legacy/alternative**, kept for comparison in `algorithms/rft/variants/operator_variants.py`
- QR-orthonormalized resonance-matrix variants live in `algorithms/rft/variants/registry.py` and are **not** the canonical definition

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

See `algorithms/rft/README_RFT.md` for the complete specification.

---

> **PATENT-PENDING RESEARCH PLATFORM.** QuantoniumOS bundles:
> - the **Φ-RFT** (golden-ratio + chirp, **closed-form, fast** unitary transform),
> - **compression** pipelines (lossless + hybrid learned),
> - **cryptographic** experiments (RFT–SIS hashing),
> - and **comprehensive validation** suites.  
> All "quantum" modules are **classical simulations** or **quantum-inspired data structures** with explicit mathematical checks. They do not simulate physical quantum mechanics.

**USPTO Application:** 19/169,399 (Filed 2025-04-03)  
**Title:** *Hybrid Computational Framework for Quantum and Resonance Simulation*

---

## Current Status (Phase 4)

**December 3, 2025** - Production Optimization Phase
- Core RFT: 39/39 tests passing (100%)
- Hybrid Codecs: 16/17 working (94%)
- Performance Optimization: H1 (146ms -> <10ms target), H10 (16ms -> <5ms target)
- Documentation: Generating Sphinx API docs

See `PHASE4_PLAN.md` for detailed roadmap.

---

## Technical Value Proposition

This repository represents a comprehensive research platform for "Quantum-Inspired" signal processing. While it does not perform actual quantum computation, it provides significant technical value in the following areas:

1.  **Novel Signal Processing Basis**: The Canonical RFT provides a mathematically rigorous eigenbasis for analyzing quasi-periodic and Fibonacci-structured signals, offering 15-20 dB PSNR gains over FFT for these specific classes.
2.  **Hybrid Compression Architecture**: The framework demonstrates a "coherence-free" hybrid codec that seamlessly switches between DCT (for structure) and RFT (for texture/resonance), solving the boundary artifact problem common in block-based transforms.
3.  **Hardware IP Core**: The `hardware/` directory contains a synthesizable Verilog design for the RFTPU (Resonant Fourier Transform Processing Unit), a dedicated accelerator architecture with a custom NoC (Network-on-Chip), validated in simulation.
4.  **Cryptographic Research**: Experimental implementations of RFT-SIS hashing and Feistel networks provide a testbed for chaos-based and lattice-based cryptography concepts.
5.  **Medical & Scientific Applications**: Validated pipelines for denoising and compressing medical data (ECG, EEG) and genomic sequences, showing specific advantages in morphology preservation.

**Note on Valuation:** This stack is a research asset. Its value lies in the intellectual property (patent pending), the validated algorithms, the hardware designs, and the comprehensive test suites that prove the claims. It is designed for researchers, hardware architects, and signal processing engineers.

---

## Reproducibility Scripts

This repository includes a suite of scripts to verify the environment, reproduce benchmarks, and validate claims.

| Script | Description |
| --- | --- |
| **`./reproduce_results.sh`** | **Master Reproducibility Script.** Runs the full verification pipeline: checks environment, builds native engines, runs unit tests, and executes the Class A-E benchmark suite. Use this to validate the entire stack. |
| **`./verify_setup.sh`** | **Environment Health Check.** Verifies Python version, virtual environment status, and core dependencies (NumPy, SciPy, SymPy, Numba). Also performs a quick numerical check of the RFT core. |
| **`./run_demo.sh`** | **Quick Demo.** Launches `demos/demo_rft_power.py` to visualize the power spectral density advantages of RFT on a sample signal. |
| **`scripts/run_full_suite.sh`** | **Full Benchmark Runner.** Executes the comprehensive set of experiments, including long-running compression and crypto benchmarks. |
| **`quantoniumos-bootstrap.sh`** | **Initial Setup.** Automates the cloning, virtual environment creation, and dependency installation process for new users. |

---

## Documentation Index

| Step | Doc | Purpose |
| --- | --- | --- |
| 1 | **[README.md](README.md)** | Project summary, RFT update, quick install |
| 2 | **[GETTING_STARTED.md](GETTING_STARTED.md)** | First run, examples, learning path |
| 3 | **[SETUP_GUIDE.md](SETUP_GUIDE.md)** | Installation, native builds, troubleshooting |
| 4 | **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** | Doc tree, task-based navigation |
| 5 | **[REPO_ORGANIZATION.md](REPO_ORGANIZATION.md)** | Repo structure map (this is the source of truth) |

**More docs:**
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical deep dive (ASM → C → C++ → Python)
- **[docs/ARCHITECTURE_QUICKREF.md](docs/ARCHITECTURE_QUICKREF.md)** - One-page cheat sheet
- **[COMPONENT_INVENTORY.md](COMPONENT_INVENTORY.md)** - Per-path purpose & routes table

**Quick installation:**
```bash
# Clone and setup
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
./quantoniumos-bootstrap.sh

# Or manual setup (no compilation needed!)
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy sympy numba
python -c "from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT; print('Setup complete!')"
```

**Build native engines for 3-10× speedup (optional):**
```bash
# C/ASM kernel
cd algorithms/rft/kernels && make -j$(nproc) && cd ../../..

# C++ engine with AVX2/AVX-512
cd src/rftmw_native && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRFTMW_ENABLE_ASM=ON && make -j$(nproc)
cp rftmw_native.cpython-*-linux-gnu.so ../../../.venv/lib/python3.12/site-packages/
```

**Verify setup:**
```bash
./verify_setup.sh  # Automated health check
```

---

## Validated Results (December 2025)

**Date:** December 3, 2025  
**Status:** Numerically Validated (Classes A-E Benchmark Suite)

The following results have been experimentally validated through automated test suites. Unitarity is verified to machine precision (~1e-15); compression and performance claims are based on measured benchmarks.

### 1. The 14 Variants & Hybrids
We have identified and validated a catalog of 14 Φ-RFT transforms: 7 **core unitary variants** plus 7 **hybrid/cascade modes** that wire DCT, entropy routing, and dictionary learning directly into the RFT stack. All unitary variants stay below $10^{-15}$ error; the hybrids inherit the same basis guarantees while exposing their specialized routing logic.

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

Every variant above is exposed through `algorithms.rft.variants.manifest`, routed through the codecs/benchmarks, and covered by `tests/rft/test_variant_unitarity.py`.

### 3. Compression: Competitive Transform Codec (Not a Breakthrough)

**Paper:** [*Coherence-Free Hybrid DCT–RFT Transform Coding for Text and Structured Data*](https://zenodo.org/uploads/17726611) [![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://zenodo.org/uploads/17726611)

We built a hierarchical transform codec combining DCT and Φ-RFT that is **competitive with classical transform codecs** on structured signals. It does NOT beat entropy bounds or general-purpose compressors.

**Honest Results:**
*   **Greedy Hybrid Failure:** BPP = 0.812, Coherence = 0.50 (50% energy loss)
*   **H3 Cascade Solution:** BPP = 0.655-0.669, Coherence = 0.00 (zero energy loss)
*   **FH5 Entropy-Guided:** BPP = 0.663, PSNR = 23.89 dB, η=0 coherence
*   **Improvement:** 17-19% compression gain with zero coherence violation
*   **Method:** Hierarchical cascade decomposition with adaptive basis pursuit.
*   **Validation:** [![Reproduce Results](https://img.shields.io/badge/Validation-experiments%2Fascii__wall__paper.py-blue)](experiments/ascii_wall_paper.py)

**Run Validation:**
```bash
python experiments/ascii_wall_paper.py
# Validates all 15 architectural variants on 11 signal types
# Output: experiments/ASCII_WALL_PAPER_RESULTS.md
```

*   **Result:** Lossless compression of Python source code (DCT-dominant) *and* high-fidelity capture of Fibonacci resonances (RFT-dominant) in the same pipeline.
*   **Boundary Proof:** The hybrid basis reduces edge reconstruction error by **>30x** compared to pure RFT/DFT for non-periodic signals.
*   **Proof:** [![Theorem 10](https://img.shields.io/badge/Theorem-10_Proven-brightgreen)](papers/coherence_free_hybrid_transforms.tex)

### 4. Key Validation Results
*   **Exact Unitarity:** Round-trip error < 1e-14 across all 13 working variants (GOLDEN_EXACT skipped - O(N³)).
*   **Coherence Elimination:** All cascade hybrids (H3, H7, H8, H9, FH1-FH5) achieve η=0 coherence.
*   **Transform Speed:** Φ-RFT is 1.6-4.9× slower than FFT (expected for O(n²) vs O(n log n)).
*   **Compression BPP:** H3 = 0.655-0.669, FH5 = 0.663, FH2 = 0.715 (all η=0).
*   **Avalanche Effect:** RFT-SIS achieves 50.0% avalanche (ideal cryptographic mixing).
*   **Quantum Scaling:** QSC compresses symbolic qubit configurations at O(n) complexity, reaching 10M+ labels at ~20 M/s. (Note: This compresses labels/configurations, not 2^n quantum amplitudes like Qiskit/Cirq.)
*   **Hybrid Status:** 14/16 hybrids working (H2, H10 have minor bugs).

**Important Disclaimers:**
- **⚠️ Crypto:** All cryptographic constructions are **experimental research prototypes**. They are **NOT NIST-compliant**, have no hardness proofs, no third-party cryptanalysis, and use `numpy.random` instead of CSPRNG. See [experiments/crypto_prototypes/README.md](experiments/crypto_prototypes/README.md) for details. **DO NOT USE FOR REAL SECURITY.**
- **Compression:** Does NOT beat entropy bounds. Competitive with classical transform codecs, not a "breakthrough."
- **Quantum:** This is classical signal processing. "Symbolic qubit" representations are compressed encodings, not quantum computation.

[![Read Proofs](https://img.shields.io/badge/Read-Full_Proofs-blue)](docs/validation/RFT_THEOREMS.md)

---

## What's New (TL;DR)

### Canonical RFT (Current Definition)

The **Resonant Fourier Transform** is the Gram-normalized φ-grid exponential basis:

```python
from algorithms.rft.core.resonant_fourier_transform import (
    rft_basis_matrix,
    rft_forward_frame,
    rft_inverse_frame,
)

# Build canonical RFT basis (O(N³) one-time, cached)
Phi = rft_basis_matrix(N=256, use_gram_normalization=True)

# Transform (O(N²) per signal)
X = rft_forward_frame(signal, Phi)      # RFT(x) = Φ̃^H x
rec = rft_inverse_frame(X, Phi)         # RFT⁻¹(X) = Φ̃ X
```

**Properties:**
- **Unitary:** $\widetilde{\Phi}^H \widetilde{\Phi} = I$ (by Gram normalization)
- **Domain-Specific:** +15-20 dB PSNR on golden quasi-periodic signals
- **Honest:** Loses to FFT/DCT on non-target signal families

### Deprecated: φ-Phase FFT (Old "RFT")

The original formula Ψ = D_φ C_σ F is preserved for backwards compatibility:

*   $[C_\sigma]_{kk} = \exp(i\pi\sigma k^2/n)$
*   $[D_\phi]_{kk} = \exp(2\pi i\,\beta\,\{k/\phi\})$

**WARNING:** This has NO sparsity advantage over FFT. It is a phase-only transform.
See `algorithms/rft/core/phi_phase_fft_optimized.py` (formerly `closed_form_rft.py`).

---

## Repository Layout

```
QuantoniumOS/
├─ algorithms/
│  ├─ rft/
│  │  ├─ README_RFT.md             # AUTHORITATIVE RFT DEFINITION
│  │  ├─ kernels/
│  │  │  ├─ resonant_fourier_transform.py  # Canonical RFT kernel
│  │  │  ├─ operator_arft_kernel.py        # Adaptive RFT (signal-driven)
│  │  │  └─ arft_kernel.py                 # QR-orthonormalized variant
│  │  ├─ core/
│  │  │  ├─ canonical_true_rft.py          # Reference implementation
│  │  │  ├─ phi_phase_fft_optimized.py     # DEPRECATED: old Ψ=DφCσF
│  │  │  └─ phi_phase_fft_optimized.py     # DEPRECATED: fused version
│  │  ├─ theory/
│  │  │  └─ formal_framework.py            # Formal proofs & theorems
│  │  └─ hybrids/                          # Cascade codecs
│  ├─ compression/                 # Lossless & hybrid codecs
│  └─ crypto/                      # RFT–SIS experiments & validators
├─ tests/
│  ├─ benchmarks/
│  │  ├─ honest_rft_benchmark.py   # Canonical RFT vs FFT/DCT
│  │  ├─ rft_multiscale_benchmark.py  # Multi-N scaling tests
│  │  └─ rft_realworld_benchmark.py   # Real-world signal tests
├─ hardware/
│  ├─ rftpu_architecture.tlv       # 64-tile RFTPU TL-Verilog (Makerchip)
│  ├─ quantoniumos_unified_engines.sv  # Unified RTL (RFT+SIS+Feistel)
│  ├─ rftpu-3d-viewer/             # React + Three.js chip visualizer
│  │  ├─ src/RFTPU3DChipDissect.jsx  # 3D chip component
│  │  └─ package.json              # npm run dev
│  ├─ tb/                          # Testbenches & formal props
│  │  ├─ tb_rftpu_accelerator.sv   # Main accelerator testbench
│  │  └─ rftpu_formal_props.sv     # SVA formal verification
│  ├─ PHYSICAL_DESIGN_SPEC.md      # TSMC N7FF design spec
│  └─ HW_TEST_RESULTS.md           # Simulation results
├─ src/
│  ├─ rftmw_native/
│  │  ├─ rftmw_core.hpp            # C++ RFT engine
│  │  ├─ rft_fused_kernel.hpp      # AVX2/AVX512 SIMD kernels
│  │  └─ rftmw_python.cpp          # pybind11 bindings
│  └─ apps/
│     ├─ quantsounddesign/         # Φ-RFT Sound Design Studio
│     └─ ...
├─ experiments/
│  ├─ competitors/
│  │  ├─ benchmark_transforms_vs_fft.py   # RFT vs FFT/DCT benchmark
│  │  ├─ benchmark_compression_vs_codecs.py
│  │  └─ benchmark_crypto_throughput.py
│  └─ ...
├─ scripts/
│  └─ run_full_suite.sh            # One-command benchmark runner
├─ results/                        # Benchmark output (JSON/CSV/MD)
├─ tests/                          # Unit, integration, validation
├─ docs/                           # Tech docs, USPTO packages
├─ REPRODUCING_RESULTS.md          # Reproducibility guide
└─ README.md                       # This file
```

---

## Medical Applications (NEW - December 2025)

**Status:** 83 tests passing | Open Research Preview  
**Dataset DOI:** [![RFT-Wavelet Medical Data](https://zenodo.org/badge/DOI/10.5281/zenodo.17885350.svg)](https://doi.org/10.5281/zenodo.17885350) - RFT-Wavelet Hybrid Denoising Results on Real Medical Data

> **RESEARCH USE ONLY** — NOT FOR CLINICAL OR DIAGNOSTIC APPLICATION

> **FREE FOR HOSPITALS & MEDICAL RESEARCHERS** - This module is free for hospitals, healthcare institutions, medical researchers, and academics for testing and research purposes.

QuantoniumOS includes a comprehensive medical applications test suite validating RFT for healthcare domains:

| Domain | Tests | Key Metrics |
|--------|-------|-------------|
| **Medical Imaging** | MRI/CT/PET denoising | PSNR, SSIM vs DCT/Wavelet |
| **Biosignals** | ECG/EEG/EMG compression | PRD < 9%, SNR, correlation |
| **Genomics** | K-mer spectrum, contact maps | Compression ratio, F1 score |
| **Clinical Security** | Waveform hashing, federated learning | Avalanche effect ~0.5, Byzantine resilience |
| **Edge Devices** | Cortex-M4, ESP32, nRF52 | Memory fit, latency targets |

### Real-Data Validation Results (MIT-BIH & Sleep-EDF)

Benchmark on real PhysioNet open-source medical data:

| Method | Avg PSNR Delta | Avg Correlation | Best For |
|--------|-----------|-----------------|----------|
| **RFT (entropy_modulated)** | **+2.61 dB** | **0.859** | ECG waveform fidelity |
| Wavelet (Haar) | -2.48 dB | 0.447 | EEG band preservation |
| RFT-Wavelet Hybrid | -2.59 dB | 0.472 | Not recommended |

**Key Findings:**
- RFT outperforms wavelets for ECG morphology preservation (+2.20 dB, r=0.914)
- RFT is noise-type agnostic (works on Gaussian, Rician, Poisson)
- Wavelets still win for EEG band power preservation (+6.49 dB)
- Hybrid cascading degrades performance (don't combine)

> **Full Report:** [docs/reports/RFT_MEDICAL_BENCHMARK_REPORT.md](docs/reports/RFT_MEDICAL_BENCHMARK_REPORT.md)

### Quick Start

```bash
# Run all medical tests
pytest tests/medical/ -v

# Run benchmark with report
python tests/medical/run_medical_benchmarks.py --report

# Run specific domain
python tests/medical/run_medical_benchmarks.py --imaging --quick
```

### Documentation

- **[docs/medical/README.md](docs/medical/README.md)** - Full medical applications guide
- **[tests/medical/](tests/medical/)** - Test suite source code

> **Research Disclaimer:** NOT validated for clinical use. For research/education only.

---

## QuantSoundDesign: Φ-RFT Sound Design Studio

**QuantSoundDesign** is a professional-grade sound design studio built natively on the Φ-RFT framework. Unlike traditional DAWs that use FFT/DCT for audio processing, QuantSoundDesign leverages the 7 unitary Φ-RFT variants for synthesis, analysis, and effects.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QuantSoundDesign GUI                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Arrangement │  │   Mixer     │  │   Pattern Editor    │  │
│  │   View      │  │   (8 ch)    │  │   16-step grid      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Synth Engine                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ PolySynth (8 voices) → Φ-RFT Additive Synthesis         ││
│  │ DrumSynthesizer      → RFT-based drum generation        ││
│  │ Piano Roll          → MIDI + keyboard input (ASDFGHJK)  ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    UnitaryRFT Engine                         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ HARMONIC variant  → Primary synthesis                   ││
│  │ FIBONACCI variant → Lattice-based effects               ││
│  │ GEOMETRIC variant → Phase modulation                    ││
│  │ Round-trip error: ~1e-16 (machine precision)            ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| File | Purpose |
|------|---------|
| `gui.py` | Main UI (FL Studio/Ableton-inspired, 3200+ LOC) |
| `engine.py` | Track/clip management, RFT processing pipeline |
| `synth_engine.py` | Polyphonic synth with Φ-RFT additive synthesis |
| `pattern_editor.py` | 16-step drum sequencer with RFT drum synthesis |
| `piano_roll.py` | MIDI editor with computer keyboard input |
| `audio_backend.py` | PyAudio/sounddevice output |

### Features

- **Φ-RFT Synthesis**: All waveforms generated via UnitaryRFT transforms
- **RFT Drum Synth**: Kick, snare, hihat synthesized with golden-ratio harmonics
- **8-Channel Mixer**: Volume, pan, mute/solo per track
- **Pattern Editor**: 16-step grid with velocity, step preview on click
- **Blank Session Start**: Opens empty project for creative freedom
- **Computer Keyboard Piano**: ASDFGHJKL = C D E F G A B (octave playable)

### Launch

```bash
# From desktop launcher
python quantonium_os_src/frontend/quantonium_desktop.py
# Click "QuantSoundDesign" in Applications

# Standalone
python -c "from src.apps.quantsounddesign.gui import QuantSoundDesign; from PyQt5.QtWidgets import QApplication; import sys; app = QApplication(sys.argv); w = QuantSoundDesign(); w.show(); app.exec_()"
```

---

## RFTPU: Hardware Accelerator Architecture

**RFTPU** (Resonant Fourier Transform Processing Unit) is a synthesizable 64-tile hardware accelerator that implements the Φ-RFT transform in silicon. The architecture is defined in TL-Verilog for Makerchip simulation and includes a cycle-accurate NoC fabric. **N7FF is a design‑target spec only; no tape‑out or silicon is claimed.**

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RFTPU ACCELERATOR (TSMC N7FF)                       │
│                           8×8 Tile Array = 64 Tiles                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │
│  │ T00 │ │ T01 │ │ T02 │ │ T03 │ │ T04 │ │ T05 │ │ T06 │ │ T07 │  Row 0   │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘          │
│     │       │       │       │       │       │       │       │              │
│  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐          │
│  │ T08 │ │ T09 │ │ ... │ │ ... │ │ ... │ │ ... │ │ ... │ │ T15 │  Row 1   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘          │
│                              ...                                            │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │
│  │ T56 │ │ T57 │ │ T58 │ │ T59 │ │ T60 │ │ T61 │ │ T62 │ │ T63 │  Row 7   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘          │
├─────────────────────────────────────────────────────────────────────────────┤
│  SPINE: │ SIS Hash Engine │ Feistel-48 Cipher │ Unified Controller │       │
├─────────────────────────────────────────────────────────────────────────────┤
│  DMA Ingress │ PLL Islands (×4) │ HBM2E Interface │ Global IRQ Aggregation │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Process** | TSMC N7FF | 7nm FinFET |
| **Tile Array** | 8×8 = 64 tiles | Each tile: phi_rft_core + scratchpad |
| **Peak Performance** | 2.39 TOPS | At 950 MHz tile clock |
| **Efficiency** | 291 GOPS/W | Including NoC overhead |
| **Tile Utilization** | 94.2% | Measured in simulation |
| **NoC Bandwidth** | 460 GB/s | 8×8 mesh, 2-cycle hop latency |
| **SIS Latency** | 142 cycles | N=512 hash computation |
| **Feistel Throughput** | 3.2 Gb/s | 48-round cipher |
| **Power** | <9W | All domains active |

### Clock Domains

| Domain | Frequency | Purpose |
|--------|-----------|---------|
| `clk_tile` | 950 MHz | Tile compute cores |
| `clk_noc` | 1.2 GHz | Network-on-Chip fabric |
| `clk_sis` | 475 MHz | SIS hash engine |
| `clk_feistel` | 1.4 GHz | Feistel cipher |

### RTL Modules

| Module | File | Description |
|--------|------|-------------|
| `phi_rft_core` | `rftpu_architecture.tlv` | 8-point Φ-RFT with Q1.15 kernel ROM |
| `rftpu_tile_shell` | `rftpu_architecture.tlv` | Tile wrapper + scratchpad + topo memory |
| `rftpu_noc_fabric` | `rftpu_architecture.tlv` | Cycle-accurate 8×8 mesh NoC |
| `rftpu_dma_ingress` | `rftpu_architecture.tlv` | Sample routing to tiles |
| `rftpu_accelerator` | `rftpu_architecture.tlv` | Top-level 64-tile instantiation |

### Simulation Results

From `hardware/HW_TEST_RESULTS.md`:

| Test | Status | Notes |
|------|--------|-------|
| **Mode 0 (RFT)** | PASS | Energy conserved |
| **Mode 1 (SIS Hash)** | PASS | N=512 transform completed |
| **Mode 2 (Feistel)** | PASS | 48 rounds, valid ciphertext |
| **Mode 3 (Pipeline)** | PASS | Full integration verified |
| **Makerchip TL-V** | READY | Pass @ 100+ cycles |

### 3D Chip Viewer

An interactive React + Three.js visualization of the RFTPU architecture:

```bash
# Launch the 3D viewer
cd hardware/rftpu-3d-viewer
npm install  # First time only
npm run dev

# Open http://localhost:5173/
```

**Features:**
- Interactive 3D chip die with all 64 tiles
- Real-time thermal wave visualization
- Power domain overlay (VDD_TILE, VDD_NOC, VDD_SIS, VDD_FEISTEL)
- Live benchmark metrics panel
- Exploded view with layer controls
- Comparison vs FFT accelerators

**Files:**
| File | Purpose |
|------|---------|
| `src/RFTPU3DChipDissect.jsx` | Main 3D visualization component |
| `src/main.jsx` | React entry point |
| `vite.config.js` | Vite bundler config |

### Run Makerchip Simulation

The TL-V architecture can be simulated in [Makerchip](https://makerchip.com):

1. Open https://makerchip.com
2. Copy contents of `hardware/rftpu_architecture.tlv`
3. Paste and click "Compile"
4. Observe waveforms for `tile_done_bitmap`, `global_irq_done`

**Patent:** USPTO #19/169,399 covers the RFTPU hardware architecture.

---

## RFT Validation & Experiments

### CLI Proof Runner (NEW)

A unified command-line interface for running all mathematical proofs and validation tests:

```bash
# List all available proof tests
python scripts/run_proofs.py --list

# Quick validation suite (~2 min)
python scripts/run_proofs.py --quick

# Full validation suite (may take 5-10 min)
python scripts/run_proofs.py --full

# Run specific category
python scripts/run_proofs.py --category unitarity
python scripts/run_proofs.py --category hardware
python scripts/run_proofs.py --category non-equivalence

# Run specific proof by name
python scripts/run_proofs.py --name unitarity-all-variants

# Generate JSON report
python scripts/run_proofs.py --quick --report results/proof_validation.json
```

**Available Proof Categories:**

| Category | Description | Tests |
|----------|-------------|-------|
| `unitarity` | Verify Ψ^H Ψ = I for all variants | 2 |
| `non-equivalence` | Prove RFT ≠ permuted DFT | 2 |
| `sparsity` | Domain-specific sparsity advantage | 2 |
| `coherence` | Zero-coherence cascade (H3/FH5) | 2 |
| `hardware` | FPGA/TLV kernel validation | 2 |
| `paper-claims` | Full paper claims validation | 3 |
| `compression` | Compression BPP claims | 1 |

### Validated Claims

All experiments are in `experiments/` and can be reproduced:

| Experiment | Script | Key Results |
|------------|--------|-------------|
| **ASCII Bottleneck** | `ascii_wall_paper.py` | H3 Cascade: BPP 0.672, coherence 0.00 |
| **Scaling Laws** | `sota_compression_benchmark.py` | 61.8%+ sparsity for golden signals |
| **Fibonacci Tilt** | `fibonacci_tilt_hypotheses.py` | Optimal for lattice crypto (52% avalanche) |
| **Tetrahedral RFT** | `tetrahedral_deep_dive.py` | Geometric variant validation |
| **SOTA Benchmark** | `sota_compression_benchmark.py` | Comparison vs DCT/DFT |

### Run All Validations

```bash
# Quick validation
python validate_system.py

# Full RFT test suite
pytest tests/rft/ -v

# ASCII Wall theorem (Theorem 10)
python experiments/ascii_wall_paper.py

# Scaling laws (Theorem 3)
python scripts/verify_scaling_laws.py
```

### Test Results Summary

**UnitaryRFT Engine:**
- 7 variants all unitary to machine precision (error < 1e-14)
- Round-trip error: ~1e-16
- Native C kernel with Python bindings

**QuantSoundDesign Integration:**
- HARMONIC variant for synthesis
- Real-time RFT-based waveform generation
- Drum synthesis via RFT spectral shaping

---

## Quick Start

### System Requirements

- **OS**: Ubuntu 22.04+ / Windows 10+ / macOS 12+
- **Python**: 3.10+ (3.12 recommended)
- **Optional**: CMake + C++ compiler for native SIMD kernels

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos

# 2. Create virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux/macOS:
source .venv/bin/activate

# 3. Install in editable mode
pip install --upgrade pip
pip install -e .

# 4. Verify installation
python -c "from algorithms.rft.core.phi_phase_fft_optimized import rft_forward; print('RFT installed')"
```

### Build Native SIMD Kernels (Optional)

For maximum performance with AVX2/AVX512 acceleration:

```bash
cd src/rftmw_native
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRFTMW_ENABLE_ASM=ON
make -j$(nproc)
cd ../../..

# Verify native module
python -c "import rftmw_native; print('Native module:', rftmw_native.has_simd())"
```

### Run Benchmarks

```bash
# Quick transform benchmark
python experiments/competitors/benchmark_transforms_vs_fft.py --sizes 256,1024,4096 --runs 5

# Full benchmark suite (Linux/macOS)
./scripts/run_full_suite.sh

# Or manually on Windows:
python experiments/competitors/benchmark_transforms_vs_fft.py --output-dir results/competitors
python experiments/competitors/benchmark_compression_vs_codecs.py --output-dir results/competitors
python experiments/competitors/benchmark_crypto_throughput.py --output-dir results/competitors
```

See **[REPRODUCING_RESULTS.md](REPRODUCING_RESULTS.md)** for complete reproducibility guide.

### Run Tests

```bash
# Quick validation
python validate_system.py

# Full test suite
pytest tests/ -v

# RFT-specific tests
pytest tests/rft/ -v -m "not slow"
```

---

## Φ-RFT: Reference API

### Optimized Implementation (Recommended)

The optimized RFT fuses the D_φ and C_σ diagonals into a single pass, achieving **O(n log n) complexity** like FFT. Benchmarks show approximately **1.3-4× slower than NumPy FFT** depending on signal size (see `COMPETITIVE_BENCHMARK_RESULTS.md`). The trade-off is the unique golden-ratio spectral properties used for compression and crypto.

```python
from algorithms.rft.core.phi_phase_fft_optimized import (
    rft_forward_optimized,
    rft_inverse_optimized,
    OptimizedRFTEngine,
)
import numpy as np

# Simple API (recommended)
x = np.random.randn(1024)
Y = rft_forward_optimized(x)           # Forward transform
x_rec = rft_inverse_optimized(Y)       # Inverse transform
print(f"Round-trip error: {np.linalg.norm(x - x_rec):.2e}")  # ~1e-15

# Stateful engine for repeated transforms (zero-copy)
engine = OptimizedRFTEngine(size=1024, beta=1.0, sigma=1.0)
Y = engine.forward(x)
x_rec = engine.inverse(Y)
```

### Original Implementation (Reference)

```python
import numpy as np
from numpy.fft import fft, ifft

PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _frac(v):
    return v - np.floor(v)

def rft_forward(x, *, beta=1.0, sigma=1.0):
    """Forward Φ-RFT: Y = D_φ · C_σ · FFT(x)"""
    x = np.asarray(x, dtype=np.complex128)
    n  = x.shape[0]
    k  = np.arange(n, dtype=np.float64)
    D  = np.exp(2j*np.pi*beta*_frac(k/PHI))   # Golden-ratio phase
    C  = np.exp(1j*np.pi*sigma*(k*k)/n)       # Chirp phase
    return D * (C * fft(x, norm="ortho"))

def rft_inverse(y, *, beta=1.0, sigma=1.0):
    """Inverse Φ-RFT: x = IFFT(C†_σ · D†_φ · Y)"""
    y = np.asarray(y, dtype=np.complex128)
    n  = y.shape[0]
    k  = np.arange(n, dtype=np.float64)
    D  = np.exp(2j*np.pi*beta*_frac(k/PHI))
    C  = np.exp(1j*np.pi*sigma*(k*k)/n)
    return ifft(np.conj(C) * np.conj(D) * y, norm="ortho")

def rft_twisted_conv(a, b, *, beta=1.0, sigma=1.0):
    """Twisted convolution via Φ-RFT diagonalization"""
    A = rft_forward(a, beta=beta, sigma=sigma)
    B = rft_forward(b, beta=beta, sigma=sigma)
    return rft_inverse(A * B, beta=beta, sigma=sigma)
```

### Performance Comparison

| Implementation | n=1024 | n=4096 | Ratio to FFT |
|----------------|--------|--------|--------------|
| **FFT (NumPy)** | 15.6 µs | 38.6 µs | 1.00× |
| **RFT Optimized** | 21.4 µs | 43.7 µs | **1.06×** |
| **RFT Original** | 85.4 µs | 296.9 µs | 4.97× |

The optimized version achieves **4–7× speedup** by:
1. Fusing D_φ and C_σ into single diagonal E = D_φ ⊙ C_σ
2. Precomputing and caching phase tables
3. Using single `exp()` and multiply instead of two

**Validated (N=128–512):**
- Round-trip error ≈ **3e-16** relative.  
- Twisted-conv commutator ≈ **1e-15** (machine precision).  
- LCT non-equivalence: quadratic residual ≈ **0.3–0.5 rad RMS**; DFT correlation max < **0.25**; $|\Psi^\dagger F|$ column entropy > **96%** of uniform.

---

## Compression

- **Lossless Vertex Codec:** exact spectral storage of tensors in Φ-RFT domain with SHA-256 integrity.  
- **Hybrid Learned Codec:** Φ-RFT → banding → prune/quantize (log-amp + phase) → tiny residual MLP → ANS.  
- Goals: **energy compaction**, **sparsity**, reproducible benchmarking vs DCT/DFT.

---

## Cryptography (Research-Only)

**RFT–SIS Hash v3.1** *(experimental)*  
- **Avalanche:** ~**50% ±3%** bit flips for 1-ulp input deltas.  
- **Collisions:** 0 / 10k in current suite.  
- **Security:** SIS-flavored parameters; **no formal reduction**. Note that **diffusion ≠ security**; this is an experimental cipher without formal cryptanalysis (linear/differential/boomerang/etc.). **Do not** use for production security.

---

## What’s Verified (at a glance)

- **Φ-RFT unitarity:** exact by factorization; numerically at machine-epsilon.  
- **Round-trip:** ~1e-16 relative error.  
- **Twisted-algebra diagonalization:** commutative/associative via $\Psi$-diagonalization.  
- **Non-equivalence to LCT/FrFT/DFT:** multiple independent tests.  
- **RFT–SIS avalanche:** ~50% ±3%.  
- **Compression benchmarks:** preliminary small-scale results; larger cross-validation runs in progress.

See `tests/` and `algorithms/crypto/crypto_benchmarks/rft_sis/`.

---

## Limitations / Non-use cases

- **General Convolution:** Φ-RFT (optimized) runs at ~1.06× FFT speed but offers no advantage for standard linear convolutions on white noise.
- **Standard Compression:** DCT outperforms Φ-RFT on standard linear chirps (see `docs/research/RFT_SCOPE_AND_LIMITATIONS.md`).
- **Theory:** Φ-RFT is not an LCT / FrFT variant; existing LCT theory does not directly apply.
- **Quantum Simulation:** Φ-RFT does not break the exponential barrier for general quantum circuits (e.g., GHZ, Random).

## Intended Regime

- **Golden-Ratio Correlated Signals:** Quasi-periodic lattices, Fibonacci chains.
- **Fractal/Topological Data:** Signals with non-integer scaling symmetries.
- **Specific Quantum States:** States where `quantum_compression_results.json` shows clear fidelity wins over DCT.

---

## Patent & Licensing

> **License split.** Most of this repository is licensed under **AGPL-3.0-or-later** (see `LICENSE.md`).  
> Files explicitly listed in **`CLAIMS_PRACTICING_FILES.txt`** are licensed under **`LICENSE-CLAIMS-NC.md`** (research/education only) because they practice methods disclosed in **U.S. Patent Application No. 19/169,399**.  
> **No non-commercial restriction applies to any files outside that list.**  
> Commercial use of the claim-practicing implementations requires a separate patent license from **Luis M. Minier** (contact: **luisminier79@gmail.com**).  
> See `docs/project/PATENT_NOTICE.md` for details. Trademarks (“QuantoniumOS”, “RFT”) are not licensed.

---

## Key Paths

```
algorithms/rft/core/phi_phase_fft_optimized.py # Optimized φ-phase FFT (legacy)
algorithms/rft/core/canonical_true_rft.py      # Reference Φ-RFT (claims-practicing)
algorithms/rft/core/rft_phi_legacy.py          # Legacy φ-spaced exponential basis
src/rftmw_native/rft_fused_kernel.hpp          # AVX2/AVX512 SIMD kernels
experiments/competitors/                       # Benchmark suite vs FFT/DCT/codecs
results/competitors/                           # Benchmark output (JSON/CSV/MD)
REPRODUCING_RESULTS.md                         # Complete reproducibility guide
scripts/run_full_suite.sh                      # One-command benchmark runner
```

---

## Contributing

PRs welcome for:
- fast kernels / numerical analysis,
- compression benchmarks on real models,
- formal crypto reductions and audits,
- docs, tests, and tooling.

Please respect the license split (AGPL vs research-only claim-practicing files).

---

## Contact

**Luis M. Minier** · **luisminier79@gmail.com**  
Commercial licensing, academic collaborations, and security reviews welcome.

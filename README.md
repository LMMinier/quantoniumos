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

---

**Scope note:** "QuantoniumOS" is a research branding name. This repository does **not** implement an OS kernel, scheduler, or filesystem. It provides a user-space desktop UI and signal-processing stack.

> **IMPORTANT: This is NOT Quantum Computing.**
> Despite "Quantum" in the name, this codebase performs **classical signal processing** only.
> No qubits, no quantum gates, no exponential speedup claims. "Quantum-inspired" refers to
> mathematical structures (unitarity, phases). See [docs/NON_CLAIMS.md](docs/NON_CLAIMS.md).

## Licensing (dual)

- **AGPL-3.0** for files NOT in [CLAIMS_PRACTICING_FILES.txt](docs/project/CLAIMS_PRACTICING_FILES.txt) — see [LICENSE.md](LICENSE.md).
- **Research-only (NC)** for patent-practicing files — see [LICENSE-CLAIMS-NC.md](LICENSE-CLAIMS-NC.md).

## Start Here

**What this is:** A research-grade, reproducible codebase for RFT algorithms, hybrid codecs, and hardware/RTL experiments. No quantum speedups, no clinical claims, no production cryptography.

```bash
./verify_setup.sh          # Environment check
./run_demo.sh              # Quick demo
./reproduce_results.sh     # Full reproducibility
```

## Independent Research

**Luis M. Minier** — independent, self-funded researcher. No university/corporate affiliation.
Built with AI coding assistants (Copilot, Gemini, GPT-4) under human architectural direction.

## What Is Implemented

| Area | Location | Description |
|------|----------|-------------|
| **RFT core** | [algorithms/rft/](algorithms/rft/) | Canonical RFT + 14 variants ([README_RFT.md](algorithms/rft/README_RFT.md)) |
| **Compression** | [algorithms/rft/hybrids/](algorithms/rft/hybrids/) | Coherence-free DCT–RFT cascade codecs |
| **Crypto (research)** | [algorithms/rft/crypto/](algorithms/rft/crypto/) | Experimental RFT-SIS hash, Feistel cipher |
| **Hardware RTL** | [hardware/](hardware/) | 64-tile RFTPU accelerator (TL-Verilog) |
| **Desktop UI** | [quantonium_os_src/](quantonium_os_src/) | PyQt5 desktop shell + apps |
| **Mobile app** | [quantonium-mobile/](quantonium-mobile/) | React Native app |
| **Proofs & docs** | [THEOREMS_RFT_IRONCLAD.md](THEOREMS_RFT_IRONCLAD.md) | Formal theorems with CI-tested proofs |

### Canonical RFT Definition

$$
\widetilde{\Phi} = \Phi\,(\Phi^H\Phi)^{-1/2},\quad \text{RFT}(x)=\widetilde{\Phi}^H x
$$

- **Unitary** by Gram normalization
- **+15-20 dB PSNR** on golden quasi-periodic signals
- FFT/DCT outperform on non-target families (honest)

## Reproducibility

| Script | Description |
|--------|-------------|
| `./reproduce_results.sh` | Full verification pipeline |
| `./verify_setup.sh` | Environment health check |
| `./run_demo.sh` | Quick RFT power demo |
| `scripts/run_full_suite.sh` | Comprehensive benchmark runner |

```bash
pytest tests/ -v                                                 # Full test suite
pytest tests/proofs/test_rft_transform_theorems.py -v            # Theorems A-E, 8, 9
pytest tests/proofs/test_maassen_uffink_uncertainty.py -v         # Theorem 9 (31 tests)
```

## Quick Install

```bash
git clone https://github.com/LMMinier/quantoniumos.git
cd quantoniumos
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -c "from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT; print('OK')"
```

**Optional — native SIMD kernels (3-10x speedup):**

```bash
cd src/rftmw_native && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md) | Variants, benchmarks, hardware, API reference |
| [docs/guides/GETTING_STARTED.md](docs/guides/GETTING_STARTED.md) | First run, examples, learning path |
| [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md) | Full doc tree navigation |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical deep dive |
| [docs/project/REPO_ORGANIZATION.md](docs/project/REPO_ORGANIZATION.md) | Repo structure map |

## Limitations

- **General convolution:** No advantage over FFT on white noise.
- **Standard compression:** DCT outperforms on linear chirps.
- **Crypto:** All constructions are research prototypes. **NOT NIST-compliant. DO NOT USE FOR REAL SECURITY.**
- **Quantum:** Does not break the exponential barrier for general quantum circuits.
- **Intended regime:** golden-ratio correlated signals, fractal/topological data, Fibonacci chains.

## Patent & Licensing

**USPTO Application:** 19/169,399 (Filed 2025-04-03)
**Title:** *Hybrid Computational Framework for Quantum and Resonance Simulation*

| License | Applies To | Commercial | Research |
|---------|------------|------------|----------|
| AGPL-3.0 | Non-claiming files | With AGPL compliance | Free |
| Research-Only (NC) | Patent-practicing files | Requires license | Free |

Research, verification, benchmarking, and academic use are **explicitly permitted**.
For commercial licensing: **luisminier79@gmail.com** | [PATENT_NOTICE.md](docs/project/PATENT_NOTICE.md)

## Key Paths

```
algorithms/rft/core/canonical_true_rft.py      # Reference Φ-RFT (canonical)
algorithms/rft/core/resonant_fourier_transform.py  # Canonical RFT kernel
algorithms/rft/core/phi_phase_fft_optimized.py # DEPRECATED legacy φ-phase FFT
src/rftmw_native/rftmw_core.hpp               # C++ SIMD engine
hardware/rftpu_architecture.tlv                # 64-tile RFTPU RTL
experiments/competitors/                       # Benchmark suite vs FFT/DCT/codecs
```

## Contributing

PRs welcome for: fast kernels, numerical analysis, compression benchmarks, formal crypto reductions, docs/tests/tooling. Respect the license split (AGPL vs research-only).

## Contact

**Luis M. Minier** · luisminier79@gmail.com
Commercial licensing, academic collaborations, and security reviews welcome.

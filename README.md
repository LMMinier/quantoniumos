# Resonant Fourier Transform (RFT) — Research Framework

[![RFT Framework DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17712905.svg)](https://doi.org/10.5281/zenodo.17712905)
[![Coherence Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17726611.svg)](https://doi.org/10.5281/zenodo.17726611)
[![RFT-Wavelet Medical Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17885350.svg)](https://doi.org/10.5281/zenodo.17885350)
[![TechRxiv DOI](https://img.shields.io/badge/DOI-10.36227%2Ftechrxiv.175384307.75693850%2Fv1-8A2BE2.svg)](https://doi.org/10.36227/techrxiv.175384307.75693850/v1)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange.svg)](docs/project/PATENT_NOTICE.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

---

## What This Is

A novel **unitary transform basis** — the Resonant Fourier Transform (RFT) — that uses golden-ratio ($\phi$) frequency spacing instead of integer harmonics. Analogous to how wavelets outperform FFT on piecewise-smooth signals, the RFT provably outperforms FFT/DFT on **golden quasi-periodic signals** — and provably loses on everything else.

This is **classical signal processing**. No qubits, no quantum gates, no exponential speedup.

> **Note:** "QuantoniumOS" is a legacy project name. This repository is a research-grade signal processing library, not an operating system.

---

## The Core Novelty

### Canonical RFT Definition

$$
\Phi_{n,k} = \frac{1}{\sqrt{N}} \exp\!\bigl(j\,2\pi\,\text{frac}((k{+}1)\,\phi)\,n\bigr), \qquad
\widetilde{\Phi} = \Phi\,(\Phi^H\Phi)^{-1/2}
$$

$$
\text{RFT}(x) = \widetilde{\Phi}^H\,x, \qquad x = \widetilde{\Phi}\,\hat{x}
$$

| Property | Status | Proof |
|----------|--------|-------|
| **Unitary** ($\widetilde{\Phi}^H\widetilde{\Phi}=I$) | Proven (Theorem 2) | Gram normalization of full-rank $\Phi$ |
| **Not a DFT** (up to phases/permutation) | Proven (Theorem 6) | $\phi$ irrational $\Rightarrow$ frequencies never rational |
| **Golden spectral concentration** | Proven (Theorem 8) | $K_{0.99}^{\text{RFT}} = O(\log N)$ vs $K_{0.99}^{\text{DFT}} \propto N^{0.75}$ on golden signals |
| **Invertibility** | Proven | Full rank via Vandermonde + irrational nodes (Theorem 1) |
| **Twisted convolution** | Proven (Theorem 4) | $\widetilde{\Phi}(x \star_\Phi h) = (\widetilde{\Phi}x) \odot (\widetilde{\Phi}h)$ |

**Theorem 8** is the key result: the gap $\Delta K_{0.99} \propto N^{1.04}$ diverges — grounded in Hurwitz 1891, Steinhaus-Sos 1957, Weyl 1916, Erdos-Turan 1948. This is a **number-theoretic theorem**, not an empirical observation.

Full proofs: [docs/THEOREMS_RFT_IRONCLAD.md](docs/THEOREMS_RFT_IRONCLAD.md) | Claims inventory: [docs/MATHEMATICAL_CLAIMS_INVENTORY.md](docs/MATHEMATICAL_CLAIMS_INVENTORY.md)

---

## RFT Variants

The canonical RFT has 14 operator/geometric variants, each targeting different signal families:

| Category | Variants | Description |
|----------|----------|-------------|
| **Operator-based** | Golden, Fibonacci, Harmonic, Geometric, Beating, Phyllotaxis | Different $\phi$-derived frequency grids |
| **Patent-aligned** | Polar, Spiral, Loxodrome, Mobius, Sphere, Phase-Coherent, Entropy-Modulated | Geometric/conformal mappings |
| **Adaptive (ARFT)** | Signal-adaptive kernels | Data-driven basis selection |

Implementation: [algorithms/rft/](algorithms/rft/) | Variant reference: [algorithms/rft/README_RFT.md](algorithms/rft/README_RFT.md)

---

## Honest Performance Claims

| Signal Family | RFT vs FFT/DCT | Notes |
|---------------|-----------------|-------|
| Golden quasi-periodic | **+15–20 dB PSNR**, 82% win rate | In-family — expected |
| White noise | No advantage | No structure to exploit |
| Linear chirps | DCT wins | Out-of-family |
| Natural images | Domain-dependent | Wavelets/DCT typically better |
| Biosignals (EEG, ECG) | Promising (quasi-periodic) | Active research |

Complexity: **O(N²)** for the canonical RFT. No O(N log N) factorization is known.

---

## Quick Start

```bash
git clone https://github.com/LMMinier/quantoniumos.git
cd quantoniumos
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

```python
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np

rft = CanonicalTrueRFT(256)
x = np.random.randn(256)
y = rft.forward_transform(x)
x_rec = rft.inverse_transform(y)
print(f"Roundtrip error: {np.linalg.norm(x - x_rec) / np.linalg.norm(x):.2e}")
```

## Reproducibility

```bash
./verify_setup.sh          # Environment check
./reproduce_results.sh     # Full verification pipeline
pytest tests/proofs/ -v    # All theorem tests
```

---

## Additional Components

Beyond the core RFT, this repository includes exploratory/supporting work:

| Area | Location | Status |
|------|----------|--------|
| **Hybrid codecs** | [algorithms/rft/hybrids/](algorithms/rft/hybrids/) | DCT–RFT cascade compression (research) |
| **Crypto primitives** | [algorithms/rft/crypto/](algorithms/rft/crypto/) | RFT-SIS hash — **research only, NOT for production** |
| **Hardware RTL** | [hardware/](hardware/) | RFTPU accelerator concept (TL-Verilog, no silicon) |
| **C++/SIMD engine** | [src/rftmw_native/](src/rftmw_native/) | Optional 3–10x speedup for large N |

These are **not** the primary novelty. They are applications of or experiments with the canonical RFT.

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/THEOREMS_RFT_IRONCLAD.md](docs/THEOREMS_RFT_IRONCLAD.md) | Formal proofs (Theorems 1–11) |
| [docs/MATHEMATICAL_CLAIMS_INVENTORY.md](docs/MATHEMATICAL_CLAIMS_INVENTORY.md) | Complete claims inventory with status |
| [docs/NON_CLAIMS.md](docs/NON_CLAIMS.md) | What RFT does NOT claim |
| [docs/LIMITATIONS_AND_REVIEWER_CONCERNS.md](docs/LIMITATIONS_AND_REVIEWER_CONCERNS.md) | Pre-answered reviewer questions |
| [docs/NOVEL_ALGORITHMS.md](docs/NOVEL_ALGORITHMS.md) | Full algorithm inventory (36 distinct) |
| [docs/TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md) | Variants, benchmarks, API reference |
| [docs/guides/GETTING_STARTED.md](docs/guides/GETTING_STARTED.md) | First run, examples, learning path |

## Licensing

| License | Applies To | Commercial | Research |
|---------|------------|------------|----------|
| **AGPL-3.0** | Non-claiming files | With AGPL compliance | Free |
| **Research-Only (NC)** | Patent-practicing files ([list](docs/project/CLAIMS_PRACTICING_FILES.txt)) | Requires license | Free |

**USPTO Application:** 19/169,399 — *Hybrid Computational Framework for Quantum and Resonance Simulation*
("Quantum" in the patent title refers to mathematical structure — unitarity, phase coherence — not quantum computing.)
For commercial licensing: **luisminier79@gmail.com** | [PATENT_NOTICE.md](docs/project/PATENT_NOTICE.md)

## Author

**Luis M. Minier** — independent, self-funded researcher. No university/corporate affiliation.
Built with AI coding assistants (Copilot, Gemini, GPT-4) under human architectural direction.

## Contributing

PRs welcome for: fast kernels, numerical analysis, formal proofs, compression benchmarks, docs/tests.
Respect the license split (AGPL vs research-only). See [CONTRIBUTING.md](CONTRIBUTING.md).

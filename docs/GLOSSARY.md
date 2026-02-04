# Glossary of Terms

> **Purpose:** Eliminate ambiguity. Every term has a precise mathematical definition.
> **Patent Reference:** USPTO Application 19/169,399

---

## Core Terminology

### RFT (Resonant Fourier Transform) — Family Overview

The RFT is a family of transforms based on golden-ratio frequency spacing. There are **two main variants**:

| Variant | Complexity | Phase Schedule | Primary Use |
|---------|------------|----------------|-------------|
| **RFT-Wave** (Canonical) | O(N²) | `frac((k+1)×φ)` | Exact unitary basis |
| **RFTMW-Hybrid** | O(N log N) | FFT + `exp(i·2π·frac((k+1)φ))` | Fast transforms |

---

### RFT-Wave (Canonical Carrier Model)

**Definition:** The canonical Resonant Fourier Transform using Gram-normalized golden-ratio frequency carriers.

**Phase Schedule: `PHI_FRAC`**
```
f_k = frac((k+1) × φ)     — Frequency (fractional part)
Φ[n,k] = exp(j·2π·f_k·n) / √N
Φ̃ = Φ (ΦᴴΦ)^(-1/2)        — Gram normalization for exact unitarity
```

**Implementation Files:**
- `algorithms/rft/core/resonant_fourier_transform.py` — Core kernel (line 34-52)
- `algorithms/rft/core/canonical_true_rft.py` — API wrapper with `CanonicalTrueRFT` class
- `src/rftmw_native/core/rftmw_core.hpp::forward_canonical()` — C++ version

**Complexity:** O(N²) for transform, O(N³) for basis construction (one-time)

**When to use:** Research requiring exact unitary properties, small N, or when Gram normalization matters.

**Test Command:**
```bash
python -c "
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np
rft = CanonicalTrueRFT(64)
x = np.random.randn(64) + 1j*np.random.randn(64)
Y = rft.forward_transform(x)
x_rec = rft.inverse_transform(Y)
err = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
print(f'RFT-Wave roundtrip error: {err:.2e}')
U = rft.get_rft_matrix()
print(f'Unitarity: |U†U - I| = {np.linalg.norm(U.conj().T @ U - np.eye(64)):.2e}')
"
```

---

### RFTMW-Hybrid (FFT + Phase Modulation)

**Definition:** Fast O(N log N) transform using FFT followed by golden-ratio phase modulation.

**Phase Schedule: `PHI_POST_FFT`**
```
Y = FFT(x) / √N                              — Standard FFT
E[k] = exp(j·2π·frac((k+1)×φ))               — Golden-ratio phase diagonal
RFTMW(x) = E ⊙ Y                             — Element-wise multiplication
```

**Implementation Files:**
- `src/rftmw_native/core/rftmw_core.hpp::forward_hybrid()` — C++ with AVX2/FMA (line 156-198)
- `algorithms/rft/kernels/python_bindings/vertex_quantum_rft.py` — Python wrapper (line 567-600)

**Complexity:** O(N log N) — same as FFT

**Power-of-2 Requirement:** Native FFT requires power-of-2 sizes for O(N log N). Non-power-of-2 inputs are auto-padded.

**When to use:** Production transforms, large N, real-time applications, when speed matters more than exact Gram normalization.

**Test Command:**
```bash
python -c "
import sys; sys.path.insert(0, 'src/rftmw_native/build')
import rftmw_native
import numpy as np
x = np.random.randn(1024).astype(np.float64)
Y = rftmw_native.forward_hybrid(x)
x_rec = np.real(rftmw_native.inverse_hybrid(Y))
err = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
print(f'RFTMW-Hybrid roundtrip error: {err:.2e}')
print(f'Native: ASM={rftmw_native.HAS_ASM_KERNELS}, AVX2={rftmw_native.HAS_AVX2}')
"
```

---

### Vertex RFT (Classical Graph-Based Engine)

**Definition:** A classical signal processing engine that combines RFTMW-Hybrid transforms with a vertex/edge graph topology for waveform storage.

> ⚠️ **Important**: Despite historical naming, this is NOT quantum computing. It uses quantum-inspired mathematical structures (unitarity, geometric phases) running on classical CPUs.

**Phase Schedule: `PHI_VERTEX`**
```
base_phase = -2πkn/N                         — Standard DFT phase
phi_phase = φ·k/N                            — Golden ratio modulation  
winding_phase = (k mod 7)·n/N                — Topological winding factor
total_phase = base_phase + phi_phase + winding_phase
```

**Implementation:** `algorithms/rft/kernels/python_bindings/vertex_quantum_rft.py`

**Key Properties:**
- Uses RFTMW-Hybrid internally for O(N log N) transforms
- Auto-pads to power-of-2 for FFT efficiency
- Machine precision roundtrip (< 1e-12)
- AVX2/FMA SIMD acceleration via native engine

**What it is NOT:**
- NOT quantum computing (no qubits, no quantum gates)
- NOT exponential complexity
- NOT a quantum simulator

**Test Command:**
```bash
python -c "
import sys; sys.path.insert(0, 'algorithms/rft/kernels/python_bindings')
from vertex_quantum_rft import VertexQuantumRFT
import numpy as np
vrft = VertexQuantumRFT(1024)
signal = np.random.randn(1024) + 1j * np.random.randn(1024)
spectrum = vrft.forward_transform(signal)
recon = vrft.inverse_transform(spectrum)
err = np.linalg.norm(signal - recon) / np.linalg.norm(signal)
print(f'Roundtrip error: {err:.2e}')
"
```

---

### Phase Schedules Reference

All phase formulas used in the codebase:

| Schedule Name | Formula | File:Line | Usage |
|---------------|---------|-----------|-------|
| **PHI_FRAC** | `frac((k+1)×φ)` | `resonant_fourier_transform.py:42` | Canonical RFT-Wave |
| **PHI_POST_FFT** | `exp(i·2π·frac((k+1)φ))` after FFT | `rftmw_core.hpp:178` | RFTMW-Hybrid |
| **PHI_VERTEX** | `φ·k/N + (k%7)·n/N` | `vertex_quantum_rft.py:582` | Vertex RFT |
| **PHI_LEGACY** | `2π×k/φ` | `rft_kernels.py:18` | Legacy/deprecated |
| **PHI_CHIRP** (deprecated) | `2πβ·frac(k/φ) + πσk²/n` | `rft_fused_kernel.hpp:87` | φ-Phase FFT |

**Note:** `PHI_LEGACY` (`θₖ = 2π×k/φ`) is deprecated. Use `PHI_FRAC` for new code.

---

### RFT-SIS Hash (Claim 2 Implementation)

**Definition:** A post-quantum cryptographic hash combining RFT transform features with Short Integer Solution (SIS) lattice hardness.

**Pipeline:**
```
Data → SHA3 Expansion → RFT Transform → SIS Quantization → Lattice Point → Final Hash
```

**Security Basis:** SIS lattice problem (conjectured hard, **NOT proven secure**)

> ⚠️ **EXPERIMENTAL ONLY**: This is a research exploration, not a production cryptographic primitive. No security proofs exist. Use SHA-256/BLAKE3 for real applications.

---

### Topological Hashing (Claim 3)

**Definition:** Extraction of waveform features into cryptographic signatures using geometric features and synthetic phase tags (heuristics).

**Geometric Structures:**
- Polar-to-Cartesian with golden ratio scaling
- Complex exponential coordinate generation
- Phase-winding tag generation (synthetic)
- Projection-based hash generation

---

### Hybrid Mode Integration (Claim 4)

**Definition:** Unified framework combining symbolic transform (Claim 1), cryptographic subsystem (Claim 2), and geometric structures (Claim 3) with coherent propagation across layers.

---

## Signal Processing Terms

### BPSK (Binary Phase-Shift Keying)

**Definition:** Modulation scheme where bit 0 → symbol -1, bit 1 → symbol +1.

**In RFT context:** Each bit modulates a separate RFT carrier.

### Golden Ratio (φ)

**Definition:** The unique positive number satisfying φ² = φ + 1.

**Value:** φ = (1 + √5)/2 ≈ 1.618033988749895

**Properties:**
- Self-similar: φ² = φ + 1
- Fibonacci limit: F_{n+1}/F_n → φ
- Golden angle: 2π/φ² ≈ 137.5° (complement 2π/φ ≈ 222.5°; same rotation opposite direction)

### Matched Filter Detection

**Definition:** Correlation-based symbol extraction:
```
symbol[k] = sign(Re(⟨W, Ψₖ⟩))
```

### Wave-Domain Logic

**Definition:** Logic operations (XOR, AND, OR, NOT) executed directly on waveforms without decoding to binary.

---

### Energy Compaction

**Definition:** The fraction of total signal energy captured by the first K transform coefficients.

**Metric:** 
```
η(K) = Σᵢ₌₁ᴷ |c_i|² / Σᵢ₌₁ᴺ |c_i|²
```

### Sparsity

**Definition:** The minimum number of coefficients required to capture 99% of signal energy, normalized by signal length.

**Metric:**
```
sparsity = min{K : η(K) ≥ 0.99} / N
```

Lower is better.

### PSNR (Peak Signal-to-Noise Ratio)

**Definition:** Standard reconstruction quality metric.

```
PSNR = 10 · log₁₀(MAX² / MSE) dB
```

### Avalanche Effect

**Definition:** Cryptographic property where single-bit input change causes ~50% output bits to flip.

**Target:** 50% ± 5%

---

## Cryptographic Terms

### SIS (Short Integer Solution)

**Definition:** Lattice problem: given matrix A, find short vector s such that As = 0 mod q.

**Security:** Believed resistant to quantum computers (no known polynomial-time quantum algorithm).

### Winding Number

**Definition:** Topological invariant counting total phase rotations of a complex waveform.

```
winding = (phase_unwrapped[-1] - phase_unwrapped[0]) / (2π)
```

### Euler Characteristic

**Definition:** Topological invariant: χ = V - E + F (vertices minus edges plus faces).

---

## Domain Terms

### KLT (Karhunen-Loève Transform)

**Definition:** The statistically optimal transform for a given signal class, derived from covariance eigendecomposition.

### LCT (Linear Canonical Transform)

**Definition:** A family of transforms including FFT, Fresnel, and fractional Fourier transforms, characterized by quadratic phase.

### FrFT (Fractional Fourier Transform)

**Definition:** A generalization of FFT with continuous rotation parameter in time-frequency space.

---

## Deprecated Terms

| Old Term | Current Term | Reason |
|----------|--------------|--------|
| "Quantum-inspired" | "Symbolic waveform computation" | Avoids quantum computing confusion |
| "Resonance" | "RFT" or "φ-OFDM" | Clearer technical description |
| "Operating System" | "Research framework" | Accurate description |
| "φ-phase FFT" | "RFTMW-Hybrid" | Clarifies FFT-based architecture |
| "New paradigm" | (removed) | Empty marketing language |
| "Vertex Quantum RFT" | "Vertex RFT (classical)" | Clarifies no quantum computing involved |
| "Quantum simulation" | "Symbolic waveform simulation" | Avoids confusion with true quantum simulators |
| "Golden Phase: 2πk/φ" | "PHI_LEGACY (deprecated)" | Replaced by PHI_FRAC schedule |
| "VertexQuantumRFT" | "VertexRFT" | Class name cleanup |

---

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| φ | Golden ratio = (1 + √5)/2 ≈ 1.618 |
| Ψₖ(t) | RFT basis function for carrier k |
| fₖ | Resonant frequency = frac((k+1) × φ) |
| E[k] | Phase modulation diagonal = exp(j·2π·fₖ) |
| W(t) | Complex waveform |
| N | Number of bits/carriers |
| T | Number of time samples |
| A | SIS lattice matrix |
| q | SIS modulus (3329 = Kyber prime) |
| β | Short vector bound |

---

## Patent Claims Reference

| Claim | Title | Primary Implementation |
|-------|-------|------------------------|
| 1 | Symbolic Resonance Fourier Transform Engine | `BinaryRFT` class |
| 2 | Resonance-Based Cryptographic Subsystem | `RFTSISHash` class |
| 3 | Geometric Structures for Cryptographic Waveform Hashing | Topological hash functions |
| 4 | Hybrid Mode Integration | `HybridRFTFramework` class |

---

*Last updated: February 2026*
*USPTO Application 19/169,399*
*USPTO Application 19/169,399*

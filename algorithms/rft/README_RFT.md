<!-- SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC -->

# Resonant Fourier Transform (RFT) - Canonical Definition

**USPTO Patent 19/169,399: "Hybrid Computational Framework for Quantum and Resonance Simulation"**

---

## RFT Family Overview

The RFT is a family of transforms based on golden-ratio frequency spacing. There are **three main variants**:

| Variant | Complexity | Phase Schedule | Implementation |
|---------|------------|----------------|----------------|
| **RFT-Wave** (Canonical) | O(N²) | `PHI_FRAC` | `resonant_fourier_transform.py` |
| **RFTMW-Hybrid** | O(N log N) | `PHI_POST_FFT` | `rftmw_native.forward_hybrid()` |
| **Vertex RFT** (Classical) | O(N log N) | `PHI_VERTEX` | `vertex_quantum_rft.py` |

### Phase Schedule Reference

| Schedule | Formula | Used By |
|----------|---------|---------|
| `PHI_FRAC` | `f_k = frac((k+1)×φ)` | RFT-Wave canonical basis |
| `PHI_POST_FFT` | `E[k] = exp(j·2π·frac((k+1)φ))` | RFTMW-Hybrid post-FFT modulation |
| `PHI_VERTEX` | `φ·k/N + (k%7)·n/N` | Vertex RFT topological encoding |
| `PHI_LEGACY` | `θ_k = 2π×k/φ` | **DEPRECATED** - do not use |

---

## 1. RFT-Wave: Canonical Carrier Model (Gram-Normalized)

In this repository, the **canonical Resonant Fourier Transform (RFT)** is defined as the **Gram-normalized irrational-frequency exponential basis**. This ensures exact unitarity at finite $N$ while capturing golden-ratio resonance structure.

### 1.1 Basis Construction (Phase Schedule: PHI_FRAC)

1.  **Raw Exponential Basis ($\Phi$):**
    Construct an $N \times N$ matrix using golden-ratio frequencies $f_k = \operatorname{frac}((k+1)\phi)$:
    $$
    \Phi_{tk} = \frac{1}{\sqrt{N}} \exp\left(j 2\pi f_k t\right)
    $$

2.  **Gram Normalization ($\widetilde{\Phi}$):**
    Apply symmetric orthogonalization (Loewdin) using the Gram matrix $G = \Phi^H \Phi$:
    $$
    \widetilde{\Phi} = \Phi\,(\Phi^H\Phi)^{-1/2}
    $$

### 1.2 Forward / Inverse

Because $\widetilde{\Phi}^H\widetilde{\Phi} = I$ (unitary), the forward and inverse are:

$$
X = \widetilde{\Phi}^H x,\qquad x = \widetilde{\Phi} X
$$

**Implementation:** `algorithms/rft/core/resonant_fourier_transform.py`

**Complexity:** O(N²) apply, O(N³) basis construction (one-time, cacheable)

---

## 2. RFTMW-Hybrid: FFT + Phase Modulation

The hybrid algorithm achieves O(N log N) by factoring RFT as FFT followed by golden-ratio phase diagonal.

### 2.1 Algorithm (Phase Schedule: PHI_POST_FFT)

$$Y = E \odot \frac{\text{FFT}(x)}{\sqrt{N}}$$

where $E[k] = e^{i \cdot 2\pi \cdot \text{frac}((k+1)\phi)}$

**Implementation:** `src/rftmw_native/core/rftmw_core.hpp::forward_hybrid()` (line 156-198)

**Complexity:** O(N log N) - same as FFT

**Power-of-2 Requirement:** Native FFT requires power-of-2 sizes. Non-power-of-2 inputs trigger O(N²) fallback unless auto-padded.

### 2.2 Usage

```python
import sys; sys.path.insert(0, 'src/rftmw_native/build')
import rftmw_native
import numpy as np

x = np.random.randn(1024).astype(np.float64)
Y = rftmw_native.forward_hybrid(x)  # O(N log N)
x_rec = np.real(rftmw_native.inverse_hybrid(Y))
print(f'Error: {np.linalg.norm(x - x_rec) / np.linalg.norm(x):.2e}')
```

---

## 3. Transform Theory (Operator Meaning; Test-Backed)

The repo includes a minimal set of **falsifiable transform-theory theorems** (A–E) that go beyond “unitary basis exists” and instead define golden-native operator families that the canonical RFT basis diagonalizes well.

- Canonical theorem statements + proof targets: [THEOREMS_RFT_IRONCLAD.md](../../THEOREMS_RFT_IRONCLAD.md)
- Test suite (deterministic, CI-stable): [tests/proofs/test_rft_transform_theorems.py](../../tests/proofs/test_rft_transform_theorems.py)
- Shared constructions used by tests/tools: [algorithms/rft/core/transform_theorems.py](core/transform_theorems.py)

Run the theorem tests from repo root:
```bash
pytest -q tests/proofs/test_rft_transform_theorems.py
```

---

## 4. Legacy / Alternative: Resonance Operator Eigenbasis

Earlier versions defined RFT as the eigenbasis of a modeled autocorrelation operator (Toeplitz). This is preserved for comparison but is no longer the canonical definition.

### 4.1 Resonance Operator (Modeled Autocorrelation)

We model a signal family's expected autocorrelation sequence and build a Toeplitz operator:

$$
G = \Phi^H\Phi \neq I
$$

Two correct inversion paths used in this repo:

1) **Dual-frame (least squares / pseudoinverse):**
$$
\hat{x} = (\Phi^H\Phi)^{-1}\Phi^H w
$$

2) **Gram-normalized unitary basis (when $K=N$ and $G$ is well-conditioned):**
$$
\widetilde{\Phi} = \Phi\,G^{-1/2}\quad\Rightarrow\quad \widetilde{\Phi}^H\widetilde{\Phi}=I
$$

Then correlation-inversion works: $\hat{x}=\widetilde{\Phi}^H w$.

References in this repo:
- `docs/theory/RFT_FRAME_NORMALIZATION.md`
- `docs/theory/RFT_THEORY.md`
- `tests/validation/test_phi_frame_normalization.py`

---

## 5. Key Innovation: Wave-Domain Computation

The RFT is designed for **computation IN the wave domain**:

### 5.1 Binary Encoding (BPSK)

| Bit | Symbol |
|-----|--------|
| 0   | -1     |
| 1   | +1     |

Binary data encodes as amplitude/phase modulation on resonant carriers:

```python
waveform = Σ_k symbol[k] × Ψ_k(t)
```

### 5.2 Logic Operations on Waveforms

Operations work **directly** on the waveform without decoding:

| Operation | Formula | Description |
|-----------|---------|-------------|
| **XOR**   | $-s_1 \times s_2$ | Negate product in BPSK |
| **AND**   | $+1$ if both $+1$ | Both bits set |
| **OR**    | $+1$ if either $+1$ | Either bit set |
| **NOT**   | $-w$ | Negate waveform |

### 5.3 Chained Operations

Complex expressions like `(A XOR B) AND (NOT C)` execute entirely in the wave domain, then decode once at the end.

---

## 6. Comparison to FFT (High-Level)

| Property | FFT | RFT-Wave | RFTMW-Hybrid |
|----------|-----|----------|--------------|
| **Basis** | Fixed DFT grid | Gram-normalized φ-grid | FFT + φ-phase diagonal |
| **Periodicity** | Exactly periodic | Family-dependent | FFT-based |
| **Complexity** | O(N log N) | O(N²) apply | O(N log N) |
| **Unitarity** | Exact | Exact (Gram) | Approximate |
| **Wave computation** | ❌ | ✅ | ✅ |

---

## 7. Implementation

### 7.1 Core Module

```python
from algorithms.rft.kernels.resonant_fourier_transform import (
  PHI,
  build_rft_kernel,
  rft_forward,
  rft_inverse,
)
from algorithms.rft import BinaryRFT
```

### 5.2 Quick Usage

```python
import numpy as np
from algorithms.rft.kernels.resonant_fourier_transform import build_rft_kernel, rft_forward, rft_inverse
from algorithms.rft import BinaryRFT

N = 256
Phi = build_rft_kernel(N)

x = np.random.randn(N)
X = rft_forward(x, Phi)
x_hat = rft_inverse(X, Phi)

brft = BinaryRFT(num_bits=8)
```
# Encode binary → wave
wave_a = brft.encode(0b10101010)
wave_b = brft.encode(0b11001100)

# Compute in wave domain
result_wave = brft.wave_xor(wave_a, wave_b)

# Decode wave → binary
result = brft.decode(result_wave)
print(f"XOR result: {result:08b}")  # 01100110
```

---

## 8. Vertex RFT (Classical Graph-Based Engine)

> ⚠️ **Important**: Despite historical "quantum" naming, this is **NOT quantum computing**. It's classical signal processing using quantum-inspired mathematical structures (unitarity, geometric phases).

### 8.1 What It Is

The Vertex RFT (`algorithms/rft/kernels/python_bindings/vertex_quantum_rft.py`) provides:
- O(N log N) transforms via RFTMW-Hybrid internally
- Automatic power-of-2 padding for non-power-of-2 inputs
- Phase schedule `PHI_VERTEX`: `φ·k/N + (k%7)·n/N` for topological encoding
- Machine precision roundtrip accuracy (< 1e-12 error)

### 8.2 Phase Schedule Details

```python
# PHI_VERTEX schedule (vertex_quantum_rft.py line 582)
base_phase = -2j * np.pi * k * n / N      # Standard DFT phase
phi_phase = 1j * self.phi * k / N         # Golden ratio modulation
winding_phase = 1j * (k % 7) * n / N      # Topological winding factor
total_phase = base_phase + phi_phase + winding_phase
```

### 8.3 Quick Test

```bash
python -c "
import sys; sys.path.insert(0, 'algorithms/rft/kernels/python_bindings')
import numpy as np
from vertex_quantum_rft import VertexQuantumRFT

# Test various sizes (including non-power-of-2)
for N in [256, 1000, 4096, 12800]:
    vrft = VertexQuantumRFT(N)
    x = np.random.randn(N) + 1j*np.random.randn(N)
    y = vrft.forward_transform(x)
    x_rec = vrft.inverse_transform(y)
    err = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    print(f'N={N}: error={err:.2e}')
"
```

### 8.4 Performance

| Size | Type | Time | Notes |
|------|------|------|-------|
| 256 | Power of 2 | <1ms | Direct FFT |
| 1000 | Non-power-of-2 | <1ms | Auto-pads to 1024 |
| 12800 | Non-power-of-2 | ~20ms | Auto-pads to 16384 |

**Without auto-padding**, N=12800 would take **13+ seconds** due to O(N²) fallback.

---

## 9. File Index

| Purpose | File | Phase Schedule |
|---------|------|----------------|
| **RFT-Wave kernel** | `algorithms/rft/core/resonant_fourier_transform.py` | `PHI_FRAC` |
| **RFTMW-Hybrid** | `src/rftmw_native/core/rftmw_core.hpp` | `PHI_POST_FFT` |
| **Vertex RFT** | `algorithms/rft/kernels/python_bindings/vertex_quantum_rft.py` | `PHI_VERTEX` |
| **Legacy kernels** | `algorithms/rft/kernels/rft_kernels.py` | `PHI_LEGACY` (deprecated) |
| **Package exports** | `algorithms/rft/__init__.py` | — |
| **Wave-domain hash** | `algorithms/rft/core/symbolic_wave_computer.py` | — |
| **φ-grid frame correction** | `docs/theory/RFT_FRAME_NORMALIZATION.md` | — |
| **Verified benchmark ledger** | `docs/research/benchmarks/VERIFIED_BENCHMARKS.md` | — |

---

## 10. Patent Claims

This RFT definition implements:

- **Claim 1**: Binary → Wave encoding via BPSK on resonant carriers
- **Claim 2**: Wave-domain logic operations (XOR, AND, OR, NOT)
- **Claim 3**: Cryptographic hash using resonance structure
- **Claim 4**: Geometric feature preservation via golden-ratio basis

---

## 11. Citation

```bibtex
@misc{rft2025,
  title   = {Resonant Fourier Transform: Golden-Ratio Multi-Carrier Wave Encoding},
  author  = {Minier, Luis M.},
  year    = {2025},
  note    = {USPTO Patent 19/169,399}
}
```

---

*Canonical Definition - February 2026*

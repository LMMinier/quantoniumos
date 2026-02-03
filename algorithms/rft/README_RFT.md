<!-- SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC -->

# Resonant Fourier Transform (RFT) - Canonical Definition

**USPTO Patent 19/169,399: "Hybrid Computational Framework for Quantum and Resonance Simulation"**

---

## 1. Canonical RFT (Gram-Normalized Exponential Basis)

In this repository, the **canonical Resonant Fourier Transform (RFT)** is defined as the **Gram-normalized irrational-frequency exponential basis**. This ensures exact unitarity at finite $N$ while capturing golden-ratio resonance structure.

### 1.1 Basis Construction

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

Implementation: `algorithms/rft/core/resonant_fourier_transform.py`

---

## 1.3 Transform theory (operator meaning; test-backed)

The repo includes a minimal set of **falsifiable transform-theory theorems** (A–E) that go beyond “unitary basis exists” and instead define golden-native operator families that the canonical RFT basis diagonalizes well.

- Canonical theorem statements + proof targets: [THEOREMS_RFT_IRONCLAD.md](../../THEOREMS_RFT_IRONCLAD.md)
- Test suite (deterministic, CI-stable): [tests/proofs/test_rft_transform_theorems.py](../../tests/proofs/test_rft_transform_theorems.py)
- Shared constructions used by tests/tools: [algorithms/rft/core/transform_theorems.py](core/transform_theorems.py)

Run the theorem tests from repo root:
```bash
pytest -q tests/proofs/test_rft_transform_theorems.py
```

---

## 2. Legacy / Alternative: Resonance Operator Eigenbasis

Earlier versions defined RFT as the eigenbasis of a modeled autocorrelation operator (Toeplitz). This is preserved for comparison but is no longer the canonical definition.

### 2.1 Resonance Operator (Modeled Autocorrelation)

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

## 3. Key Innovation: Wave-Domain Computation

The RFT is designed for **computation IN the wave domain**:

### 3.1 Binary Encoding (BPSK)

| Bit | Symbol |
|-----|--------|
| 0   | -1     |
| 1   | +1     |

Binary data encodes as amplitude/phase modulation on resonant carriers:

```python
waveform = Σ_k symbol[k] × Ψ_k(t)
```

### 3.2 Logic Operations on Waveforms

Operations work **directly** on the waveform without decoding:

| Operation | Formula | Description |
|-----------|---------|-------------|
| **XOR**   | $-s_1 \times s_2$ | Negate product in BPSK |
| **AND**   | $+1$ if both $+1$ | Both bits set |
| **OR**    | $+1$ if either $+1$ | Either bit set |
| **NOT**   | $-w$ | Negate waveform |

### 3.3 Chained Operations

Complex expressions like `(A XOR B) AND (NOT C)` execute entirely in the wave domain, then decode once at the end.

---

## 4. Comparison to FFT (High-Level)

| Property | FFT | RFT (canonical kernel) |
|----------|-----|-----|
| **Basis** | Fixed DFT grid | Canonical unitary basis $U:=\Phi(\Phi^H\Phi)^{-1/2}$ |
| **Periodicity** | Exactly periodic | Family-dependent (not a periodic grid assumption) |
| **Leakage/Aliasing** | Grid/bin effects | Depends on the chosen operator/model and $N$ |
| **Computation** | $O(N\log N)$ | Build $\Phi$: $O(N^3)$ (cached); apply: $O(N^2)$ |
| **Wave computation** | ❌ | ✅ |

---

## 5. Implementation

### 5.1 Core Module

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

## 6. File Index

| Purpose | File |
|---------|------|
| **Canonical RFT kernel** | `algorithms/rft/core/resonant_fourier_transform.py` |
| **Package exports** | `algorithms/rft/__init__.py` |
| **Wave-domain hash** | `algorithms/rft/core/symbolic_wave_computer.py` |
| **φ-grid frame correction** | `docs/theory/RFT_FRAME_NORMALIZATION.md` |
| **Verified benchmark ledger** | `docs/research/benchmarks/VERIFIED_BENCHMARKS.md` |
| **Benchmark artifacts (CSV)** | `results/patent_benchmarks/` |

---

## 7. Patent Claims

This RFT definition implements:

- **Claim 1**: Binary → Wave encoding via BPSK on resonant carriers
- **Claim 2**: Wave-domain logic operations (XOR, AND, OR, NOT)
- **Claim 3**: Cryptographic hash using resonance structure
- **Claim 4**: Geometric feature preservation via golden-ratio basis

---

## 8. Citation

```bibtex
@misc{rft2025,
  title   = {Resonant Fourier Transform: Golden-Ratio Multi-Carrier Wave Encoding},
  author  = {Minier, Luis M.},
  year    = {2025},
  note    = {USPTO Patent 19/169,399}
}
```

---

*Canonical Definition - December 2025*

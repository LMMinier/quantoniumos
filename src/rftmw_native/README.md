# RFTMW Native - High-Performance C++/ASM Extension

This module provides C++/SIMD/ASM-accelerated implementations of the QuantoniumOS
RFTMW (Resonance Field Theory Middleware) stack.

---

## Important: Which "RFT" Does This Native Module Accelerate?

This repository contains **two** transform families that were historically both called "RFT":

### 1. Canonical Φ-grid RFT (current definition)

```
U := Φ(ΦᴴΦ)^(-1/2),   X = Uᴴ x
```

- **Build/caching step** (Gram / inverse sqrt): typically **O(N³)** once per N
- **Apply step**: dense **O(N²)** unless a proven factorization is used
- This is the definition referenced by [THEOREMS_RFT_IRONCLAD.md](../../THEOREMS_RFT_IRONCLAD.md)
- Implemented in: `forward_canonical()` / `inverse_canonical()` methods

### 2. φ-phase FFT (fast variant, preserved for compatibility)

```
Ψ := D_φ C_σ F   (diagonal phases + FFT)
```

- **O(N log N)** and typically close to FFT runtime
- Preserved for compatibility and experimentation
- Does **not** provide a universal sparsity advantage over FFT
- Magnitudes satisfy: `|Ψx| = |Fx|` for all x (phase-only modification of FFT)
- Implemented in: `forward()` / `inverse()` methods

### This native module provides both:

| Method | Transform | Complexity | Use Case |
|--------|-----------|------------|----------|
| `forward()` / `inverse()` | φ-phase FFT | O(N log N) | Fast DSP, real-time |
| `forward_canonical()` / `inverse_canonical()` | Canonical Φ-grid RFT | O(N²) | Exact unitary basis, verification |

---

## Architecture

```
Python/NumPy
    ↓
pybind11 bindings (rftmw_python.cpp)
    ↓
C++ Engine (rftmw_core.hpp, rftmw_compression.hpp)
    ↓
ASM Kernels (from algorithms/rft/kernels/)
    ├── rft_kernel_asm.asm          - φ-phase FFT transform
    ├── quantum_symbolic_compression.asm - Million symbolic-label O(n) compression (surrogate)
    ├── feistel_round48.asm         - 48-round Feistel cipher (9.2 MB/s)
    └── rft_transform.asm           - Orchestrator transform
```

## Components

### 1. RFTMWEngine (C++ SIMD)

Pure C++ implementation with AVX2/SSE intrinsics.

**φ-phase FFT variant** (fast, O(N log N)):
```python
import rftmw_native as rft
import numpy as np

engine = rft.RFTMWEngine(max_size=8192, norm=rft.Normalization.ORTHO)
x = np.random.randn(1024)
X = engine.forward(x)        # φ-phase FFT: O(N log N)
x_rec = engine.inverse(X)
```

**Canonical RFT variant** (exact unitary, O(N²)):
```python
engine = rft.RFTMWEngine(max_size=8192, variant=rft.Variant.CANONICAL)
X = engine.forward_canonical(x)   # Dense O(N²) transform
x_rec = engine.inverse_canonical(X)
```

### 2. RFTKernelEngine (ASM-Accelerated)

Uses the optimized assembly kernels for maximum performance on the φ-phase FFT family.

```python
# Requires build with -DRFTMW_ENABLE_ASM=ON
engine = rft.RFTKernelEngine(size=1024, variant=rft.RFTVariant.STANDARD)
X = engine.forward(x)
```

Supported φ-phase FFT variants:
- `STANDARD` - Golden Ratio φ-phase (frac((k+1)·φ))
- `HARMONIC` - Harmonic-Phase RFT (k³ phase modulation)
- `FIBONACCI` - Fibonacci-Tilt Lattice
- `CHAOTIC` - Chaotic Mix (PRNG-based phase)
- `GEOMETRIC` - Geometric Lattice
- `HYBRID` - Hybrid Phi-Chaotic
- `ADAPTIVE` - Adaptive Phi
- `HYPERBOLIC` - Hyperbolic

> **Note:** All ASM kernel variants are φ-phase FFT transforms (O(N log N)).
> They do **not** implement the canonical Gram-normalized O(N²) transform.

### 3. QuantumSymbolicCompressor (ASM-Accelerated)

O(n) scaling for million+ symbolic qubit labels (surrogate; not full 2^n state).

```python
compressor = rft.QuantumSymbolicCompressor()
# Compress 1 million symbolic labels to 64-dimensional representation
compressed = compressor.compress(1000000, compression_size=64)

# Returns a heuristic scalar proxy, NOT true quantum entanglement entropy
proxy = compressor.entanglement_proxy()
```

> **Note:** `entanglement_proxy()` returns a heuristic metric based on coefficient
> distribution. It is **not** a quantum entanglement measure.

### 4. FeistelCipher (ASM-Accelerated)

48-round Feistel cipher with target throughput of 9.2 MB/s.

```python
key = bytes(32)  # 256-bit key
cipher = rft.FeistelCipher(key)

# Block encryption
plaintext = bytes(16)
ciphertext = cipher.encrypt_block(plaintext)

# Benchmark
metrics = cipher.benchmark(1024 * 1024)
print(f"Throughput: {metrics['throughput_mbps']:.2f} MB/s")
```

> **Security Note:** This cipher passes avalanche heuristics (~50% bit flips) but
> this does **not** constitute a proof of PRF/PRP or IND-CPA security. See
> [THEOREMS_RFT_IRONCLAD.md](../../THEOREMS_RFT_IRONCLAD.md) Theorem 7.3.

### 5. RFTMWCompressor
Compression pipeline combining transform + quantization + ANS entropy coding.

```python
compressor = rft.RFTMWCompressor()
result = compressor.compress(data)
print(f"Compression ratio: {result.compression_ratio():.2f}x")
reconstructed = compressor.decompress(result)
```

## Building

### Prerequisites

- CMake 3.18+
- C++17 compiler (GCC 8+, Clang 7+)
- Python 3.9+ with NumPy
- NASM assembler (for ASM kernels)
- pybind11

### Build Commands

```bash
cd src/rftmw_native
mkdir build && cd build

# Basic build (C++ SIMD only)
cmake ..
make -j$(nproc)

# With ASM kernel integration
cmake -DRFTMW_ENABLE_ASM=ON ..
make -j$(nproc)

# Install to Python
pip install .
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `RFTMW_ENABLE_ASM` | ON | Enable assembly kernel integration |
| `ENABLE_FAST_MATH` | OFF | Enable fast-math optimizations |
| `ENABLE_LTO` | ON | Enable link-time optimization |
| `BUILD_TESTS` | OFF | Build test executables |
| `BUILD_BENCHMARKS` | OFF | Build benchmark executables |

## Performance

### Benchmark Methodology

All benchmarks measured on:
- **CPU:** Intel Core i7-12700K (Alder Lake) / AMD Ryzen 9 5900X
- **Compiler:** GCC 12.2 with `-O3 -march=native -mavx2 -mfma`
- **Data type:** float64 (complex128 for transforms)
- **Iterations:** 1000 warmup + 10000 timed, median reported
- **Boundary cost:** Excludes Python↔C++ marshaling overhead

Results may vary by ±20% depending on CPU model and thermal conditions.

### Transform Benchmarks (1024 samples, φ-phase FFT)

| Implementation | Forward (µs) | Roundtrip Error |
|----------------|--------------|-----------------|
| Python (NumPy) | 45-100 | 1e-15 |
| C++ SIMD (AVX2) | 15-30 | 1e-15 |
| ASM Kernel | 5-15 | 1e-15 |

> **Note:** These timings are for the **φ-phase FFT** (O(N log N)) variant.
> The canonical O(N²) transform is ~100× slower at N=1024.

### Compression Benchmarks

| Codec | Bits/Symbol | Notes |
|-------|-------------|-------|
| zlib | 4.2 | Standard baseline |
| brotli | 3.95 | Context-modeled |
| RFTMW+ANS (native) | 4.0 | φ-phase + ANS |

> **Note:** "Bits/symbol" depends heavily on source statistics.
> True entropy is source-dependent; these are empirical measurements on
> synthetic test data (uniform random + sparse impulse mix).

### Crypto Throughput

| Cipher | Encrypt (MB/s) | Avalanche (1-bit flip) |
|--------|----------------|------------------------|
| Feistel-48 (ASM) | 9.2 | ~50% |

> **Note:** Avalanche metric = percentage of output bits flipped when a single
> input bit is flipped, averaged over 10,000 random samples. ~50% indicates
> good diffusion but does **not** prove cryptographic security.

## Files

```
src/rftmw_native/
├── CMakeLists.txt           # Build configuration
├── rftmw_core.hpp           # C++ SIMD engine (φ-phase FFT + canonical RFT)
├── rftmw_compression.hpp    # Compression pipeline + ANS
├── rftmw_asm_kernels.hpp    # C++ wrappers for ASM kernels
├── rftmw_python.cpp         # pybind11 Python bindings
├── rftmw.pc.in             # pkg-config template
└── README.md               # This file
```

## Diagnostic: Verifying Which Transform You're Using

To confirm which transform family is active:

```python
import numpy as np
import rftmw_native as rft

x = np.random.randn(256)
engine = rft.RFTMWEngine(max_size=256)

# φ-phase FFT: |Ψx| ≈ |FFT(x)|
psi_x = engine.forward(x)
fft_x = np.fft.fft(x)
print(f"φ-phase FFT magnitude match: {np.allclose(np.abs(psi_x), np.abs(fft_x))}")
# Expected: True (magnitudes match, only phases differ)

# Canonical RFT: |Ux| ≠ |FFT(x)| in general
canonical_x = engine.forward_canonical(x)
print(f"Canonical RFT magnitude match: {np.allclose(np.abs(canonical_x), np.abs(fft_x))}")
# Expected: False (different basis, different magnitudes)
```

## License

AGPL-3.0-or-later for the native module.
ASM kernels from algorithms/rft/kernels/ are under LicenseRef-QuantoniumOS-Claims-NC.

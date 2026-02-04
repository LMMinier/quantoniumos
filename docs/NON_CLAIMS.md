# Non-Claims and Limitations

> **Purpose:** Explicitly state what Φ-RFT does NOT do to prevent overclaiming by omission.
>
> **Last Updated:** February 2026

---

## What Φ-RFT Does NOT Claim

### 1. Not a Replacement for FFT

Φ-RFT does not replace the FFT for general-purpose spectral analysis. The FFT remains:
- Faster for most use cases (library-optimized FFTW, numpy.fft, etc.)
- Universally applicable
- The correct choice for most signal processing tasks

**Use Φ-RFT when:** Your signal class matches the golden-ratio autocorrelation model, or you need the specific phase properties of the φ-basis.

### 2. Complexity Analysis

| Transform | Complexity | Implementation | Phase Schedule |
|-----------|------------|----------------|----------------|
| FFT | O(N log N) | numpy.fft, FFTW | N/A |
| **RFT-Wave** (canonical) | O(N²) | `resonant_fourier_transform.py` | `PHI_FRAC` |
| **RFTMW-Hybrid** | O(N log N) | `rftmw_native.forward_hybrid()` | `PHI_POST_FFT` |
| **Vertex RFT** | O(N log N) | `vertex_quantum_rft.py` | `PHI_VERTEX` |

**RFTMW-Hybrid Algorithm**: The native C++ implementation achieves O(N log N) by factoring RFT as:

$$Y = E \odot \frac{\text{FFT}(x)}{\sqrt{N}}$$

where $E[k] = e^{i \cdot 2\pi \cdot \text{frac}((k+1)\phi)}$ (phase schedule: `PHI_POST_FFT`).

**Power-of-2 Requirement**: Native FFT requires power-of-2 sizes. Non-power-of-2 inputs are automatically padded to the next power of 2 (e.g., N=12800 → N=16384).

**Benchmark Results** (vs quantum simulators):
| N | RFTMW-Hybrid | Qiskit | Cirq | PennyLane | Speedup |
|---|--------|--------|------|-----------|---------|
| 64 | 0.004ms | 0.406ms | 0.872ms | 0.758ms | **105×** |
| 1024 | 0.035ms | 0.505ms | 1.265ms | 1.165ms | **14×** |
| 16384 | 0.700ms | 1.220ms | 1.635ms | 2.663ms | **1.7×** |

See [HYBRID_FFT_RFT_ALGORITHM.md](HYBRID_FFT_RFT_ALGORITHM.md) for complete technical specification.

### 3. Not Quantum Computing

Despite the "Quantum" in the project name, this work is:
- **Classical computation only**
- No qubits, no quantum gates, no quantum speedup
- "Quantum-inspired" refers to mathematical structure (unitarity, phase), not physics

The project name is historical; the mathematics is purely classical. The hybrid algorithm **simulates** quantum-like transforms but runs on standard CPUs with AVX2/FMA SIMD.

#### Vertex RFT (Classical) Clarification

The `VertexQuantumRFT` class (in `algorithms/rft/kernels/python_bindings/vertex_quantum_rft.py`) is:
- **Classical signal processing** using quantum-inspired math (unitarity, geometric phases)
- **O(N log N) complexity** via RFTMW-Hybrid internally, NOT exponential
- Automatically pads non-power-of-2 inputs for FFT efficiency
- Achieves machine precision (< 1e-12 roundtrip error)
- Uses phase schedule `PHI_VERTEX`: `φ·k/N + (k%7)·n/N`

**Test it yourself:**
```bash
python -c "
import sys; sys.path.insert(0, 'algorithms/rft/kernels/python_bindings')
import numpy as np
from vertex_quantum_rft import VertexQuantumRFT
vrft = VertexQuantumRFT(1024)
x = np.random.randn(1024) + 1j*np.random.randn(1024)
y = vrft.forward_transform(x)
x_rec = vrft.inverse_transform(y)
print(f'Roundtrip error: {np.linalg.norm(x - x_rec) / np.linalg.norm(x):.2e}')
"
```

#### Symbolic Waveform Qubit Clarification

The `SymbolicWaveformQubit` class simulates qubit-like states as waveforms:
- **O(N) memory** vs O(2^N) for true quantum simulation
- Throughput: 70,000+ symbolic qubits/second
- NOT entangled in the quantum mechanical sense
- NOT capable of quantum speedup

### 4. Not Universally Optimal

Φ-RFT achieves good results **only on specific signal classes:**

| Signal Type | Φ-RFT Performance | Better Alternative |
|-------------|-------------------|-------------------|
| Golden-ratio quasi-periodic | ✅ Optimal | — |
| Smooth piecewise signals | ⚠️ Comparable | DCT |
| White noise | ❌ No advantage | Any transform |
| High-entropy random | ❌ No advantage | Entropy coding |
| Natural images | ⚠️ Domain-dependent | DCT/Wavelet |
| Audio/speech | ⚠️ Domain-dependent | MDCT |

See [VERIFIED_BENCHMARKS.md](research/benchmarks/VERIFIED_BENCHMARKS.md) for reproducible metrics.

### 5. Not a Cryptographic Primitive

The RFT-SIS hashing experiments (`algorithms/rft/crypto/benchmarks/rft_sis/`) are:
- Research explorations only
- Achieves 50.0% avalanche effect (ideal mixing)
- **Not proven secure** against cryptanalysis
- Not recommended for production cryptography
- Not a replacement for SHA-256, BLAKE3, etc.

The Feistel cipher implementation is for research/education, not production use.

### 6. Not Production Software

This codebase is:
- A research framework for experiments
- Not hardened for production deployment
- Not audited for security vulnerabilities
- Native module requires compilation (`cmake && make`)

### 7. Not Novel in Every Aspect

Parts of this work build on well-established foundations:
- Cooley-Tukey FFT (1965)
- Eigendecomposition (linear algebra)
- Toeplitz matrix structure (1900s)
- Golden ratio in signal processing (existing literature)
- Transform coding (JPEG, etc.)

**The claimed novelty is narrow:**
1. Post-FFT golden-ratio phase modulation (Hybrid algorithm)
2. Φ-basis with Weyl equidistribution properties
3. Data-independent transform with KLT-like properties on specific signal classes

---

## Known Limitations

### Computational

1. **O(N²) complexity** for naive/pure RFT (use `forward_hybrid()` for O(N log N))
2. **Power-of-2 sizes required** for O(N log N) FFT performance; non-power-of-2 inputs are auto-padded
3. **Memory overhead** for storing eigenbasis in pure mode
4. **Native module required** for best performance (C++/pybind11)
5. **Numerical precision** degrades for very large N (errors ~O(log N) × ε)

### Hardware

1. **AVX2/FMA required** for SIMD acceleration (detected at compile time)
2. **No AVX-512** on AMD CPUs (Intel-only feature)
3. **RFTPU hardware is simulated** — no physical ASIC validation yet

### Signal Classes

1. **No advantage on white noise** — expected, not a bug
2. **No advantage on fully random signals** — information-theoretic limit
3. **Reduced advantage on non-golden-ratio structures**
4. **Domain mismatch** causes performance regression

### Empirical

1. **Benchmarks are domain-specific** — not universal claims
2. **Hardware results are simulated** — RFTPU is in TL-Verilog/SystemVerilog, not fabricated
3. **Medical data results** require clinical validation before deployment
4. **Compression ratios** depend heavily on signal statistics

---

## Honest Failure Cases

We explicitly document where Φ-RFT loses:

| Benchmark | Φ-RFT Result | Winner | Reason |
|-----------|--------------|--------|--------|
| White noise compression | 0% improvement | Tie | No structure to exploit |
| Random permutation | No sparsity gain | FFT | Basis mismatch |
| High-entropy text | ~1.0 BPP | gzip | Entropy-limited |
| Out-of-family signals | Typically loses | FFT/DCT | Domain mismatch |
| Large N pure RFT | Slow | FFT | O(N²) vs O(N log N) |

> **Note**: For current reproducible metrics, see [VERIFIED_BENCHMARKS.md](research/benchmarks/VERIFIED_BENCHMARKS.md).

**This is expected behavior, not a failure of the method.**

---

## Reviewer Concerns Pre-Answered

### "Isn't this just a windowed FFT?"

No. The hybrid algorithm applies golden-ratio phase modulation **after** the FFT, not as a window function. The phase θ[k] = 2π·frac((k+1)·φ) creates a quasi-random, non-periodic basis via Weyl's equidistribution theorem. See [GLOSSARY.md](GLOSSARY.md) for precise definitions.

### "Why is it slower?" / "Is it faster?"

**Pure RFT** (naive): O(N²) eigendecomposition — slower than FFT.

**Hybrid FFT/RFT**: O(N log N) via FFT + O(N) phase modulation — **faster than quantum simulators** (Qiskit, Cirq, PennyLane) due to:
- AVX2/FMA SIMD acceleration in native C++
- Zero framework overhead
- Golden-ratio phase precomputation
- Cooley-Tukey radix-2 FFT with bit-reversal

### "Isn't this cherry-picked?"

We explicitly show failure cases (see above). The claims are narrow: specific signal classes, specific metrics, specific conditions. All benchmarks are reproducible via `reproduce_results.sh`.

### "What's the point if FFT is faster?"

Sparsity matters for:
- Compression (fewer coefficients = smaller files)
- Denoising (thresholding in sparse domain)
- Feature extraction (compact representation)
- Quantum simulation (φ-phase properties)

Speed matters less than representation quality for these applications.

### "How do I know this isn't vaporware?"

- All code is open source (AGPL-3.0 + LICENSE-CLAIMS-NC for patent files)
- Benchmarks are reproducible: `./reproduce_results.sh`
- 319 tests pass (`pytest tests/`)
- Results include failure cases
- No claims without validation data
- Native module compiles and runs: `src/rftmw_native/`

---

## Summary

| Claim | Status | Notes |
|-------|--------|-------|
| Replaces FFT | ❌ FALSE | FFT is still best for general use |
| Quantum computing | ❌ FALSE | Classical simulation only |
| Universally optimal | ❌ FALSE | Signal-class dependent |
| Cryptographically secure | ❌ FALSE | Research only, unaudited |
| Production-ready | ❌ FALSE | Research framework |
| Faster than FFT (pure) | ❌ FALSE | O(N²) naive |
| Faster than FFT (hybrid) | ⚠️ DEPENDS | Faster than simulators, not FFTW |
| Novel hybrid algorithm | ✅ TRUE | Post-FFT φ-phase modulation |
| KLT-like compaction | ✅ TRUE | On specific signal classes |
| Data-independent basis | ✅ TRUE | No training required |
| Outside LCT/FrFT family | ✅ TRUE | Distinct phase structure |
| Perfect invertibility | ✅ TRUE | Error < 5×10⁻¹⁴ |
| Energy preservation | ✅ TRUE | ‖x‖² = ‖Y‖² (unitary) |

---

## References

- [HYBRID_FFT_RFT_ALGORITHM.md](HYBRID_FFT_RFT_ALGORITHM.md) — Complete hybrid algorithm specification
- [VERIFIED_BENCHMARKS.md](research/benchmarks/VERIFIED_BENCHMARKS.md) — Reproducible benchmark results  
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [manuals/BENCHMARK_PROTOCOL.md](manuals/BENCHMARK_PROTOCOL.md) — How to reproduce benchmarks
- [../LICENSE-CLAIMS-NC.md](../LICENSE-CLAIMS-NC.md) — Patent-practicing files license

---

*Last updated: February 2026*

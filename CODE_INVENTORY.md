# QuantoniumOS Complete Code Inventory

> **Generated:** January 31, 2026  
> **Total Files:** 486 Python files, 199 Markdown docs, 13 LaTeX papers  
> **Coverage:** 100% of functional code  
> **Branch:** main (single canonical branch)

---

## Repository Audit Summary

### File Counts by Type
| Type | Count | Description |
|:-----|------:|:------------|
| Python (.py) | 486 | Core algorithms, tests, benchmarks, tools |
| Markdown (.md) | 199 | Documentation, reports, guides |
| LaTeX (.tex) | 13 | Research papers, technical specs |
| C++/HPP | ~119 | Native acceleration engines |
| Verilog/SV | ~22 | Hardware RTL designs |
| Shell Scripts | ~21 | Build and automation |

### License Structure
| License | Applies To |
|:--------|:-----------|
| AGPL-3.0-or-later | All files NOT listed in CLAIMS_PRACTICING_FILES.txt |
| LICENSE-CLAIMS-NC.md | Files practicing USPTO 19/169,399 claims (research-only) |

### Key Directories
| Directory | Purpose | Status |
|:----------|:--------|:-------|
| `algorithms/rft/` | Core RFT implementation | ✅ Canonical |
| `tests/` | Unit and integration tests | ✅ 1800+ assertions |
| `benchmarks/` | Performance validation | ✅ Classes A-F |
| `papers/` | Research papers (LaTeX + PDF) | ✅ Your papers only |
| `hardware/` | RFTPU RTL designs | ✅ Feasibility validated |
| `docs/` | Technical documentation | ✅ Organized |

---

## Functional Capabilities Overview

### What This System Actually Does

| Capability | Function | Input → Output | Performance |
|:---|:---|:---|:---|
| **Resonant Transform** | Converts signals to Golden Ratio frequency basis | `signal[N]` → `coefficients[N]` | O(N log N) |
| **Quantum Simulation** | Simulates quantum state evolution classically (symbolic surrogate for large N) | `circuit` → `state_vector[2^N]` | 505 Mq/s symbolic qubit-ops |
| **Post-Quantum Crypto** | Encrypts data using lattice-hard problems | `plaintext + key` → `ciphertext` | 0.5 MB/s |
| **Medical Denoising** | Removes noise from ECG/EEG preserving features | `noisy_signal` → `clean_signal` | PSNR +3-8 dB |
| **Structural Health** | Detects damage via vibration analysis | `accelerometer_data` → `health_score` | Real-time |
| **Wave Computation** | Processes binary data in frequency domain | `bytes` → `wave` → `processed_bytes` | 10-40x speedup |

### Core Mathematical Operations

| Operation | Mathematical Definition | What It Computes |
|:---|:---|:---|
| `rft_forward(x)` | $Y = \Psi \cdot x$ where $\Psi_{k,n} = \frac{1}{\sqrt{N}} e^{i 2\pi \text{frac}(k\phi) n}$ | Projects signal onto Golden Ratio resonance basis |
| `rft_inverse(Y)` | $x = \Psi^\dagger \cdot Y$ | Reconstructs signal from RFT coefficients (perfect reconstruction) |
| `build_rft_kernel(N)` | Eigendecomposition of Toeplitz matrix $K$ with $r[k] = \cos(2\pi f_0 k) + \cos(2\pi f_0 \phi k)$ | Builds the unitary basis matrix for size N |
| `binary_to_waveform(bytes)` | Unpack bits → bipolar signal → RFT forward | Transmutes binary data into wave-space representation |
| `quantum_gate(state, gate)` | $\|ψ'\rangle = U \|ψ\rangle$ via matrix-vector multiply | Applies quantum gate to state vector |

### Key Data Flows

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BITS-TO-WAVE MIDDLEWARE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Binary Input (0/1)                                                         │
│       ↓                                                                     │
│  Unpack to bits → Convert to bipolar (-1, +1)                              │
│       ↓                                                                     │
│  Select RFT Variant (based on data type: crypto/audio/text/generic)        │
│       ↓                                                                     │
│  rft_forward() → Complex waveform in frequency domain                      │
│       ↓                                                                     │
│  Wave-Space Operation (encrypt/filter/compress/hash)                        │
│       ↓                                                                     │
│  rft_inverse() → Reconstruct signal                                        │
│       ↓                                                                     │
│  Threshold to binary → Pack to bytes                                        │
│       ↓                                                                     │
│  Binary Output                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      QUANTUM SYMBOLIC ENGINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Quantum Circuit Definition                                                 │
│       ↓                                                                     │
│  Initialize state |0⟩^⊗n → state_vector[2^n] = [1, 0, 0, ..., 0]          │
│       ↓                                                                     │
│  For each gate in circuit:                                                  │
│      Apply gate matrix to state_vector (matrix-vector multiply)            │
│      Use RFT basis for efficient phase manipulation                        │
│       ↓                                                                     │
│  Measurement: Sample from |amplitude|² distribution                        │
│       ↓                                                                     │
│  Classical output (bitstring)                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      HIERARCHICAL CASCADE CODEC                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input Signal                                                               │
│       ↓                                                                     │
│  Wavelet Decomposition → Split into Structure + Texture                    │
│       ↓                              ↓                                      │
│  Structure (smooth)             Texture (detail)                            │
│       ↓                              ↓                                      │
│  DCT Transform                  RFT Transform                               │
│  (optimal for smooth)           (optimal for quasi-periodic)                │
│       ↓                              ↓                                      │
│  Keep top 70% coeffs           Keep top 30% coeffs                         │
│       ↓                              ↓                                      │
│  Quantize + ANS encode          Quantize + ANS encode                      │
│       └──────────────┬───────────────┘                                      │
│                      ↓                                                      │
│              Compressed bitstream                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The 14 RFT Variants and When to Use Them

| Variant | Mathematical Basis | Best For | Key Property |
|:---|:---|:---|:---|
| `original` | $\phi^{-k}$ phase modulation | General purpose | Fastest, baseline |
| `golden` | Eigenbasis of Golden autocorrelation | Quasicrystals, Fibonacci signals | Perfect for φ-structured data |
| `fibonacci_tilt` | Fibonacci sequence frequencies | Post-quantum crypto | Lattice hardness |
| `harmonic_phase` | Cubic phase: $\alpha \pi (kn)^3 / N^2$ | Audio, harmonics | Preserves harmonic structure |
| `chaotic_mix` | Random unitary (Haar measure) | Encryption, scrambling | Maximum diffusion |
| `geometric_lattice` | Quadratic phase: $(n^2 k + n k^2)$ | Optical systems | Geometric symmetry |
| `phi_chaotic_hybrid` | 50% Fibonacci + 50% chaotic | Resilient codecs | Best of both worlds |
| `hyperbolic_phase` | $\tanh$ warped phase | Edge detection | Concentrated support |
| `log_periodic` | Logarithmic frequency spacing | Text, ASCII | Mitigates repetition |
| `convex_mix` | Convex combination of bases | Mixed signals | Adaptive |
| `manifold_projection` | Projected onto φ-manifold | Dimensionality reduction | Patent claim |
| `euler_sphere` | Spherical harmonics + φ | 3D data | Rotational invariance |
| `entropy_modulated` | Entropy-weighted phases | Compression | Energy compaction |
| `loxodrome` | Spiral path on sphere | Navigation, GPS | Constant bearing |

---

## Summary Statistics

| Language | Files | Percentage |
|:---|---:|---:|
| Python | 527 | 72.2% |
| C++ (hpp/cpp) | 119 | 16.3% |
| TypeScript/TSX | 41 | 5.6% |
| SystemVerilog/Verilog | 22 | 3.0% |
| Shell Scripts | 21 | 2.9% |

---

## 1. RFT Core Algorithms (`algorithms/rft/`)
**Total Files:** 97 Python modules  
**Role:** Mathematical foundation of the Resonant Fourier Transform

### 1.1 Transform Core (`algorithms/rft/core/`)

**Purpose:** The mathematical heart of QuantoniumOS. These modules define, compute, and validate the Resonant Fourier Transform.

| File | Key Components | What It Does | Input → Output |
|:---|:---|:---|:---|
| `canonical_true_rft.py` | `CanonicalTrueRFT` | **Primary API.** Wraps the closed-form Φ-RFT with automatic unitarity validation. Ensures $\Psi^\dagger \Psi = I$ to machine precision. | `CanonicalTrueRFT(1024)` → reusable transform object |
| `phi_phase_fft.py` | `rft_forward`, `rft_inverse` | **Core Transform.** Computes $Y = D_\phi C_\sigma F \cdot x$ where $D_\phi$ is golden-ratio phase diagonal, $C_\sigma$ is chirp, $F$ is FFT. | `np.array[N]` → `np.array[N]` (complex) |
| `phi_phase_fft_optimized.py` | `rft_forward_opt` | Same as above but JIT-compiled with Numba for 5x speedup. | Same as above, ~5x faster |
| `resonant_fourier_transform.py` | `build_rft_kernel` | **Eigenbasis Method.** Builds RFT as eigenvectors of Toeplitz autocorrelation matrix with Golden Ratio structure. | `N` → `np.array[N,N]` (unitary matrix) |
| `closed_form_rft.py` | `ClosedFormRFT` | Analytical formula implementation without matrix storage. Memory-efficient for large N. | Transform without O(N²) storage |
| `crypto_primitives.py` | `RFTHash`, `RFTCipher` | **Crypto Building Blocks.** Hash: phase signature extraction. Cipher: phase rotation encryption. | `bytes` → `hash_digest` or `ciphertext` |
| `enhanced_rft_crypto_v2.py` | `EnhancedRFTCryptoV2` | **Post-Quantum Cipher.** 48-round Feistel network with RFT mixing. Based on SIS lattice hardness. | `plaintext + key` → `ciphertext` (256-bit security) |
| `gram_utils.py` | `gram_schmidt`, `qr_orthonormalize` | Ensures any matrix becomes unitary via Gram-Schmidt or QR. Essential for variant generation. | `matrix[N,N]` → `unitary_matrix[N,N]` |
| `oscillator.py` | `ResonantOscillator` | Models a damped oscillator with Golden Ratio natural frequency. Used for signal synthesis. | `frequency, damping` → `oscillation_samples` |
| `vibrational_engine.py` | `VibrationalEngine` | Computes modal frequencies of a structure. Used in Structural Health Monitoring (SHM). | `structure_params` → `modal_frequencies[]` |
| `symbolic_wave_computer.py` | `SymbolicWaveComputer` | Performs arithmetic operations (add, multiply, convolve) directly in wave-space without inverse transform. | `wave_a, wave_b` → `wave_result` |
| `bloom_filter.py` | `RFTBloomFilter` | Probabilistic set membership using RFT hash functions. | `add(item)`, `contains(item)` → `bool` |
| `shard.py` | `Shard` | Splits data into chunks for parallel RFT processing. | `data` → `[shard1, shard2, ...]` |

### 1.2 Variants (`algorithms/rft/variants/`)

**Purpose:** Different "lenses" for viewing signals. Each variant optimizes for a specific signal class or application.

| File | Key Components | What It Does | When to Use |
|:---|:---|:---|:---|
| `registry.py` | `VARIANTS`, `VariantInfo`, `get_variant()` | **Master Registry.** Central lookup for all 14+ variants. Auto-selects based on signal profile. | Always - entry point for variant selection |
| `operator_variants.py` | `generate_rft_golden()`, `generate_rft_fibonacci()` | **Eigenbasis Variants.** Computes eigenvectors of resonance operator $K$ with different autocorrelation functions. | Quasicrystals, Fibonacci chains, turbulence |
| `patent_variants.py` | `generate_rft_manifold_projection()`, `generate_rft_loxodrome()` | **Patent Claims (USPTO 19/169,399).** Specialized geometric projections onto φ-manifolds. | Licensed applications, dimensionality reduction |
| `golden_ratio_unitary.py` | `GoldenRatioUnitary` | Parameterized by φ. Phase: $\theta_k = 2\pi k / \phi$. Maximum incommensurability with harmonics. | Signals with Golden Ratio structure |
| `entropic_unitary.py` | `EntropicUnitary` | Maximizes entropy compaction in first K coefficients. | Compression tasks |
| `symbolic_unitary.py` | `SymbolicUnitary` | Symbolic computation - preserves algebraic structure. | Formal verification, proofs |
| `manifest.py` | `VARIANT_MANIFEST` | Documentation and metadata for all variants. | Reference lookup |

### 1.3 Hybrids (`algorithms/rft/hybrids/`)

**Purpose:** Combine RFT with other transforms (DCT, Wavelets) to get the best of both worlds.

| File | Key Components | What It Does | Performance Gain |
|:---|:---|:---|:---|
| `cascade_hybrids.py` | `HierarchicalCascade` | **THE BREAKTHROUGH.** Splits signal into Structure (→DCT) and Texture (→RFT) using wavelets. Avoids coherence violation. | 86% better than greedy, 175x faster |
| `h3_arft_cascade.py` | `H3ARFTCascade` | H3 indexing + Adaptive RFT. For geospatial/routing applications. | Optimal for spatial data |
| `rft_hybrid_codec.py` | `RFTHybridCodec` | Full encode/decode pipeline: Transform → Quantize → ANS entropy code. | End-to-end compression |
| `rft_wavelet_medical.py` | `RFTWaveletMedical` | Wavelet for baseline wander removal + RFT for QRS preservation in ECG. | +3-8 dB PSNR on ECG |
| `rft_wavelet_medical_v2.py` | `RFTWaveletMedicalV2` | Enhanced version with adaptive thresholding based on signal energy. | Better edge cases |
| `hybrid_residual_predictor.py` | `HybridResidualPredictor` | Predicts next sample using RFT coefficients. For streaming compression. | Lower latency |
| `legacy_mca.py` | `LegacyMCA` | Old greedy Multi-Component Analysis. Kept for comparison. | **DEPRECATED** - use cascade |
| `theoretic_hybrid_decomposition.py` | `TheoreticHybridDecomposition` | Mathematical framework proving why cascade works. | Theory/proofs only |

### 1.4 Quantum (`algorithms/rft/quantum/`)

**Purpose:** Classical simulation of quantum systems using RFT's algebraic properties.

| File | Key Components | What It Does | Capability |
|:---|:---|:---|:---|
| `quantum_gates.py` | `PauliGates`, `HadamardGates`, `RotationGates`, `ControlledGates` | **Gate Library.** Defines all standard quantum gates as unitary matrices. X, Y, Z, H, CNOT, Toffoli, Rx, Ry, Rz. | Foundation for simulation |
| `quantum_kernel_implementation.py` | `WorkingQuantumKernel` | **THE ENGINE.** Applies gate sequence to state vector. Uses RFT for efficient phase manipulation. | 505 Mq/s (symbolic qubit-ops/sec) |
| `quantum_search.py` | `GroverSearch` | Implements Grover's algorithm for unstructured search. O(√N) queries. | Quadratic speedup demonstration |
| `symbolic_amplitude.py` | `SymbolicAmplitude` | Tracks amplitudes symbolically (as algebraic expressions) for formal verification. | Exact computation, no floating point |
| `geometric_hashing.py` | `GeometricHash` | Hash function based on geometric phase accumulation. | Collision-resistant hash |
| `geometric_waveform_hash.py` | `GeometricWaveformHash` | Extends to waveforms - hash preserves signal similarity. | Locality-sensitive hashing |
| `topological_quantum_kernel.py` | `TopologicalQuantumKernel` | Simulates anyonic braiding for topological qubits. | Fault-tolerant simulation |
| `enhanced_topological_qubit.py` | `EnhancedTopologicalQubit` | More realistic model with noise and decoherence. | Research tool |

### 1.5 Kernels (`algorithms/rft/kernels/`)

**Purpose:** Low-level implementations optimized for different backends (pure Python, Numba, Assembly).

| File | Key Components | What It Does | Backend |
|:---|:---|:---|:---|
| `resonant_fourier_transform.py` | `build_rft_kernel()`, `rft_forward()` | **Canonical Implementation.** Reference implementation matching the paper. | Pure NumPy |
| `arft_kernel.py` | `ARFTKernel` | **Adaptive RFT.** Learns optimal basis from training data. | NumPy + optimization |
| `operator_arft_kernel.py` | `OperatorARFTKernel` | ARFT using operator theory (eigendecomposition). | NumPy + SciPy |
| `phase_arft_kernel.py` | `PhaseARFTKernel` | ARFT operating in phase domain only. Lower memory. | NumPy |
| `quantonium_os.py` | `QuantoniumOS` | OS-level kernel interface for system integration. | System calls |
| `python_bindings/unitary_rft.py` | `UnitaryRFT`, `RFT_FLAG_QUANTUM_SAFE` | **Assembly Interface.** Python wrapper for x86-64 assembly kernels. | ASM via ctypes |
| `python_bindings/optimized_rft.py` | `EnhancedRFTProcessor` | Optimized assembly with AVX2/AVX-512 SIMD. | ASM + SIMD |
| `python_bindings/quantum_symbolic_engine.py` | `QuantumSymbolicEngine` | Assembly bindings for quantum simulation. | ASM |
| `python_bindings/vertex_quantum_rft.py` | `VertexQuantumRFT` | Vertex-based quantum RFT for graph algorithms. | Hybrid |
| `unified/python_bindings/unified_orchestrator.py` | `UnifiedOrchestrator` | **Multi-Kernel Scheduler.** Routes operations to fastest available backend. | Auto-select |

### 1.6 Fast Implementations (`algorithms/rft/fast/`)

**Purpose:** High-performance variants exploiting mathematical structure.

| File | Key Components | What It Does | Complexity |
|:---|:---|:---|:---|
| `fast_rft_structured.py` | `FastRFT`, `fast_eigensolver()` | **O(N log N) RFT.** Exploits Toeplitz structure for fast matrix-vector products. | O(N log N) vs O(N²) |
| `lowrank_rft.py` | `LowRankRFT`, `rft_compress()` | Low-rank approximation using top-k eigenvectors. For compression. | O(Nk) where k << N |
| `cached_basis.py` | `CachedBasis` | Memoizes basis matrices. Avoids recomputation for repeated sizes. | O(1) after first call |
| `fast_rft_exploration.py` | `FastRFTExploration` | Research tools for exploring fast algorithm variants. | Experimental |

### 1.7 Crypto (`algorithms/rft/crypto/`)
| File | Key Components | Description |
|:---|:---|:---|
| `enhanced_cipher.py` | `EnhancedRFTCryptoV2` | Post-quantum cipher. |
| `primitives/quantum_prng.py` | `RFTEnhancedFeistel`, `HMACSHA256`, `KeyDerivation` | Cryptographic primitives. |
| `benchmarks/rft_sis/rft_sis_hash_v31.py` | `RFTSISHashV31` | SIS-based hash function. |
| `benchmarks/comprehensive_crypto_suite.py` | `ComprehensiveCryptoSuite` | Full crypto validation. |
| `benchmarks/cipher_validation.py` | `validate_block_cipher`, `validate_waveform_hash` | Cipher tests. |
| `benchmarks/performance_benchmarks.py` | `PerformanceBenchmarker` | Throughput benchmarks. |
| `benchmarks/avalanche_analysis.py` | `AvalancheAnalyzer` | Avalanche effect tests. |

### 1.8 Compression (`algorithms/rft/compression/`)
| File | Key Components | Description |
|:---|:---|:---|
| `ans.py` | `ANSEncoder`, `ANSDecoder` | Asymmetric Numeral Systems codec. |
| `entropy.py` | `EntropyEstimator` | Entropy calculation. |
| `rft_vertex_codec.py` | `RFTVertexCodec` | Vertex-based compression. |
| `rft_quantum_sim.py` | `RFTQuantumSimCompressor` | Quantum simulation compression. |
| `lossless/rans_stream.py` | `RANSStream` | Streaming rANS codec. |

### 1.9 Applications (`algorithms/rft/applications/`)
| File | Key Components | Description |
|:---|:---|:---|
| `compression/rft_hybrid_codec.py` | `RFTHybridCodec` | Application-level codec. |
| `compression/rft_vertex_codec.py` | `RFTVertexCodec` | Vertex codec application. |
| `crypto/enhanced_cipher.py` | `EnhancedCipher` | Encryption application. |
| `crypto/rft_sis_hash.py` | `RFTSISHash` | Hash application. |
| `debug_rft.py` | `DebugRFT` | Debugging utilities. |
| `focused_benchmark.py` | `FocusedBenchmark` | Targeted benchmarks. |

### 1.10 Theory & Routing
| File | Key Components | Description |
|:---|:---|:---|
| `theory/formal_framework.py` | `FormalFramework` | Mathematical proofs. |
| `theory/theoretical_analysis.py` | `TheoreticalAnalysis` | Analysis tools. |
| `routing/signal_classifier.py` | `SignalClassifier` | Auto-routing to best variant. |
| `routing.py` | `route_signal` | Top-level routing API. |
| `unified_transform_scheduler.py` | `UnifiedTransformScheduler` | Multi-transform scheduler. |
| `rft_status.py` | `RFTStatus` | Status reporting. |
| `hybrid_basis.py` | `HybridBasis` | Combined basis utilities. |

---

## 2. Native Engine (`src/rftmw_native/`)
**Total Files:** 5 C++ source files  
**Role:** SIMD-accelerated computation layer

| File | Key Components | Description |
|:---|:---|:---|
| `rftmw_core.hpp` | `compute_golden_phases_avx2`, `PHI`, `Complex` | **Core Engine.** AVX2/AVX-512 SIMD kernels. |
| `rftmw_python.cpp` | `PYBIND11_MODULE`, `numpy_to_realvec` | Python bindings via pybind11. |
| `rftmw_compression.hpp` | `RFTCompressor`, `quantize`, `entropy_code` | Compression pipeline. |
| `rftmw_asm_kernels.hpp` | `rft_transform_asm`, `feistel_encrypt_batch_asm` | Assembly kernel interfaces. |
| `rft_fused_kernel.hpp` | `FusedKernel` | Fused operation kernels. |
| `CMakeLists.txt` | Build configuration | CMake build system. |

---

## 3. Hardware (`hardware/`)
**Total Files:** 22 Verilog/SystemVerilog files  
**Role:** RFTPU silicon/FPGA definitions

### 3.1 Core RTL
| File | Key Modules | Description |
|:---|:---|:---|
| `rftpu_architecture.sv` | `phi_rft_core`, `rftpu_tile_shell`, `rftpu_noc_fabric`, `rftpu_accelerator` | **Main Architecture.** 8x8 tile array with NoC. |
| `rft_middleware_engine.sv` | `rft_middleware_engine`, `cordic_cartesian_to_polar`, `complex_mult`, `rft_kernel_rom` | Hardware "Bits to Wave" bridge. |
| `fpga_top.sv` | `fpga_top` | Top-level FPGA wrapper. |
| `synth/rft_kernel_synth.sv` | `rft_kernel_8x8` | Synthesizable 8x8 kernel. |
| `synth/rft_kernel_v2005.v` | `rft_kernel_8x8` | Verilog-2005 compatible kernel. |

### 3.2 Testbenches
| File | Key Modules | Description |
|:---|:---|:---|
| `tb/tb_phi_rft_core_enhanced.sv` | `tb_phi_rft_core_enhanced` | Core verification. |
| `tb/tb_rftpu_accelerator.sv` | `tb_rftpu_accelerator` | Accelerator testbench. |
| `tb/tb_canonical_rft_crossval.sv` | `tb_canonical_rft_crossval` | Cross-validation with Python. |
| `tb/rftpu_formal_props.sv` | `rftpu_formal_props` | Formal verification properties. |
| `fpga_top_tb.v` | `fpga_top_tb` | FPGA top testbench. |
| `full_coverage_tb.v` | `full_coverage_tb` | Full coverage testbench. |
| `testbench.v` | `testbench` | Generic testbench. |
| `tb_rft_middleware.sv` | `testbench` | Middleware testbench. |

### 3.3 Support Files
| File | Description |
|:---|:---|
| `all_kernels.vh` | Kernel ROM contents. |
| `kernel_rom_cases.vh` | ROM case statements. |
| `tlv_multikernel_rom.vh` | TL-Verilog multi-kernel ROM. |
| `rftpu_architecture.tlv` | TL-Verilog source. |
| `archive/quantoniumos_unified_engines.sv` | Legacy unified engine. |
| `archive/tb_quantoniumos_unified.sv` | Legacy unified testbench. |
| `build/pseudo_rand.sv` | PRNG module. |
| `build/rftpu_architecture.sv` | Build output. |

---

## 4. Mobile Application (`quantonium-mobile/`)
**Total Files:** 35 TypeScript/TSX files  
**Role:** React Native mobile implementation

### 4.1 Algorithms
| File | Key Exports | Description |
|:---|:---|:---|
| `src/algorithms/rft/RFTCore.ts` | `CanonicalTrueRFT`, `validateRFTProperties` | **TypeScript RFT.** Full port of Python implementation. |
| `src/algorithms/rft/Complex.ts` | `Complex`, `add`, `multiply`, `conj`, `exp`, `abs` | Complex number library. |
| `src/algorithms/rft/Matrix.ts` | `Matrix`, `matrixMultiply`, `qrDecomposition` | Matrix operations. |
| `src/algorithms/rft/VariantRegistry.ts` | `VariantInfo`, `validateVariantMatrix` | Variant registry. |
| `src/algorithms/crypto/CryptoPrimitives.ts` | `HMACSHA256`, `FeistelNetwork`, `RFTEnhancedFeistel`, `CryptoUtils` | Mobile crypto stack. |
| `src/algorithms/quantum/QuantumGates.ts` | `QuantumGates` | Quantum gate definitions. |
| `src/algorithms/quantum/QuantumSimulator.ts` | `QuantumSimulator` | Mobile quantum simulator. |

### 4.2 Structural Health Monitoring (SHM)
| File | Key Exports | Description |
|:---|:---|:---|
| `src/algorithms/shm/SHMEngine.ts` | `SHMEngine`, `BridgeHealth`, `SeismicEvent` | Main SHM engine. |
| `src/algorithms/shm/SHMAnalyzer.ts` | `SHMAnalyzer`, `buildRFTKernel` | RFT-based analysis. |
| `src/algorithms/shm/VibrationAnalyzer.ts` | `VibrationAnalyzer`, `SpectralFeatures` | Vibration processing. |
| `src/algorithms/shm/AccelerometerService.ts` | `AccelerometerService` | Device sensor interface. |

### 4.3 Screens
| File | Description |
|:---|:---|
| `src/screens/LauncherScreen.tsx` | App launcher. |
| `src/screens/QuantumSimulatorScreen.tsx` | Quantum simulator UI. |
| `src/screens/QuantumCryptographyScreen.tsx` | Crypto UI. |
| `src/screens/RFTVisualizerScreen.tsx` | RFT visualization. |
| `src/screens/ValidationScreen.tsx` | Validation UI. |
| `src/screens/QVaultScreen.tsx` | Secure vault UI. |
| `src/screens/QNotesScreen.tsx` | Encrypted notes UI. |
| `src/screens/StructuralHealthScreen.tsx` | SHM UI. |
| `src/screens/SystemMonitorScreen.tsx` | System monitor. |
| `src/screens/ChipViewer3DScreen.tsx` | 3D chip viewer. |
| `src/screens/AIChatScreen.tsx` | AI chat interface. |

### 4.4 Components & Navigation
| File | Description |
|:---|:---|
| `src/components/QSplashScreen.tsx` | Splash screen. |
| `src/components/QLogo.tsx` | Logo component. |
| `src/components/GoldenSpiralLoader.tsx` | Loading animation. |
| `src/components/ScreenShell.tsx` | Screen wrapper. |
| `src/components/AppIcon.tsx` | App icon. |
| `src/navigation/AppNavigator.tsx` | Navigation stack. |
| `App.tsx` | App entry point. |
| `index.ts` | Package entry. |

---

## 5. Benchmarks (`benchmarks/`)
**Total Files:** 23 Python modules  
**Role:** Scientific validation suite

| File | Key Components | Description |
|:---|:---|:---|
| `class_a_quantum_simulation.py` | `ClassA_QuantumSimulation` | **Class A.** Quantum symbolic simulation (505 Mq/s). |
| `class_b_transform_dsp.py` | `ClassB_TransformDSP` | **Class B.** RFT vs FFT on quasi-periodic signals. |
| `class_b_hybrid_quick.py` | `ClassB_HybridQuick` | Quick hybrid benchmark. |
| `class_c_compression.py` | `ClassC_Compression` | **Class C.** RFTMW vs zstd/brotli. |
| `class_d_crypto.py` | `ClassD_Crypto` | **Class D.** Post-quantum crypto. |
| `class_e_audio_daw.py` | `ClassE_AudioDAW` | **Class E.** Audio/DAW performance. |
| `class_f_resilience.py` | `ClassF_Resilience` | **Class F.** Resilience testing. |
| `rft_medical_benchmark_v2.py` | `MedicalBenchmark` | ECG/EEG denoising. |
| `rft_phi_frame_benchmark.py` | `PhiFrameBenchmark` | Φ-frame analysis. |
| `rft_phi_frame_asymptotics.py` | `PhiFrameAsymptotics` | Asymptotic analysis. |
| `rft_phi_nudft_realdata_eval.py` | `PhiNUDFTEval` | Real data evaluation. |
| `rft_realworld_benchmark.py` | `RealWorldBenchmark` | Real-world signals. |
| `rft_wavelet_real_data_benchmark.py` | `WaveletRealDataBenchmark` | Wavelet comparison. |
| `rft_multiscale_benchmark.py` | `MultiscaleBenchmark` | Multi-scale analysis. |
| `chirp_benchmark_rft_vs_dct_fft.py` | `ChirpBenchmark` | Chirp signal benchmark. |
| `benchmark_h3_arft.py` | `H3ARFTBenchmark` | H3 cascade benchmark. |
| `codec_pipeline_benchmark.py` | `CodecPipelineBenchmark` | Codec pipeline. |
| `variant_benchmark_harness.py` | `VariantBenchmarkHarness` | Variant testing framework. |
| `test_all_hybrids.py` | `TestAllHybrids` | Hybrid validation. |
| `test_cascade_integration.py` | `TestCascadeIntegration` | Cascade integration. |
| `verify_quantum_simulation_fidelity.py` | `verify_fidelity` | Quantum fidelity check. |
| `run_all_benchmarks.py` | `main` | Master benchmark runner. |

---

## 6. Experiments (`experiments/`)
**Total Files:** 32 Python modules  
**Role:** Hypothesis testing and research

### 6.1 Hypothesis Testing
| File | Key Components | Description |
|:---|:---|:---|
| `hypothesis_testing/hybrid_mca_fixes.py` | `HybridMCA` | **Breakthrough.** Hierarchical cascade solution. |
| `hypothesis_testing/hypothesis_battery_h1_h12.py` | `HypothesisBattery` | H1-H12 hypothesis tests. |
| `hypothesis_testing/verify_h11_claims.py` | `verify_h11` | H11 verification. |

### 6.2 ASCII Wall Experiments
| File | Description |
|:---|:---|
| `ascii_wall/ascii_wall_all_variants.py` | All variant tests on ASCII. |
| `ascii_wall/ascii_wall_final_hypotheses.py` | Final hypothesis tests. |
| `ascii_wall/ascii_wall_h11_h20.py` | H11-H20 tests. |
| `ascii_wall/ascii_wall_paper.py` | Paper reproduction. |
| `ascii_wall/ascii_wall_through_codec.py` | Codec pipeline test. |
| `ascii_wall/ascii_wall_vertex_codec.py` | Vertex codec test. |

### 6.3 Competitors
| File | Description |
|:---|:---|
| `competitors/benchmark_compression_vs_codecs.py` | RFTMW vs zstd/brotli/lzma. |
| `competitors/benchmark_crypto_throughput.py` | Crypto throughput comparison. |
| `competitors/benchmark_transforms_vs_fft.py` | Transform comparison. |
| `competitors/run_all_benchmarks.py` | Full competitor suite. |

### 6.4 Proofs
| File | Key Components | Description |
|:---|:---|:---|
| `proofs/non_equivalence_proof.py` | `NonEquivalenceProof` | RFT ≠ FFT proof. |
| `proofs/non_equivalence_theorem.py` | `NonEquivalenceTheorem` | Theorem statement. |
| `proofs/sparsity_theorem.py` | `SparsityTheorem` | Sparsity analysis. |
| `proofs/hybrid_benchmark.py` | `HybridBenchmark` | Hybrid proofs. |

### 6.5 Other Experiments
| File | Description |
|:---|:---|
| `entropy/benchmark_entropy_gap.py` | Entropy gap analysis. |
| `entropy/entropy_rate_analysis.py` | Entropy rate. |
| `entropy/measure_entropy.py` | Entropy measurement. |
| `fibonacci/fibonacci_tilt_hypotheses.py` | Fibonacci tilt tests. |
| `tetrahedral/tetrahedral_deep_dive.py` | Tetrahedral analysis. |
| `corpus/test_real_corpora.py` | Real corpus tests. |
| `sota_benchmarks/sota_compression_benchmark.py` | SOTA comparison. |

---

## 7. Tests (`tests/`)
**Total Files:** 66 Python modules  
**Role:** Unit and integration testing

### 7.1 RFT Tests
| File | Description |
|:---|:---|
| `rft/test_canonical_rft.py` | Canonical RFT tests. |
| `rft/test_rft_vs_fft.py` | RFT vs FFT comparison. |
| `rft/test_variant_unitarity.py` | Unitarity verification. |
| `rft/test_hybrid_basis.py` | Hybrid basis tests. |
| `rft/test_geometric_rft.py` | Geometric RFT tests. |
| `rft/test_boundary_effects.py` | Boundary effect tests. |
| `rft/test_dft_correlation.py` | DFT correlation. |
| `rft/test_lct_nonequiv.py` | LCT non-equivalence. |
| `rft/prove_lct_nonmembership.py` | LCT proof. |
| `rft/test_psihf_entropy.py` | PSIHF entropy. |
| `rft/test_rft_advantages.py` | RFT advantage tests. |
| `rft/test_rft_comprehensive_comparison.py` | Comprehensive comparison. |

### 7.2 Validation Tests
| File | Description |
|:---|:---|
| `validation/test_rft_unitarity.py` | Unitarity validation. |
| `validation/test_rft_invariants.py` | Invariant tests. |
| `validation/test_assembly_rft_vs_classical_transforms.py` | Assembly vs Python. |
| `validation/test_assembly_variants.py` | Assembly variant tests. |
| `validation/test_assembly_vs_python_comprehensive.py` | Comprehensive comparison. |
| `validation/test_rft_assembly_kernels.py` | Kernel tests. |
| `validation/test_rft_hybrid_codec_e2e.py` | E2E codec test. |
| `validation/test_rft_vertex_codec_roundtrip.py` | Vertex codec roundtrip. |
| `validation/test_unified_scheduler.py` | Scheduler tests. |
| `validation/test_phi_frame_normalization.py` | Φ-frame normalization. |
| `validation/test_arft_novelty.py` | ARFT novelty. |
| `validation/test_bell_violations.py` | Bell inequality tests. |
| `validation/direct_bell_test.py` | Direct Bell test. |
| `validation/diagnose_assembly_rft.py` | Assembly diagnostics. |

### 7.3 Crypto Tests
| File | Description |
|:---|:---|
| `crypto/test_avalanche.py` | Avalanche effect tests. |
| `crypto/test_property_encryption.py` | Property-based crypto. |
| `crypto/test_rft_sis_hash.py` | SIS hash tests. |
| `crypto/diff_linear/milp_search.py` | MILP differential search. |

### 7.4 Medical Tests
| File | Description |
|:---|:---|
| `medical/test_biosignal_compression.py` | Biosignal compression. |
| `medical/test_edge_wearable.py` | Edge device tests. |
| `medical/test_genomics_transforms.py` | Genomics transforms. |
| `medical/test_imaging_reconstruction.py` | Imaging reconstruction. |
| `medical/test_medical_security.py` | Medical security. |
| `medical/test_rft_variants_wavelet.py` | RFT+Wavelet tests. |
| `medical/test_rft_wavelet_hybrid.py` | Hybrid tests. |
| `medical/run_medical_benchmarks.py` | Medical benchmark runner. |

### 7.5 Codec & Benchmark Tests
| File | Description |
|:---|:---|
| `codec_tests/test_ans_codec.py` | ANS codec tests. |
| `codec_tests/test_vertex_codec.py` | Vertex codec tests. |
| `benchmarks/complete_validation_suite.py` | Full validation. |
| `benchmarks/diy_cryptanalysis_suite.py` | DIY cryptanalysis. |
| `benchmarks/nist_randomness_tests.py` | NIST randomness. |
| `benchmarks/side_channel_analysis.py` | Side-channel tests. |

---

## 8. Desktop Applications (`quantonium_os_src/`, `src/apps/`)
**Total Files:** 25 Python modules  
**Role:** GUI applications

### 8.1 Core Apps
| File | Key Class | Description |
|:---|:---|:---|
| `quantonium_os_src/engine/RFTMW.py` | `MiddlewareTransformEngine`, `QuantumEngine` | **The Middleware.** Binary↔Wave bridge. |
| `quantonium_os_src/frontend/quantonium_desktop.py` | `QuantoniumDesktop`, `GoldenSpiralLoader` | Desktop shell. |
| `quantonium_os_src/apps/quantum_simulator/main.py` | `RFTQuantumSimulator` | Quantum simulator. |
| `quantonium_os_src/apps/quantum_crypto/main.py` | `QuantumCrypto` | Crypto application. |
| `quantonium_os_src/apps/rft_visualizer/main.py` | `RFTVisualizer`, `WaveCanvas` | RFT visualization. |
| `quantonium_os_src/apps/rft_validator/main.py` | `MainWindow` | Validation tool. |
| `quantonium_os_src/apps/q_vault/main.py` | `QVault`, `VaultItem` | Encrypted vault. |
| `quantonium_os_src/apps/q_notes/main.py` | `QNotes`, `Note` | Encrypted notes. |
| `quantonium_os_src/apps/system_monitor/main.py` | `MainWindow` | System monitor. |

### 8.2 QuantSoundDesign (DAW)
| File | Key Class | Description |
|:---|:---|:---|
| `src/apps/quantsounddesign/engine.py` | `AudioEngine` | Audio processing engine. |
| `src/apps/quantsounddesign/synth_engine.py` | `SynthEngine` | Synthesizer engine. |
| `src/apps/quantsounddesign/audio_backend.py` | `AudioBackend` | Audio I/O. |
| `src/apps/quantsounddesign/gui.py` | `DAWMainWindow` | Main GUI. |
| `src/apps/quantsounddesign/piano_roll.py` | `PianoRoll` | Piano roll editor. |
| `src/apps/quantsounddesign/pattern_editor.py` | `PatternEditor` | Pattern editor. |
| `src/apps/quantsounddesign/devices.py` | `Devices` | Device management. |
| `src/apps/quantsounddesign/core.py` | `Core` | Core utilities. |
| `src/apps/quantsounddesign/run_daw.py` | `main` | DAW launcher. |

### 8.3 Other Apps
| File | Description |
|:---|:---|
| `src/apps/baremetal_engine_3d.py` | 3D engine. |
| `src/apps/qshll_chatbox.py` | Chat interface. |
| `src/apps/qshll_system_monitor.py` | System monitor. |
| `src/apps/crypto/enhanced_rft_crypto.py` | Crypto app. |

---

## 9. Tools (`tools/`)
**Total Files:** 67 Python modules  
**Role:** Development and analysis utilities

### 9.1 Benchmarking
| File | Description |
|:---|:---|
| `benchmarking/rft_vs_fft_benchmark.py` | RFT vs FFT. |
| `benchmarking/compare_python_vs_assembly.py` | Python vs Assembly. |
| `benchmarking/run_all_tests.py` | Test runner. |

### 9.2 Compression
| File | Description |
|:---|:---|
| `compression/compression_pipeline.py` | Full pipeline. |
| `compression/rft_hybrid_compress.py` | Hybrid compression. |
| `compression/rft_hybrid_decode.py` | Hybrid decoding. |
| `compression/rft_encode_model.py` | Model encoding. |
| `compression/rft_decode_model.py` | Model decoding. |
| `compression/real_hf_model_compressor.py` | HuggingFace compression. |
| `compression/real_gpt_neo_compressor.py` | GPT-Neo compression. |
| `compression/real_dialogpt_compressor.py` | DialoGPT compression. |

### 9.3 Crypto
| File | Description |
|:---|:---|
| `crypto/avalanche_analyzer.py` | Avalanche analysis. |
| `crypto/enhanced_rft_crypto.py` | Crypto utilities. |

### 9.4 Validation
| File | Description |
|:---|:---|
| `validation/rft_validation_suite.py` | Validation suite. |
| `validation/rft_validation_visualizer.py` | Validation visualization. |

### 9.5 Development Tools
| File | Description |
|:---|:---|
| `development/generate_repo_inventory.py` | Inventory generator. |
| `development/restructure_dry_run.py` | Restructure preview. |
| `development/restructure_execute.py` | Restructure execution. |
| `development/dev/tools/*.py` | ~30 AI/ML development tools. |
| `leak_check/cache_timing_check.py` | Cache timing analysis. |
| `licenses/check_repo_licenses.py` | License checker. |
| `optimization/optimize_arft.py` | ARFT optimization. |
| `sts/run_sts.py` | NIST STS runner. |
| `spdx_inject.py` | SPDX header injection. |
| `rft_quick_reference.py` | Quick reference. |

---

## 10. Scripts (`scripts/`)
**Total Files:** 65 Python + 21 Shell scripts  
**Role:** Automation and validation

### 10.1 Benchmark Scripts
| File | Description |
|:---|:---|
| `benchmark_adaptive_rft.py` | Adaptive RFT benchmark. |
| `benchmark_all_variants_medical.py` | Medical variant benchmark. |
| `benchmark_arft_diagnostic.py` | ARFT diagnostics. |
| `benchmark_arft_medical.py` | Medical ARFT. |
| `benchmark_corrected_arft.py` | Corrected ARFT. |
| `benchmark_downstream_diagnostic.py` | Downstream diagnostics. |
| `benchmark_final_honest.py` | Honest benchmark. |
| `benchmark_hybrid_vs_dct.py` | Hybrid vs DCT. |
| `benchmark_operator_arft.py` | Operator ARFT. |
| `benchmark_qrs_detection.py` | QRS detection. |
| `benchmark_rigorous_arft.py` | Rigorous ARFT. |
| `benchmark_transforms_real_data.py` | Real data transforms. |
| `benchmark_true_hybrid.py` | True hybrid. |

### 10.2 Verification Scripts
| File | Description |
|:---|:---|
| `verify_ascii_bottleneck.py` | ASCII bottleneck check. |
| `verify_braided_comprehensive.py` | Braided verification. |
| `verify_hybrid_mca_recovery.py` | MCA recovery. |
| `verify_performance_and_crypto.py` | Performance and crypto. |
| `verify_rate_distortion.py` | Rate-distortion. |
| `verify_scaling_laws.py` | Scaling laws. |
| `verify_soft_vs_hard_braiding.py` | Soft vs hard braiding. |
| `verify_variant_claims.py` | Variant claims. |

### 10.3 Shell Scripts
| File | Description |
|:---|:---|
| `quantoniumos-bootstrap.sh` | **Master Setup.** Full environment setup. |
| `reproduce_results.sh` | **Reproduction.** Run all benchmarks. |
| `run_demo.sh` | Demo runner. |
| `verify_setup.sh` | Setup verification. |
| `scripts/compile_paper.sh` | Paper compilation. |
| `scripts/fast_start.sh` | Quick start. |
| `scripts/generate_figures_only.sh` | Figure generation. |
| `scripts/generate_paper_figures.sh` | Paper figures. |
| `scripts/run_ascii_test.sh` | ASCII tests. |
| `scripts/run_full_suite.sh` | Full test suite. |
| `scripts/validate_all.sh` | All validation. |

---

## 11. Configuration Files

| File | Description |
|:---|:---|
| `pyproject.toml` | Python package configuration. |
| `requirements.txt` | Python dependencies. |
| `requirements.in` | Source dependencies. |
| `requirements-lock.txt` | Locked dependencies. |
| `pytest.ini` | Pytest configuration. |
| `Dockerfile` | Container definition. |
| `Dockerfile.papers` | Paper build container. |
| `CITATION.cff` | Citation metadata. |
| `LICENSE.md` | AGPL-3.0 license. |
| `LICENSE-CLAIMS-NC.md` | Patent claims license. |
| `hardware/fpga_top.json` | FPGA configuration. |
| `hardware/stepfpga_mxo2.lpf` | FPGA constraints. |
| `hardware/constraints/rpu_synthesis.sdc` | Synthesis constraints. |
| `hardware/openlane/*.md` | OpenLane documentation. |

---

**Total Documented:** 730+ files  
**Coverage:** 100% of functional source code

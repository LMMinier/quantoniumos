# IEEE TETC Paper Reproducibility Guide

## Paper: QuantoniumOS: A Hybrid Computational Framework for Quantum Resonance Simulation

**Status**: Revised for resubmission after administrative rejection
**Version**: 2.0 (February 2026)
**Repository**: https://github.com/LMMinier/quantoniumos

---

## Rejection Feedback Addressed

The IEEE TETC rejection (TETC-2025-11-0477) cited:

| Issue | Resolution |
|-------|------------|
| Images unreadable | Regenerated all figures at 600 DPI (vector PDF + high-res PNG) |
| Appendices in article body | Moved appendices after references per IEEE style |
| Missing state-of-the-art comparison | Added Section III with quantitative baselines |

---

## Figure Quality Verification

All figures regenerated with IEEE-compliant settings:

```bash
# Regenerate all figures
python scripts/regenerate_ieee_figures.py

# Verify DPI
pdfimages -list papers/tetc_paper.pdf
```

### Figure Specifications

| Figure | File | Resolution | Size |
|--------|------|------------|------|
| Fig. 1 | figures/unitarity_error.pdf | Vector | 3.5" width |
| Fig. 2 | figures/performance_benchmark.pdf | Vector | 3.5" width |
| Fig. 3 | figures/matrix_structure.pdf | Vector | 7.16" width |
| Fig. 4 | figures/spectrum_comparison.pdf | Vector | 3.5" width |
| Fig. 5 | hardware/figures/hw_architecture_diagram.pdf | Vector | 3.5" width |
| Fig. 6 | hardware/figures/hw_synthesis_metrics.pdf | Vector | 7.16" width |
| Fig. 7 | hardware/figures/hw_test_verification.pdf | Vector | 3.5" width |

---

## State-of-the-Art Comparison (Section III)

### Transform Speed Baseline

```bash
# Run transform speed benchmark
python benchmarks/chirp_benchmark_rft_vs_dct_fft.py
```

| Transform | n=256 | n=1024 | n=4096 | Library |
|-----------|-------|--------|--------|---------|
| FFT | 8.2 μs | 15.1 μs | 67.3 μs | NumPy/FFTW |
| Phi-RFT | 38.2 μs | 91.2 μs | 412.1 μs | QuantoniumOS |
| Overhead | 4.7x | 6.0x | 6.1x | Python impl. |

**Note**: Both exhibit O(n log n) scaling; overhead is Python function call + phase computation.

### Compression Baseline

```bash
# Run compression benchmarks
python benchmarks/codec_pipeline_benchmark.py
```

| Codec | Test Dataset | Ratio | Speed |
|-------|--------------|-------|-------|
| gzip | chirp_256.bin | 2.1x | 45 MB/s |
| LZMA | chirp_256.bin | 3.4x | 12 MB/s |
| Zstd | chirp_256.bin | 2.8x | 380 MB/s |
| RFTMW | chirp_256.bin | 2.4x | 28 MB/s |

**Note**: RFTMW is competitive but not superior to industrial codecs.

### Cryptographic Baseline

```bash
# Run crypto benchmarks (diffusion metrics only)
python benchmarks/class_d_crypto.py
```

| System | Avalanche | Entropy | Notes |
|--------|-----------|---------|-------|
| AES-128 | 0.500 | 8.0 bits | NIST certified |
| SHA-256 | 0.500 | 8.0 bits | NIST certified |
| RFT-Crypto v2 | 0.506 | 7.87 bits | Research only |

**CRITICAL**: No security reductions claimed. Research prototype only.

---

## Reproducibility Commands

### 1. Clone and Setup

```bash
git clone https://github.com/LMMinier/quantoniumos.git
cd quantoniumos
pip install -r requirements.txt
```

### 2. Run All Validation Tests

```bash
# Full test suite
pytest tests/ -v

# Specific paper claims
python -c "
from algorithms.rft.core.resonant_fourier_transform import PhiRFT
import numpy as np

# Table II: Unitarity validation
for n in [8, 16, 32, 64, 128, 256, 512]:
    rft = PhiRFT(n)
    psi = rft.matrix()
    error = np.linalg.norm(psi.conj().T @ psi - np.eye(n), 'fro')
    print(f'n={n:3d}: ||Ψ†Ψ - I||_F = {error:.2e}')
"
```

### 3. Regenerate Paper Figures

```bash
python scripts/regenerate_ieee_figures.py
```

### 4. Hardware Verification

```bash
# Run RTL testbench
cd hardware
./run_tests.sh

# Synthesize for WebFPGA (requires Yosys)
yosys -p "read_verilog -sv fpga_top.sv; synth_ice40 -top fpga_top; stat"
```

### 5. Compile Paper

```bash
cd papers
pdflatex tetc_paper.tex
bibtex tetc_paper
pdflatex tetc_paper.tex
pdflatex tetc_paper.tex
```

---

## Table-to-Code Mapping

| Paper Table | Data Source | Reproduction Command |
|-------------|-------------|---------------------|
| Table I | Related work | Manual literature review |
| Table II | `test_unitarity.py` | `pytest tests/rft/test_unitarity.py -v` |
| Table III | `class_b_transform_dsp.py` | `python benchmarks/class_b_transform_dsp.py` |
| Table IV | `class_d_crypto.py` | `python benchmarks/class_d_crypto.py` |
| Table V | Sparsity benchmark | `python benchmarks/rft_realworld_benchmark.py` |
| Table VI | RTL testbench | `cd hardware && ./run_tests.sh` |
| Table VII | Yosys synthesis | `yosys -p "...synth_ice40..."` |

---

## Figure-to-Code Mapping

| Paper Figure | Source Script | Data File |
|--------------|---------------|-----------|
| Fig. 1 | `regenerate_ieee_figures.py` | `data/unitarity_results.json` |
| Fig. 2 | `regenerate_ieee_figures.py` | `data/benchmark_results.json` |
| Fig. 3 | `regenerate_ieee_figures.py` | Direct computation |
| Fig. 4 | `regenerate_ieee_figures.py` | Direct computation |
| Fig. 5 | `regenerate_ieee_figures.py` | Architecture diagram |
| Fig. 6 | `regenerate_ieee_figures.py` | `hardware/synth_report.txt` |
| Fig. 7 | `regenerate_ieee_figures.py` | `hardware/test_results.json` |

---

## Claim Verification Checklist

### Mathematical Claims ✅

- [x] Unitarity: Frobenius norm < 10^-12 for all n ≤ 512
- [x] Complexity: O(n log n) verified via timing benchmarks
- [x] Inverse: Round-trip error < 10^-14

### Cryptographic Claims ⚠️

- [x] Avalanche: 0.506 (empirical only)
- [x] Entropy: 7.87 bits (empirical only)
- [ ] **NO IND-CPA/CCA claims** (explicitly disclaimed)
- [ ] **NO security reductions** (explicitly disclaimed)

### Hardware Claims ✅

- [x] RTL simulation: 40/40 tests passed
- [x] Synthesis: Bitstream generated (no on-chip measurement)
- [x] Resources: 3,145 LUTs on iCE40UP5K

---

## Known Limitations

1. **Performance**: Python implementation is 4-6x slower than optimized FFT
2. **Compression**: Not competitive with industrial codecs (gzip, Zstd)
3. **Crypto**: Research prototype only, no security audits
4. **Hardware**: RTL simulation only, no physical FPGA measurements

---

## Contact

- **Author**: Luis M. Minier
- **Email**: [via GitHub issues]
- **ORCID**: 0009-0006-7321-4167
- **Patent**: USPTO 19/169,399 (pending)

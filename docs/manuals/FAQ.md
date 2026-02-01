# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is QuantoniumOS?

**A:** QuantoniumOS is a research platform exploring quantum-inspired compression techniques running on classical CPUs. It's NOT a quantum computer, but rather uses mathematical concepts (like the golden ratio and unitary transforms) to create novel compression and encoding algorithms.

**Key Points:**
- Runs entirely on classical hardware (no quantum computer needed)
- Experimental research prototype, not production software
- Focus on symbolic compression of specific data structures

---

### Q: Do I need a quantum computer to run this?

**A:** No! QuantoniumOS runs on regular CPUs. The "quantum" terminology refers to:
- Mathematical properties (unitarity, like quantum operators)
- Quantum state *simulation* (classical computer simulating small quantum systems)
- Quantum-*inspired* algorithms (using ideas from quantum mechanics)

You need:
- A regular computer (Linux, Windows, or macOS)
- Python 3.8+
- Standard libraries (NumPy, SciPy, PyQt5)

---

### Q: Is this peer-reviewed research?

**A:** Currently, QuantoniumOS is **not peer-reviewed**. It is:
- A research prototype in active development
- Patent-pending (application filed, not granted)
- Seeking validation through reproducible benchmarks

For academic or production use, independent validation would be required.

---

### Q: What models can actually run on this?

**A:** Currently, only **tiny-gpt2** (2.3M parameters) has been fully verified end-to-end:
- Successfully compressed using RFT codecs
- Reconstructed and validated for inference
- Perplexity benchmarked against original

Claims about larger models (billions of parameters) are **theoretical projections** that have not been validated with reconstructed, working models.

---

## Technical Questions

### Q: What is RFT (Resonance Fourier Transform)?

**A:** RFT is a unitary matrix transformation constructed via QR decomposition of a golden-ratio-weighted kernel.

**Key Properties:**
- Mathematically distinct from DFT (Discrete Fourier Transform)
- Maintains unitarity to machine precision (<1e-12 error)
- Uses golden ratio (φ ≈ 1.618) parameterization

**What makes it unique:**
- Specific kernel construction using φ^|i-j| weighting
- Claimed numerical stability advantages (under investigation)
- Structured for certain classes of problems

**What it's NOT:**
- Not a replacement for all Fourier transforms
- Not proven to be universally "better" than DFT
- Not a quantum algorithm (runs classically)

---

### Q: Is the compression lossless or lossy?

**A:** It depends on the codec:

**Vertex Codec:**
- **Lossless** for φ-structured, low-treewidth quantum states
- **Lossy** for general states
- Typical reconstruction error: <1e-6 for structured states

**Hybrid Codec:**
- **Always lossy** (like JPEG for images)
- Configurable quality levels (0.5x to 100x compression)
- Typical error: 0.1% to 10% depending on quality setting

**Important:** Claims of "15,134:1 lossless compression" are **incorrect**. This would violate Shannon's information theory. The actual system is lossy with measured reconstruction errors.

---

### Q: How does compression ratio of 21:1 on tiny-gpt2 compare to other methods?

**A:** We don't know yet! The 21.9:1 ratio is against **uncompressed fp32 weights**, not against modern compression methods.

**Missing comparisons:**
- GPTQ (GPU-friendly quantization)
- GGUF (efficient inference format)
- bitsandbytes (8-bit/4-bit quantization)
- Standard compression (gzip, zstd)

These benchmarks are needed to assess competitive advantage.

---

### Q: Can this simulate real quantum algorithms?

**A:** Yes, with verified quantum simulation capabilities:

**Verified & Tested:**
- ✅ **Bell States**: Perfect |Φ⁺⟩ states with fidelity = 1.0
- ✅ **CHSH Inequality**: Achieved maximum 2.828427 (Tsirelson bound)
- ✅ **Quantum Entanglement**: Maximum entanglement validated
- ✅ **Up to 20 qubits**: Full state vector simulation
- ⚠️ **>20 qubits**: Symbolic surrogate with fixed-size representation (not full 2^n amplitudes)

**Test Files:**
- `tests/validation/direct_bell_test.py` - CHSH > 2.7 achieved
- `tests/validation/test_bell_violations.py` - Comprehensive Bell tests
- `quantonium_os_src/apps/symbolic_simulator/main.py` - symbolic simulator UI
- `docs/research/BELL_VIOLATION_ACHIEVEMENT.md` - Full results

**Simulation Capabilities:**
- Small circuits (<20 qubits): Full quantum simulation
- Larger qubit counts: Symbolic surrogate with capped amplitude vector
- Bell states, GHZ states: Perfect generation verified
- Measurement operators: Pauli measurements at arbitrary angles
- Decoherence modeling: Depolarizing and amplitude damping noise

**Current Limitations:**
- Shor's algorithm: Not yet implemented (requires phase estimation)
- General deep circuits: Limited by compression efficiency
- Quantum supremacy circuits: Beyond current scope

**What's Unique:**
The RFT kernel enables a symbolic surrogate for structured demos beyond full-state limits. Fidelity and CHSH benchmarks are validated for small, explicit state vectors; large-qubit UI modes do not represent general 2^n states.

**Important Limitation:**
For >20 qubits, this app does **not** represent the full 2^n state vector. It is a compressed surrogate intended for specific structured states and demos. It does **not** support arbitrary circuits, arbitrary entanglement, or general-purpose quantum algorithm simulation at n≫20.

---

## Installation & Setup

### Q: What are the system requirements?

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB disk space
- x86_64 CPU

**Recommended:**
- Python 3.10+
- 16GB RAM
- 10GB disk space
- Modern CPU with AVX2 support
- Linux or macOS

**For Windows:**
- Use WSL2 (Windows Subsystem for Linux)
- Or use the dev container setup

---

### Q: How do I install dependencies?

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: for GUI
pip install PyQt5
```

---

### Q: Do I need to compile the C kernels?

**A:** No, it's optional but recommended for performance.

**Without C kernels:**
- System uses Python fallback implementations
- 10-100x slower performance
- All features still work

**With C kernels:**
- Build with: `make -C src/assembly all`
- Requires GCC or compatible C compiler
- Provides significant speedup

---

### Q: The build fails on my system. What should I do?

**Common solutions:**

1. **No compiler found:**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install

# Windows (WSL2)
sudo apt-get install gcc make
```

2. **Build errors:**
```bash
# Clean and rebuild
cd src/assembly
make clean
make all
```

3. **Skip C kernels:**
```bash
# Just use Python implementations
python quantonium_boot.py
# System will automatically use fallback
```

---

## Usage Questions

### Q: How do I get started?

**Quickest path:**

```bash
# 1. Clone repository
git clone https://github.com/mandcony/quantoniumos
cd quantoniumos

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run quick validation
python tests/validation/quick_validation.py

# 4. Boot the system
python quantonium_boot.py
```

See [Quick Start Guide](../onboarding/QUICK_START.md) for detailed instructions.

---

### Q: How do I use the RFT in my own code?

**Basic example:**

```python
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np

# Create RFT engine
rft = CanonicalTrueRFT(size=256)

# Transform data
data = np.random.rand(256) + 1j * np.random.rand(256)
transformed = rft.transform(data)

# Inverse transform
reconstructed = rft.inverse_transform(transformed)
```

See [Working with RFT Kernel](../technical/guides/WORKING_WITH_RFT_KERNEL.md) for complete guide.

---

### Q: How do I compress my own AI model?

**Important:** This is experimental. Only tiny-gpt2 has been fully validated.

```python
from algorithms.compression.hybrid.rft_hybrid_codec import RFTHybridCodec
import torch

# Load your model
model = torch.load('your_model.pth')

# Extract weights
weights = model.state_dict()

# Compress each layer
codec = RFTHybridCodec(quality=0.9)
compressed_weights = {}

for name, weight in weights.items():
    weight_np = weight.cpu().numpy()
    compressed = codec.encode(weight_np)
    compressed_weights[name] = compressed

# Save compressed model
# ... your saving logic ...
```

**Warning:** Thorough testing required to ensure model still functions correctly after compression!

---

## Troubleshooting

### Q: I get "ImportError: cannot import name 'UnitaryRFT'"

**Solution:** The C kernels aren't built or installed.

```bash
# Option 1: Build kernels
cd src/assembly
make clean && make all

# Option 2: Use Python fallback
# Just import the Python version instead
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
```

---

### Q: Python crashes with "Segmentation fault"

**Likely causes:**
1. Array size mismatch with RFT engine
2. Data type issues (not complex128)
3. Non-contiguous arrays

**Solutions:**

```python
# Ensure correct size
assert len(data) == rft.size

# Ensure correct type
data = data.astype(np.complex128)

# Ensure contiguous memory
data = np.ascontiguousarray(data)

# Or rebuild with debug symbols
cd src/assembly
make clean && make asan
```

---

### Q: Desktop won't launch / shows errors

**Common issues:**

1. **PyQt5 not installed:**
```bash
pip install PyQt5
```

2. **Headless environment (no display):**
```bash
# Use console mode
python quantonium_boot.py --mode console

# Or set up virtual display
QT_QPA_PLATFORM=offscreen python quantonium_boot.py
```

3. **Permission errors:**
```bash
# Check file permissions
chmod +x quantonium_boot.py
```

---

### Q: Tests are failing

**First steps:**

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Run quick validation
python tests/validation/quick_validation.py

# Check specific failing test
pytest tests/path/to/test.py -v
```

**Common test failures:**

| Error | Solution |
|-------|----------|
| `import pyqt5 failed` | Install PyQt5 or skip GUI tests |
| `RFT unitarity failed` | Check NumPy version, rebuild kernels |
| `Codec round-trip failed` | Expected for some test data (ANS fallback) |
| `Timeout` | Increase timeout or use faster machine |

---

## Performance Questions

### Q: Why is it so slow?

**Possible reasons:**

1. **Using Python fallback:**
   - Build C kernels for 10-100x speedup
   - Check: `make -C src/assembly all`

2. **Large transform sizes:**
   - RFT is O(n²) for matrix operations
   - Use smaller sizes or batch processing

3. **Inefficient data types:**
   - Use `np.complex128` consistently
   - Keep arrays contiguous

4. **Unoptimized code:**
   - Profile with: `python -m cProfile script.py`
   - Optimize hot paths

---

### Q: How much memory does it use?

**Typical usage:**
- RFT engine (size 256): ~8 MB
- Desktop environment: ~350 MB
- Per application: 30-90 MB
- Total system: <1 GB for normal operation

**For large models:**
- Compression buffer: ~2x uncompressed size temporarily
- Final compressed: 5-50x smaller than original

---

### Q: Can I run this on a Raspberry Pi?

**Possibly, but not recommended.**

- Minimum: Raspberry Pi 4 (4GB model)
- Expect much slower performance
- May hit memory limits with larger operations
- Desktop UI may be sluggish

Better platforms:
- Modern laptop/desktop
- Cloud instance (2+ vCPU, 8GB RAM)
- Development workstation

---

## Contributing

### Q: How can I contribute?

We welcome contributions! See [Contributing Guide](../technical/guides/CONTRIBUTING.md).

**Ways to help:**
1. Run benchmarks and report results
2. Test on different platforms
3. Add missing test coverage
4. Improve documentation
5. Report bugs
6. Suggest features

---

### Q: I found a bug. What should I do?

1. **Check if it's known:**
   - Search GitHub issues
   - Check [Known Issues](#known-issues)

2. **Gather information:**
   - Python version, OS, hardware
   - Full error message and traceback
   - Minimal reproduction steps

3. **Report:**
   - Open GitHub issue
   - Provide all gathered information
   - Tag appropriately

---

### Q: Can I use this in my research/project?

**For research:**
- ✅ Yes, it's open source
- ⚠️ Cite appropriately
- ⚠️ Note it's experimental/not peer-reviewed
- ⚠️ Validate results independently

**For production:**
- ⚠️ Not recommended (experimental stage)
- ⚠️ No security guarantees
- ⚠️ Crypto primitives are experimental
- ✅ OK for prototyping/exploration

---

## Known Issues

### Current Limitations

1. **Large model compression**: Only tiny-gpt2 (2.3M params) fully validated
2. **Crypto primitives**: Experimental only, not production-ready
3. **Test coverage**: ~73% overall (some modules <50%)
4. **Documentation**: Some advanced features under-documented
5. **Windows support**: Best via WSL2 or dev containers
6. **QuTiP fidelity**: Known low-fidelity case (~0.468) in entanglement tests

### Planned Improvements

See [Roadmap](../ROADMAP.md) for upcoming features and fixes.

---

## Getting Help

### Resources

- **Documentation**: Start with [Quick Start](../onboarding/QUICK_START.md)
- **API Reference**: [API Docs](../technical/API_REFERENCE.md)
- **Examples**: Check `tests/` directory
- **Community**: GitHub Discussions

### Before Asking

1. Check this FAQ
2. Search documentation
3. Review existing GitHub issues
4. Try [troubleshooting steps](#troubleshooting)

### How to Ask

Good questions include:
- What you're trying to do
- What you expected
- What actually happened
- Steps to reproduce
- System information

Example:
```
I'm trying to compress a custom model but getting a unitarity error.

Expected: Model compresses successfully
Actual: "UnitarityError: Matrix not unitary (error: 0.01)"

Steps:
1. Load model with torch.load()
2. Extract weights
3. Call codec.encode(weights)
4. Error occurs

System: Ubuntu 22.04, Python 3.10, 16GB RAM
```

---

## Philosophy & Vision

### Q: What's the goal of this project?

**Short term:**
- Validate mathematical foundations
- Benchmark against state-of-the-art methods
- Build reproducible research platform

**Long term:**
- Advance compression techniques for AI models
- Explore quantum-inspired classical algorithms
- Contribute to open-source research community

---

### Q: Why open source?

- **Transparency**: Research should be reproducible
- **Collaboration**: Best results through community
- **Validation**: Independent verification builds credibility
- **Education**: Help others learn these techniques

---

### Q: What's the difference between QuantoniumOS and traditional compression?

**Traditional compression** (gzip, LZMA):
- General-purpose
- Lossless for arbitrary data
- Well-understood algorithms
- Production-ready

**QuantoniumOS compression**:
- Domain-specific (quantum states, AI models)
- Lossy with bounded distortion
- Novel mathematical approaches
- Research/experimental stage

**Use cases:**
- Traditional: Compress files, data transfer
- QuantoniumOS: Compress AI models, quantum simulation data

---

Still have questions? 

Check our [Documentation Hub](../README.md) or open a GitHub issue!

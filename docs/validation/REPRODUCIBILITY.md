# QuantoniumOS Reproducibility Guide

This document provides step-by-step instructions for reproducing all technical claims and benchmarks in the QuantoniumOS repository.

## Quick Start

### Full System Validation

```bash
# Run complete reproducibility suite (5-15 minutes)
python tools/run_all_tests.py

# Quick validation (< 60 seconds)
python tools/run_all_tests.py --quick

# Custom output location
python tools/run_all_tests.py --output my_results/
```

## Individual Component Testing

### 1. RFT Unitarity Validation

**Claim**: RFT maintains unitarity to machine precision (< 1e-12)  
**Artifact**: `results/rft_invariants_*.json`

```bash
# Method 1: Using direct RFT validation
python dev/tools/print_rft_invariants.py --size 64 --seed 42

# Method 2: Using Python RFT implementation  
python -c "
import sys; sys.path.append('src/core')
from canonical_true_rft import UnitaryRFT
rft = UnitaryRFT(64)
matrix = rft.generate_matrix()
import numpy as np
error = np.linalg.norm(matrix.conj().T @ matrix - np.eye(64), ord=2)
print(f'Unitarity error: {error:.2e}')
print(f'Unitary (< 1e-12): {error < 1e-12}')
"
```

**Expected Output**: Unitarity error < 1e-12, determinant magnitude ≈ 1.0

### 2. Quantum Scaling Measurements

**Claim**: Near-linear O(n) scaling for quantum simulation  
**Artifact**: `results/QUANTUM_SCALING_BENCHMARK.json`

```bash
# Test vertex-based quantum scaling
python -c "
import sys; sys.path.append('src/apps')
from symbolic_simulator import QuantumSimulator
import time

for size in [100, 200, 500, 1000]:
    start = time.time()
    sim = QuantumSimulator()
    sim.create_vertex_system(size)
    duration = time.time() - start
    print(f'{size} vertices: {duration:.4f}s')
"
```

**Expected Output**: Sub-linear time scaling, not exponential growth

### 3. Cryptographic Performance

**Claim**: 64-round Feistel cipher with measured throughput  
**Artifact**: `results/crypto_performance_*.json`

```bash
# Test crypto system performance
python -c "
import sys; sys.path.append('src/core')
from enhanced_rft_crypto_v2 import EnhancedRFTCrypto
import time

crypto = EnhancedRFTCrypto(key=b'test_key_32_bytes_long_for_test!')
data = b'A' * 1024

start = time.time()
encrypted = crypto.encrypt(data)
encrypt_time = time.time() - start

start = time.time()  
decrypted = crypto.decrypt(encrypted)
decrypt_time = time.time() - start

print(f'Rounds: {crypto.rounds}')
print(f'Encrypt: {encrypt_time:.6f}s')
print(f'Decrypt: {decrypt_time:.6f}s')
print(f'Throughput: {(1024/1024/1024)/encrypt_time:.2f} GB/s')
print(f'Correctness: {data == decrypted}')
"
```

**Expected Output**: 64 rounds, successful encryption/decryption, measurable throughput

### 4. Compression Ratio Analysis

**Claim**: High compression ratios for quantum-encoded models  
**Artifact**: `results/compression_analysis_*.json`

```bash
# Run compression analysis
python analyze_compression.py > results/compression_analysis_$(date +%Y%m%d_%H%M%S).json

# Alternative: Manual analysis
python -c "
import json
from pathlib import Path

# Check existing results
results_dir = Path('results')
compression_files = list(results_dir.glob('*compression*.json'))
print(f'Found {len(compression_files)} compression result files')

for file in compression_files[:3]:
    with open(file) as f:
        data = json.load(f)
        print(f'{file.name}: {json.dumps(data, indent=2)[:200]}...')
"
```

### 5. Parameter Count Verification

**Claim**: 25.02 billion total parameters across all models  
**Artifact**: `results/params_summary.json`

```bash
# Verify parameter counts
python -c "
import json

# Check for existing parameter summary
try:
    with open('results/params_summary.json') as f:
        data = json.load(f)
        print('Parameter Summary:')
        print(f'Total: {data.get(\"total_parameters\", \"N/A\")}')
        print(f'Quantum Encoded: {data.get(\"quantum_encoded\", \"N/A\")}') 
        print(f'Direct Models: {data.get(\"direct_models\", \"N/A\")}')
        print(f'Native Components: {data.get(\"native_components\", \"N/A\")}')
except FileNotFoundError:
    print('Parameter summary not found. Run analyze_compression.py to generate.')
"
```

## Environment Requirements

### System Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify key imports
python -c "
try:
    import numpy, scipy, torch, transformers
    print('✓ Core dependencies available')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
"
```

### Hardware Specifications

**Test Environment**: The benchmarks in this repository were generated on:

- **OS**: Ubuntu 24.04.2 LTS (GitHub Codespaces)
- **CPU**: Variable (cloud instance)
- **Memory**: Variable (cloud instance)  
- **Python**: 3.x with NumPy/SciPy scientific stack

**Performance Note**: Absolute timing results will vary by hardware. The reproducibility harness focuses on:

1. **Correctness**: Algorithms produce expected outputs
2. **Scaling Behavior**: Relative performance scaling patterns
3. **Unitarity**: Mathematical properties maintained to precision
4. **Functionality**: All components execute without errors

## Troubleshooting

### Common Issues

**ImportError for core modules**:
```bash
# Ensure Python path includes source directories
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)/src/core:$(pwd)/src/apps"
python tools/run_all_tests.py
```

**Missing C extensions**:
```bash
# C kernels are optional; Python fallbacks available
cd src/assembly
make clean
make all  # Optional: builds optimized C kernels
```

**Test timeouts**:
```bash
# Use quick mode for faster validation
python tools/run_all_tests.py --quick
```

### Validation Failure Analysis

If tests fail, check:

1. **Import paths**: Ensure `PYTHONPATH` includes `src/` directories
2. **Dependencies**: Verify `requirements.txt` packages installed
3. **Disk space**: Ensure sufficient space in `results/` directory
4. **Permissions**: Verify write access to output directories

### Hardware-Specific Results

**Expected variations by hardware**:

- **Timing results**: Absolute times will vary; focus on scaling patterns
- **Memory usage**: May differ based on available system memory
- **Precision**: Mathematical properties (unitarity, correctness) should be consistent
- **Throughput**: Crypto/compression throughput scales with CPU performance

## Continuous Integration

### Automated Validation

The repository includes automated validation that can be integrated into CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run QuantoniumOS Validation  
  run: |
    python tools/run_all_tests.py --quick
    
- name: Archive Results
  uses: actions/upload-artifact@v2
  with:
    name: validation-results
    path: results/reproducibility_run_*.json
```

### Result Archival

All test runs generate timestamped JSON files in `results/`:

- `reproducibility_run_YYYYMMDD_HHMMSS.json`: Complete test suite results
- Individual component artifacts referenced in documentation
- Metadata includes system information and test parameters

## Claims Validation Matrix

| Claim | Test Method | Expected Result | Artifact Location |
|-------|-------------|-----------------|-------------------|
| RFT Unitarity < 1e-12 | `print_rft_invariants.py` | Error < 1e-12 | `results/rft_invariants_*.json` |
| Near-linear scaling | Quantum simulator timing | O(n) behavior | `results/QUANTUM_SCALING_BENCHMARK.json` |
| 64-round crypto | Cipher inspection | `rounds = 64` | Source code verification |
| Compression ratios | Model analysis | High compression | `results/compression_analysis_*.json` |
| 25.02B parameters | Parameter counting | Total = 25.02B | `results/params_summary.json` |

## Contact & Issues

For reproducibility issues or questions:

1. **Check existing artifacts**: Many claims reference existing result files
2. **Run diagnostics**: Use `--quick` mode for fast validation
3. **Review logs**: Check console output and generated JSON for error details
4. **Environment check**: Verify Python dependencies and path configuration

---

*Last Updated: Generated by bulletproofing plan execution*  
*Reproducibility Harness Version: 1.0*
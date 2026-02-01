# QuantoniumOS Project Structure

## Implementation Architecture (September 2025)

### 1. Source Code (`src/`)
- `core/` - Mathematical algorithms and core implementations
  - `canonical_true_rft.py` - RFT implementation with golden ratio parameterization
  - `enhanced_rft_crypto_v2.py` - 48-round Feistel cryptographic system
  - `working_quantum_kernel.py` - Quantum simulation kernel
- `assembly/kernel/` - C implementation with SIMD optimization
  - `rft_kernel.c` - Unitary RFT kernel with QR decomposition
  - `rft_kernel.h` - Header definitions and constants
- `apps/` - PyQt5 desktop applications
  - `symbolic_simulator.py` - symbolic surrogate quantum simulator (artifact: results/QUANTUM_SCALING_BENCHMARK.json)
  - `q_notes.py` - Note-taking application
  - `q_vault.py` - Secure storage application
  - Additional utility and monitoring applications
- `frontend/` - Desktop environment
  - `quantonium_desktop.py` - Main desktop manager with app integration

### 2. Configuration and Data (`data/`)
- `config/` - System configuration files
  - `app_registry.json` - Application registry
  - `build_config.json` - Build configuration
- `logs/` - System logs and chat records
- `weights/` - Model weights and parameters

### 3. Testing Framework (`tests/`)
- `crypto/` - Cryptographic validation and statistical analysis
- `benchmarks/` - Performance testing suites
- `results/` - Test results and validation reports

### 4. Documentation (`docs/`)
- `technical/` - Technical specifications and implementation details
- `reports/` - Analysis and validation reports
- `safety/` - AI safety documentation

### 5. User Interface Assets (`ui/`)
- `icons/` - Application icons (SVG format)
- `styles_dark.qss`, `styles_light.qss` - PyQt5 themes
- `styles.qss` - Base stylesheet

## Architecture Overview

```
QuantoniumOS Implementation:
┌─────────────────────────────────────┐
│  PyQt5 Applications               │ ← 7 integrated desktop apps
├─────────────────────────────────────┤
│  QuantoniumOS Desktop Manager      │ ← Main desktop environment
├─────────────────────────────────────┤
│  Python Core Algorithms           │ ← Mathematical implementations
├─────────────────────────────────────┤
│  C RFT Kernel                     │ ← SIMD-optimized performance
└─────────────────────────────────────┘
```

## Running the System

### Prerequisites
```bash
pip install -r requirements.txt
```

### Launch QuantoniumOS
```bash
python quantonium_boot.py
```

### Individual Components
```bash
# Run specific applications
python src/apps/symbolic_simulator.py
python src/apps/q_notes.py

# Test RFT kernel
python src/core/canonical_true_rft.py
```

## Implementation Principles

### 1. **Mathematical Foundation**
- Machine precision unitarity (errors < 1e-15)
- Golden ratio parameterization throughout
- Validated energy conservation

### 2. **System Integration**
- In-process app launching for unified environment
- Dynamic Python imports for extensibility
- Shared mathematical kernels across applications

### 3. **Performance**
- SIMD-optimized C kernels for critical operations
- Vertex encoding for large-scale symbolic surrogate simulation
- Linear complexity algorithms where possible

### 4. **User Experience**
- Cohesive PyQt5 desktop environment
- Golden ratio proportions in UI design
- Dark/light theme support
3. **Consistent UI**: All apps use the unified stylesheet
4. **Error Handling**: Apps must handle RFT unavailability gracefully
5. **Documentation**: Update this file when adding new components

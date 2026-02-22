# Validation & Testing Workflow

This guide describes the comprehensive testing and validation process for QuantoniumOS.

## Testing Philosophy

QuantoniumOS follows a multi-layered testing approach:

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Validation Tests**: Verify mathematical properties
4. **Performance Benchmarks**: Measure and track performance
5. **End-to-End Tests**: Test complete workflows

## Test Structure

```
tests/
├── algorithms/              # Algorithm unit tests
│   ├── compression/
│   ├── crypto/
│   └── rft/
├── integration/             # Integration tests
│   ├── test_rft_integration.py
│   └── test_codec_pipeline.py
├── benchmarks/              # Performance benchmarks
│   ├── benchmark_rft.py
│   └── benchmark_compression.py
└── validation/              # Mathematical validation
    ├── comprehensive_validation_suite.py
    └── quick_validation.py
```

## Running Tests

### Quick Validation

Run the quick validation suite for rapid feedback:

```bash
python tests/validation/quick_validation.py
```

Expected output:
```
🧪 QuantoniumOS Quick Validation Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ RFT Unitarity Test: PASSED
✅ Vertex Codec Round-trip: PASSED
✅ Crypto System Test: PASSED
✅ Desktop Boot Test: PASSED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 All Quick Tests Passed! (4/4)
```

### Comprehensive Validation

Run the full validation suite:

```bash
python tests/validation/comprehensive_validation_suite.py
```

This runs all tests including:
- Mathematical property verification
- Codec accuracy tests
- Cryptographic primitive tests
- Integration tests
- Performance regression tests

### Unit Tests with pytest

Run specific test modules:

```bash
# All tests
pytest tests/

# Specific module
pytest tests/algorithms/rft/test_canonical_rft.py

# With coverage
pytest --cov=algorithms --cov-report=html tests/

# Verbose output
pytest -v tests/

# Stop on first failure
pytest -x tests/
```

### Integration Tests

```bash
# Run all integration tests
pytest tests/integration/

# Specific integration test
pytest tests/integration/test_rft_integration.py
```

### Performance Benchmarks

```bash
# Run all benchmarks
python -m pytest tests/benchmarks/ --benchmark-only

# Specific benchmark
python tests/benchmarks/benchmark_rft.py

# Save baseline
pytest tests/benchmarks/ --benchmark-save=baseline

# Compare with baseline
pytest tests/benchmarks/ --benchmark-compare=baseline
```

## Writing Tests

### Unit Test Template

```python
import pytest
import numpy as np
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT


class TestCanonicalRFT:
    """Test suite for Canonical RFT implementation"""
    
    @pytest.fixture
    def rft_engine(self):
        """Create RFT engine for testing"""
        return CanonicalTrueRFT(size=128)
    
    @pytest.fixture
    def random_state(self):
        """Generate random quantum state"""
        state = np.random.rand(128) + 1j * np.random.rand(128)
        return state / np.linalg.norm(state)
    
    def test_initialization(self, rft_engine):
        """Test that RFT engine initializes correctly"""
        assert rft_engine is not None
        assert rft_engine.size == 128
        
    def test_unitarity(self, rft_engine, random_state):
        """Test that RFT preserves unitarity"""
        norm_before = np.linalg.norm(random_state)
        transformed = rft_engine.transform(random_state)
        norm_after = np.linalg.norm(transformed)
        
        error = abs(norm_before - norm_after)
        assert error < 1e-12, f"Unitarity error: {error}"
        
    def test_round_trip(self, rft_engine, random_state):
        """Test forward and inverse transforms"""
        transformed = rft_engine.transform(random_state)
        reconstructed = rft_engine.inverse_transform(transformed)
        
        error = np.linalg.norm(random_state - reconstructed)
        assert error < 1e-10, f"Round-trip error: {error}"
        
    @pytest.mark.parametrize("size", [64, 128, 256, 512])
    def test_multiple_sizes(self, size):
        """Test RFT with various sizes"""
        engine = CanonicalTrueRFT(size=size)
        state = np.random.rand(size) + 1j * np.random.rand(size)
        
        result = engine.transform(state)
        assert result.shape == state.shape
```

### Integration Test Template

```python
import pytest
from algorithms.compression.vertex.rft_vertex_codec import RFTVertexCodec
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT


class TestRFTCodecIntegration:
    """Test RFT and Codec integration"""
    
    @pytest.fixture
    def setup(self):
        """Setup RFT and codec"""
        rft = CanonicalTrueRFT(size=128)
        codec = RFTVertexCodec(rft_engine=rft)
        return rft, codec
    
    def test_encode_decode_pipeline(self, setup):
        """Test complete encode-decode pipeline"""
        rft, codec = setup
        
        # Create test data
        original_data = np.random.rand(128) + 1j * np.random.rand(128)
        
        # Encode
        encoded = codec.encode(original_data)
        assert len(encoded) < len(original_data) * 16  # Check compression
        
        # Decode
        decoded = codec.decode(encoded)
        
        # Verify accuracy
        error = np.linalg.norm(original_data - decoded)
        assert error < 1e-6, f"Decode error: {error}"
```

### Benchmark Template

```python
import pytest
import numpy as np
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT


@pytest.mark.benchmark
class TestRFTPerformance:
    """Benchmark RFT performance"""
    
    @pytest.fixture
    def data(self):
        """Generate benchmark data"""
        return {
            'small': np.random.rand(128) + 1j * np.random.rand(128),
            'medium': np.random.rand(512) + 1j * np.random.rand(512),
            'large': np.random.rand(2048) + 1j * np.random.rand(2048),
        }
    
    def test_small_transform(self, benchmark, data):
        """Benchmark small transform"""
        engine = CanonicalTrueRFT(size=128)
        result = benchmark(engine.transform, data['small'])
        assert result is not None
        
    def test_medium_transform(self, benchmark, data):
        """Benchmark medium transform"""
        engine = CanonicalTrueRFT(size=512)
        result = benchmark(engine.transform, data['medium'])
        assert result is not None
```

## Validation Checklist

Use this checklist after making changes:

### Before Committing

- [ ] All unit tests pass: `pytest tests/algorithms/`
- [ ] No new linting errors: `flake8 algorithms/ os/`
- [ ] Code formatted: `black algorithms/ os/`
- [ ] Documentation updated
- [ ] Type hints added (if applicable)

### Before Pull Request

- [ ] All integration tests pass: `pytest tests/integration/`
- [ ] Quick validation passes: `python tests/validation/quick_validation.py`
- [ ] No performance regression: `pytest tests/benchmarks/ --benchmark-compare`
- [ ] New features have tests
- [ ] CHANGELOG updated

### After Merge

- [ ] Full validation suite passes: `python tests/validation/comprehensive_validation_suite.py`
- [ ] CI/CD pipeline succeeds
- [ ] Documentation deployed
- [ ] Release notes updated (if applicable)

## Continuous Integration

### GitHub Actions Workflow

`.github/workflows/test.yml`:

```yaml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e '.[dev]'
    
    - name: Lint with flake8
      run: flake8 algorithms/ os/
    
    - name: Check formatting with black
      run: black --check algorithms/ os/
    
    - name: Run unit tests
      run: pytest tests/ --cov=algorithms --cov=os --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Test Coverage

### Measure Coverage

```bash
# Generate coverage report
pytest --cov=algorithms --cov=os --cov-report=html tests/

# View report
open htmlcov/index.html
```

### Coverage Goals

- **Core algorithms**: 90%+ coverage
- **Critical paths**: 100% coverage
- **Integration points**: 85%+ coverage
- **Overall**: 80%+ coverage

## Debugging Failed Tests

### Enable Verbose Output

```bash
pytest -vv tests/algorithms/test_failing.py
```

### Use pdb Debugger

```python
def test_something():
    result = function_under_test()
    
    # Drop into debugger
    import pdb; pdb.set_trace()
    
    assert result == expected
```

Or run with:
```bash
pytest --pdb tests/
```

### Print Debugging

```python
def test_with_debug_output(capfd):
    """Test with captured output"""
    print("Debug info")
    result = function_under_test()
    
    # Capture printed output
    captured = capfd.readouterr()
    print(f"Captured: {captured.out}")
    
    assert result is not None
```

### Isolate Failing Tests

```bash
# Run only failed tests from last run
pytest --lf

# Run failed tests first
pytest --ff

# Increase verbosity for failed tests
pytest --tb=long
```

## Performance Regression Detection

### Establish Baseline

```bash
# Run benchmarks and save baseline
pytest tests/benchmarks/ --benchmark-save=v1.0.0
```

### Compare Against Baseline

```bash
# Compare current performance
pytest tests/benchmarks/ --benchmark-compare=v1.0.0

# Fail if regression > 10%
pytest tests/benchmarks/ --benchmark-compare=v1.0.0 \
  --benchmark-compare-fail=mean:10%
```

### Track Over Time

```bash
# Save after each release
pytest tests/benchmarks/ --benchmark-save=v1.1.0
pytest tests/benchmarks/ --benchmark-save=v1.2.0

# Compare versions
pytest tests/benchmarks/ --benchmark-compare=v1.0.0,v1.1.0,v1.2.0
```

## Test Data Management

### Fixtures for Reusable Data

```python
# conftest.py - shared fixtures
import pytest
import numpy as np

@pytest.fixture(scope="session")
def golden_ratio():
    """Golden ratio constant"""
    return (1 + np.sqrt(5)) / 2

@pytest.fixture(scope="module")
def test_quantum_states():
    """Generate test quantum states"""
    states = []
    for size in [64, 128, 256]:
        state = np.random.rand(size) + 1j * np.random.rand(size)
        state = state / np.linalg.norm(state)
        states.append(state)
    return states
```

### Test Data Files

Store large test datasets in `tests/data/`:

```python
import json
from pathlib import Path

def load_test_data(filename):
    """Load test data from file"""
    data_dir = Path(__file__).parent / "data"
    with open(data_dir / filename) as f:
        return json.load(f)

def test_with_stored_data():
    data = load_test_data("test_case_1.json")
    result = process(data)
    assert result == expected
```

## Best Practices

### 1. Test One Thing at a Time

```python
# Bad: Testing multiple things
def test_everything():
    assert rft.transform(data) is not None
    assert rft.size == 128
    assert rft.is_unitary()

# Good: Separate tests
def test_transform_returns_result():
    assert rft.transform(data) is not None
    
def test_size_is_correct():
    assert rft.size == 128
    
def test_unitarity_preserved():
    assert rft.is_unitary()
```

### 2. Use Descriptive Test Names

```python
# Bad
def test_1():
    ...

# Good
def test_rft_preserves_norm_after_forward_transform():
    ...
```

### 3. Arrange-Act-Assert Pattern

```python
def test_codec_compression():
    # Arrange
    codec = RFTVertexCodec()
    data = np.random.rand(128)
    
    # Act
    compressed = codec.encode(data)
    
    # Assert
    assert len(compressed) < len(data) * 16
```

### 4. Use Parameterization

```python
@pytest.mark.parametrize("size,expected_time", [
    (128, 0.01),
    (256, 0.05),
    (512, 0.20),
])
def test_performance_scales(size, expected_time):
    engine = RFT(size)
    time = measure_time(engine.transform, data)
    assert time < expected_time
```

## Troubleshooting

### Tests Timeout

```bash
# Increase timeout
pytest --timeout=300 tests/
```

### Memory Issues

```bash
# Run tests in separate processes
pytest -n auto tests/
```

### Random Failures

```python
# Set random seed for reproducibility
@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)
    random.seed(42)
```

## Further Reading

- [Contributing Guidelines](./CONTRIBUTING.md)
- [Component Deep Dive](../COMPONENT_DEEP_DIVE.md)
- [Working with RFT Kernel](./WORKING_WITH_RFT_KERNEL.md)

# Algorithm Test Suite

Comprehensive tests for all QuantoniumOS mathematical algorithms.

## Test Categories

### RFT Algorithm Tests
- **Unitarity preservation** - Verify RFT maintains unitary properties
- **Golden ratio parameterization** - Test φ-based parameter construction
- **Transform accuracy** - Compare against DFT and validate distinctness
- **Assembly kernel integration** - Test C/Python implementation consistency

### Compression Algorithm Tests
- **Vertex codec validation** - Round-trip encoding/decoding tests
- **Hybrid compression** - Multi-stage compression pipeline tests
- **Model compression** - AI model encode/decode validation
- **Error bounds** - Lossy compression quality metrics

### Cryptographic Algorithm Tests
- **Encryption primitives** - Test quantum-inspired crypto functions
- **Key derivation** - Golden ratio based key generation
- **Hash functions** - Geometric hash validation
- **Security properties** - Avalanche effect and differential analysis

## Test Files

Place algorithm-specific test files here:
- `test_rft_core.py` - Core RFT algorithm tests
- `test_vertex_codec.py` - Vertex encoding tests
- `test_hybrid_compression.py` - Compression pipeline tests
- `test_crypto_primitives.py` - Cryptographic function tests

## Running Tests

```bash
pytest tests/algorithms/           # Run all algorithm tests
pytest tests/algorithms/test_rft_core.py  # Specific test file
```
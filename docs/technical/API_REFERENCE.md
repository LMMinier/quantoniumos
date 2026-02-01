# API Reference

> **RESEARCH DISCLAIMER:** This API is for experimental research only.
> Cryptographic features are NOT certified. Do NOT use for production security.

This document provides a comprehensive API reference for QuantoniumOS components.

## Core RFT API

### `algorithms.rft.core.canonical_true_rft`

#### `CanonicalTrueRFT`

Main RFT transform engine (Python reference implementation).

```python
class CanonicalTrueRFT:
    """
    Canonical implementation of the Resonance Fourier Transform.
    
    This is the reference Python implementation. For performance-critical
    applications, use the C kernel if available (UnitaryRFT).
    
    Attributes:
        size (int): Transform size (must be power of 2)
        phi (float): Golden ratio constant (1.618...)
        matrix (ndarray): Precomputed RFT matrix
    """
    
    def __init__(self, size: int):
        """
        Initialize RFT engine.
        
        Args:
            size: Transform size (recommended: power of 2)
            
        Raises:
            ValueError: If size < 2 or not supported
        """
        
    def transform(self, input_state: np.ndarray) -> np.ndarray:
        """
        Apply forward RFT transform.
        
        Args:
            input_state: Complex array of length self.size
            
        Returns:
            Transformed state (same shape as input)
            
        Raises:
            ValueError: If input size doesn't match engine size
        """
        
    def inverse_transform(self, transformed_state: np.ndarray) -> np.ndarray:
        """
        Apply inverse RFT transform.
        
        Args:
            transformed_state: Complex array of length self.size
            
        Returns:
            Reconstructed state
        """
        
    def verify_unitarity(self) -> float:
        """
        Verify that the RFT matrix is unitary.
        
        Returns:
            Unitarity error (should be < 1e-12)
        """
```

**Example Usage**:
```python
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np

# Create engine
rft = CanonicalTrueRFT(size=256)

# Generate test state
state = np.random.rand(256) + 1j * np.random.rand(256)
state = state / np.linalg.norm(state)  # Normalize

# Transform
transformed = rft.transform(state)

# Inverse transform
reconstructed = rft.inverse_transform(transformed)

# Verify round-trip
error = np.linalg.norm(state - reconstructed)
print(f"Round-trip error: {error:.2e}")  # Should be ~1e-14
```

---

## Assembly Kernel API

### `ASSEMBLY.python_bindings.unitary_rft`

#### `UnitaryRFT`

High-performance C kernel wrapper.

```python
class UnitaryRFT:
    """
    C kernel implementation of RFT with Python bindings.
    
    This provides 10-100x performance improvement over the Python
    implementation for large transform sizes.
    
    Attributes:
        size (int): Transform size
        flags (int): Configuration flags
    """
    
    def __init__(self, size: int, flags: int = 0):
        """
        Initialize RFT kernel.
        
        Args:
            size: Transform size
            flags: Bitwise OR of RFT_FLAG_* constants
                - RFT_FLAG_QUANTUM_SAFE: Enable quantum-safe operations
                - RFT_FLAG_USE_RESONANCE: Use resonance mode
                - RFT_FLAG_OPTIMIZE_MEMORY: Optimize for memory
                
        Raises:
            RuntimeError: If kernel initialization fails
        """
        
    def transform(self, input_data: np.ndarray) -> np.ndarray:
        """
        Apply forward transform using C kernel.
        
        Args:
            input_data: Complex128 array
            
        Returns:
            Transformed array
            
        Note:
            Input must be contiguous and correct type.
            Use np.ascontiguousarray() if needed.
        """
        
    def process_quantum_field(self, field_data: np.ndarray) -> np.ndarray:
        """
        Process quantum field data with RFT.
        
        This is an optimized path for quantum state evolution.
        
        Args:
            field_data: Quantum field amplitudes
            
        Returns:
            Evolved field state
        """
```

**Flag Constants**:
```python
# WARNING: "QUANTUM_SAFE" is legacy naming - crypto is NOT certified
RFT_FLAG_QUANTUM_SAFE = 0x01    # Experimental crypto mode (RESEARCH ONLY)
RFT_FLAG_USE_RESONANCE = 0x02   # Use resonance algorithms
RFT_FLAG_OPTIMIZE_MEMORY = 0x04 # Memory-optimized mode
RFT_FLAG_DEBUG = 0x08           # Enable debug output
```

**Example Usage**:
```python
from ASSEMBLY.python_bindings.unitary_rft import (
    UnitaryRFT, 
    RFT_FLAG_QUANTUM_SAFE,
    RFT_FLAG_USE_RESONANCE
)
import numpy as np

# Create engine with flags
engine = UnitaryRFT(
    size=1024,
    flags=RFT_FLAG_QUANTUM_SAFE | RFT_FLAG_USE_RESONANCE
)

# Prepare data
data = np.random.rand(1024) + 1j * np.random.rand(1024)
data = np.ascontiguousarray(data, dtype=np.complex128)

# Transform
result = engine.transform(data)
```

---

## Compression API

### `algorithms.compression.vertex.rft_vertex_codec`

#### `RFTVertexCodec`

Symbolic quantum state encoder/decoder.

```python
class RFTVertexCodec:
    """
    Vertex-based quantum state codec using modular arithmetic.
    
    This codec is designed for φ-structured, low-treewidth quantum
    states and provides efficient compression for this specific class.
    
    Attributes:
        rft_engine: RFT engine instance
        modulus: Prime modulus for encoding
    """
    
    def __init__(self, 
                 rft_engine=None,
                 modulus: int = None,
                 quality: float = 1.0):
        """
        Initialize vertex codec.
        
        Args:
            rft_engine: RFT engine (optional, will create if None)
            modulus: Prime modulus (auto-selected if None)
            quality: Encoding quality 0.0-1.0 (higher = less lossy)
        """
        
    def encode(self, quantum_state: np.ndarray) -> bytes:
        """
        Encode quantum state to compressed bytes.
        
        Args:
            quantum_state: Complex array representing quantum state
            
        Returns:
            Compressed byte representation
            
        Note:
            Best results for φ-structured states. General states
            may not compress well.
        """
        
    def decode(self, encoded_data: bytes, size: int) -> np.ndarray:
        """
        Decode compressed bytes back to quantum state.
        
        Args:
            encoded_data: Compressed byte data
            size: Expected output size
            
        Returns:
            Reconstructed quantum state
            
        Raises:
            ValueError: If data is corrupted
        """
        
    def get_compression_ratio(self, 
                              original_size: int, 
                              compressed_size: int) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_size: Original data size in bytes
            compressed_size: Compressed size in bytes
            
        Returns:
            Compression ratio (original / compressed)
        """
```

**Example Usage**:
```python
from algorithms.compression.vertex.rft_vertex_codec import RFTVertexCodec
import numpy as np

# Create codec
codec = RFTVertexCodec(quality=0.95)

# Create structured quantum state (e.g., GHZ-like)
n_qubits = 10
state_size = 2 ** n_qubits  # 1024
state = np.zeros(state_size, dtype=np.complex128)
state[0] = 1/np.sqrt(2)  # |00...0⟩
state[-1] = 1/np.sqrt(2)  # |11...1⟩

# Encode
encoded = codec.encode(state)
print(f"Original: {state.nbytes} bytes")
print(f"Compressed: {len(encoded)} bytes")
print(f"Ratio: {state.nbytes / len(encoded):.1f}:1")

# Decode
decoded = codec.decode(encoded, size=state_size)

# Verify
error = np.linalg.norm(state - decoded)
print(f"Reconstruction error: {error:.2e}")
```

---

### `algorithms.compression.hybrid.rft_hybrid_codec`

#### `RFTHybridCodec`

Multi-stage lossy compression codec.

```python
class RFTHybridCodec:
    """
    Hybrid compression pipeline: RFT + Quantization + Residual + Entropy.
    
    This is a lossy codec suitable for AI model weights and general
    numerical data where some precision loss is acceptable.
    
    Attributes:
        quality: Compression quality (0.0 to 1.0)
        use_rft: Whether to use RFT preprocessing
        quantization_bits: Bit depth for quantization
        use_residual: Whether to use residual encoding
    """
    
    def __init__(self,
                 quality: float = 0.9,
                 use_rft: bool = True,
                 quantization_bits: int = 8,
                 use_residual: bool = True):
        """
        Initialize hybrid codec.
        
        Args:
            quality: 0.0 (max compression) to 1.0 (max quality)
            use_rft: Enable RFT preprocessing
            quantization_bits: Bits per quantized value (4-16)
            use_residual: Enable residual prediction
        """
        
    def encode(self, 
               data: np.ndarray,
               **kwargs) -> dict:
        """
        Encode data with hybrid codec.
        
        Args:
            data: Input array (any shape)
            **kwargs: Additional encoding options
            
        Returns:
            Dictionary with:
                - 'compressed': Compressed bytes
                - 'metadata': Reconstruction metadata
                - 'stats': Compression statistics
        """
        
    def decode(self, encoded: dict) -> np.ndarray:
        """
        Decode compressed data.
        
        Args:
            encoded: Dictionary from encode()
            
        Returns:
            Reconstructed array (lossy)
        """
        
    def benchmark(self, data: np.ndarray) -> dict:
        """
        Benchmark codec on given data.
        
        Args:
            data: Test data
            
        Returns:
            Dictionary with:
                - 'compression_ratio': Original / compressed size
                - 'encode_time': Encoding time (seconds)
                - 'decode_time': Decoding time (seconds)
                - 'rmse': Root mean squared error
                - 'psnr': Peak signal-to-noise ratio
        """
```

**Example Usage**:
```python
from algorithms.compression.hybrid.rft_hybrid_codec import RFTHybridCodec
import numpy as np

# Create codec with different quality settings
codec_high_quality = RFTHybridCodec(quality=0.95, quantization_bits=12)
codec_balanced = RFTHybridCodec(quality=0.90, quantization_bits=8)
codec_max_compression = RFTHybridCodec(quality=0.70, quantization_bits=4)

# Test data (e.g., neural network weights)
weights = np.random.randn(1000, 1000).astype(np.float32)

# Encode
encoded = codec_balanced.encode(weights)
print(f"Original: {weights.nbytes / 1024:.1f} KB")
print(f"Compressed: {len(encoded['compressed']) / 1024:.1f} KB")
print(f"Ratio: {encoded['stats']['compression_ratio']:.1f}:1")

# Decode
decoded = codec_balanced.decode(encoded)

# Measure quality
rmse = np.sqrt(np.mean((weights - decoded) ** 2))
print(f"RMSE: {rmse:.4f}")

# Benchmark
results = codec_balanced.benchmark(weights)
print(f"Encode time: {results['encode_time']:.3f}s")
print(f"Decode time: {results['decode_time']:.3f}s")
print(f"PSNR: {results['psnr']:.1f} dB")
```

---

## Desktop Application API

### `os.frontend.quantonium_desktop`

#### `QuantoniumDesktop`

Main desktop environment manager.

```python
class QuantoniumDesktop(QMainWindow):
    """
    QuantoniumOS desktop environment.
    
    Manages application launching, windowing, and system integration.
    
    Attributes:
        applications: List of registered applications
        active_windows: Currently running application windows
        phi: Golden ratio constant for UI layout
    """
    
    def __init__(self):
        """Initialize desktop environment."""
        
    def register_application(self, 
                            name: str,
                            icon_path: str,
                            launch_path: str,
                            description: str = "",
                            category: str = "general"):
        """
        Register an application with the desktop.
        
        Args:
            name: Application display name
            icon_path: Path to icon file (SVG recommended)
            launch_path: Path to application main file
            description: Brief description
            category: Category for organization
        """
        
    def launch_application(self, app_name: str):
        """
        Launch an application by name.
        
        Args:
            app_name: Name of registered application
            
        Raises:
            ValueError: If application not found
        """
```

---

## Utility Functions

### Golden Ratio Constants

```python
# Golden ratio
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895

# Related constants
PHI_SQUARED = PHI + 1  # ≈ 2.618033988749895
PHI_INVERSE = 1 / PHI  # ≈ 0.618033988749895

# Fibonacci relationship
def phi_power(n: int) -> float:
    """
    Calculate φ^n using Fibonacci relationship.
    
    φ^n = F_n * φ + F_{n-1}
    
    Args:
        n: Power to raise φ to
        
    Returns:
        φ^n
    """
```

### Validation Helpers

```python
def validate_unitarity(matrix: np.ndarray, tolerance: float = 1e-12) -> bool:
    """
    Check if a matrix is unitary.
    
    Args:
        matrix: Square complex matrix
        tolerance: Maximum allowed error
        
    Returns:
        True if unitary within tolerance
    """
    identity = np.eye(len(matrix))
    product = matrix.conj().T @ matrix
    error = np.linalg.norm(product - identity, ord=2)
    return error < tolerance

def measure_compression_ratio(original: np.ndarray, 
                              compressed: bytes) -> float:
    """
    Calculate compression ratio.
    
    Args:
        original: Original array
        compressed: Compressed bytes
        
    Returns:
        Compression ratio
    """
    return original.nbytes / len(compressed)
```

---

## Error Handling

### Common Exceptions

```python
class RFTError(Exception):
    """Base exception for RFT-related errors."""
    pass

class CodecError(Exception):
    """Base exception for codec-related errors."""
    pass

class UnitarityError(RFTError):
    """Raised when unitarity is violated."""
    pass

class CompressionError(CodecError):
    """Raised when compression fails."""
    pass
```

---

## Type Hints

Common type aliases used throughout the codebase:

```python
from typing import Union, Tuple, Optional
import numpy as np

# Common types
ComplexArray = np.ndarray  # Complex128 array
RealArray = np.ndarray     # Float64 array
ByteData = bytes           # Compressed byte data

# Function signatures
def transform(state: ComplexArray) -> ComplexArray: ...
def encode(data: Union[ComplexArray, RealArray]) -> ByteData: ...
def decode(data: ByteData, size: int) -> ComplexArray: ...
```

---

For implementation details, see the source code in:
- `algorithms/rft/core/`
- `algorithms/compression/`
- `ASSEMBLY/python_bindings/`
- `os/frontend/`

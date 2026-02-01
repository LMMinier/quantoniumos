#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Geometric Hashing - QuantoniumOS
================================

Geometric hashing algorithms for spatial data structures and collision detection.
Uses RFT-enhanced geometric transformations for experimental hashing research.

**RESEARCH ONLY DISCLAIMER:**
This module is for academic research purposes only. The cryptographic properties
have NOT been formally verified or certified. Do NOT use in production security
applications. "Quantum-inspired" refers to mathematical structure, not actual
quantum computing or proven post-quantum security.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
import hashlib

# Try to import RFT components
try:
    from ..assembly.python_bindings import UnitaryRFT, RFT_FLAG_UNITARY, RFT_FLAG_QUANTUM_SAFE
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False

class Point2D:
    """2D point representation."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point2D({self.x}, {self.y})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point2D):
            return False
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __hash__(self) -> int:
        return hash((round(self.x, 6), round(self.y, 6)))

    def distance(self, other: 'Point2D') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def angle(self, other: 'Point2D') -> float:
        """Calculate angle to another point in radians."""
        return math.atan2(other.y - self.y, other.x - self.x)

class Point3D:
    """3D point representation."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Point3D({self.x}, {self.y}, {self.z})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point3D):
            return False
        return (math.isclose(self.x, other.x) and
                math.isclose(self.y, other.y) and
                math.isclose(self.z, other.z))

    def __hash__(self) -> int:
        return hash((round(self.x, 6), round(self.y, 6), round(self.z, 6)))

    def distance(self, other: 'Point3D') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 +
                        (self.y - other.y)**2 +
                        (self.z - other.z)**2)

class GeometricHash:
    """Base class for geometric hashing algorithms."""

    def __init__(self, dimensions: int = 2, hash_size: int = 256):
        """Initialize geometric hash.

        Args:
            dimensions: Number of spatial dimensions (2 or 3)
            hash_size: Size of hash output in bits
        """
        self.dimensions = dimensions
        self.hash_size = hash_size
        self.byte_size = hash_size // 8

    def _normalize_point(self, point: Union[Point2D, Point3D, Tuple, List]) -> np.ndarray:
        """Normalize point to numpy array."""
        if isinstance(point, (Point2D, Point3D)):
            coords = [point.x, point.y]
            if isinstance(point, Point3D):
                coords.append(point.z)
        elif isinstance(point, (tuple, list)):
            coords = list(point)
        else:
            raise ValueError("Unsupported point type")

        if len(coords) != self.dimensions:
            raise ValueError(f"Point must have {self.dimensions} dimensions")

        return np.array(coords, dtype=np.float64)

    def hash_point(self, point: Union[Point2D, Point3D, Tuple, List]) -> bytes:
        """Hash a single point."""
        raise NotImplementedError("Subclasses must implement hash_point")

    def hash_points(self, points: List[Union[Point2D, Point3D, Tuple, List]]) -> bytes:
        """Hash multiple points."""
        raise NotImplementedError("Subclasses must implement hash_points")

class SpatialHash(GeometricHash):
    """Spatial hashing using coordinate quantization."""

    def __init__(self, dimensions: int = 2, hash_size: int = 256,
                 precision: float = 1e-6, grid_size: float = 1.0):
        """Initialize spatial hash.

        Args:
            dimensions: Number of spatial dimensions
            hash_size: Size of hash output in bits
            precision: Coordinate precision for quantization
            grid_size: Size of spatial grid cells
        """
        super().__init__(dimensions, hash_size)
        self.precision = precision
        self.grid_size = grid_size

    def _quantize_coordinate(self, coord: float) -> int:
        """Quantize a coordinate to grid."""
        return int(round(coord / self.grid_size))

    def hash_point(self, point: Union[Point2D, Point3D, Tuple, List]) -> bytes:
        """Hash a single point using spatial quantization."""
        coords = self._normalize_point(point)

        # Quantize coordinates
        quantized = [self._quantize_coordinate(c) for c in coords]

        # Convert to bytes and hash
        coord_bytes = b''.join(i.to_bytes(8, 'big', signed=True) for i in quantized)
        return hashlib.sha256(coord_bytes).digest()[:self.byte_size]

    def hash_points(self, points: List[Union[Point2D, Point3D, Tuple, List]]) -> bytes:
        """Hash multiple points by combining their spatial hashes."""
        if not points:
            return b'\x00' * self.byte_size

        # Hash each point and combine
        point_hashes = [self.hash_point(p) for p in points]

        # XOR all point hashes together
        combined = point_hashes[0]
        for ph in point_hashes[1:]:
            combined = bytes(a ^ b for a, b in zip(combined, ph))

        return combined

class GeometricTransformHash(GeometricHash):
    """Geometric hashing using coordinate transformations."""

    def __init__(self, dimensions: int = 2, hash_size: int = 256,
                 transforms: Optional[List[Callable]] = None):
        """Initialize geometric transform hash.

        Args:
            dimensions: Number of spatial dimensions
            hash_size: Size of hash output in bits
            transforms: List of transformation functions
        """
        super().__init__(dimensions, hash_size)

        if transforms is None:
            # Default transforms: identity, rotation, scaling
            self.transforms = [
                lambda x: x,  # Identity
                lambda x: np.roll(x, 1),  # Cyclic shift
                lambda x: x * 2.0,  # Scaling
                lambda x: -x,  # Reflection
            ]
        else:
            self.transforms = transforms

    def hash_point(self, point: Union[Point2D, Point3D, Tuple, List]) -> bytes:
        """Hash a point using geometric transformations."""
        coords = self._normalize_point(point)

        # Apply all transforms and collect results
        transform_results = []
        for transform in self.transforms:
            transformed = transform(coords)
            # Convert to bytes
            coord_bytes = transformed.tobytes()
            transform_results.append(coord_bytes)

        # Combine all transform results
        combined = b''.join(transform_results)
        return hashlib.sha256(combined).digest()[:self.byte_size]

    def hash_points(self, points: List[Union[Point2D, Point3D, Tuple, List]]) -> bytes:
        """Hash multiple points using geometric transformations."""
        if not points:
            return b'\x00' * self.byte_size

        # Hash each point
        point_hashes = [self.hash_point(p) for p in points]

        # Combine using XOR
        combined = point_hashes[0]
        for ph in point_hashes[1:]:
            combined = bytes(a ^ b for a, b in zip(combined, ph))

        return combined

class RFTGeometricHash(GeometricHash):
    """RFT-enhanced geometric hashing for experimental research applications.
    
    WARNING: This is research code. Cryptographic security is NOT verified.
    """

    def __init__(self, dimensions: int = 2, hash_size: int = 256,
                 rft_size: int = 64):
        """Initialize RFT geometric hash.

        Args:
            dimensions: Number of spatial dimensions
            hash_size: Size of hash output in bits
            rft_size: Size of RFT transform
        """
        super().__init__(dimensions, hash_size)
        self.rft_size = rft_size

        if RFT_AVAILABLE:
            self.rft = UnitaryRFT(rft_size, RFT_FLAG_QUANTUM_SAFE)
        else:
            self.rft = None

    def _apply_rft_transform(self, coords: np.ndarray) -> np.ndarray:
        """Apply RFT transform to coordinates."""
        if self.rft is None:
            # Fallback: return original coordinates
            return coords

        try:
            # Pad or truncate coordinates to RFT size
            if len(coords) < self.rft_size:
                padded = np.pad(coords, (0, self.rft_size - len(coords)))
            else:
                padded = coords[:self.rft_size]

            # Convert to complex for RFT
            complex_coords = padded.astype(np.complex128)

            # Apply RFT transform
            transformed = self.rft.forward(complex_coords)

            # Return real part
            return transformed.real

        except Exception:
            # Fallback to original coordinates
            return coords

    def hash_point(self, point: Union[Point2D, Point3D, Tuple, List]) -> bytes:
        """Hash a point using RFT-enhanced geometric hashing."""
        coords = self._normalize_point(point)

        # Apply RFT transform
        rft_coords = self._apply_rft_transform(coords)

        # Convert to bytes and hash
        coord_bytes = rft_coords.tobytes()
        return hashlib.sha256(coord_bytes).digest()[:self.byte_size]

    def hash_points(self, points: List[Union[Point2D, Point3D, Tuple, List]]) -> bytes:
        """Hash multiple points using RFT-enhanced geometric hashing."""
        if not points:
            return b'\x00' * self.byte_size

        # Hash each point
        point_hashes = [self.hash_point(p) for p in points]

        # Combine using XOR
        combined = point_hashes[0]
        for ph in point_hashes[1:]:
            combined = bytes(a ^ b for a, b in zip(combined, ph))

        return combined

class CollisionDetector:
    """Geometric collision detection using hashing."""

    def __init__(self, hash_algorithm: GeometricHash,
                 cell_size: float = 1.0):
        """Initialize collision detector.

        Args:
            hash_algorithm: Geometric hash algorithm to use
            cell_size: Size of spatial cells for bucketing
        """
        self.hash_algorithm = hash_algorithm
        self.cell_size = cell_size
        self.buckets: Dict[bytes, List] = {}

    def _get_bucket_key(self, point: Union[Point2D, Point3D, Tuple, List]) -> bytes:
        """Get bucket key for a point."""
        # Use spatial hash for bucketing
        spatial_hash = SpatialHash(dimensions=self.hash_algorithm.dimensions,
                                  hash_size=64, grid_size=self.cell_size)
        return spatial_hash.hash_point(point)

    def insert(self, point: Union[Point2D, Point3D, Tuple, List],
               data: Optional[object] = None):
        """Insert a point into the collision detector."""
        bucket_key = self._get_bucket_key(point)

        if bucket_key not in self.buckets:
            self.buckets[bucket_key] = []

        self.buckets[bucket_key].append((point, data))

    def query(self, point: Union[Point2D, Point3D, Tuple, List],
              radius: float = 0.0) -> List[Tuple]:
        """Query points near the given point."""
        bucket_key = self._get_bucket_key(point)
        candidates = self.buckets.get(bucket_key, [])

        # If radius is 0, return all points in the same bucket
        if radius == 0.0:
            return candidates

        # Otherwise, filter by actual distance
        results = []
        query_coords = self.hash_algorithm._normalize_point(point)

        for candidate_point, data in candidates:
            candidate_coords = self.hash_algorithm._normalize_point(candidate_point)
            distance = np.linalg.norm(query_coords - candidate_coords)

            if distance <= radius:
                results.append((candidate_point, data, distance))

        return results

    def clear(self):
        """Clear all points from the collision detector."""
        self.buckets.clear()

# Export all geometric hashing classes
__all__ = [
    'Point2D', 'Point3D',
    'GeometricHash', 'SpatialHash', 'GeometricTransformHash', 'RFTGeometricHash',
    'CollisionDetector'
]
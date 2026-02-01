#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Test Geometric RFT Components
=============================

Tests for GeometricContainer and QuantumSearch.
"""

import unittest
import numpy as np
from algorithms.rft.core.geometric_container import GeometricContainer
from algorithms.rft.quantum_inspired.quantum_search import QuantumSearch

class TestGeometricRFT(unittest.TestCase):

    def test_geometric_container_encoding(self):
        """Test that GeometricContainer can encode and decode data."""
        container = GeometricContainer(id="test_1", capacity_bits=64)
        data = "Hello Quantum"
        container.encode_data(data)
        
        # Check if data is encoded
        self.assertIsNotNone(container.wave_form)
        self.assertEqual(container.encoded_data_len, len(data.encode('utf-8')))
        
        # Check decoding
        decoded = container.get_data()
        # Note: SymbolicWaveComputer decode might not be perfect without error correction 
        # or if capacity is too small, but for small strings it should work.
        # The current implementation of decode_bytes in SWC seems correct for BPSK.
        # However, SWC uses BPSK which is robust.
        
        # If decode fails to match exactly due to float precision/noise in a real system,
        # we might need to relax this, but for the simulation it should be exact.
        self.assertEqual(decoded, data)

    def test_geometric_container_resonance(self):
        """Test resonance frequency checking."""
        container = GeometricContainer(id="test_2", capacity_bits=8)
        container.encode_data("A") # 1 byte = 8 bits
        
        # Check a known resonant frequency
        # f_k = (k + 1) * phi
        phi = (1 + np.sqrt(5)) / 2
        f_0 = 1 * phi
        
        self.assertTrue(container.check_resonance(f_0))
        self.assertFalse(container.check_resonance(f_0 + 0.5))

    def test_quantum_search(self):
        """Test Grover's search simulation."""
        # Create a list of containers
        containers = [GeometricContainer(id=f"c_{i}") for i in range(4)]
        
        # We want to find the one at index 2
        target_index = 2
        
        qs = QuantumSearch()
        
        # The search method returns the found container
        found_container = qs.search(containers, target_index)
        
        self.assertIsNotNone(found_container)
        self.assertEqual(found_container.id, f"c_{target_index}")

if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
RFT Integration Demo - QuantoniumOS
===================================

Demonstrates the full pipeline:
1. Encoding data into GeometricContainers using RFT waveforms.
2. Searching for containers using Classical Resonance (VibrationalEngine).
3. Searching for containers using Quantum Search (Grover's Algorithm).
4. Decoding the data from the found container.
"""

import sys
import os
import numpy as np

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

from algorithms.rft.core import Oscillator, GeometricContainer, Shard, VibrationalEngine
from algorithms.rft.quantum import QuantumSearch

def main():
    print("=" * 60)
    print("QuantoniumOS - RFT Integration Demo")
    print("=" * 60)

    # 1. Setup Data and Containers
    secret_message = "QuantumOS"
    print(f"\n[1] Encoding Data: '{secret_message}'")
    
    # Create a container with enough capacity (bits)
    # "QuantumOS" is 9 bytes = 72 bits. 128 bits is safe.
    container_a = GeometricContainer(id="Container_A", capacity_bits=128)
    container_a.encode_data(secret_message)
    
    # Create a dummy container for search noise
    container_b = GeometricContainer(id="Container_B", capacity_bits=128)
    container_b.encode_data("NoiseData")
    
    print(f"    Container A encoded length: {container_a.encoded_data_len} bytes")
    print(f"    Container A resonant frequencies: {len(container_a.resonant_frequencies)} carriers active")
    
    # 2. Classical Resonance Search (VibrationalEngine)
    print(f"\n[2] Classical Resonance Search")
    
    # We want to find the container that resonates at the frequency of the first bit of our data
    # In RFT, f_0 = PHI. Let's check if Container A resonates at f_0.
    osc = Oscillator(mode=0) # f = PHI
    target_freq = osc.frequency
    print(f"    Target Frequency (Mode 0): {target_freq:.4f} Hz")
    
    vib_engine = VibrationalEngine()
    
    # Check Container A
    data_a = vib_engine.retrieve_data(container_a, target_freq)
    if data_a:
        print(f"    [MATCH] Container A resonates! Data: '{data_a}'")
    else:
        print(f"    [NO MATCH] Container A does not resonate.")
        
    # Check Container B
    data_b = vib_engine.retrieve_data(container_b, target_freq)
    if data_b:
        print(f"    [MATCH] Container B resonates! Data: '{data_b}'")
    else:
        print(f"    [NO MATCH] Container B does not resonate.")

    # 3. Shard Search (Bloom Filter)
    print(f"\n[3] Shard Search (Bloom Filter)")
    shard = Shard([container_a, container_b])
    results = shard.search(target_freq)
    print(f"    Shard found {len(results)} candidates for frequency {target_freq:.4f} Hz")
    for res in results:
        print(f"    - Candidate: {res.id}")

    # 4. Quantum Search (Grover's Algorithm)
    print(f"\n[4] Quantum Search (Grover's Algorithm)")
    
    # We want to find the container that resonates at target_freq among a list
    # Let's add more dummy containers to make it interesting
    containers = [container_a, container_b]
    for i in range(6):
        c = GeometricContainer(id=f"Dummy_{i}", capacity_bits=128)
        c.encode_data(f"Noise_{i}")
        containers.append(c)
        
    print(f"    Searching among {len(containers)} containers...")
    
    # The target is Container A (index 0 in this list)
    # In a real scenario, the Oracle would check the resonance condition.
    # Our QuantumSearch simulation takes the target index to construct the Oracle.
    # Here we simulate finding the index that satisfies the condition.
    target_index = 0 
    
    qs = QuantumSearch()
    found_container = qs.search(containers, target_index)
    
    if found_container:
        print(f"    [SUCCESS] Quantum Search found: {found_container.id}")
        
        # 5. Decode and Verify
        print(f"\n[5] Verification")
        decoded = found_container.get_data()
        print(f"    Decoded Data: '{decoded}'")
        
        if decoded == secret_message:
            print("    STATUS: VERIFIED ✓")
        else:
            print("    STATUS: FAILED ✗")
    else:
        print("    [FAILURE] Quantum Search did not find the container.")

if __name__ == "__main__":
    main()

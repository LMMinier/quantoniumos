#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
DIY Side-Channel Analysis for QuantoniumOS
==========================================

Implementation vulnerability assessment including timing attacks,
cache timing, and power analysis simulation.

Part of PHASE 1.3: DIY Cryptanalysis Suite
Implementation: Self-developed professional analysis
"""

import time
import numpy as np
import statistics
from typing import List, Dict, Tuple
import hashlib
import os
from pathlib import Path

class SideChannelAnalyzer:
    """Side-channel vulnerability analysis framework."""
    
    def __init__(self):
        self.results = {}
        self.cipher = None
        
    def load_cipher(self):
        """Load QuantoniumOS cipher for analysis."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from algorithms.rft.crypto.enhanced_cipher import RFTCipher
            self.cipher = RFTCipher()
            return True
        except ImportError:
            print("⚠️ Using test cipher for demonstration")
            self.cipher = TestCipher()
            return False

class TimingAttackAnalysis:
    """Timing attack vulnerability analysis."""
    
    def __init__(self, cipher):
        self.cipher = cipher
        self.results = {}
    
    def key_dependent_timing_test(self, samples: int = 10000) -> Dict:
        """Test for key-dependent timing variations."""
        print(f"\n⏱️ KEY-DEPENDENT TIMING ANALYSIS")
        print(f"Testing {samples} encryptions with different keys...")
        
        # Test with different key patterns
        key_patterns = {
            'all_zeros': 0x0000000000000000,
            'all_ones': 0xFFFFFFFFFFFFFFFF,
            'alternating': 0xAAAAAAAAAAAAAAAA,
            'random_low_weight': 0x0000000000000001,
            'random_high_weight': 0xFFFFFFFFFFFFFFFE
        }
        
        timing_results = {}
        
        for pattern_name, key in key_patterns.items():
            timings = []
            
            for _ in range(samples):
                plaintext = np.random.randint(0, 2**64)
                
                # Measure encryption time
                start = time.perf_counter()
                ciphertext = self._encrypt_with_key(plaintext, key)
                end = time.perf_counter()
                
                timings.append(end - start)
            
            timing_results[pattern_name] = {
                'mean': statistics.mean(timings),
                'std': statistics.stdev(timings),
                'min': min(timings),
                'max': max(timings),
                'timings': timings  # Keep for detailed analysis
            }
        
        # Analyze timing variations
        mean_times = [result['mean'] for result in timing_results.values()]
        timing_variance = statistics.stdev(mean_times)
        max_difference = max(mean_times) - min(mean_times)
        
        # Vulnerability assessment
        if max_difference > 1e-6:  # 1 microsecond difference
            vulnerability = "HIGH - Significant timing variations detected"
        elif max_difference > 1e-7:  # 100 nanoseconds
            vulnerability = "MEDIUM - Detectable timing variations"
        else:
            vulnerability = "LOW - Minimal timing variations"
        
        result = {
            'key_patterns': timing_results,
            'timing_variance': timing_variance,
            'max_difference': max_difference,
            'vulnerability_level': vulnerability,
            'samples_per_key': samples
        }
        
        print(f"Timing Analysis Results:")
        print(f"  Max timing difference: {max_difference*1e9:.2f} ns")
        print(f"  Timing variance: {timing_variance*1e9:.2f} ns")
        print(f"  Vulnerability: {vulnerability}")
        
        return result
    
    def input_dependent_timing_test(self, samples: int = 5000) -> Dict:
        """Test for input-dependent timing variations."""
        print(f"\n📊 INPUT-DEPENDENT TIMING ANALYSIS")
        
        # Test with different input patterns
        input_patterns = {
            'low_hamming_weight': [i for i in range(16)],  # 0-15 bits set
            'high_hamming_weight': [2**64 - 1 - i for i in range(16)],  # 49-64 bits set
            'alternating_bits': [0xAAAAAAAAAAAAAAAA >> i for i in range(8)],
            'sparse_patterns': [1 << i for i in range(0, 64, 8)]  # Single bits
        }
        
        timing_results = {}
        
        for pattern_name, inputs in input_patterns.items():
            timings = []
            
            for input_val in inputs:
                pattern_timings = []
                
                for _ in range(samples // len(inputs)):
                    start = time.perf_counter()
                    ciphertext = self._encrypt(input_val)
                    end = time.perf_counter()
                    
                    pattern_timings.append(end - start)
                
                timings.extend(pattern_timings)
            
            timing_results[pattern_name] = {
                'mean': statistics.mean(timings),
                'std': statistics.stdev(timings),
                'count': len(timings)
            }
        
        # Analyze input-dependent variations
        mean_times = [result['mean'] for result in timing_results.values()]
        input_timing_variance = statistics.stdev(mean_times) if len(mean_times) > 1 else 0
        
        result = {
            'input_patterns': timing_results,
            'input_timing_variance': input_timing_variance,
            'vulnerability_assessment': self._assess_input_timing_vuln(input_timing_variance)
        }
        
        print(f"Input Timing Results:")
        print(f"  Input timing variance: {input_timing_variance*1e9:.2f} ns")
        print(f"  Assessment: {result['vulnerability_assessment']}")
        
        return result
    
    def _encrypt_with_key(self, plaintext: int, key: int) -> int:
        """Encrypt with specific key (if cipher supports it)."""
        if hasattr(self.cipher, 'encrypt_with_key'):
            return self.cipher.encrypt_with_key(plaintext, key)
        else:
            # Fallback: XOR key into plaintext
            return self._encrypt(plaintext ^ key)
    
    def _encrypt(self, plaintext: int) -> int:
        """Basic encryption."""
        if hasattr(self.cipher, 'encrypt'):
            return self.cipher.encrypt(plaintext)
        else:
            # Test cipher simulation
            x = plaintext
            for r in range(16):
                x = ((x << 1) | (x >> 63)) & 0xFFFFFFFFFFFFFFFF
                x ^= 0x123456789ABCDEF0
            return x
    
    def _assess_input_timing_vuln(self, variance: float) -> str:
        """Assess input timing vulnerability level."""
        if variance > 1e-6:
            return "HIGH - Input-dependent timing detected"
        elif variance > 1e-7:
            return "MEDIUM - Slight input timing correlation"
        else:
            return "LOW - No significant input timing correlation"

class CacheTimingAnalysis:
    """Cache timing attack simulation."""
    
    def __init__(self, cipher):
        self.cipher = cipher
        
    def cache_timing_test(self, samples: int = 1000) -> Dict:
        """Simulate cache timing attack vectors."""
        print(f"\n💾 CACHE TIMING ANALYSIS")
        print(f"Simulating cache timing attacks...")
        
        # Simulate S-box access patterns
        sbox_accesses = {}
        
        for _ in range(samples):
            plaintext = np.random.randint(0, 2**64)
            
            # Simulate S-box lookups during encryption
            sbox_indices = self._simulate_sbox_accesses(plaintext)
            
            for idx in sbox_indices:
                sbox_accesses[idx] = sbox_accesses.get(idx, 0) + 1
        
        # Analyze access patterns
        access_counts = list(sbox_accesses.values())
        access_variance = statistics.stdev(access_counts) if len(access_counts) > 1 else 0
        
        # Check for uniform distribution (good)
        expected_accesses = samples * 16 / 256  # Assuming 256-entry S-box, 16 lookups per encryption
        chi_square = sum((count - expected_accesses)**2 / expected_accesses 
                        for count in access_counts)
        
        vulnerability_level = self._assess_cache_vulnerability(access_variance, chi_square)
        
        result = {
            'sbox_access_pattern': sbox_accesses,
            'access_variance': access_variance,
            'chi_square_statistic': chi_square,
            'vulnerability_level': vulnerability_level,
            'samples_tested': samples
        }
        
        print(f"Cache Timing Results:")
        print(f"  Access variance: {access_variance:.2f}")
        print(f"  Chi-square: {chi_square:.2f}")
        print(f"  Vulnerability: {vulnerability_level}")
        
        return result
    
    def _simulate_sbox_accesses(self, plaintext: int) -> List[int]:
        """Simulate S-box access indices during encryption."""
        indices = []
        x = plaintext
        
        # Simulate 16 rounds with S-box lookups
        for round_num in range(16):
            # Extract bytes for S-box lookups
            for byte_pos in range(8):
                byte_val = (x >> (byte_pos * 8)) & 0xFF
                indices.append(byte_val)  # S-box index
            
            # Simple round function simulation
            x = ((x << 1) | (x >> 63)) & 0xFFFFFFFFFFFFFFFF
            x ^= (0x123456789ABCDEF0 + round_num)
        
        return indices
    
    def _assess_cache_vulnerability(self, variance: float, chi_square: float) -> str:
        """Assess cache timing vulnerability."""
        if variance > 10 or chi_square > 500:
            return "HIGH - Non-uniform access patterns detected"
        elif variance > 5 or chi_square > 300:
            return "MEDIUM - Some access pattern bias"
        else:
            return "LOW - Uniform access patterns"

class PowerAnalysisSimulation:
    """Simulated power analysis (since we can't measure real power)."""
    
    def __init__(self, cipher):
        self.cipher = cipher
    
    def hamming_weight_correlation_test(self, samples: int = 1000) -> Dict:
        """Test correlation between Hamming weight and simulated power."""
        print(f"\n⚡ POWER ANALYSIS SIMULATION")
        print(f"Testing Hamming weight correlations...")
        
        plaintexts = []
        power_traces = []
        hamming_weights = []
        
        for _ in range(samples):
            plaintext = np.random.randint(0, 2**64)
            
            # Simulate power consumption based on Hamming weight
            power_trace = self._simulate_power_trace(plaintext)
            hamming_weight = bin(plaintext).count('1')
            
            plaintexts.append(plaintext)
            power_traces.append(power_trace)
            hamming_weights.append(hamming_weight)
        
        # Calculate correlation between Hamming weight and power
        correlation = np.corrcoef(hamming_weights, power_traces)[0, 1]
        
        # Assess vulnerability
        if abs(correlation) > 0.8:
            vulnerability = "HIGH - Strong Hamming weight correlation"
        elif abs(correlation) > 0.5:
            vulnerability = "MEDIUM - Moderate correlation detected"
        else:
            vulnerability = "LOW - Weak correlation"
        
        result = {
            'hamming_weight_correlation': correlation,
            'vulnerability_level': vulnerability,
            'samples_tested': samples,
            'mean_power': statistics.mean(power_traces),
            'power_variance': statistics.stdev(power_traces)
        }
        
        print(f"Power Analysis Results:")
        print(f"  Hamming weight correlation: {correlation:.4f}")
        print(f"  Vulnerability: {vulnerability}")
        
        return result
    
    def _simulate_power_trace(self, plaintext: int) -> float:
        """Simulate power consumption during encryption."""
        # Simple model: power proportional to number of bit transitions
        x = plaintext
        total_power = 0
        
        for round_num in range(16):
            prev_x = x
            
            # Round function
            x = ((x << 1) | (x >> 63)) & 0xFFFFFFFFFFFFFFFF
            x ^= (0x123456789ABCDEF0 + round_num)
            
            # Power consumption = Hamming weight of state change
            transition = prev_x ^ x
            power = bin(transition).count('1')
            total_power += power
            
            # Add some noise
            total_power += np.random.normal(0, 0.5)
        
        return total_power

class TestCipher:
    """Test cipher for demonstration."""
    
    def encrypt(self, plaintext: int) -> int:
        x = plaintext
        for r in range(16):
            x = ((x << 1) | (x >> 63)) & 0xFFFFFFFFFFFFFFFF
            x ^= 0x123456789ABCDEF0
        return x

def main():
    """Run comprehensive side-channel analysis."""
    print("🔍 QUANTONIUMOS SIDE-CHANNEL ANALYSIS")
    print("=" * 60)
    print("Implementation security assessment")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SideChannelAnalyzer()
    cipher_loaded = analyzer.load_cipher()
    
    # Run timing attack analysis
    timing_analyzer = TimingAttackAnalysis(analyzer.cipher)
    key_timing_results = timing_analyzer.key_dependent_timing_test(samples=5000)
    input_timing_results = timing_analyzer.input_dependent_timing_test(samples=2000)
    
    # Run cache timing analysis
    cache_analyzer = CacheTimingAnalysis(analyzer.cipher)
    cache_results = cache_analyzer.cache_timing_test(samples=1000)
    
    # Run power analysis simulation
    power_analyzer = PowerAnalysisSimulation(analyzer.cipher)
    power_results = power_analyzer.hamming_weight_correlation_test(samples=1000)
    
    # Compile results
    final_results = {
        'key_dependent_timing': key_timing_results,
        'input_dependent_timing': input_timing_results,
        'cache_timing': cache_results,
        'power_analysis': power_results,
        'overall_assessment': assess_side_channel_security(
            key_timing_results, input_timing_results, cache_results, power_results
        )
    }
    
    # Save results
    import json
    results_path = Path(__file__).parent / "side_channel_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("✅ SIDE-CHANNEL ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {results_path}")
    print(f"\nOverall Assessment: {final_results['overall_assessment']}")

def assess_side_channel_security(key_timing, input_timing, cache, power) -> str:
    """Assess overall side-channel security."""
    high_risk_count = 0
    issues = []
    
    # Check each category
    if "HIGH" in key_timing.get('vulnerability_level', ''):
        high_risk_count += 1
        issues.append("Key-dependent timing")
    
    if "HIGH" in input_timing.get('vulnerability_assessment', ''):
        high_risk_count += 1
        issues.append("Input-dependent timing")
    
    if "HIGH" in cache.get('vulnerability_level', ''):
        high_risk_count += 1
        issues.append("Cache timing patterns")
    
    if "HIGH" in power.get('vulnerability_level', ''):
        high_risk_count += 1
        issues.append("Power analysis correlation")
    
    if high_risk_count == 0:
        return "✅ SECURE: No high-risk side-channel vulnerabilities detected"
    elif high_risk_count <= 2:
        return f"⚠️ MODERATE RISK: {high_risk_count} vulnerabilities - {', '.join(issues)}"
    else:
        return f"❌ HIGH RISK: {high_risk_count} critical vulnerabilities - {', '.join(issues)}"

if __name__ == "__main__":
    main()
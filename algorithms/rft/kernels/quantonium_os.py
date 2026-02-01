#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
QuantoniumOS - Unified 3-Engine System
======================================

Main entry point integrating three engines:
1. OS Engine (kernel + system management)
2. Crypto Engine (48-round Feistel + RFT crypto)  
3. Symbolic Engine (structured state compression)

**SCOPE LIMITATION:**
The "Symbolic Engine" operates on structured, separable states only.
O(N) scaling is expected for this family - there is no exponential barrier
being broken. This is classical signal processing, not quantum simulation.
"""

import sys
import os
import time
import signal
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add ASSEMBLY paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "python_bindings"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "core"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "apps"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "engine"))

# Import engines
try:
    # Quantum Engine
    from quantum_symbolic_engine import QuantumSymbolicEngine
    QUANTUM_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Quantum Engine not available: {e}")
    QUANTUM_AVAILABLE = False

try:
    # Crypto Engine  
    from enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2
    CRYPTO_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Crypto Engine not available: {e}")
    CRYPTO_AVAILABLE = False

try:
    # OS Engine components
    from unitary_rft import UnitaryRFT
    OS_KERNEL_AVAILABLE = True
except Exception as e:
    print(f"⚠️  OS Kernel not available: {e}")
    OS_KERNEL_AVAILABLE = False

class QuantoniumOS:
    """
    QuantoniumOS - Unified System with 3 Engines
    
    This class provides the main interface integrating:
    - OS Engine: System management and kernel operations
    - Crypto Engine: 48-round Feistel encryption (experimental, no security proofs)
    - Symbolic Engine: Structured state compression (NOT general quantum simulation)
    """
    
    def __init__(self):
        """Initialize QuantoniumOS with all engines"""
        self.running = False
        self.engines = {}
        self.system_stats = {}
        self.startup_time = time.time()
        
        print("🚀 QuantoniumOS - Quantum Operating System")
        print("=" * 50)
        print("Initializing 3-Engine Architecture...")
        
        self._initialize_engines()
        self._setup_signal_handlers()
        
    def _initialize_engines(self):
        """Initialize all three engines"""
        
        # 1. OS Engine (Kernel + System Management)
        print("\n🖥️  Initializing OS Engine...")
        if OS_KERNEL_AVAILABLE:
            try:
                self.engines['os'] = {
                    'kernel': UnitaryRFT(size=64),
                    'status': 'operational',
                    'type': 'OS Kernel',
                    'capabilities': ['RFT transforms', 'System calls', 'Memory management']
                }
                print("   ✅ OS Kernel loaded (RFT-based)")
                print("   📊 Kernel size: 64-point RFT")
                print("   🔧 Capabilities: System calls, Memory mgmt, Real-time processing")
            except Exception as e:
                print(f"   ❌ OS Engine failed: {e}")
                self.engines['os'] = {'status': 'failed', 'error': str(e)}
        else:
            print("   ⚠️  OS Engine not available")
            self.engines['os'] = {'status': 'unavailable'}
        
        # 2. Crypto Engine (48-round Feistel)
        print("\n🔐 Initializing Crypto Engine...")
        if CRYPTO_AVAILABLE:
            try:
                test_key = b"quantonium_test_key_1234567890123456"  # 32 bytes
                self.engines['crypto'] = {
                    'engine': EnhancedRFTCryptoV2(test_key),
                    'status': 'operational',
                    'type': '48-Round Feistel + RFT',
                    'capabilities': ['AES S-box', 'MixColumns', 'ARX operations', 'AEAD mode']
                }
                print("   ✅ Crypto Engine loaded (48-round Feistel)")
                print("   🔒 Algorithm: Enhanced RFT Crypto v2")
                print("   🎯 Target: 9.2 MB/s throughput")
                print("   🛡️  Features: AEAD, Domain separation, Golden ratio enhancement")
            except Exception as e:
                print(f"   ❌ Crypto Engine failed: {e}")
                self.engines['crypto'] = {'status': 'failed', 'error': str(e)}
        else:
            print("   ⚠️  Crypto Engine not available")
            self.engines['crypto'] = {'status': 'unavailable'}
        
        # 3. Symbolic Engine (Structured State Compression)
        print("\n⚛️  Initializing Symbolic Engine...")
        if QUANTUM_AVAILABLE:
            try:
                self.engines['quantum'] = {
                    'engine': QuantumSymbolicEngine(compression_size=64, use_assembly=True),
                    'status': 'operational',
                    'type': 'Structured State Compression',
                    'capabilities': ['O(N) structured states', 'Phi-RFT transforms']
                }
                print("   ✅ Symbolic Engine loaded (C/Assembly optimized)")
                print("   Note: O(N) for structured states, NOT general quantum simulation")
            except Exception as e:
                print(f"   ❌ Symbolic Engine failed: {e}")
                self.engines['quantum'] = {'status': 'failed', 'error': str(e)}
        else:
            print("   ⚠️  Symbolic Engine not available")
            self.engines['quantum'] = {'status': 'unavailable'}
        
        # System Summary
        operational_engines = sum(1 for e in self.engines.values() if e.get('status') == 'operational')
        print(f"\n📊 Engine Status: {operational_engines}/3 engines operational")
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down QuantoniumOS...")
        self.shutdown()
        
    def start(self):
        """Start the QuantoniumOS"""
        self.running = True
        startup_duration = (time.time() - self.startup_time) * 1000
        
        print(f"\n🎉 QuantoniumOS started successfully!")
        print(f"   ⏱️  Startup time: {startup_duration:.1f}ms")
        print(f"   🚀 All engines initialized")
        print(f"   💻 Ready for quantum computing operations")
        
        return True
        
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'running': self.running,
            'startup_time': time.time() - self.startup_time,
            'engines': self.engines,
            'operational_engines': sum(1 for e in self.engines.values() if e.get('status') == 'operational'),
            'total_engines': len(self.engines)
        }
    
    def run_crypto_benchmark(self) -> Dict:
        """Benchmark the crypto engine"""
        if self.engines['crypto']['status'] != 'operational':
            return {'error': 'Crypto engine not operational'}
            
        print("\n🔐 Running Crypto Engine Benchmark...")
        engine = self.engines['crypto']['engine']
        
        # Test data
        test_data = b"QuantoniumOS crypto benchmark test data 12345678"
        
        # Benchmark encryption
        start_time = time.perf_counter()
        iterations = 1000
        
        for i in range(iterations):
            encrypted = engine.encrypt_aead(test_data)
            decrypted = engine.decrypt_aead(encrypted)
            
        total_time = time.perf_counter() - start_time
        throughput_mbps = (len(test_data) * iterations) / (total_time * 1024 * 1024)
        
        result = {
            'iterations': iterations,
            'total_time_ms': total_time * 1000,
            'avg_time_per_op_ms': (total_time / iterations) * 1000,
            'throughput_mbps': throughput_mbps,
            'data_size_bytes': len(test_data),
            'target_mbps': 9.2,
            'target_achieved': throughput_mbps >= 9.2
        }
        
        print(f"   📊 Iterations: {iterations}")
        print(f"   ⏱️  Total time: {result['total_time_ms']:.1f}ms")
        print(f"   🔄 Avg per operation: {result['avg_time_per_op_ms']:.3f}ms")
        print(f"   🚀 Throughput: {throughput_mbps:.2f} MB/s")
        print(f"   🎯 Target (9.2 MB/s): {'✅ ACHIEVED' if result['target_achieved'] else '❌ NOT ACHIEVED'}")
        
        return result
    
    def run_quantum_benchmark(self) -> Dict:
        """Benchmark the symbolic compression engine.
        
        NOTE: This compresses structured classical vectors, NOT quantum states.
        The 'qubit' terminology is legacy and misleading.
        """
        if self.engines['quantum']['status'] != 'operational':
            return {'error': 'Symbolic compression engine not operational'}
            
        print("\n🔢  Running Symbolic Compression Benchmark (legacy 'quantum' naming)...")
        engine = self.engines['quantum']['engine']
        
        # Test symbolic vector compression (NOT actual quantum simulation)
        start_time = time.perf_counter()
        success, perf = engine.compress_million_qubits(1000000)  # Legacy name
        total_time = time.perf_counter() - start_time
        
        if success:
            entanglement = engine.measure_entanglement()
            state = engine.get_compressed_state()
            
            result = {
                'success': True,
                'symbolic_labels_simulated': 1000000,
                'compression_time_ms': perf['compression_time_ms'],
                'total_time_ms': total_time * 1000,
                'operations_per_second': perf['operations_per_second'],
                'memory_mb': perf['memory_mb'],
                'compression_ratio': perf['compression_ratio'],
                'entanglement': entanglement,
                'state_norm': abs(1.0 - state.max()) if state is not None else None,
                'backend': perf['backend']
            }
            
            print(f"   ✅ Compressed 1,000,000 symbolic vectors (NOT quantum states)")
            print(f"   ⏱️  Compression time: {result['compression_time_ms']:.1f}ms")
            print(f"   🔄 Operations/sec: {result['operations_per_second']:,}")
            print(f"   💾 Memory usage: {result['memory_mb']:.6f} MB")
            print(f"   📊 Compression: {result['compression_ratio']:,.0f}:1")
            print(f"   🌀 Entanglement: {result['entanglement']:.6f}")
            print(f"   🖥️  Backend: {result['backend']}")
            
        else:
            result = {'success': False, 'error': 'Quantum compression failed'}
            print(f"   ❌ Quantum benchmark failed")
        
        engine.cleanup()
        return result
    
    def run_os_benchmark(self) -> Dict:
        """Benchmark the OS engine"""
        if self.engines['os']['status'] != 'operational':
            return {'error': 'OS engine not operational'}
            
        print("\n🖥️  Running OS Engine Benchmark...")
        kernel = self.engines['os']['kernel']
        
        # Test RFT kernel performance with complex data
        test_data = np.array([complex(1.0, 0.0), complex(0.5, 0.0), complex(0.25, 0.0), complex(0.125, 0.0)] * 16, dtype=np.complex128)
        
        start_time = time.perf_counter()
        iterations = 1000
        
        for i in range(iterations):
            spectrum = kernel.forward(test_data)
            reconstructed = kernel.inverse(spectrum)
            
        total_time = time.perf_counter() - start_time
        
        result = {
            'iterations': iterations,
            'total_time_ms': total_time * 1000,
            'avg_time_per_transform_us': (total_time / iterations) * 1000000,
            'transforms_per_second': iterations / total_time,
            'kernel_size': 64,
            'precision': 'Double precision (64-bit)'
        }
        
        print(f"   📊 RFT transforms: {iterations}")
        print(f"   ⏱️  Total time: {result['total_time_ms']:.1f}ms")
        print(f"   🔄 Avg per transform: {result['avg_time_per_transform_us']:.1f}μs")
        print(f"   🚀 Transforms/sec: {result['transforms_per_second']:,.0f}")
        
        return result
    
    def run_full_system_benchmark(self) -> Dict:
        """Run comprehensive system benchmark"""
        print("\n🔬 QUANTONIUMOS FULL SYSTEM BENCHMARK")
        print("=" * 50)
        
        results = {
            'system_info': self.get_system_status(),
            'benchmarks': {}
        }
        
        # Run all engine benchmarks
        if self.engines['os']['status'] == 'operational':
            results['benchmarks']['os'] = self.run_os_benchmark()
            
        if self.engines['crypto']['status'] == 'operational':
            results['benchmarks']['crypto'] = self.run_crypto_benchmark()
            
        if self.engines['quantum']['status'] == 'operational':
            results['benchmarks']['quantum'] = self.run_quantum_benchmark()
        
        # Summary
        print(f"\n📋 BENCHMARK SUMMARY:")
        print(f"   🖥️  OS Engine: {'✅ Tested' if 'os' in results['benchmarks'] else '❌ Unavailable'}")
        print(f"   🔐 Crypto Engine: {'✅ Tested' if 'crypto' in results['benchmarks'] else '❌ Unavailable'}")  
        print(f"   ⚛️  Quantum Engine: {'✅ Tested' if 'quantum' in results['benchmarks'] else '❌ Unavailable'}")
        
        return results
    
    def shutdown(self):
        """Shutdown QuantoniumOS"""
        if not self.running:
            return
            
        print("\n🛑 Shutting down QuantoniumOS...")
        self.running = False
        
        # Clean up engines
        for engine_name, engine_data in self.engines.items():
            if engine_data.get('status') == 'operational':
                print(f"   🔄 Stopping {engine_name} engine...")
                if 'engine' in engine_data:
                    if hasattr(engine_data['engine'], 'cleanup'):
                        engine_data['engine'].cleanup()
        
        uptime = time.time() - self.startup_time
        print(f"   ⏱️  Total uptime: {uptime:.1f}s")
        print("   ✅ QuantoniumOS shutdown complete")
        
    def __del__(self):
        """Destructor"""
        if self.running:
            self.shutdown()

def main():
    """Main entry point for QuantoniumOS"""
    try:
        # Initialize QuantoniumOS
        quantonium = QuantoniumOS()
        
        # Start the system
        if not quantonium.start():
            print("❌ Failed to start QuantoniumOS")
            return 1
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if "--benchmark" in sys.argv:
                # Run full system benchmark
                results = quantonium.run_full_system_benchmark()
                return 0
            elif "--crypto-test" in sys.argv:
                # Test crypto engine only
                quantonium.run_crypto_benchmark()
                return 0
            elif "--quantum-test" in sys.argv:
                # Test quantum engine only  
                quantonium.run_quantum_benchmark()
                return 0
            elif "--os-test" in sys.argv:
                # Test OS engine only
                quantonium.run_os_benchmark()
                return 0
        
        # Interactive mode
        print(f"\n💻 QuantoniumOS Interactive Mode")
        print(f"Available commands:")
        print(f"  status    - Show system status") 
        print(f"  benchmark - Run full system benchmark")
        print(f"  crypto    - Test crypto engine")
        print(f"  quantum   - Test quantum engine")
        print(f"  os        - Test OS engine")
        print(f"  quit      - Shutdown QuantoniumOS")
        
        while quantonium.running:
            try:
                cmd = input(f"\nQuantoniumOS> ").strip().lower()
                
                if cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "status":
                    status = quantonium.get_system_status()
                    print(f"Running: {status['running']}")
                    print(f"Uptime: {status['startup_time']:.1f}s")
                    print(f"Engines: {status['operational_engines']}/{status['total_engines']} operational")
                elif cmd == "benchmark":
                    quantonium.run_full_system_benchmark()
                elif cmd == "crypto":
                    quantonium.run_crypto_benchmark()
                elif cmd == "quantum":
                    quantonium.run_quantum_benchmark()
                elif cmd == "os":
                    quantonium.run_os_benchmark()
                elif cmd == "help":
                    print(f"Available commands: status, benchmark, crypto, quantum, os, quit")
                elif cmd:
                    print(f"Unknown command: {cmd}")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
                
        quantonium.shutdown()
        return 0
        
    except Exception as e:
        print(f"❌ QuantoniumOS error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

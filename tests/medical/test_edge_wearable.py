#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# MEDICAL RESEARCH LICENSE:
# FREE for hospitals, medical researchers, academics, and healthcare
# institutions for testing, validation, and research purposes.
# Commercial medical device use: See LICENSE-CLAIMS-NC.md
#
"""
Edge/Wearable Device Tests
===========================

Tests RFT performance characteristics for edge and wearable medical devices:
- On-device latency for real-time processing
- Memory footprint estimation
- Battery impact simulation (compute intensity)
- Packet loss resilience with forward error correction
- Streaming/chunked processing for continuous monitoring

Simulates constraints of common MCU/SoC platforms (Cortex-M4, ESP32, etc.)
"""

import numpy as np
import time
import pytest
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import struct
import zlib


# =============================================================================
# Device Simulation Parameters
# =============================================================================

@dataclass
class DeviceProfile:
    """Simulated device capability profile."""
    name: str
    cpu_mhz: int
    ram_kb: int
    flash_kb: int
    power_mw_active: float
    power_mw_sleep: float
    target_latency_ms: float


# Common embedded device profiles
DEVICE_PROFILES = {
    'cortex_m4': DeviceProfile(
        name='ARM Cortex-M4 (STM32F4)',
        cpu_mhz=168,
        ram_kb=192,
        flash_kb=1024,
        power_mw_active=150,
        power_mw_sleep=0.5,
        target_latency_ms=50
    ),
    'esp32': DeviceProfile(
        name='ESP32',
        cpu_mhz=240,
        ram_kb=520,
        flash_kb=4096,
        power_mw_active=250,
        power_mw_sleep=10,
        target_latency_ms=100
    ),
    'nrf52': DeviceProfile(
        name='Nordic nRF52840',
        cpu_mhz=64,
        ram_kb=256,
        flash_kb=1024,
        power_mw_active=50,
        power_mw_sleep=0.3,
        target_latency_ms=100
    ),
    'rpi_pico': DeviceProfile(
        name='Raspberry Pi Pico',
        cpu_mhz=133,
        ram_kb=264,
        flash_kb=2048,
        power_mw_active=100,
        power_mw_sleep=1,
        target_latency_ms=50
    ),
}


# =============================================================================
# Memory Footprint Estimation
# =============================================================================

def estimate_memory_footprint(signal_length: int, 
                              precision: str = 'float32',
                              n_buffers: int = 3) -> Dict[str, int]:
    """
    Estimate memory footprint for RFT processing.
    
    Args:
        signal_length: Length of signal buffer
        precision: 'float32' or 'float64'
        n_buffers: Number of concurrent buffers needed
        
    Returns:
        Memory breakdown in bytes
    """
    bytes_per_sample = 4 if precision == 'float32' else 8
    
    # Input buffer
    input_buffer = signal_length * bytes_per_sample
    
    # Complex coefficient buffer (2x for real/imag)
    coeff_buffer = signal_length * bytes_per_sample * 2
    
    # Twiddle factors (precomputed)
    twiddle_factors = signal_length * bytes_per_sample * 2
    
    # Working buffers
    working_memory = n_buffers * signal_length * bytes_per_sample
    
    # Output buffer
    output_buffer = signal_length * bytes_per_sample
    
    total = input_buffer + coeff_buffer + twiddle_factors + working_memory + output_buffer
    
    return {
        'input_buffer': input_buffer,
        'coefficient_buffer': coeff_buffer,
        'twiddle_factors': twiddle_factors,
        'working_memory': working_memory,
        'output_buffer': output_buffer,
        'total_bytes': total,
        'total_kb': total / 1024
    }


def check_device_memory_fit(signal_length: int, 
                             device: DeviceProfile,
                             precision: str = 'float32') -> Tuple[bool, Dict]:
    """
    Check if processing fits in device RAM.
    
    Args:
        signal_length: Signal buffer size
        device: Target device profile
        precision: Numeric precision
        
    Returns:
        (fits, memory_breakdown)
    """
    footprint = estimate_memory_footprint(signal_length, precision)
    available_kb = device.ram_kb * 0.7  # Reserve 30% for stack/OS
    
    fits = footprint['total_kb'] < available_kb
    
    footprint['available_kb'] = available_kb
    footprint['utilization'] = footprint['total_kb'] / available_kb
    
    return fits, footprint


# =============================================================================
# Latency Benchmarking
# =============================================================================

def benchmark_rft_latency(signal_length: int,
                          n_iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark RFT processing latency.
    
    Args:
        signal_length: Input signal length
        n_iterations: Number of timing iterations
        
    Returns:
        Latency statistics
    """
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import (
            rft_forward,
            rft_inverse,
        )
    except ImportError:
        pytest.skip("RFT not available")
    
    # Generate test signal
    signal = np.random.randn(signal_length).astype(np.complex128)
    
    # Warm-up
    for _ in range(5):
        coeffs = rft_forward(signal)
        _ = rft_inverse(coeffs)
    
    # Benchmark forward transform
    forward_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        coeffs = rft_forward(signal)
        forward_times.append((time.perf_counter() - t0) * 1000)
    
    # Benchmark inverse transform
    inverse_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = rft_inverse(coeffs)
        inverse_times.append((time.perf_counter() - t0) * 1000)
    
    # Benchmark roundtrip
    roundtrip_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        coeffs = rft_forward(signal)
        _ = rft_inverse(coeffs)
        roundtrip_times.append((time.perf_counter() - t0) * 1000)
    
    return {
        'forward_mean_ms': np.mean(forward_times),
        'forward_std_ms': np.std(forward_times),
        'forward_max_ms': np.max(forward_times),
        'inverse_mean_ms': np.mean(inverse_times),
        'inverse_std_ms': np.std(inverse_times),
        'roundtrip_mean_ms': np.mean(roundtrip_times),
        'roundtrip_p99_ms': np.percentile(roundtrip_times, 99),
        'samples_per_ms': signal_length / np.mean(roundtrip_times)
    }


def simulate_embedded_latency(host_latency_ms: float,
                              host_cpu_mhz: float,
                              target_cpu_mhz: float) -> float:
    """
    Estimate latency on embedded device based on CPU clock ratio.
    
    This is a rough approximation; actual performance varies.
    
    Args:
        host_latency_ms: Measured latency on host
        host_cpu_mhz: Host CPU frequency
        target_cpu_mhz: Target device CPU frequency
        
    Returns:
        Estimated latency on target device
    """
    # Assume linear scaling (optimistic for cache effects)
    ratio = host_cpu_mhz / target_cpu_mhz
    return host_latency_ms * ratio


# =============================================================================
# Power/Battery Estimation
# =============================================================================

def estimate_battery_impact(processing_time_ms: float,
                            device: DeviceProfile,
                            duty_cycle: float = 1.0) -> Dict[str, float]:
    """
    Estimate battery impact of continuous processing.
    
    Args:
        processing_time_ms: Processing time per sample window
        device: Target device profile
        duty_cycle: Fraction of time spent processing (0-1)
        
    Returns:
        Power and battery estimates
    """
    # Average power
    avg_power_mw = (duty_cycle * device.power_mw_active + 
                   (1 - duty_cycle) * device.power_mw_sleep)
    
    # Common battery capacities (mAh at 3.7V nominal)
    battery_mah = 200  # Small wearable (CR2032 ~200mAh, coin cell ~40mAh)
    battery_wh = battery_mah * 3.7 / 1000
    
    # Battery life
    hours = (battery_wh * 1000) / avg_power_mw
    
    # Processing energy per operation
    energy_per_op_mj = device.power_mw_active * processing_time_ms / 1000
    
    return {
        'average_power_mw': avg_power_mw,
        'battery_life_hours': hours,
        'battery_life_days': hours / 24,
        'energy_per_operation_mj': energy_per_op_mj,
        'operations_per_mwh': (1000 * 3600) / (energy_per_op_mj if energy_per_op_mj > 0 else 1)
    }


# =============================================================================
# Packet Loss Resilience
# =============================================================================

def simulate_packet_transmission(data: bytes,
                                 packet_size: int = 20,
                                 loss_rate: float = 0.0) -> List[Optional[bytes]]:
    """
    Simulate packetized transmission with potential losses.
    
    Args:
        data: Raw data bytes
        packet_size: Size of each packet (BLE MTU is typically 20-244 bytes)
        loss_rate: Probability of packet loss
        
    Returns:
        List of received packets (None for lost packets)
    """
    packets = []
    for i in range(0, len(data), packet_size):
        chunk = data[i:i + packet_size]
        if np.random.random() < loss_rate:
            packets.append(None)  # Lost
        else:
            packets.append(chunk)
    
    return packets


def add_fec_redundancy(data: bytes, 
                       redundancy_factor: float = 0.25) -> bytes:
    """
    Add simple forward error correction redundancy.
    
    Uses duplication + CRC for simplicity. Real FEC would use Reed-Solomon etc.
    
    Args:
        data: Original data
        redundancy_factor: Extra data fraction (0.25 = 25% overhead)
        
    Returns:
        Data with FEC protection
    """
    # Simple scheme: prepend length and CRC, add parity bytes
    length = len(data)
    crc = zlib.crc32(data) & 0xFFFFFFFF
    
    header = struct.pack('<I I', length, crc)
    
    # Add XOR parity every N bytes
    n = int(1 / redundancy_factor) if redundancy_factor > 0 else len(data)
    parity_bytes = bytearray()
    
    for i in range(0, len(data), n):
        block = data[i:i + n]
        parity = 0
        for b in block:
            parity ^= b
        parity_bytes.append(parity)
    
    return header + data + bytes(parity_bytes)


def decode_with_fec(received_packets: List[Optional[bytes]],
                    original_length: int) -> Tuple[Optional[bytes], Dict]:
    """
    Attempt to decode data from potentially lossy packets.
    
    Args:
        received_packets: List of packets (None for lost)
        original_length: Expected data length
        
    Returns:
        (decoded_data or None, stats)
    """
    # Reassemble received data
    received = b''.join(p for p in received_packets if p is not None)
    
    lost_count = sum(1 for p in received_packets if p is None)
    total_count = len(received_packets)
    
    stats = {
        'total_packets': total_count,
        'lost_packets': lost_count,
        'loss_rate': lost_count / total_count if total_count > 0 else 0,
        'received_bytes': len(received),
        'expected_bytes': original_length
    }
    
    # Check if we have enough data
    if len(received) < 8:  # Need at least header
        stats['status'] = 'insufficient_data'
        return None, stats
    
    # Parse header
    try:
        length, crc = struct.unpack('<I I', received[:8])
        data = received[8:8 + length]
        
        if len(data) < length:
            stats['status'] = 'incomplete'
            return None, stats
        
        # Verify CRC
        actual_crc = zlib.crc32(data) & 0xFFFFFFFF
        if actual_crc != crc:
            stats['status'] = 'crc_mismatch'
            return None, stats
        
        stats['status'] = 'success'
        return data, stats
        
    except Exception as e:
        stats['status'] = f'decode_error: {e}'
        return None, stats


def rft_compress_for_transmission(signal: np.ndarray,
                                   keep_ratio: float = 0.3) -> bytes:
    """
    Compress signal using RFT for efficient transmission.
    
    Args:
        signal: Input signal
        keep_ratio: Coefficient retention ratio
        
    Returns:
        Compressed bytes
    """
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import rft_forward
    except ImportError:
        pytest.skip("RFT not available")
    
    # RFT transform
    coeffs = rft_forward(signal.astype(np.complex128))
    
    # Keep top coefficients
    magnitudes = np.abs(coeffs)
    n_keep = int(keep_ratio * len(coeffs))
    indices = np.argsort(magnitudes)[-n_keep:]
    
    # Pack sparse representation
    # Format: [n_total(4B)][n_kept(4B)][indices(4B each)][values(16B each)]
    header = struct.pack('<I I', len(coeffs), n_keep)
    index_bytes = indices.astype(np.int32).tobytes()
    value_bytes = coeffs[indices].tobytes()
    
    return header + index_bytes + value_bytes


def rft_decompress_from_transmission(data: bytes) -> np.ndarray:
    """
    Decompress RFT-encoded signal.
    
    Args:
        data: Compressed bytes
        
    Returns:
        Reconstructed signal
    """
    try:
        from algorithms.rft.kernels.operator_eigenbasis_rft import rft_inverse
    except ImportError:
        pytest.skip("RFT not available")
    
    # Parse header
    n_total, n_kept = struct.unpack('<I I', data[:8])
    
    # Parse indices
    index_start = 8
    index_end = index_start + n_kept * 4
    indices = np.frombuffer(data[index_start:index_end], dtype=np.int32)
    
    # Parse values
    values = np.frombuffer(data[index_end:], dtype=np.complex128)
    
    # Reconstruct sparse coefficients
    coeffs = np.zeros(n_total, dtype=np.complex128)
    coeffs[indices] = values
    
    # Inverse transform
    return rft_inverse(coeffs).real


# =============================================================================
# Streaming Processing
# =============================================================================

class StreamingRFTProcessor:
    """
    Streaming RFT processor for continuous monitoring.
    
    Processes data in overlapping chunks to maintain continuity.
    """
    
    def __init__(self, chunk_size: int = 256, 
                 overlap_ratio: float = 0.5,
                 keep_ratio: float = 0.5):
        """
        Initialize streaming processor.
        
        Args:
            chunk_size: Size of processing chunks
            overlap_ratio: Overlap between consecutive chunks (0-0.75)
            keep_ratio: Coefficient retention for compression
        """
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)
        self.hop = chunk_size - self.overlap
        self.keep_ratio = keep_ratio
        
        self.buffer = np.zeros(chunk_size)
        self.buffer_fill = 0
        
    def process_samples(self, samples: np.ndarray) -> List[np.ndarray]:
        """
        Process incoming samples.
        
        Args:
            samples: New samples to process
            
        Returns:
            List of processed chunks (may be empty if not enough data)
        """
        try:
            from algorithms.rft.kernels.operator_eigenbasis_rft import (
                rft_forward,
                rft_inverse,
            )
        except ImportError:
            pytest.skip("RFT not available")
        
        outputs = []
        
        for sample in samples:
            self.buffer[self.buffer_fill] = sample
            self.buffer_fill += 1
            
            if self.buffer_fill >= self.chunk_size:
                # Process full chunk
                coeffs = rft_forward(self.buffer.astype(np.complex128))
                
                # Threshold (compression)
                n_keep = int(self.keep_ratio * len(coeffs))
                magnitudes = np.abs(coeffs)
                threshold = np.sort(magnitudes)[-n_keep] if n_keep > 0 else 0
                compressed = np.where(magnitudes >= threshold, coeffs, 0)
                
                # Reconstruct
                output = rft_inverse(compressed).real
                outputs.append(output)
                
                # Shift buffer (keep overlap portion)
                self.buffer[:self.overlap] = self.buffer[-self.overlap:]
                self.buffer_fill = self.overlap
        
        return outputs
    
    def flush(self) -> Optional[np.ndarray]:
        """Process remaining buffer content."""
        if self.buffer_fill > 0:
            # Zero-pad to chunk size
            chunk = np.zeros(self.chunk_size)
            chunk[:self.buffer_fill] = self.buffer[:self.buffer_fill]
            
            try:
                from algorithms.rft.kernels.operator_eigenbasis_rft import (
                    rft_forward,
                    rft_inverse,
                )
                coeffs = rft_forward(chunk.astype(np.complex128))
                return rft_inverse(coeffs).real[:self.buffer_fill]
            except ImportError:
                return chunk[:self.buffer_fill]
        
        return None


# =============================================================================
# Test Data Structures
# =============================================================================

@dataclass
class EdgeTestResult:
    """Result container for edge device tests."""
    test_name: str
    device: str
    metric: str
    value: float
    target: float
    passed: bool


# =============================================================================
# Pytest Test Cases
# =============================================================================

class TestMemoryFootprint:
    """Test suite for memory footprint estimation."""
    
    @pytest.mark.parametrize("signal_length", [64, 128, 256, 512])
    def test_memory_estimation(self, signal_length):
        """Test memory footprint calculation."""
        footprint = estimate_memory_footprint(signal_length, 'float32')
        
        print(f"\n  Signal length {signal_length}:")
        print(f"    Total: {footprint['total_kb']:.2f} KB")
        
        # Sanity check
        assert footprint['total_bytes'] > 0
        assert footprint['total_kb'] < 1000  # Should be reasonable
    
    @pytest.mark.parametrize("device_name", ['cortex_m4', 'esp32', 'nrf52'])
    def test_device_fit(self, device_name):
        """Test if processing fits on various devices."""
        device = DEVICE_PROFILES[device_name]
        
        results = []
        for signal_length in [64, 128, 256, 512, 1024]:
            fits, footprint = check_device_memory_fit(signal_length, device)
            results.append((signal_length, fits, footprint['utilization']))
        
        print(f"\n  {device.name} memory fit:")
        for length, fits, util in results:
            status = "✓" if fits else "✗"
            print(f"    {status} {length} samples: {util:.1%} RAM")
        
        # At least small buffers should fit
        assert results[0][1], f"Even 64 samples doesn't fit on {device_name}"


class TestLatency:
    """Test suite for processing latency."""
    
    @pytest.mark.parametrize("signal_length", [64, 128, 256, 512])
    def test_latency_measurement(self, signal_length):
        """Measure RFT latency for various sizes."""
        stats = benchmark_rft_latency(signal_length, n_iterations=50)
        
        print(f"\n  Latency ({signal_length} samples):")
        print(f"    Forward: {stats['forward_mean_ms']:.3f} ± {stats['forward_std_ms']:.3f} ms")
        print(f"    Roundtrip: {stats['roundtrip_mean_ms']:.3f} ms (P99: {stats['roundtrip_p99_ms']:.3f} ms)")
        
        # Should be reasonably fast on host
        assert stats['roundtrip_mean_ms'] < 100, "Processing too slow"
    
    def test_embedded_latency_estimation(self):
        """Estimate latency on embedded devices."""
        # Measure on host
        host_stats = benchmark_rft_latency(256, n_iterations=30)
        host_cpu_mhz = 3000  # Approximate host CPU
        
        print("\n  Estimated embedded latency (256 samples):")
        for device_name, device in DEVICE_PROFILES.items():
            estimated = simulate_embedded_latency(
                host_stats['roundtrip_mean_ms'],
                host_cpu_mhz,
                device.cpu_mhz
            )
            meets_target = estimated < device.target_latency_ms
            status = "✓" if meets_target else "✗"
            print(f"    {status} {device.name}: ~{estimated:.1f}ms (target: {device.target_latency_ms}ms)")


class TestBatteryImpact:
    """Test suite for battery/power estimation."""
    
    def test_power_estimation(self):
        """Estimate power consumption for continuous monitoring."""
        # Simulate continuous ECG monitoring at 360 Hz
        sample_rate = 360
        chunk_size = 256
        
        # Measure processing time
        stats = benchmark_rft_latency(chunk_size, n_iterations=30)
        processing_time_ms = stats['roundtrip_mean_ms']
        
        # Calculate duty cycle for real-time processing
        # Time to collect one chunk: chunk_size / sample_rate seconds
        collection_time_ms = (chunk_size / sample_rate) * 1000
        duty_cycle = processing_time_ms / collection_time_ms
        
        print("\n  Power estimation (continuous ECG at 360 Hz):")
        print(f"    Processing time: {processing_time_ms:.2f} ms")
        print(f"    Collection time: {collection_time_ms:.2f} ms")
        print(f"    Duty cycle: {duty_cycle:.1%}")
        
        for device_name, device in DEVICE_PROFILES.items():
            power = estimate_battery_impact(processing_time_ms, device, duty_cycle)
            print(f"    {device.name}:")
            print(f"      Battery life: {power['battery_life_days']:.1f} days")


class TestPacketLoss:
    """Test suite for packet loss resilience."""
    
    @pytest.mark.parametrize("loss_rate", [0.0, 0.05, 0.10, 0.20])
    def test_transmission_resilience(self, loss_rate):
        """Test data recovery under packet loss."""
        # Generate and compress signal
        signal = np.random.randn(256)
        compressed = rft_compress_for_transmission(signal, keep_ratio=0.3)
        
        # Add FEC
        protected = add_fec_redundancy(compressed, redundancy_factor=0.25)
        
        # Simulate transmission
        packets = simulate_packet_transmission(protected, packet_size=20, loss_rate=loss_rate)
        
        # Attempt recovery
        decoded, stats = decode_with_fec(packets, len(protected))
        
        print(f"\n  Packet loss {loss_rate:.0%}:")
        print(f"    Lost: {stats['lost_packets']}/{stats['total_packets']} packets")
        print(f"    Status: {stats['status']}")
        
        if loss_rate == 0:
            assert stats['status'] == 'success', "Should succeed with no loss"
    
    def test_compression_for_transmission(self):
        """Test RFT compression for efficient transmission."""
        signal = np.random.randn(512)
        
        for keep_ratio in [0.2, 0.3, 0.5]:
            compressed = rft_compress_for_transmission(signal, keep_ratio)
            decompressed = rft_decompress_from_transmission(compressed)
            
            # Quality check
            error = np.max(np.abs(signal - decompressed))
            compression_ratio = len(signal) * 8 / len(compressed)  # Assuming float64 input
            
            print(f"\n  Transmission compression (keep={keep_ratio}):")
            print(f"    Compressed size: {len(compressed)} bytes")
            print(f"    Compression ratio: {compression_ratio:.2f}x")
            print(f"    Max error: {error:.4f}")


class TestStreaming:
    """Test suite for streaming processing."""
    
    def test_streaming_processor(self):
        """Test streaming RFT processor."""
        processor = StreamingRFTProcessor(
            chunk_size=256,
            overlap_ratio=0.5,
            keep_ratio=0.5
        )
        
        # Simulate streaming data (e.g., 10 seconds of ECG at 360 Hz)
        total_samples = 3600
        signal = np.sin(2 * np.pi * 1 * np.arange(total_samples) / 360)  # 1 Hz sine
        signal += 0.1 * np.random.randn(total_samples)  # Add noise
        
        # Process in small batches (simulating real-time arrival)
        batch_size = 100
        all_outputs = []
        
        t0 = time.perf_counter()
        for i in range(0, total_samples, batch_size):
            batch = signal[i:i + batch_size]
            outputs = processor.process_samples(batch)
            all_outputs.extend(outputs)
        
        # Flush remaining
        final = processor.flush()
        if final is not None:
            all_outputs.append(final)
        
        elapsed = time.perf_counter() - t0
        
        print(f"\n  Streaming processing (3600 samples):")
        print(f"    Total time: {elapsed * 1000:.1f} ms")
        print(f"    Output chunks: {len(all_outputs)}")
        print(f"    Throughput: {total_samples / elapsed:.0f} samples/s")
        
        # Should be faster than real-time
        assert elapsed < 10, "Streaming too slow"
    
    def test_streaming_latency(self):
        """Measure per-chunk latency in streaming mode."""
        processor = StreamingRFTProcessor(chunk_size=256, overlap_ratio=0.5)
        
        # Pre-fill buffer
        _ = processor.process_samples(np.random.randn(256))
        
        # Measure latency for subsequent chunks
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = processor.process_samples(np.random.randn(128))  # Half chunk (hop size)
            latencies.append((time.perf_counter() - t0) * 1000)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        print(f"\n  Streaming latency (256-sample chunks):")
        print(f"    Average: {avg_latency:.2f} ms")
        print(f"    Maximum: {max_latency:.2f} ms")


# =============================================================================
# Standalone Runner
# =============================================================================

def run_comprehensive_edge_benchmark():
    """Run comprehensive edge device benchmark."""
    print("=" * 70)
    print("EDGE/WEARABLE DEVICE BENCHMARK")
    print("=" * 70)
    
    results: List[EdgeTestResult] = []
    
    # Memory footprint
    print("\n[1] Memory Footprint Analysis")
    for device_name, device in DEVICE_PROFILES.items():
        max_fit_size = 0
        for size in [64, 128, 256, 512, 1024, 2048]:
            fits, _ = check_device_memory_fit(size, device, 'float32')
            if fits:
                max_fit_size = size
            else:
                break
        
        results.append(EdgeTestResult(
            test_name='Max Buffer Size',
            device=device.name,
            metric='samples',
            value=max_fit_size,
            target=256,
            passed=max_fit_size >= 256
        ))
        print(f"  {device.name}: max {max_fit_size} samples")
    
    # Latency
    print("\n[2] Processing Latency")
    host_cpu_mhz = 3000
    
    for size in [128, 256]:
        stats = benchmark_rft_latency(size, n_iterations=30)
        
        for device_name, device in DEVICE_PROFILES.items():
            estimated = simulate_embedded_latency(
                stats['roundtrip_mean_ms'],
                host_cpu_mhz,
                device.cpu_mhz
            )
            
            results.append(EdgeTestResult(
                test_name=f'Latency {size}',
                device=device.name,
                metric='ms',
                value=estimated,
                target=device.target_latency_ms,
                passed=estimated < device.target_latency_ms
            ))
    
    # Battery life
    print("\n[3] Battery Life Estimation")
    processing_stats = benchmark_rft_latency(256)
    
    for device_name, device in DEVICE_PROFILES.items():
        power = estimate_battery_impact(
            processing_stats['roundtrip_mean_ms'],
            device,
            duty_cycle=0.1  # 10% duty cycle
        )
        
        results.append(EdgeTestResult(
            test_name='Battery Life',
            device=device.name,
            metric='days',
            value=power['battery_life_days'],
            target=1.0,  # At least 1 day
            passed=power['battery_life_days'] >= 1.0
        ))
        print(f"  {device.name}: {power['battery_life_days']:.1f} days")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EDGE DEVICE TEST SUMMARY")
    print("=" * 70)
    print(f"\n{'Test':<20} {'Device':<25} {'Value':<12} {'Target':<12} {'Status':<8}")
    print("-" * 80)
    
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"{r.test_name:<20} {r.device:<25} {r.value:<12.2f} {r.target:<12.2f} {status:<8}")
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return results


if __name__ == "__main__":
    run_comprehensive_edge_benchmark()

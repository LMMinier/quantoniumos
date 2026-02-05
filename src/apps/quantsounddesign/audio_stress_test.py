#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
QuantSoundDesign Audio Stress Test

EPIC 1: Stress test for audio backend stability.

Tests:
1. Buffer size stability at 64, 128, 256 samples
2. CPU load under synthetic DSP workload
3. XRun detection and reporting
4. Extended playback (configurable duration)

Usage:
    python audio_stress_test.py [--duration MINUTES] [--buffer-size SIZE]

Example:
    python audio_stress_test.py --duration 10  # Run for 10 minutes
    python audio_stress_test.py --buffer-size 64  # Test only 64 sample buffer
"""

import sys
import time
import argparse
import numpy as np
import warnings
from dataclasses import dataclass
from typing import List, Optional

# Add parent path for imports
sys.path.insert(0, '.')

try:
    from audio_backend import (
        AudioSettings, AudioBackend, LatencyMode, PerformanceStats,
        get_audio_devices, BUFFER_SIZES
    )
except ImportError:
    from src.apps.quantsounddesign.audio_backend import (
        AudioSettings, AudioBackend, LatencyMode, PerformanceStats,
        get_audio_devices, BUFFER_SIZES
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """Result of a single stress test"""
    buffer_size: int
    sample_rate: int
    duration_seconds: float
    callbacks_total: int
    avg_callback_us: float
    max_callback_us: float
    cpu_load_percent: float
    underruns: int
    overruns: int
    stable: bool
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DSP WORKLOAD
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2


class SyntheticDSP:
    """
    Synthetic DSP workload to stress the audio callback.
    
    Simulates real DAW processing with:
    - Sine wave oscillators
    - RFT-style transforms
    - Filter operations
    - Mixing and summing
    """
    
    def __init__(self, sample_rate: int, num_oscillators: int = 8):
        self.sample_rate = sample_rate
        self.num_oscillators = num_oscillators
        
        # Oscillator state
        self.phases = np.random.random(num_oscillators) * 2 * np.pi
        self.frequencies = 220 * (PHI ** np.arange(num_oscillators))  # Phi-based harmonics
        nyquist = sample_rate / 2
        if np.any(self.frequencies >= nyquist):
            warnings.warn(
                "Nyquist guard: clamping stress-test oscillator frequencies to avoid aliasing.",
                RuntimeWarning,
            )
            self.frequencies = np.minimum(self.frequencies, nyquist * 0.98)
        
        # Filter state (simple IIR)
        self.filter_state = np.zeros(2)
        
    def process(self, frames: int) -> np.ndarray:
        """Generate synthetic audio with controlled CPU load"""
        output = np.zeros(frames, dtype=np.float32)
        
        # Generate oscillators (simulates synth voices)
        for i in range(self.num_oscillators):
            t = np.arange(frames) / self.sample_rate
            phase_inc = 2 * np.pi * self.frequencies[i] * t
            osc = 0.1 * np.sin(self.phases[i] + phase_inc)
            output += osc.astype(np.float32)
            self.phases[i] += phase_inc[-1]
        
        # Simulate FFT-style transform (common in spectral effects)
        if frames >= 64:
            spectrum = np.fft.rfft(output)
            # Apply phi-weighted spectral modification
            weights = PHI ** (-np.arange(len(spectrum)) / len(spectrum))
            spectrum *= weights
            output = np.fft.irfft(spectrum, n=frames).astype(np.float32)
        
        # Simple low-pass filter (simulates EQ/filter plugins)
        alpha = 0.1
        for i in range(frames):
            output[i] = alpha * output[i] + (1 - alpha) * self.filter_state[0]
            self.filter_state[0] = output[i]
        
        # Soft clip (simulates saturation/limiting)
        output = np.tanh(output * 2)
        
        return output * 0.3  # Scale down


# ═══════════════════════════════════════════════════════════════════════════════
# STRESS TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_stress_test(
    buffer_size: int,
    sample_rate: int = 48000,
    duration_seconds: float = 60.0,
    dsp_load: int = 8,  # Number of synth voices
    verbose: bool = True
) -> TestResult:
    """
    Run a stress test at the specified buffer size.
    
    Args:
        buffer_size: Audio buffer size in samples
        sample_rate: Sample rate in Hz
        duration_seconds: How long to run the test
        dsp_load: Number of synthetic DSP voices
        verbose: Print progress
    
    Returns:
        TestResult with performance data
    """
    if verbose:
        latency_ms = (buffer_size / sample_rate) * 1000
        print(f"\n{'='*60}")
        print(f"  STRESS TEST: {buffer_size} samples ({latency_ms:.2f}ms)")
        print(f"  Duration: {duration_seconds}s | DSP Load: {dsp_load} voices")
        print(f"{'='*60}")
    
    # Create settings
    settings = AudioSettings(
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        channels=2,
        enable_performance_monitoring=True,
        use_low_latency=True
    )
    
    # Create backend
    backend = AudioBackend(settings=settings)
    
    # Create DSP workload
    dsp = SyntheticDSP(sample_rate, num_oscillators=dsp_load)
    
    # Track XRun callbacks
    xrun_log = []
    
    def on_xrun(kind: str):
        xrun_log.append((time.time(), kind))
        if verbose:
            print(f"  ⚠ XRun ({kind}) at {len(xrun_log)} total")
    
    settings.xrun_callback = on_xrun
    
    # Store original callback
    original_callback = backend._audio_callback
    
    # Wrap callback to add DSP workload
    def stress_callback(outdata, frames, time_info, status):
        # Run original callback
        original_callback(outdata, frames, time_info, status)
        
        # Add DSP workload
        dsp_audio = dsp.process(frames)
        outdata[:, 0] += dsp_audio
        outdata[:, 1] += dsp_audio
    
    backend._audio_callback = stress_callback
    
    # Start audio
    if not backend.start():
        return TestResult(
            buffer_size=buffer_size,
            sample_rate=sample_rate,
            duration_seconds=0,
            callbacks_total=0,
            avg_callback_us=0,
            max_callback_us=0,
            cpu_load_percent=0,
            underruns=0,
            overruns=0,
            stable=False,
            notes="Failed to start audio backend"
        )
    
    # Run test
    start_time = time.time()
    last_report = start_time
    
    try:
        while (time.time() - start_time) < duration_seconds:
            time.sleep(0.5)
            
            # Periodic status report
            if verbose and (time.time() - last_report) >= 10:
                last_report = time.time()
                stats = backend.get_stats()
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.0f}s] CPU: {stats.cpu_load_percent:.1f}% | "
                      f"XRuns: {stats.underruns + stats.overruns} | "
                      f"Callbacks: {stats.callbacks_total}")
    
    except KeyboardInterrupt:
        if verbose:
            print("\n  Test interrupted by user")
    
    finally:
        backend.stop()
    
    # Collect results
    stats = backend.get_stats()
    actual_duration = time.time() - start_time
    
    # Determine if test was stable
    total_xruns = stats.underruns + stats.overruns
    stable = total_xruns == 0 and stats.cpu_load_percent < 90
    
    result = TestResult(
        buffer_size=buffer_size,
        sample_rate=sample_rate,
        duration_seconds=actual_duration,
        callbacks_total=stats.callbacks_total,
        avg_callback_us=stats.callback_time_us,
        max_callback_us=stats.max_callback_time_us,
        cpu_load_percent=stats.cpu_load_percent,
        underruns=stats.underruns,
        overruns=stats.overruns,
        stable=stable
    )
    
    if verbose:
        print(f"\n  Results:")
        print(f"    Callbacks: {stats.callbacks_total}")
        print(f"    Avg callback: {stats.callback_time_us:.1f} µs")
        print(f"    Max callback: {stats.max_callback_time_us:.1f} µs")
        print(f"    CPU load: {stats.cpu_load_percent:.1f}%")
        print(f"    XRuns: {stats.underruns} underruns, {stats.overruns} overruns")
        print(f"    Status: {'✓ STABLE' if stable else '✗ UNSTABLE'}")
    
    return result


def run_full_test_suite(
    duration_per_test: float = 60.0,
    buffer_sizes: List[int] = None
) -> List[TestResult]:
    """
    Run the full test suite across multiple buffer sizes.
    
    Args:
        duration_per_test: Duration for each buffer size test
        buffer_sizes: List of buffer sizes to test (default: 64, 128, 256)
    
    Returns:
        List of TestResult objects
    """
    if buffer_sizes is None:
        buffer_sizes = [64, 128, 256]
    
    print("\n" + "="*70)
    print("  QUANTSOUNDDESIGN AUDIO STRESS TEST SUITE")
    print("="*70)
    
    # Show available devices
    input_devs, output_devs = get_audio_devices()
    print(f"\n  Output devices: {len(output_devs)}")
    for dev in output_devs[:3]:  # Show first 3
        default = " (default)" if dev.is_default_output else ""
        print(f"    - {dev.name}{default}")
    if len(output_devs) > 3:
        print(f"    ... and {len(output_devs) - 3} more")
    
    results = []
    
    for buffer_size in buffer_sizes:
        result = run_stress_test(
            buffer_size=buffer_size,
            duration_seconds=duration_per_test
        )
        results.append(result)
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUITE SUMMARY")
    print("="*70)
    print(f"\n  {'Buffer':<10} {'Latency':<10} {'CPU':<8} {'XRuns':<10} {'Status':<10}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
    
    all_stable = True
    for r in results:
        latency_ms = (r.buffer_size / r.sample_rate) * 1000
        xruns = r.underruns + r.overruns
        status = "✓ STABLE" if r.stable else "✗ UNSTABLE"
        if not r.stable:
            all_stable = False
        print(f"  {r.buffer_size:<10} {latency_ms:<10.2f} {r.cpu_load_percent:<8.1f} "
              f"{xruns:<10} {status:<10}")
    
    print(f"\n  {'='*50}")
    if all_stable:
        print("  ✓ ALL TESTS PASSED - Audio backend stress tests passed")
    else:
        print("  ⚠ SOME TESTS FAILED - Review results above")
    print(f"  {'='*50}\n")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="QuantSoundDesign Audio Stress Test"
    )
    parser.add_argument(
        "--duration", type=float, default=1.0,
        help="Duration per test in minutes (default: 1)"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=None,
        help="Test only this buffer size (default: test 64, 128, 256)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test (10 seconds per buffer size)"
    )
    
    args = parser.parse_args()
    
    duration = 10 if args.quick else args.duration * 60
    
    if args.buffer_size:
        # Single buffer size test
        run_stress_test(
            buffer_size=args.buffer_size,
            duration_seconds=duration
        )
    else:
        # Full suite
        run_full_test_suite(duration_per_test=duration)


if __name__ == "__main__":
    main()

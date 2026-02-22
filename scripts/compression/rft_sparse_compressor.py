#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
🎯 RFT SPARSE MODEL COMPRESSOR
==============================
Exploits sparsity in RFT coefficients for actual compression.

Strategy:
1. Transform weights → RFT coefficients
2. Threshold small coefficients → create sparsity  
3. Store only non-zero coefficients (sparse format)
4. Quantize to int8/int16 for smaller values
5. Compress with zlib

Author: QuantoniumOS Team
Date: December 19, 2025
"""

import json
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import zlib

sys.path.insert(0, '/workspaces/quantoniumos')

# Try to load native C++/ASM accelerated RFT
NATIVE_AVAILABLE = False
try:
    native_lib_path = '/workspaces/quantoniumos/src/rftmw_native/build'
    if native_lib_path not in sys.path:
        sys.path.insert(0, native_lib_path)
    import rftmw_native
    NATIVE_AVAILABLE = True
    print("✅ Native C++/ASM RFT library loaded (AVX2/AVX-512 accelerated)")
except ImportError as e:
    print(f"⚠️ Native library not available: {e}")

# Try optimized Python bindings
OPTIMIZED_BINDINGS = False
try:
    from algorithms.rft.kernels.python_bindings.optimized_rft import (
        OptimizedRFT, RFT_OPT_AVX2, RFT_OPT_PARALLEL
    )
    OPTIMIZED_BINDINGS = True
    print("✅ Optimized RFT Python bindings available")
except ImportError:
    pass

from algorithms.rft.routing.signal_classifier import classify_signal, TransformType
from algorithms.rft.variants.registry import (
    generate_original_phi_rft,
    generate_fibonacci_tilt,
    generate_harmonic_phase,
    generate_chaotic_mix,
    generate_geometric_lattice,
    generate_phi_chaotic_hybrid,
    generate_hyperbolic_phase,
)
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from huggingface_hub import snapshot_download
    import torch
    print("✅ Dependencies loaded")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    exit(1)

PHI = (1.0 + np.sqrt(5.0)) / 2.0

VARIANT_GENERATORS = {
    'rft_golden': generate_original_phi_rft,
    'rft_fibonacci': generate_fibonacci_tilt,
    'rft_harmonic': generate_harmonic_phase,
    'rft_geometric': generate_geometric_lattice,
    'rft_chaotic': generate_chaotic_mix,
    'rft_hybrid': generate_phi_chaotic_hybrid,
    'rft_hyperbolic': generate_hyperbolic_phase,
}


class RFTSparseCompressor:
    """Sparse RFT compressor - exploits coefficient sparsity for real compression."""
    
    def __init__(self, threshold_percentile: float = 90.0, quantization_bits: int = 8):
        """
        Args:
            threshold_percentile: Zero out coefficients below this percentile (0-100).
                                  90 = keep top 10% of coefficients
                                  95 = keep top 5% of coefficients
                                  99 = keep top 1% of coefficients
            quantization_bits: 8 for int8, 16 for int16
        """
        self.threshold_percentile = threshold_percentile
        self.quantization_bits = quantization_bits
        self.phi = PHI
        self.basis_cache = {}
        
        print(f"🎯 Sparse RFT Compressor initialized")
        print(f"   Threshold: {threshold_percentile}th percentile (keep top {100-threshold_percentile}%)")
        print(f"   Quantization: int{quantization_bits}")
    
    def _native_forward(self, data: np.ndarray) -> np.ndarray:
        """Use native library for forward transform."""
        if NATIVE_AVAILABLE:
            try:
                return rftmw_native.rft_forward(data.astype(np.complex128))
            except Exception:
                pass
        return rft_forward(data)
    
    def _native_inverse(self, data: np.ndarray) -> np.ndarray:
        """Use native library for inverse transform."""
        if NATIVE_AVAILABLE:
            try:
                return rftmw_native.rft_inverse(data.astype(np.complex128))
            except Exception:
                pass
        return rft_inverse(data)
    
    def get_basis(self, n: int, variant: str) -> Optional[np.ndarray]:
        """Get or generate basis matrix."""
        if n > 4096:
            return None
        cache_key = (n, variant)
        if cache_key not in self.basis_cache:
            generator = VARIANT_GENERATORS.get(variant, generate_original_phi_rft)
            self.basis_cache[cache_key] = generator(n)
        return self.basis_cache[cache_key]
    
    def select_variant(self, weights: np.ndarray) -> Tuple[str, float]:
        """Select optimal RFT variant - try each and pick sparsest."""
        flat = weights.flatten()
        
        # Sample if too large
        if len(flat) > 1024:
            indices = np.linspace(0, len(flat)-1, 1024, dtype=int)
            sample = flat[indices]
        else:
            sample = flat
        
        # Pad to power of 2
        n = max(64, 2**int(np.ceil(np.log2(len(sample)))))
        if len(sample) < n:
            sample = np.pad(sample, (0, n - len(sample)))
        else:
            sample = sample[:n]
        
        # Test each variant for sparsity
        best_variant = 'rft_golden'
        best_sparsity = 0.0
        
        for variant_name, generator in VARIANT_GENERATORS.items():
            if generator is None:
                continue
            try:
                basis = generator(n)
                coeffs = basis.conj().T @ sample
                
                # Calculate sparsity after thresholding
                magnitudes = np.abs(coeffs)
                threshold = np.percentile(magnitudes, self.threshold_percentile)
                sparsity = np.mean(magnitudes < threshold)
                
                # Also consider energy concentration (higher is better)
                sorted_mags = np.sort(magnitudes)[::-1]
                top_10_energy = np.sum(sorted_mags[:n//10]**2)
                total_energy = np.sum(magnitudes**2) + 1e-10
                energy_concentration = top_10_energy / total_energy
                
                # Combined score
                score = sparsity * 0.5 + energy_concentration * 0.5
                
                if score > best_sparsity:
                    best_sparsity = score
                    best_variant = variant_name
            except Exception:
                continue
        
        return best_variant, best_sparsity
    
    def compress_tensor(self, weights: np.ndarray, name: str) -> Dict:
        """Compress tensor using sparse RFT representation."""
        original_shape = weights.shape
        flat = weights.flatten().astype(np.float32)
        original_bytes = flat.nbytes
        
        # Select best variant for this tensor
        variant, sparsity_score = self.select_variant(flat)
        
        # Process in chunks for large tensors
        CHUNK_SIZE = 2048
        
        if len(flat) > CHUNK_SIZE:
            return self._compress_chunked(flat, name, original_shape, variant, CHUNK_SIZE)
        
        # Pad to power of 2
        n = 2**int(np.ceil(np.log2(max(len(flat), 64))))
        if len(flat) < n:
            flat_padded = np.pad(flat, (0, n - len(flat)))
        else:
            flat_padded = flat
        
        # Transform to RFT domain
        coeffs = self._native_forward(flat_padded)
        
        # Apply thresholding to create sparsity
        magnitudes = np.abs(coeffs)
        threshold = np.percentile(magnitudes, self.threshold_percentile)
        
        # Zero out small coefficients
        mask = magnitudes >= threshold
        sparse_coeffs = coeffs * mask
        
        # Count non-zero coefficients
        nonzero_count = np.sum(mask)
        sparsity_ratio = 1.0 - (nonzero_count / len(coeffs))
        
        # Get indices and values of non-zero coefficients
        nonzero_indices = np.where(mask)[0].astype(np.uint16 if n <= 65536 else np.uint32)
        nonzero_real = sparse_coeffs[mask].real.astype(np.float32)
        nonzero_imag = sparse_coeffs[mask].imag.astype(np.float32)
        
        # Quantize to int8 or int16
        max_real = np.max(np.abs(nonzero_real)) if len(nonzero_real) > 0 else 1.0
        max_imag = np.max(np.abs(nonzero_imag)) if len(nonzero_imag) > 0 else 1.0
        max_val = max(max_real, max_imag, 1e-10)
        
        if self.quantization_bits == 8:
            scale = 127.0 / max_val
            quant_real = np.round(nonzero_real * scale).astype(np.int8)
            quant_imag = np.round(nonzero_imag * scale).astype(np.int8)
        else:
            scale = 32767.0 / max_val
            quant_real = np.round(nonzero_real * scale).astype(np.int16)
            quant_imag = np.round(nonzero_imag * scale).astype(np.int16)
        
        # Compress everything with zlib
        compressed_indices = zlib.compress(nonzero_indices.tobytes(), level=9)
        compressed_real = zlib.compress(quant_real.tobytes(), level=9)
        compressed_imag = zlib.compress(quant_imag.tobytes(), level=9)
        
        compressed_bytes = len(compressed_indices) + len(compressed_real) + len(compressed_imag)
        compression_ratio = original_bytes / max(compressed_bytes, 1)
        
        return {
            'name': name,
            'original_shape': list(original_shape),
            'original_length': len(flat),
            'padded_length': n,
            'variant': variant,
            'threshold_percentile': self.threshold_percentile,
            'nonzero_count': int(nonzero_count),
            'sparsity': float(sparsity_ratio),
            'scale': float(scale),
            'max_val': float(max_val),
            'quantization_bits': self.quantization_bits,
            'compressed_indices': compressed_indices.hex(),
            'compressed_real': compressed_real.hex(),
            'compressed_imag': compressed_imag.hex(),
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': compression_ratio,
            'chunked': False,
        }
    
    def _compress_chunked(self, flat: np.ndarray, name: str, original_shape: tuple,
                          variant: str, chunk_size: int) -> Dict:
        """Compress large tensor in sparse chunks."""
        chunks_data = []
        total_compressed = 0
        total_nonzero = 0
        
        for i in range(0, len(flat), chunk_size):
            chunk = flat[i:i+chunk_size]
            
            # Pad chunk
            n = 2**int(np.ceil(np.log2(max(len(chunk), 64))))
            if len(chunk) < n:
                chunk_padded = np.pad(chunk, (0, n - len(chunk)))
            else:
                chunk_padded = chunk
            
            # Transform
            coeffs = self._native_forward(chunk_padded)
            
            # Threshold
            magnitudes = np.abs(coeffs)
            threshold = np.percentile(magnitudes, self.threshold_percentile)
            mask = magnitudes >= threshold
            sparse_coeffs = coeffs * mask
            
            nonzero_count = np.sum(mask)
            total_nonzero += nonzero_count
            
            # Sparse representation
            nonzero_indices = np.where(mask)[0].astype(np.uint16 if n <= 65536 else np.uint32)
            nonzero_real = sparse_coeffs[mask].real.astype(np.float32)
            nonzero_imag = sparse_coeffs[mask].imag.astype(np.float32)
            
            # Quantize
            max_val = max(np.max(np.abs(nonzero_real)) if len(nonzero_real) > 0 else 1.0,
                         np.max(np.abs(nonzero_imag)) if len(nonzero_imag) > 0 else 1.0,
                         1e-10)
            
            if self.quantization_bits == 8:
                scale = 127.0 / max_val
                quant_real = np.round(nonzero_real * scale).astype(np.int8)
                quant_imag = np.round(nonzero_imag * scale).astype(np.int8)
            else:
                scale = 32767.0 / max_val
                quant_real = np.round(nonzero_real * scale).astype(np.int16)
                quant_imag = np.round(nonzero_imag * scale).astype(np.int16)
            
            # Compress
            compressed_indices = zlib.compress(nonzero_indices.tobytes(), level=9)
            compressed_real = zlib.compress(quant_real.tobytes(), level=9)
            compressed_imag = zlib.compress(quant_imag.tobytes(), level=9)
            
            chunk_compressed = len(compressed_indices) + len(compressed_real) + len(compressed_imag)
            total_compressed += chunk_compressed
            
            chunks_data.append({
                'chunk_length': len(chunk),
                'padded_length': n,
                'nonzero_count': int(nonzero_count),
                'scale': float(scale),
                'compressed_indices': compressed_indices.hex(),
                'compressed_real': compressed_real.hex(),
                'compressed_imag': compressed_imag.hex(),
            })
        
        original_bytes = flat.nbytes
        
        return {
            'name': name,
            'original_shape': list(original_shape),
            'original_length': len(flat),
            'variant': variant,
            'threshold_percentile': self.threshold_percentile,
            'total_nonzero': int(total_nonzero),
            'sparsity': float(1.0 - total_nonzero / len(flat)),
            'quantization_bits': self.quantization_bits,
            'chunks': chunks_data,
            'original_bytes': original_bytes,
            'compressed_bytes': total_compressed,
            'compression_ratio': original_bytes / max(total_compressed, 1),
            'chunked': True,
        }
    
    def decompress_tensor(self, compressed: Dict) -> np.ndarray:
        """Decompress sparse RFT tensor."""
        if compressed.get('chunked', False):
            return self._decompress_chunked(compressed)
        
        n = compressed['padded_length']
        original_length = compressed['original_length']
        scale = compressed['scale']
        qbits = compressed.get('quantization_bits', 16)
        
        # Decompress
        indices_bytes = bytes.fromhex(compressed['compressed_indices'])
        real_bytes = bytes.fromhex(compressed['compressed_real'])
        imag_bytes = bytes.fromhex(compressed['compressed_imag'])
        
        indices = np.frombuffer(zlib.decompress(indices_bytes), 
                               dtype=np.uint16 if n <= 65536 else np.uint32)
        
        if qbits == 8:
            quant_real = np.frombuffer(zlib.decompress(real_bytes), dtype=np.int8)
            quant_imag = np.frombuffer(zlib.decompress(imag_bytes), dtype=np.int8)
        else:
            quant_real = np.frombuffer(zlib.decompress(real_bytes), dtype=np.int16)
            quant_imag = np.frombuffer(zlib.decompress(imag_bytes), dtype=np.int16)
        
        # Dequantize
        real_vals = quant_real.astype(np.float64) / scale
        imag_vals = quant_imag.astype(np.float64) / scale
        
        # Reconstruct sparse coefficients
        coeffs = np.zeros(n, dtype=np.complex128)
        coeffs[indices] = real_vals + 1j * imag_vals
        
        # Inverse transform
        reconstructed = self._native_inverse(coeffs)
        
        return reconstructed.real[:original_length].reshape(compressed['original_shape']).astype(np.float32)
    
    def _decompress_chunked(self, compressed: Dict) -> np.ndarray:
        """Decompress chunked sparse tensor."""
        chunks = compressed['chunks']
        original_length = compressed['original_length']
        qbits = compressed.get('quantization_bits', 16)
        
        reconstructed_chunks = []
        
        for chunk_data in chunks:
            n = chunk_data['padded_length']
            chunk_length = chunk_data['chunk_length']
            scale = chunk_data['scale']
            
            # Decompress
            indices_bytes = bytes.fromhex(chunk_data['compressed_indices'])
            real_bytes = bytes.fromhex(chunk_data['compressed_real'])
            imag_bytes = bytes.fromhex(chunk_data['compressed_imag'])
            
            indices = np.frombuffer(zlib.decompress(indices_bytes),
                                   dtype=np.uint16 if n <= 65536 else np.uint32)
            
            if qbits == 8:
                quant_real = np.frombuffer(zlib.decompress(real_bytes), dtype=np.int8)
                quant_imag = np.frombuffer(zlib.decompress(imag_bytes), dtype=np.int8)
            else:
                quant_real = np.frombuffer(zlib.decompress(real_bytes), dtype=np.int16)
                quant_imag = np.frombuffer(zlib.decompress(imag_bytes), dtype=np.int16)
            
            # Dequantize
            real_vals = quant_real.astype(np.float64) / scale
            imag_vals = quant_imag.astype(np.float64) / scale
            
            # Reconstruct
            coeffs = np.zeros(n, dtype=np.complex128)
            coeffs[indices] = real_vals + 1j * imag_vals
            
            # Inverse transform
            chunk_recon = self._native_inverse(coeffs)
            reconstructed_chunks.append(chunk_recon.real[:chunk_length])
        
        flat = np.concatenate(reconstructed_chunks)[:original_length]
        return flat.reshape(compressed['original_shape']).astype(np.float32)
    
    def compress_model(self, model_path: str) -> Dict:
        """Compress entire model."""
        print(f"\n🔄 Loading model weights from {model_path}...")
        
        if os.path.isdir(model_path):
            model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        else:
            model = AutoModel.from_pretrained(
                "microsoft/DialoGPT-small",
                cache_dir=model_path,
                torch_dtype=torch.float32
            )
        
        state_dict = model.state_dict()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Total parameters: {total_params:,}")
        
        compressed_layers = []
        total_original = 0
        total_compressed = 0
        variant_counts = {}
        
        for name, tensor in state_dict.items():
            weights = tensor.numpy()
            compressed = self.compress_tensor(weights, name)
            compressed_layers.append(compressed)
            
            total_original += compressed['original_bytes']
            total_compressed += compressed['compressed_bytes']
            
            variant = compressed['variant']
            variant_counts[variant] = variant_counts.get(variant, 0) + 1
            
            ratio = compressed['compression_ratio']
            sparsity = compressed.get('sparsity', 0)
            print(f"  {name}: {variant} → {ratio:.1f}x ({sparsity*100:.0f}% sparse)")
        
        return {
            'model_id': 'microsoft/DialoGPT-small',
            'total_params': total_params,
            'threshold_percentile': self.threshold_percentile,
            'quantization_bits': self.quantization_bits,
            'layers': compressed_layers,
            'total_original_bytes': total_original,
            'total_compressed_bytes': total_compressed,
            'overall_ratio': total_original / max(total_compressed, 1),
            'variant_counts': variant_counts,
            'timestamp': datetime.now().isoformat(),
        }


def test_sparsity_levels():
    """Test different sparsity levels to find optimal compression."""
    print("\n" + "="*60)
    print("🧪 TESTING SPARSITY LEVELS FOR RFT VARIANTS")
    print("="*60)
    
    # Create test signal similar to NN weights
    np.random.seed(42)
    test_weights = np.random.randn(1024).astype(np.float32) * 0.02  # Typical NN weight scale
    
    print("\nTest signal: 1024 float32 values (typical NN weight distribution)")
    print(f"Original size: {test_weights.nbytes} bytes")
    
    results = []
    
    for variant_name, generator in VARIANT_GENERATORS.items():
        if generator is None:
            continue
        
        try:
            basis = generator(1024)
            coeffs = basis.conj().T @ test_weights
            
            # Analyze coefficient distribution
            magnitudes = np.abs(coeffs)
            
            # Energy concentration in top coefficients
            sorted_mags = np.sort(magnitudes)[::-1]
            cumulative_energy = np.cumsum(sorted_mags**2) / np.sum(magnitudes**2)
            
            # How many coefficients needed for 99% energy?
            coeffs_for_99 = np.searchsorted(cumulative_energy, 0.99) + 1
            coeffs_for_95 = np.searchsorted(cumulative_energy, 0.95) + 1
            coeffs_for_90 = np.searchsorted(cumulative_energy, 0.90) + 1
            
            results.append({
                'variant': variant_name,
                'coeffs_90': coeffs_for_90,
                'coeffs_95': coeffs_for_95,
                'coeffs_99': coeffs_for_99,
                'potential_ratio_90': 1024 / coeffs_for_90,
                'potential_ratio_95': 1024 / coeffs_for_95,
                'potential_ratio_99': 1024 / coeffs_for_99,
            })
            
            print(f"\n{variant_name}:")
            print(f"  90% energy: {coeffs_for_90} coeffs ({1024/coeffs_for_90:.1f}x potential)")
            print(f"  95% energy: {coeffs_for_95} coeffs ({1024/coeffs_for_95:.1f}x potential)")
            print(f"  99% energy: {coeffs_for_99} coeffs ({1024/coeffs_for_99:.1f}x potential)")
            
        except Exception as e:
            print(f"\n{variant_name}: ERROR - {e}")
    
    # Find best variant
    best = max(results, key=lambda x: x['potential_ratio_95'])
    print(f"\n🏆 Best variant for compression: {best['variant']}")
    print(f"   Potential compression at 95% quality: {best['potential_ratio_95']:.1f}x")
    
    return results


def main():
    import tempfile
    
    # First, analyze sparsity potential
    test_sparsity_levels()
    
    # Test compression with different threshold levels
    print("\n" + "="*60)
    print("🎯 SPARSE RFT COMPRESSION TEST")
    print("="*60)
    
    # Download model
    print("\n📥 Downloading DialoGPT-small...")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = snapshot_download(
            "microsoft/DialoGPT-small",
            cache_dir=tmpdir,
            resume_download=True
        )
        print(f"✅ Downloaded to {model_path}")
        
        # Test with 95th percentile threshold (keep top 5%)
        print("\n" + "="*60)
        print("⚛️ COMPRESSING WITH 95th PERCENTILE THRESHOLD")
        print("="*60)
        
        compressor = RFTSparseCompressor(threshold_percentile=95.0, quantization_bits=8)
        result = compressor.compress_model(model_path)
        
        print("\n" + "="*60)
        print("📊 COMPRESSION RESULTS")
        print("="*60)
        print(f"  Original size:   {result['total_original_bytes'] / 1e6:.2f} MB")
        print(f"  Compressed size: {result['total_compressed_bytes'] / 1e6:.2f} MB")
        print(f"  Ratio:          {result['overall_ratio']:.1f}x")
        print(f"\n  Variant usage:")
        for v, c in result['variant_counts'].items():
            print(f"    {v}: {c} layers")
        
        # Save compressed model
        output_path = Path("/workspaces/quantoniumos/ai/models/quantum/dialogpt_sparse_rft.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(result, f)
        
        file_size = output_path.stat().st_size / 1e6
        print(f"✅ Saved! File size: {file_size:.2f} MB")
        
        # Verify reconstruction
        print("\n" + "="*60)
        print("🔍 VERIFYING RECONSTRUCTION QUALITY")
        print("="*60)
        
        # Load original model again for comparison
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        original_state = model.state_dict()
        
        errors = []
        for layer in result['layers'][:5]:  # Check first 5 layers
            name = layer['name']
            original = original_state[name].numpy()
            reconstructed = compressor.decompress_tensor(layer)
            
            # Calculate error metrics
            abs_error = np.abs(original - reconstructed)
            rel_error = abs_error / (np.abs(original) + 1e-10)
            
            max_err = np.max(abs_error)
            mean_err = np.mean(abs_error)
            mean_rel = np.mean(rel_error) * 100
            
            errors.append({
                'name': name,
                'max_error': max_err,
                'mean_error': mean_err,
                'mean_rel_error': mean_rel,
            })
            
            print(f"  {name}:")
            print(f"    Max error: {max_err:.6f}, Mean error: {mean_err:.6f}, Rel error: {mean_rel:.2f}%")
        
        print("\n" + "="*60)
        print("🎉 SPARSE RFT COMPRESSION COMPLETE!")
        print("="*60)


if __name__ == '__main__':
    main()

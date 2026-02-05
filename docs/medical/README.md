# QuantoniumOS Medical Applications Guide

> ## ⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE  ⚠️
>
> This software is provided **strictly for research, educational, and experimental purposes**.
> It has **NOT** been validated, approved, or cleared by any regulatory authority
> (FDA, CE, etc.) for clinical, diagnostic, or therapeutic use.
>
> **DO NOT** use this software to make medical diagnoses, treatment decisions,
> or process real patient data in clinical workflows. The authors and copyright
> holders disclaim all liability for any harm arising from clinical or diagnostic use.

[![Medical Tests](https://img.shields.io/badge/tests-83%20passed-brightgreen.svg)](../../tests/medical/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](../../pyproject.toml)
[![Research Preview](https://img.shields.io/badge/status-research%20preview-orange.svg)](#safety-considerations)
[![Free for Medical Research](https://img.shields.io/badge/license-free%20for%20hospitals%20%26%20researchers-blue.svg)](#license)

**Version:** 1.0.0  
**Date:** December 2025  
**Status:** Open Research Preview (83/83 tests passing)

---

## Free for Hospitals & Medical Researchers

> **This medical applications module is FREE for:**
> - Hospitals and healthcare institutions
> - Medical researchers and academics
> - Non-profit healthcare organizations
> - Educational institutions
> - Open-source medical software projects
>
> **For research, testing, and validation purposes only.**
> **NOT for clinical diagnostics or patient care decisions.**
> 
> Commercial medical device manufacturers: See [LICENSE-CLAIMS-NC.md](../../LICENSE-CLAIMS-NC.md) for licensing terms.

---

## Overview

QuantoniumOS provides a novel signal processing framework based on the **Resonant Fourier Transform (RFT)**. This module contains **experimental simulations** exploring potential applications in:

- **Medical Imaging (Simulated)**: Synthetic Shepp-Logan phantom reconstruction (No real MRI hardware support).
- **Biosignals (Benchmarks)**: Comparison of RFT vs Wavelets on MIT-BIH/Sleep-EDF datasets.
- **Genomics (Synthetic)**: Information-theoretic compression tests on random/pseudo-random DNA strings (No support for FASTQ/BAM/SAM).
- **Clinical Security (Prototype)**: Experimental hashing concepts (Non-compliant with HIPAA/GDPR).

> **Research Disclaimer**: This software is for research and educational purposes only. It has NOT been validated for clinical use and should NOT be used for medical diagnosis or treatment decisions. See [Safety Considerations](#safety-considerations) below.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Verify installation
python -c "from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT; print('RFT Core available')"
```

### Run Medical Test Suite

```bash
# Run all medical tests
pytest tests/medical/ -v

# Run specific test categories
pytest tests/medical/test_imaging_reconstruction.py -v
pytest tests/medical/test_biosignal_compression.py -v
pytest tests/medical/test_genomics_transforms.py -v
pytest tests/medical/test_medical_security.py -v
pytest tests/medical/test_edge_wearable.py -v
```

### Run Benchmarks

```bash
# Run comprehensive benchmarks
python tests/medical/test_imaging_reconstruction.py
python tests/medical/test_biosignal_compression.py
python tests/medical/test_genomics_transforms.py
python tests/medical/test_medical_security.py
python tests/medical/test_edge_wearable.py
```

---

## Applicable RFT Variants

Different medical applications benefit from different RFT variants:

| Application | Recommended Variants | Rationale |
|-------------|---------------------|-----------|
| **MRI Denoising** | STANDARD, NOISE_SHRINK_MANIFOLD | Good noise separation in transform domain |
| **ECG Compression** | CASCADE (H3), HARMONIC | Captures periodic waveform structure |
| **EEG Analysis** | FIBONACCI, GEOMETRIC | Preserves rhythmic brain activity patterns |
| **Genomics** | STANDARD, ENTROPY_GUIDED (FH5) | Handles discrete symbols efficiently |
| **Secure Hashing** | CHAOTIC, PHI_CHAOTIC | High avalanche effect |
| **Edge Devices** | STANDARD (with quantization) | Minimal computation overhead |

### Variant Selection Code

```python
from algorithms.rft.variants.manifest import VARIANT_MANIFEST

# List all available variants
for v in VARIANT_MANIFEST:
    print(f"{v.code}: {v.info.name} - {v.info.use_cases[:50]}...")

# Get specific variant
def get_variant_matrix(variant_code: str, size: int):
    """Get transform matrix for a specific variant."""
    from algorithms.rft.variants.registry import VARIANTS
    
    manifest_entry = next(v for v in VARIANT_MANIFEST if v.code == variant_code)
    variant_info = VARIANTS[manifest_entry.registry_key]
    
    return variant_info.generator(size)
```

---

## Application Guides

### 1. Medical Imaging

#### MRI Reconstruction/Denoising

```python
import numpy as np
from algorithms.rft.core.closed_form_rft import rft_forward, rft_inverse

def rft_denoise_mri(noisy_image: np.ndarray, 
                    threshold_ratio: float = 0.1) -> np.ndarray:
    """
    Denoise MRI image using RFT coefficient thresholding.
    
    Args:
        noisy_image: 2D noisy MRI slice
        threshold_ratio: Fraction of max coefficient to threshold
        
    Returns:
        Denoised image
    """
    rows, cols = noisy_image.shape
    
    # Separable 2D RFT: rows then columns
    row_coeffs = np.zeros_like(noisy_image, dtype=np.complex128)
    for i in range(rows):
        row_coeffs[i, :] = rft_forward(noisy_image[i, :].astype(np.complex128))
    
    full_coeffs = np.zeros_like(row_coeffs)
    for j in range(cols):
        full_coeffs[:, j] = rft_forward(row_coeffs[:, j])
    
    # Threshold small coefficients
    max_mag = np.max(np.abs(full_coeffs))
    threshold = threshold_ratio * max_mag
    thresholded = np.where(np.abs(full_coeffs) < threshold, 0, full_coeffs)
    
    # Inverse transform
    inv_cols = np.zeros_like(thresholded)
    for j in range(cols):
        inv_cols[:, j] = rft_inverse(thresholded[:, j])
    
    denoised = np.zeros_like(noisy_image)
    for i in range(rows):
        denoised[i, :] = rft_inverse(inv_cols[i, :]).real
    
    return np.clip(denoised, 0, 1)
```

#### Evaluation Metrics

```python
def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Simplified SSIM (Structural Similarity Index)."""
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    
    mu_x, mu_y = np.mean(original), np.mean(reconstructed)
    sigma_x, sigma_y = np.std(original), np.std(reconstructed)
    sigma_xy = np.mean((original - mu_x) * (reconstructed - mu_y))
    
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))
```

### 2. Biosignal Processing

#### ECG Compression

```python
def compress_ecg(signal: np.ndarray, 
                 sample_rate: int = 360,
                 keep_ratio: float = 0.5) -> tuple:
    """
    Compress ECG signal using RFT.
    
    Args:
        signal: Raw ECG signal
        sample_rate: Sampling rate (Hz)
        keep_ratio: Fraction of coefficients to keep
        
    Returns:
        (reconstructed_signal, compression_stats)
    """
    from algorithms.rft.core.closed_form_rft import rft_forward, rft_inverse
    
    chunk_size = 256  # ~0.7s at 360 Hz
    n = len(signal)
    n_padded = ((n - 1) // chunk_size + 1) * chunk_size
    
    padded = np.zeros(n_padded)
    padded[:n] = signal
    
    reconstructed = np.zeros(n_padded)
    total_coeffs, kept_coeffs = 0, 0
    
    for i in range(0, n_padded, chunk_size):
        chunk = padded[i:i + chunk_size].astype(np.complex128)
        
        # Transform
        coeffs = rft_forward(chunk)
        total_coeffs += len(coeffs)
        
        # Keep top coefficients
        n_keep = int(keep_ratio * len(coeffs))
        magnitudes = np.abs(coeffs)
        threshold = np.sort(magnitudes)[-n_keep]
        
        compressed = np.where(magnitudes >= threshold, coeffs, 0)
        kept_coeffs += np.count_nonzero(compressed)
        
        # Reconstruct
        reconstructed[i:i + chunk_size] = rft_inverse(compressed).real
    
    stats = {
        'compression_ratio': total_coeffs / kept_coeffs,
        'prd': 100 * np.sqrt(np.sum((signal - reconstructed[:n])**2) / np.sum(signal**2))
    }
    
    return reconstructed[:n], stats
```

#### Quality Metrics for Biosignals

```python
def snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB."""
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - reconstructed) ** 2)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

def prd(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Percent Root-mean-square Difference (ECG standard metric)."""
    return 100 * np.sqrt(np.sum((original - reconstructed) ** 2) / np.sum(original ** 2))
```

### 3. Genomics Applications

#### K-mer Spectrum Analysis

```python
def compute_kmer_spectrum(sequence: str, k: int = 4) -> np.ndarray:
    """Compute normalized k-mer frequency spectrum."""
    n_kmers = 4 ** k
    spectrum = np.zeros(n_kmers)
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if all(b in base_to_idx for b in kmer):
            idx = sum(base_to_idx[b] * (4 ** (k - 1 - j)) for j, b in enumerate(kmer))
            spectrum[idx] += 1
    
    return spectrum / spectrum.sum() if spectrum.sum() > 0 else spectrum


def analyze_kmer_rft(sequence: str, k: int = 4):
    """Analyze sequence using RFT-transformed k-mer spectrum."""
    from algorithms.rft.core.closed_form_rft import rft_forward
    
    spectrum = compute_kmer_spectrum(sequence, k)
    coeffs = rft_forward(spectrum.astype(np.complex128))
    
    # Energy concentration analysis
    magnitudes = np.abs(coeffs)
    sorted_mags = np.sort(magnitudes)[::-1]
    cumulative_energy = np.cumsum(sorted_mags ** 2) / np.sum(sorted_mags ** 2)
    
    # Find number of coefficients for 90% energy
    n_90 = np.searchsorted(cumulative_energy, 0.9) + 1
    
    return {
        'spectrum': spectrum,
        'coefficients': coeffs,
        'sparsity_90': n_90 / len(coeffs),
        'dominant_frequency': np.argmax(magnitudes)
    }
```

### 4. Clinical Data Security

#### Waveform Hashing

```python
import hashlib

def rft_waveform_hash(waveform: np.ndarray, 
                       salt: bytes = None,
                       hash_bits: int = 256) -> bytes:
    """
    Compute privacy-preserving hash of medical waveform.
    
    Args:
        waveform: Input signal (ECG, EEG, etc.)
        salt: Optional salt for keyed hashing
        hash_bits: Output hash size (128, 256, 512)
        
    Returns:
        Hash bytes
    """
    from algorithms.rft.core.closed_form_rft import rft_forward
    
    # Normalize
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-10)
    
    # Pad to power of 2
    n = len(waveform)
    n_padded = 2 ** int(np.ceil(np.log2(n)))
    padded = np.zeros(n_padded)
    padded[:n] = waveform
    
    # Transform
    coeffs = rft_forward(padded.astype(np.complex128))
    
    # Quantize coefficients
    mag_quant = (np.abs(coeffs) * 1000).astype(np.int64)
    phase_quant = ((np.angle(coeffs) + np.pi) * 1000).astype(np.int64)
    
    # Hash
    hash_input = mag_quant.tobytes() + phase_quant.tobytes()
    if salt:
        hash_input = salt + hash_input
    
    if hash_bits == 256:
        return hashlib.sha256(hash_input).digest()
    elif hash_bits == 512:
        return hashlib.sha512(hash_input).digest()
    else:
        return hashlib.sha256(hash_input).digest()
```

---

## Public Datasets

### Medical Imaging
- **BrainWeb**: Simulated brain MRI database - [brainweb.bic.mni.mcgill.ca](https://brainweb.bic.mni.mcgill.ca/)
- **IXI Dataset**: 600 MR images - [brain-development.org/ixi-dataset](https://brain-development.org/ixi-dataset/)
- **TCIA**: The Cancer Imaging Archive - [cancerimagingarchive.net](https://www.cancerimagingarchive.net/)

### ECG/EEG
- **MIT-BIH Arrhythmia Database**: [physionet.org/content/mitdb](https://physionet.org/content/mitdb/)
- **TUH EEG Corpus**: [isip.piconepress.com/projects/tuh_eeg](https://isip.piconepress.com/projects/tuh_eeg/)
- **PTB-XL**: Large 12-lead ECG dataset - [physionet.org/content/ptb-xl](https://physionet.org/content/ptb-xl/)

### Genomics
- **NCBI SRA**: Sequence Read Archive - [ncbi.nlm.nih.gov/sra](https://www.ncbi.nlm.nih.gov/sra)
- **PDB**: Protein Data Bank - [rcsb.org](https://www.rcsb.org/)
- **1000 Genomes**: [internationalgenome.org](https://www.internationalgenome.org/)

---

## Evaluation Metrics Templates

### Imaging Metrics

```python
class ImagingMetrics:
    """Standard imaging quality metrics."""
    
    @staticmethod
    def psnr(original, reconstructed, max_val=1.0):
        mse = np.mean((original - reconstructed) ** 2)
        return 10 * np.log10(max_val ** 2 / mse) if mse > 0 else float('inf')
    
    @staticmethod
    def nmse(original, reconstructed):
        return np.sum((original - reconstructed) ** 2) / np.sum(original ** 2)
    
    @staticmethod
    def ssim(original, reconstructed, k1=0.01, k2=0.03):
        c1, c2 = (k1) ** 2, (k2) ** 2
        mu_x, mu_y = np.mean(original), np.mean(reconstructed)
        sigma_x, sigma_y = np.std(original), np.std(reconstructed)
        sigma_xy = np.mean((original - mu_x) * (reconstructed - mu_y))
        return ((2*mu_x*mu_y + c1) * (2*sigma_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))
```

### Biosignal Metrics

```python
class BiosignalMetrics:
    """Standard biosignal quality metrics."""
    
    @staticmethod
    def snr_db(original, reconstructed):
        signal = np.sum(original ** 2)
        noise = np.sum((original - reconstructed) ** 2)
        return 10 * np.log10(signal / noise) if noise > 0 else float('inf')
    
    @staticmethod
    def prd(original, reconstructed):
        """Percent Root-mean-square Difference."""
        return 100 * np.sqrt(np.sum((original - reconstructed) ** 2) / np.sum(original ** 2))
    
    @staticmethod
    def correlation(original, reconstructed):
        return np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
```

---

## Reproducible Benchmark Scripts

### Run Standard Benchmarks

```bash
# Set seed for reproducibility
export QOS_TEST_SEED=42

# Run imaging benchmark
python -c "
from tests.medical.test_imaging_reconstruction import run_comprehensive_imaging_benchmark
results = run_comprehensive_imaging_benchmark()
"

# Run biosignal benchmark
python -c "
from tests.medical.test_biosignal_compression import run_comprehensive_biosignal_benchmark
results = run_comprehensive_biosignal_benchmark()
"

# Run genomics benchmark
python -c "
from tests.medical.test_genomics_transforms import run_comprehensive_genomics_benchmark
results = run_comprehensive_genomics_benchmark()
"
```

### Reference Configurations

```python
# Reference configurations for reproducible benchmarks

IMAGING_CONFIG = {
    'phantom_size': 256,
    'noise_levels': [0.05, 0.10, 0.15],  # Rician sigma
    'threshold_ratios': [0.02, 0.05, 0.10],
    'random_seed': 42
}

BIOSIGNAL_CONFIG = {
    'ecg_duration_sec': 30,
    'ecg_sample_rate': 360,
    'eeg_duration_sec': 60,
    'eeg_sample_rate': 256,
    'keep_ratios': [0.3, 0.5, 0.7],
    'random_seed': 42
}

GENOMICS_CONFIG = {
    'sequence_lengths': [10000, 50000, 100000],
    'kmer_k': 4,
    'contact_map_sizes': [64, 128, 256],
    'keep_ratio': 0.5,
    'random_seed': 42
}
```

---

## Safety Considerations

### IMPORTANT DISCLAIMERS

1. **NOT FOR CLINICAL USE**: This software has not been validated, certified, or approved by any regulatory body (FDA, CE, etc.) for clinical or diagnostic use.

2. **RESEARCH ONLY**: All algorithms and methods are provided for research and educational purposes only.

3. **NO WARRANTY**: The authors make no guarantees about the accuracy, reliability, or safety of this software for any medical application.

4. **SYNTHETIC DATA**: Test results are based on synthetic data that may not reflect real-world clinical conditions.

5. **PHI HANDLING**: Real clinical implementations must comply with HIPAA, GDPR, and other applicable privacy regulations.

### Validation Requirements for Clinical Translation

Before any clinical use, researchers should:

1. Validate on de-identified clinical datasets with proper IRB approval
2. Compare against established clinical standards (e.g., FDA-cleared methods)
3. Document failure modes and edge cases
4. Obtain appropriate regulatory clearances
5. Implement proper quality management systems
6. Conduct formal verification and validation

---

## Contribution Guide

### Adding New Medical Evaluation Recipes

1. **Create test file** in `tests/medical/` following naming convention `test_<domain>_<application>.py`

2. **Include standard sections**:
   ```python
   # Synthetic data generators
   # Quality metrics
   # Pytest test classes
   # Standalone benchmark runner
   ```

3. **Document**:
   - Data requirements
   - Evaluation metrics
   - Expected results
   - Limitations

4. **Add to CI** in `.github/workflows/` if applicable

### Code Style

```python
# Follow project conventions
# - SPDX license header
# - Type hints
# - Docstrings with Args/Returns
# - Pytest for testing
```

---

## License

- Core RFT implementations: See `LICENSE-CLAIMS-NC.md` (research/education only)
- Medical test suite: AGPL-3.0-or-later
- Commercial use requires separate licensing agreement

---

## Contact

For questions about medical applications research:
- Open an issue on GitHub
- Email: luisminier79@gmail.com

---

*Last updated: December 2025*

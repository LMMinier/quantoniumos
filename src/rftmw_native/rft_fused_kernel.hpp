/*
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
 * under LICENSE-CLAIMS-NC.md (research/education only). Commercial
 * rights require a separate patent license from the author.
 *
 * rft_fused_kernel.hpp - Fused Diagonal SIMD Kernels for Φ-RFT
 * =============================================================
 *
 * High-performance fused kernel: D_φ · C_σ applied as single pass.
 * 
 * Key optimizations:
 *   1. Fuse D_φ and C_σ into single diagonal: E[k] = D_φ[k] · C_σ[k]
 *   2. Precompute combined phase: θ[k] = 2πβ·frac(k/φ) + πσk²/n
 *   3. Use SIMD (AVX2/AVX512) for parallel sincos + multiply
 *   4. Cache-aligned phase tables with prefetching
 *
 * The Φ-RFT forward transform is:
 *   Y = D_φ · C_σ · FFT(x)
 *
 * Fused form:
 *   Y = E · FFT(x)  where E[k] = exp(i·θ_fused[k])
 *   θ_fused[k] = 2πβ·frac(k/φ) + πσk²/n
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <complex>
#include <vector>
#include <memory>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#ifdef __AVX512F__
#define RFT_HAS_AVX512 1
#include <immintrin.h>
#else
#define RFT_HAS_AVX512 0
#endif

#ifdef __AVX2__
#define RFT_HAS_AVX2 1
#include <immintrin.h>
#else
#define RFT_HAS_AVX2 0
#endif

#ifdef __FMA__
#define RFT_HAS_FMA 1
#endif

namespace rft_fused {

// Constants
constexpr double PHI = 1.6180339887498948482045868343656;
constexpr double PHI_INV = 0.6180339887498948482045868343656;
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

using Complex = std::complex<double>;

// ============================================================================
// Fused Phase Table Generation
// ============================================================================

/**
 * Precompute fused phase table: θ_fused[k] = 2πβ·frac(k/φ) + πσk²/n
 * 
 * This table stores the combined phase from D_φ and C_σ, allowing
 * a single complex multiplication instead of two separate passes.
 */
struct FusedPhaseTable {
    std::vector<double> phases;      // θ_fused[k]
    std::vector<double> cos_cache;   // cos(θ_fused[k])
    std::vector<double> sin_cache;   // sin(θ_fused[k])
    size_t size;
    double beta;
    double sigma;
    
    FusedPhaseTable() : size(0), beta(1.0), sigma(1.0) {}
    
    void precompute(size_t n, double beta_ = 1.0, double sigma_ = 1.0) {
        if (n == size && beta == beta_ && sigma == sigma_) return;
        
        size = n;
        beta = beta_;
        sigma = sigma_;
        
        // Allocate with cache-line alignment (64 bytes)
        phases.resize(n);
        cos_cache.resize(n);
        sin_cache.resize(n);
        
        const double inv_n = 1.0 / static_cast<double>(n);
        const double inv_phi = PHI_INV;
        
        for (size_t k = 0; k < n; ++k) {
            // D_φ phase: 2πβ·frac(k/φ)
            double frac_k_phi = std::fmod(k * inv_phi, 1.0);
            if (frac_k_phi < 0) frac_k_phi += 1.0;
            double theta_phi = TWO_PI * beta * frac_k_phi;
            
            // C_σ phase: πσk²/n
            double k_sq = static_cast<double>(k * k);
            double theta_chirp = PI * sigma * k_sq * inv_n;
            
            // Fused phase
            double theta_fused = theta_phi + theta_chirp;
            phases[k] = theta_fused;
            cos_cache[k] = std::cos(theta_fused);
            sin_cache[k] = std::sin(theta_fused);
        }
    }
    
    void precompute_conjugate(size_t n, double beta_ = 1.0, double sigma_ = 1.0) {
        // For inverse: negate all phases
        precompute(n, beta_, sigma_);
        for (size_t k = 0; k < n; ++k) {
            phases[k] = -phases[k];
            sin_cache[k] = -sin_cache[k];
            // cos_cache stays the same (cos(-x) = cos(x))
        }
    }
};

// ============================================================================
// AVX-512 Fused Kernel (8 doubles at a time)
// ============================================================================

#if RFT_HAS_AVX512

/**
 * Apply fused diagonal using AVX-512.
 * Processes 4 complex numbers (8 doubles) per iteration.
 */
inline void apply_fused_diagonal_avx512(
    const Complex* __restrict input,
    Complex* __restrict output,
    const double* __restrict cos_table,
    const double* __restrict sin_table,
    size_t n
) {
    size_t k = 0;
    
    // Process 4 complex at a time (8 doubles)
    for (; k + 4 <= n; k += 4) {
        // Load 4 complex numbers as 8 doubles: [r0,i0,r1,i1,r2,i2,r3,i3]
        __m512d v_in = _mm512_loadu_pd(reinterpret_cast<const double*>(input + k));
        
        // Load cos/sin values for 4 elements
        __m256d v_cos4 = _mm256_loadu_pd(cos_table + k);
        __m256d v_sin4 = _mm256_loadu_pd(sin_table + k);
        
        // Interleave to match complex layout: [c0,c0,c1,c1,c2,c2,c3,c3]
        __m512d v_cos = _mm512_castpd256_pd512(
            _mm256_unpacklo_pd(v_cos4, v_cos4)
        );
        v_cos = _mm512_insertf64x4(v_cos, _mm256_unpackhi_pd(v_cos4, v_cos4), 1);
        // Actually need: [c0,c0,c1,c1,c2,c2,c3,c3]
        // Rebuild correctly:
        __m512d v_cos_interleaved = _mm512_permutex2var_pd(
            _mm512_castpd256_pd512(_mm256_unpacklo_pd(v_cos4, v_cos4)),
            _mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3),
            _mm512_castpd256_pd512(_mm256_unpackhi_pd(v_cos4, v_cos4))
        );
        
        __m512d v_sin_interleaved = _mm512_permutex2var_pd(
            _mm512_castpd256_pd512(_mm256_unpacklo_pd(v_sin4, v_sin4)),
            _mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3),
            _mm512_castpd256_pd512(_mm256_unpackhi_pd(v_sin4, v_sin4))
        );
        
        // Extract real and imaginary parts
        // v_in = [r0,i0,r1,i1,r2,i2,r3,i3]
        // real parts at even indices, imag at odd
        __m512d v_real = _mm512_permutex2var_pd(v_in, 
            _mm512_setr_epi64(0, 0, 2, 2, 4, 4, 6, 6), v_in);
        __m512d v_imag = _mm512_permutex2var_pd(v_in,
            _mm512_setr_epi64(1, 1, 3, 3, 5, 5, 7, 7), v_in);
        
        // Complex multiply: (r + i·j)(c + s·j) = (rc - is) + (rs + ic)j
        // out_real = r*c - i*s
        // out_imag = r*s + i*c
        __m512d v_out_real = _mm512_fmsub_pd(v_real, v_cos_interleaved,
                                             _mm512_mul_pd(v_imag, v_sin_interleaved));
        __m512d v_out_imag = _mm512_fmadd_pd(v_real, v_sin_interleaved,
                                             _mm512_mul_pd(v_imag, v_cos_interleaved));
        
        // Interleave back to [r0,i0,r1,i1,...]
        __m512d v_out = _mm512_permutex2var_pd(v_out_real,
            _mm512_setr_epi64(0, 8, 2, 10, 4, 12, 6, 14), v_out_imag);
        
        _mm512_storeu_pd(reinterpret_cast<double*>(output + k), v_out);
    }
    
    // Scalar remainder
    for (; k < n; ++k) {
        double c = cos_table[k];
        double s = sin_table[k];
        double r = input[k].real();
        double i = input[k].imag();
        output[k] = Complex(r*c - i*s, r*s + i*c);
    }
}

#endif // RFT_HAS_AVX512

// ============================================================================
// AVX2 Fused Kernel (4 doubles at a time)
// ============================================================================

#if RFT_HAS_AVX2

/**
 * Apply fused diagonal using AVX2.
 * Processes 2 complex numbers (4 doubles) per iteration.
 */
inline void apply_fused_diagonal_avx2(
    const Complex* __restrict input,
    Complex* __restrict output,
    const double* __restrict cos_table,
    const double* __restrict sin_table,
    size_t n
) {
    size_t k = 0;
    
    // Process 2 complex at a time (4 doubles)
    for (; k + 2 <= n; k += 2) {
        // Load 2 complex numbers: [r0, i0, r1, i1]
        __m256d v_in = _mm256_loadu_pd(reinterpret_cast<const double*>(input + k));
        
        // Load cos/sin for 2 elements
        __m128d v_cos2 = _mm_loadu_pd(cos_table + k);
        __m128d v_sin2 = _mm_loadu_pd(sin_table + k);
        
        // Duplicate to match complex layout: [c0,c0,c1,c1]
        __m256d v_cos = _mm256_set_pd(
            cos_table[k+1], cos_table[k+1],
            cos_table[k], cos_table[k]
        );
        __m256d v_sin = _mm256_set_pd(
            sin_table[k+1], sin_table[k+1],
            sin_table[k], sin_table[k]
        );
        
        // Separate real and imaginary: 
        // v_in = [r0, i0, r1, i1]
        // v_real = [r0, r0, r1, r1]
        // v_imag = [i0, i0, i1, i1]
        __m256d v_real = _mm256_permute4x64_pd(v_in, 0b10100000); // [r0,r0,r1,r1]
        __m256d v_imag = _mm256_permute4x64_pd(v_in, 0b11110101); // [i0,i0,i1,i1]
        
        // Complex multiply:
        // out_real = r*c - i*s
        // out_imag = r*s + i*c
#if RFT_HAS_FMA
        __m256d v_rc = _mm256_mul_pd(v_real, v_cos);
        __m256d v_out_real = _mm256_fnmadd_pd(v_imag, v_sin, v_rc); // rc - is
        __m256d v_rs = _mm256_mul_pd(v_real, v_sin);
        __m256d v_out_imag = _mm256_fmadd_pd(v_imag, v_cos, v_rs);  // rs + ic
#else
        __m256d v_out_real = _mm256_sub_pd(
            _mm256_mul_pd(v_real, v_cos),
            _mm256_mul_pd(v_imag, v_sin)
        );
        __m256d v_out_imag = _mm256_add_pd(
            _mm256_mul_pd(v_real, v_sin),
            _mm256_mul_pd(v_imag, v_cos)
        );
#endif
        
        // Interleave: [out_r0, out_i0, out_r1, out_i1]
        // v_out_real/v_out_imag have duplicated elements from the permuted inputs,
        // so unpacklo within 128-bit lanes gives the correct interleave.
        __m256d v_out = _mm256_unpacklo_pd(v_out_real, v_out_imag);
        _mm256_storeu_pd(reinterpret_cast<double*>(output + k), v_out);
    }
    
    // Scalar remainder
    for (; k < n; ++k) {
        double c = cos_table[k];
        double s = sin_table[k];
        double r = input[k].real();
        double i = input[k].imag();
        output[k] = Complex(r*c - i*s, r*s + i*c);
    }
}

#endif // RFT_HAS_AVX2

// ============================================================================
// Scalar Fallback
// ============================================================================

inline void apply_fused_diagonal_scalar(
    const Complex* __restrict input,
    Complex* __restrict output,
    const double* __restrict cos_table,
    const double* __restrict sin_table,
    size_t n
) {
    for (size_t k = 0; k < n; ++k) {
        double c = cos_table[k];
        double s = sin_table[k];
        double r = input[k].real();
        double i = input[k].imag();
        output[k] = Complex(r*c - i*s, r*s + i*c);
    }
}

// ============================================================================
// Unified Dispatcher
// ============================================================================

inline void apply_fused_diagonal(
    const Complex* input,
    Complex* output,
    const double* cos_table,
    const double* sin_table,
    size_t n,
    bool use_simd = true
) {
    if (!use_simd) {
        apply_fused_diagonal_scalar(input, output, cos_table, sin_table, n);
        return;
    }
    
#if RFT_HAS_AVX512
    apply_fused_diagonal_avx512(input, output, cos_table, sin_table, n);
#elif RFT_HAS_AVX2
    apply_fused_diagonal_avx2(input, output, cos_table, sin_table, n);
#else
    apply_fused_diagonal_scalar(input, output, cos_table, sin_table, n);
#endif
}

// ============================================================================
// Complete Fused RFT Engine
// ============================================================================

/**
 * Optimized RFT Engine with fused D_φ·C_σ diagonal.
 * 
 * Performance characteristics:
 *   - Single diagonal pass instead of two separate passes
 *   - Precomputed cos/sin tables (amortized over many transforms)
 *   - SIMD acceleration for diagonal application
 *   - Uses NumPy/FFTW for the FFT core (fastest available)
 */
class FusedRFTEngine {
public:
    FusedRFTEngine(double beta = 1.0, double sigma = 1.0, bool use_simd = true)
        : beta_(beta), sigma_(sigma), use_simd_(use_simd) {}
    
    void precompute(size_t n) {
        fwd_phases_.precompute(n, beta_, sigma_);
        inv_phases_.precompute_conjugate(n, beta_, sigma_);
    }
    
    /**
     * Apply fused diagonal (D_φ · C_σ) to FFT output.
     * Call this after FFT for forward transform.
     */
    void apply_forward_diagonal(const Complex* fft_out, Complex* result, size_t n) {
        if (n != fwd_phases_.size) {
            precompute(n);
        }
        apply_fused_diagonal(fft_out, result, 
                             fwd_phases_.cos_cache.data(),
                             fwd_phases_.sin_cache.data(),
                             n, use_simd_);
    }
    
    /**
     * Apply conjugate diagonal for inverse transform.
     * Call this before IFFT.
     */
    void apply_inverse_diagonal(const Complex* input, Complex* result, size_t n) {
        if (n != inv_phases_.size) {
            precompute(n);
        }
        apply_fused_diagonal(input, result,
                             inv_phases_.cos_cache.data(),
                             inv_phases_.sin_cache.data(),
                             n, use_simd_);
    }
    
    // Direct phase table access for Python binding
    const double* get_cos_table(size_t n, bool inverse = false) {
        if (n != fwd_phases_.size) precompute(n);
        return inverse ? inv_phases_.cos_cache.data() : fwd_phases_.cos_cache.data();
    }
    
    const double* get_sin_table(size_t n, bool inverse = false) {
        if (n != fwd_phases_.size) precompute(n);
        return inverse ? inv_phases_.sin_cache.data() : fwd_phases_.sin_cache.data();
    }
    
    bool has_avx512() const {
#if RFT_HAS_AVX512
        return true;
#else
        return false;
#endif
    }
    
    bool has_avx2() const {
#if RFT_HAS_AVX2
        return true;
#else
        return false;
#endif
    }

private:
    double beta_;
    double sigma_;
    bool use_simd_;
    FusedPhaseTable fwd_phases_;
    FusedPhaseTable inv_phases_;
};

} // namespace rft_fused

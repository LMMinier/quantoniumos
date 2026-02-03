/*
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
 * under LICENSE-CLAIMS-NC.md (research/education only). Commercial
 * rights require a separate patent license from the author.
 *
 * rftmw_core.hpp - Φ-RFT Transform C++ Core
 * ==========================================
 *
 * High-performance RFTMW implementation with SIMD acceleration.
 * This is the heavy-lifting engine; Python calls this via pybind11.
 *
 * Architecture:
 *   ASM kernels (rft_kernel_asm.asm, feistel_round48.asm, etc.)
 *       ↓
 *   C++ SIMD wrappers (rftmw_asm_kernels.hpp)
 *       ↓
 *   C++ engine (this file)
 *       ↓
 *   Python bindings (rftmw_python.cpp via pybind11)
 *
 * Assembly Kernels Available:
 *   - rft_transform_asm: Unitary RFT transform (matrix-vector)
 *   - rft_basis_multiply_asm: Matrix-vector with transpose
 *   - rft_quantum_gate_asm: Quantum gate application
 *   - qsc_symbolic_compression_asm: Million-qubit symbolic compression
 *   - feistel_encrypt_batch_asm: 48-round Feistel cipher (9.2 MB/s target)
 */

// Transform-theory reference objects (companion shift, golden drift model, K99, etc.):
//   see rftmw_transform_theorems.hpp

#pragma once

#include <complex>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#define RFTMW_HAS_AVX2 1
#else
#define RFTMW_HAS_AVX2 0
#endif

#ifdef __AVX512F__
#define RFTMW_HAS_AVX512 1
#else
#define RFTMW_HAS_AVX512 0
#endif

#ifdef __FMA__
#define RFTMW_HAS_FMA 1
#else
#define RFTMW_HAS_FMA 0
#endif

// Check if ASM kernels are available (set by build system)
#ifndef RFTMW_ENABLE_ASM
#define RFTMW_ENABLE_ASM 0
#endif

#if RFTMW_ENABLE_ASM
#include "rftmw_asm_kernels.hpp"
#endif

namespace rftmw {

// Golden ratio constant (compile-time)
constexpr double PHI = 1.6180339887498948482045868343656;
constexpr double PHI_INV = 0.6180339887498948482045868343656;
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;
using RealVec = std::vector<double>;

// ============================================================================
// Gram Normalization for True Canonical RFT
// ============================================================================

/**
 * Compute the raw φ-grid basis Φ[n,k] = exp(2πi × frac((k+1)φ) × n) / √N
 * Returns N×N matrix in row-major order.
 */
inline ComplexVec compute_raw_phi_basis(size_t N) {
    ComplexVec Phi(N * N);
    const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(N));
    
    for (size_t k = 0; k < N; ++k) {
        double freq = static_cast<double>(k + 1) * PHI;
        double f_k = freq - std::floor(freq);  // frac((k+1)φ)
        
        for (size_t n = 0; n < N; ++n) {
            double angle = TWO_PI * f_k * static_cast<double>(n);
            // Φ[n, k] stored as Phi[n * N + k]
            Phi[n * N + k] = Complex(std::cos(angle), std::sin(angle)) * inv_sqrt_n;
        }
    }
    return Phi;
}

/**
 * Compute G = Φ†Φ (Gram matrix) - Hermitian positive definite.
 * Input: Phi is N×N in row-major (Phi[row * N + col])
 * Output: G is N×N Hermitian matrix
 */
inline ComplexVec compute_gram_matrix(const ComplexVec& Phi, size_t N) {
    ComplexVec G(N * N, Complex(0.0, 0.0));
    
    // G[i,j] = Σₙ Φ*[n,i] × Φ[n,j]
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Complex sum(0.0, 0.0);
            for (size_t n = 0; n < N; ++n) {
                // Φ[n,i] and Φ[n,j]
                sum += std::conj(Phi[n * N + i]) * Phi[n * N + j];
            }
            G[i * N + j] = sum;
        }
    }
    return G;
}

/**
 * Jacobi eigendecomposition for Hermitian matrix.
 * Computes G = V D V† where D is diagonal (eigenvalues) and V is unitary.
 * 
 * This is O(N³) but numerically stable for small N.
 * For N > 256, consider using LAPACK.
 */
inline void jacobi_eigen_hermitian(
    ComplexVec& A,      // N×N input/output: eigenvalues on diagonal after
    ComplexVec& V,      // N×N output: eigenvector columns
    RealVec& eigenvals, // N output: eigenvalues
    size_t N,
    size_t max_sweeps = 50  // Each sweep processes all pairs
) {
    // Initialize V to identity
    V.assign(N * N, Complex(0.0, 0.0));
    for (size_t i = 0; i < N; ++i) {
        V[i * N + i] = Complex(1.0, 0.0);
    }
    
    eigenvals.resize(N);
    
    // Jacobi sweeps - process all upper triangular pairs each sweep
    for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_off = 0.0;
        
        // Compute max off-diagonal for convergence check
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                max_off = std::max(max_off, std::abs(A[i * N + j]));
            }
        }
        
        // Convergence check
        if (max_off < 1e-15) break;
        
        // Process all pairs in this sweep
        for (size_t p = 0; p < N - 1; ++p) {
            for (size_t q = p + 1; q < N; ++q) {
                Complex Apq = A[p * N + q];
                if (std::abs(Apq) < 1e-16) continue;
                
                double App = std::real(A[p * N + p]);
                double Aqq = std::real(A[q * N + q]);
                
                // Compute rotation angle
                double diff = Aqq - App;
                double t;  // tan(theta)
                
                if (std::abs(diff) < 1e-16) {
                    t = 1.0;
                } else {
                    double phi = diff / (2.0 * std::abs(Apq));
                    t = 1.0 / (std::abs(phi) + std::sqrt(phi * phi + 1.0));
                    if (phi < 0) t = -t;
                }
                
                double c = 1.0 / std::sqrt(1.0 + t * t);  // cos(theta)
                double s = t * c;                          // sin(theta)
                
                // Phase factor to make Apq real
                Complex tau = (std::abs(Apq) > 1e-16) ? 
                    Apq / std::abs(Apq) : Complex(1.0, 0.0);
                
                // Apply rotation to A: A' = J† A J
                // Update rows p and q
                for (size_t k = 0; k < N; ++k) {
                    if (k != p && k != q) {
                        Complex Akp = A[k * N + p];
                        Complex Akq = A[k * N + q];
                        A[k * N + p] = c * Akp - s * std::conj(tau) * Akq;
                        A[k * N + q] = s * tau * Akp + c * Akq;
                        A[p * N + k] = std::conj(A[k * N + p]);
                        A[q * N + k] = std::conj(A[k * N + q]);
                    }
                }
                
                // Update diagonal elements
                double Apq_abs = std::abs(Apq);
                A[p * N + p] = Complex(App - t * Apq_abs, 0.0);
                A[q * N + q] = Complex(Aqq + t * Apq_abs, 0.0);
                A[p * N + q] = Complex(0.0, 0.0);
                A[q * N + p] = Complex(0.0, 0.0);
                
                // Accumulate eigenvectors: V' = V J
                for (size_t k = 0; k < N; ++k) {
                    Complex Vkp = V[k * N + p];
                    Complex Vkq = V[k * N + q];
                    V[k * N + p] = c * Vkp - s * std::conj(tau) * Vkq;
                    V[k * N + q] = s * tau * Vkp + c * Vkq;
                }
            }
        }
    }
    
    // Extract eigenvalues from diagonal
    for (size_t i = 0; i < N; ++i) {
        eigenvals[i] = std::real(A[i * N + i]);
    }
}

/**
 * Compute G^{-1/2} for positive definite Hermitian G.
 * Uses eigendecomposition: G = V D V†, so G^{-1/2} = V D^{-1/2} V†
 */
inline ComplexVec compute_gram_inv_sqrt(const ComplexVec& G, size_t N) {
    // Copy G for eigendecomposition (it gets modified)
    ComplexVec A = G;
    ComplexVec V(N * N);
    RealVec eigenvals(N);
    
    jacobi_eigen_hermitian(A, V, eigenvals, N);
    
    // Compute V D^{-1/2} V†
    ComplexVec result(N * N, Complex(0.0, 0.0));
    
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Complex sum(0.0, 0.0);
            for (size_t k = 0; k < N; ++k) {
                // D^{-1/2}[k,k] = 1/sqrt(eigenvals[k])
                double d_inv_sqrt = (eigenvals[k] > 1e-14) ? 
                    1.0 / std::sqrt(eigenvals[k]) : 0.0;
                // V[i,k] × D^{-1/2}[k,k] × V†[k,j] = V[i,k] × d × conj(V[j,k])
                sum += V[i * N + k] * d_inv_sqrt * std::conj(V[j * N + k]);
            }
            result[i * N + j] = sum;
        }
    }
    
    return result;
}

/**
 * Compute the TRUE CANONICAL RFT basis: U = Φ × (Φ†Φ)^{-1/2}
 * This is the Gram-normalized unitary basis.
 * 
 * Complexity: O(N³) for eigendecomposition
 * 
 * Returns N×N unitary matrix U where U†U = I exactly.
 */
inline ComplexVec compute_canonical_basis(size_t N) {
    // Step 1: Compute raw φ-grid basis Φ
    ComplexVec Phi = compute_raw_phi_basis(N);
    
    // Step 2: Compute Gram matrix G = Φ†Φ
    ComplexVec G = compute_gram_matrix(Phi, N);
    
    // Step 3: Compute G^{-1/2}
    ComplexVec G_inv_sqrt = compute_gram_inv_sqrt(G, N);
    
    // Step 4: Compute U = Φ × G^{-1/2}
    ComplexVec U(N * N, Complex(0.0, 0.0));
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Complex sum(0.0, 0.0);
            for (size_t k = 0; k < N; ++k) {
                sum += Phi[i * N + k] * G_inv_sqrt[k * N + j];
            }
            U[i * N + j] = sum;
        }
    }
    
    return U;
}

// ============================================================================
// SIMD-Accelerated Phase Modulation Kernels
// ============================================================================

#if RFTMW_HAS_AVX2

/**
 * AVX2-accelerated golden phase computation for CANONICAL RFT.
 * 
 * CANONICAL DEFINITION (per README.md - January 2026):
 *   θ[k] = 2π·frac((k+1)·φ)
 * 
 * This computes the canonical φ-grid frequencies used in the
 * Gram-normalized RFT basis.
 */
inline void compute_golden_phases_avx2(double* phases, size_t n) {
    // Process 4 doubles at a time with AVX2
    __m256d v_two_pi = _mm256_set1_pd(TWO_PI);
    __m256d v_phi = _mm256_set1_pd(PHI);
    __m256d v_one = _mm256_set1_pd(1.0);
    
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        // k+1 values
        __m256d v_k_plus_1 = _mm256_set_pd(
            static_cast<double>(k + 4),
            static_cast<double>(k + 3),
            static_cast<double>(k + 2),
            static_cast<double>(k + 1)
        );
        
        // freq = (k+1) * PHI
        __m256d v_freq = _mm256_mul_pd(v_k_plus_1, v_phi);
        
        // frac_part = freq - floor(freq)
        __m256d v_floor = _mm256_floor_pd(v_freq);
        __m256d v_frac = _mm256_sub_pd(v_freq, v_floor);
        
        // theta = 2π * frac_part
        __m256d v_theta = _mm256_mul_pd(v_two_pi, v_frac);
        _mm256_storeu_pd(phases + k, v_theta);
    }
    
    // Handle remainder with scalar
    for (; k < n; ++k) {
        double freq = static_cast<double>(k + 1) * PHI;
        double frac_part = freq - std::floor(freq);
        phases[k] = TWO_PI * frac_part;
    }
}

/**
 * AVX2-accelerated complex multiplication with phase rotation.
 * result[k] = input[k] * exp(i * phase[k])
 */
inline void apply_phase_rotation_avx2(
    const Complex* input,
    Complex* output,
    const double* phases,
    size_t n
) {
    // Process 2 complex numbers at a time (4 doubles = 2 complex)
    for (size_t k = 0; k < n; k += 2) {
        if (k + 2 <= n) {
            // Load 2 complex numbers (4 doubles: r0, i0, r1, i1)
            __m256d v_in = _mm256_loadu_pd(reinterpret_cast<const double*>(input + k));
            
            // Compute cos/sin of phases
            double c0 = std::cos(phases[k]);
            double s0 = std::sin(phases[k]);
            double c1 = std::cos(phases[k + 1]);
            double s1 = std::sin(phases[k + 1]);
            
            // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            // input = [r0, i0, r1, i1]
            // rotation = [c0, s0, c1, s1]
            
            double r0 = input[k].real(), i0 = input[k].imag();
            double r1 = input[k+1].real(), i1 = input[k+1].imag();
            
            output[k] = Complex(r0*c0 - i0*s0, r0*s0 + i0*c0);
            output[k+1] = Complex(r1*c1 - i1*s1, r1*s1 + i1*c1);
        } else {
            // Handle last element if n is odd
            double c = std::cos(phases[k]);
            double s = std::sin(phases[k]);
            output[k] = Complex(
                input[k].real()*c - input[k].imag()*s,
                input[k].real()*s + input[k].imag()*c
            );
        }
    }
}

#endif // RFTMW_HAS_AVX2

// ============================================================================
// Scalar Fallback Implementations
// ============================================================================

/**
 * Compute fused phase diagonal for CANONICAL RFT (January 2026).
 * 
 * CANONICAL DEFINITION (per README.md and THEOREMS_RFT_IRONCLAD.md):
 *   Φ[n,k] = exp(j 2π frac((k+1)φ) n) / √N
 *   
 * For the phase-diagonal approximation used in fast transforms:
 *   θ[k] = 2π·frac((k+1)·φ)
 * 
 * where frac(x) = x mod 1 (fractional part)
 * 
 * NOTE: This is different from the deprecated φ-phase FFT which used:
 *   θ[k] = 2π·frac(k/φ) + π·k²/n  (DEPRECATED)
 */
inline void compute_golden_phases_scalar(double* phases, size_t n) {
    // CANONICAL RFT: frequencies at frac((k+1)·φ)
    for (size_t k = 0; k < n; ++k) {
        // f_k = frac((k+1) × φ) - normalized frequency in [0, 1)
        double freq = static_cast<double>(k + 1) * PHI;
        double frac_part = freq - std::floor(freq);  // frac(x) = x - floor(x)
        
        // θ[k] = 2π × frac((k+1)·φ)
        phases[k] = TWO_PI * frac_part;
    }
}

inline void apply_phase_rotation_scalar(
    const Complex* input,
    Complex* output,
    const double* phases,
    size_t n
) {
    for (size_t k = 0; k < n; ++k) {
        double c = std::cos(phases[k]);
        double s = std::sin(phases[k]);
        output[k] = Complex(
            input[k].real() * c - input[k].imag() * s,
            input[k].real() * s + input[k].imag() * c
        );
    }
}

// ============================================================================
// Cooley-Tukey FFT Implementation (for RFT core)
// ============================================================================

/**
 * In-place radix-2 Cooley-Tukey FFT.
 * n must be a power of 2.
 * 
 * NOTE: This FFT does NOT apply any normalization.
 * Normalization is handled by the calling code (RFTMWEngine).
 */
inline void fft_radix2_inplace(Complex* data, size_t n, bool inverse = false) {
    if (n <= 1) return;
    
    // Bit-reversal permutation
    size_t j = 0;
    for (size_t i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(data[i], data[j]);
        }
        size_t m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    
    // Cooley-Tukey iterative FFT
    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = (inverse ? TWO_PI : -TWO_PI) / static_cast<double>(len);
        Complex wlen(std::cos(angle), std::sin(angle));
        
        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                Complex u = data[i + k];
                Complex t = w * data[i + k + len / 2];
                data[i + k] = u + t;
                data[i + k + len / 2] = u - t;
                w *= wlen;
            }
        }
    }
    
    // NO normalization here - handled by RFTMWEngine
}

/**
 * Mixed-radix FFT for non-power-of-2 sizes.
 * Falls back to DFT for prime factors.
 * 
 * NOTE: This FFT does NOT apply any normalization.
 * Normalization is handled by the calling code (RFTMWEngine).
 */
inline void fft_mixed_radix(Complex* data, size_t n, bool inverse = false) {
    // For power-of-2, use fast path
    if ((n & (n - 1)) == 0) {
        fft_radix2_inplace(data, n, inverse);
        return;
    }
    
    // General DFT for non-power-of-2 (O(n^2) fallback)
    ComplexVec result(n);
    double sign = inverse ? 1.0 : -1.0;
    
    for (size_t k = 0; k < n; ++k) {
        result[k] = Complex(0.0, 0.0);
        for (size_t j = 0; j < n; ++j) {
            double angle = sign * TWO_PI * static_cast<double>(k * j) / static_cast<double>(n);
            result[k] += data[j] * Complex(std::cos(angle), std::sin(angle));
        }
        // NO normalization here - handled by RFTMWEngine
    }
    
    std::copy(result.begin(), result.end(), data);
}

// ============================================================================
// RFTMW Transform Class
// ============================================================================

class RFTMWEngine {
public:
    enum class Normalization {
        NONE,      // No normalization
        FORWARD,   // 1/n on forward
        ORTHO,     // 1/sqrt(n) on both
        BACKWARD   // 1/n on inverse (default FFT)
    };
    
    enum class Backend {
        AUTO,      // Auto-select best available
        ASM,       // Force assembly kernels
        SIMD,      // Force C++ SIMD (AVX2/SSE)
        SCALAR     // Force scalar fallback
    };
    
    // RFT Variant - matches C kernel variants
    enum class Variant {
        LEGACY,     // Original phase-modulated FFT (deprecated)
        CANONICAL,  // USPTO Patent 19/169,399 Claim 1: fₖ=(k+1)×φ, θₖ=2πk/φ
        BINARY_WAVE // USPTO Patent 19/169,399: BinaryRFT wave-domain logic
    };

private:
    size_t max_size_;
    Normalization norm_;
    Backend backend_;
    Variant variant_;
    RealVec phase_cache_;
    ComplexVec basis_cache_;  // For ASM kernel basis matrix
    ComplexVec canonical_U_;  // Gram-normalized canonical basis U = Φ(Φ†Φ)^{-1/2}
    size_t canonical_N_ = 0;  // Size of cached canonical basis
    bool use_simd_;
    bool use_asm_;
    size_t last_transform_size_ = 0;  // Track last computed phase size (chirp depends on n)
    
public:
    explicit RFTMWEngine(
        size_t max_size = 65536, 
        Normalization norm = Normalization::ORTHO,
        Backend backend = Backend::AUTO,
        Variant variant = Variant::CANONICAL  // Default to canonical
    )
        : max_size_(max_size)
        , norm_(norm)
        , backend_(backend)
        , variant_(variant)
        , phase_cache_(max_size)
        , use_simd_(RFTMW_HAS_AVX2)
        , use_asm_(RFTMW_ENABLE_ASM)
    {
        // Resolve AUTO backend
        if (backend_ == Backend::AUTO) {
#if RFTMW_ENABLE_ASM
            use_asm_ = true;
            use_simd_ = true;
#elif RFTMW_HAS_AVX2
            use_asm_ = false;
            use_simd_ = true;
#else
            use_asm_ = false;
            use_simd_ = false;
#endif
        } else if (backend_ == Backend::ASM) {
#if RFTMW_ENABLE_ASM
            use_asm_ = true;
#else
            throw std::runtime_error("ASM backend requested but not compiled with -DRFTMW_ENABLE_ASM=ON");
#endif
        } else if (backend_ == Backend::SIMD) {
            use_asm_ = false;
            use_simd_ = RFTMW_HAS_AVX2;
        } else {
            use_asm_ = false;
            use_simd_ = false;
        }
        
        // Pre-compute phases for max size
        precompute_phases(max_size);
        
        // Pre-compute basis matrix for ASM kernel if enabled
        if (use_asm_) {
            precompute_basis(max_size);
        }
    }
    
    /**
     * Pre-compute RFT basis matrix for ASM kernel.
     * 
     * CANONICAL DEFINITION (January 2026, per README.md):
     *   Φ[n,k] = exp(j 2π frac((k+1)φ) n) / √N
     *   
     * The Gram normalization Φ̃ = Φ (ΦᴴΦ)^{-1/2} ensures exact unitarity.
     * For the C++ engine, we use the raw φ-grid basis since the
     * Gram normalization is applied in Python for the canonical transform.
     * 
     * For LEGACY variant (deprecated):
     *   Basis[i,j] = exp(i * 2π * φ^(i*j/n))
     * 
     * Note: Only precompute for small sizes to avoid memory explosion
     * (n×n matrix would be 64GB for n=65536)
     */
    void precompute_basis(size_t n) {
        // Limit basis precomputation to avoid O(n²) memory explosion
        // For large transforms, compute on-the-fly or use ASM kernel's internal basis
        constexpr size_t MAX_BASIS_SIZE = 4096;
        
        if (n > MAX_BASIS_SIZE) {
            // Don't precompute giant basis matrices
            // ASM kernel will handle large transforms differently
            basis_cache_.clear();
            return;
        }
        
        basis_cache_.resize(n * n);
        const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n));
        
        if (variant_ == Variant::CANONICAL || variant_ == Variant::BINARY_WAVE) {
            // CANONICAL RFT (January 2026) - per README.md and THEOREMS_RFT_IRONCLAD.md
            // Φ[n,k] = exp(j 2π frac((k+1)φ) n) / √N
            //
            // This is the raw φ-grid exponential basis.
            // Full Gram normalization Φ̃ = Φ (ΦᴴΦ)^{-1/2} is applied in Python.
            for (size_t k = 0; k < n; ++k) {
                // f_k = frac((k+1) × φ) - normalized frequency in [0, 1)
                double freq = static_cast<double>(k + 1) * PHI;
                double f_k = freq - std::floor(freq);  // frac(x) = x - floor(x)
                
                for (size_t n_idx = 0; n_idx < n; ++n_idx) {
                    // Φ[n,k] = exp(j 2π f_k n) / √N
                    double angle = TWO_PI * f_k * static_cast<double>(n_idx);
                    basis_cache_[k * n + n_idx] = Complex(
                        std::cos(angle) * inv_sqrt_n,
                        std::sin(angle) * inv_sqrt_n
                    );
                }
            }
        } else {
            // Legacy phase-modulated FFT (DEPRECATED)
            const double inv_n2 = 1.0 / static_cast<double>(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double exponent = static_cast<double>(i * j) * inv_n2;
                    double phase = TWO_PI * std::pow(PHI, exponent);
                    basis_cache_[i * n + j] = Complex(std::cos(phase), std::sin(phase));
                }
            }
        }
    }
    
    void precompute_phases(size_t n) {
        if (n > phase_cache_.size()) {
            phase_cache_.resize(n);
        }
        
#if RFTMW_HAS_AVX2
        if (use_simd_) {
            compute_golden_phases_avx2(phase_cache_.data(), n);
        } else
#endif
        {
            compute_golden_phases_scalar(phase_cache_.data(), n);
        }
    }
    
    /**
     * Forward Hybrid RFT transform (O(N log N) approximation).
     * 
     * Computes:
     *   Y = E · FFT(x) / √N
     * 
     * where E[k] = exp(i · θ[k])
     *       θ[k] = 2π·frac((k+1)·φ)
     * 
     * Phase modulation is applied AFTER FFT.
     * 
     * NOTE: This is an O(N log N) APPROXIMATION, not the true canonical RFT.
     * For the true RFT (Φ†x matrix multiply), use forward().
     */
    ComplexVec forward_hybrid(const RealVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        // Always recompute phases for the actual transform size
        // (chirp term θ_chirp = πk²/n depends on n)
        if (n != last_transform_size_) {
            precompute_phases(n);
            last_transform_size_ = n;
        }
        
        // Convert real to complex
        ComplexVec data(n);
        for (size_t k = 0; k < n; ++k) {
            data[k] = Complex(input[k], 0.0);
        }
        
        // Apply FFT first
        fft_mixed_radix(data.data(), n, false);
        
        // Apply phase modulation AFTER FFT (matches Python)
        ComplexVec modulated(n);
        
#if RFTMW_HAS_AVX2
        if (use_simd_) {
            apply_phase_rotation_avx2(data.data(), modulated.data(), phase_cache_.data(), n);
        } else
#endif
        {
            apply_phase_rotation_scalar(data.data(), modulated.data(), phase_cache_.data(), n);
        }
        
        // Apply normalization
        apply_normalization(modulated, false);
        
        return modulated;
    }
    
    ComplexVec forward_hybrid_complex(const ComplexVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        // Always recompute phases for the actual transform size
        if (n != last_transform_size_) {
            precompute_phases(n);
            last_transform_size_ = n;
        }
        
        // Copy input and apply FFT
        ComplexVec data = input;
        fft_mixed_radix(data.data(), n, false);
        
        // Apply phase modulation AFTER FFT
        ComplexVec modulated(n);
        
#if RFTMW_HAS_AVX2
        if (use_simd_) {
            apply_phase_rotation_avx2(data.data(), modulated.data(), phase_cache_.data(), n);
        } else
#endif
        {
            apply_phase_rotation_scalar(data.data(), modulated.data(), phase_cache_.data(), n);
        }
        
        apply_normalization(modulated, false);
        
        return modulated;
    }
    
    /**
     * Inverse Hybrid RFT transform (O(N log N) approximation).
     * 
     * Computes:
     *   x = IFFT(E† · Y)
     * 
     * where E† = conj(E) = exp(-i · θ)
     * 
     * Inverse phase modulation is applied BEFORE IFFT.
     * 
     * NOTE: This is an O(N log N) APPROXIMATION.
     * For the true inverse RFT, use inverse().
     */
    RealVec inverse_hybrid(const ComplexVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        // Always recompute phases for the actual transform size
        if (n != last_transform_size_) {
            precompute_phases(n);
            last_transform_size_ = n;
        }
        
        // Compute conjugate phases for inverse
        RealVec phases_neg(n);
        for (size_t k = 0; k < n; ++k) {
            phases_neg[k] = -phase_cache_[k];
        }
        
        // Apply inverse phase modulation FIRST (before IFFT)
        ComplexVec demodulated(n);
        apply_phase_rotation_scalar(input.data(), demodulated.data(), phases_neg.data(), n);
        
        // Apply IFFT
        fft_mixed_radix(demodulated.data(), n, true);
        
        // Apply normalization
        apply_normalization(demodulated, true);
        
        // Extract real part
        RealVec result(n);
        for (size_t k = 0; k < n; ++k) {
            result[k] = demodulated[k].real();
        }
        
        return result;
    }
    
    ComplexVec inverse_hybrid_complex(const ComplexVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        // Always recompute phases for the actual transform size
        if (n != last_transform_size_) {
            precompute_phases(n);
            last_transform_size_ = n;
        }
        
        ComplexVec data = input;
        fft_mixed_radix(data.data(), n, true);
        
        RealVec phases_neg(n);
        for (size_t k = 0; k < n; ++k) {
            phases_neg[k] = -phase_cache_[k];
        }
        
        ComplexVec demodulated(n);
        apply_phase_rotation_scalar(data.data(), demodulated.data(), phases_neg.data(), n);
        
        apply_normalization(demodulated, true);
        
        return demodulated;
    }
    
    /**
     * Forward RFT Transform (USPTO Patent 19/169,399 Claim 1)
     * 
     * TRUE CANONICAL RFT: X = U† × x (O(N²) matrix-vector product)
     * 
     * Uses the Gram-normalized canonical basis:
     *   U = Φ × (Φ†Φ)^{-1/2}
     * 
     * where Φ[n,k] = exp(2πi × frac((k+1)φ) × n) / √N
     * 
     * This basis is EXACTLY UNITARY: U†U = I
     * 
     * The Gram normalization is computed once and cached.
     */
    ComplexVec forward(const RealVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        // Ensure canonical basis is computed for this size
        ensure_canonical_basis(n);
        
        ComplexVec result(n);
        
        // X = U† × x  (analysis transform)
        for (size_t k = 0; k < n; ++k) {
            Complex sum(0.0, 0.0);
            for (size_t m = 0; m < n; ++m) {
                // U†[k,m] = conj(U[m,k])
                sum += std::conj(canonical_U_[m * n + k]) * input[m];
            }
            result[k] = sum;
        }
        
        return result;
    }
    
    /**
     * Ensure canonical basis U = Φ(Φ†Φ)^{-1/2} is computed for size n.
     * Caches the result for reuse.
     */
    void ensure_canonical_basis(size_t n) {
        if (canonical_N_ == n && !canonical_U_.empty()) {
            return;  // Already computed
        }
        
        // Compute true canonical basis with Gram normalization
        canonical_U_ = compute_canonical_basis(n);
        canonical_N_ = n;
    }
    
    /**
     * Inverse RFT Transform (USPTO Patent 19/169,399 Claim 1)
     * 
     * TRUE CANONICAL RFT INVERSE: x = U × X (O(N²) matrix-vector product)
     * 
     * Uses the Gram-normalized canonical basis:
     *   U = Φ × (Φ†Φ)^{-1/2}
     * 
     * Since U is unitary, U† × U = I, so inverse(forward(x)) = x exactly.
     */
    RealVec inverse(const ComplexVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        // Ensure canonical basis is computed for this size
        ensure_canonical_basis(n);
        
        RealVec result(n);
        
        // x = U × X  (synthesis transform)
        for (size_t m = 0; m < n; ++m) {
            Complex sum(0.0, 0.0);
            for (size_t k = 0; k < n; ++k) {
                // U[m,k]
                sum += canonical_U_[m * n + k] * input[k];
            }
            result[m] = sum.real();
        }
        
        return result;
    }
    
    /**
     * Get the current variant.
     */
    Variant variant() const { return variant_; }
    
    /**
     * Set the variant (for runtime switching).
     */
    void set_variant(Variant v) { variant_ = v; }
    
    // Accessors
    bool has_simd() const { return use_simd_; }
    size_t max_size() const { return max_size_; }
    Normalization normalization() const { return norm_; }
    
private:
    void apply_normalization(ComplexVec& data, bool is_inverse) {
        size_t n = data.size();
        double factor = 1.0;
        
        switch (norm_) {
            case Normalization::ORTHO:
                factor = 1.0 / std::sqrt(static_cast<double>(n));
                break;
            case Normalization::FORWARD:
                factor = is_inverse ? 1.0 : (1.0 / static_cast<double>(n));
                break;
            case Normalization::BACKWARD:
                factor = is_inverse ? (1.0 / static_cast<double>(n)) : 1.0;
                break;
            case Normalization::NONE:
            default:
                return;
        }
        
        for (auto& c : data) {
            c *= factor;
        }
    }
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Forward RFT transform (true canonical Φ†x).
 */
inline ComplexVec rft_forward(const RealVec& input) {
    static thread_local RFTMWEngine engine;
    return engine.forward(input);
}

/**
 * Inverse RFT transform (true canonical Φx).
 */
inline RealVec rft_inverse(const ComplexVec& input) {
    static thread_local RFTMWEngine engine;
    return engine.inverse(input);
}

/**
 * Forward RFT hybrid transform (O(N log N) FFT approximation).
 */
inline ComplexVec rft_forward_hybrid(const RealVec& input) {
    static thread_local RFTMWEngine engine;
    return engine.forward_hybrid(input);
}

/**
 * Inverse RFT hybrid transform (O(N log N) FFT approximation).
 */
inline RealVec rft_inverse_hybrid(const ComplexVec& input) {
    static thread_local RFTMWEngine engine;
    return engine.inverse_hybrid(input);
}

} // namespace rftmw

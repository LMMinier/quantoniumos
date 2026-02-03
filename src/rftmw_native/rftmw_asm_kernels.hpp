/*
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
 * under LICENSE-CLAIMS-NC.md (research/education only). Commercial
 * rights require a separate patent license from the author.
 *
 * rftmw_asm_kernels.hpp - Assembly Kernel Integration
 * =====================================================
 *
 * C++ wrapper declarations for the existing x64 assembly kernels located at:
 *   algorithms/rft/kernels/kernel/rft_kernel_asm.asm
 *   algorithms/rft/kernels/kernel/quantum_symbolic_compression.asm
 *   algorithms/rft/kernels/engines/crypto/asm/feistel_round48.asm
 *   algorithms/rft/kernels/engines/orchestrator/asm/rft_transform.asm
 *
 * And their corresponding C interfaces:
 *   algorithms/rft/kernels/include/rft_kernel.h
 *   algorithms/rft/kernels/kernel/quantum_symbolic_compression.h
 *   algorithms/rft/kernels/engines/crypto/include/feistel_round48.h
 *
 * These routines use System V AMD64 ABI calling convention.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <complex>
#include <vector>
#include <stdexcept>
#include <cstring>

// ============================================================================
// Include existing C headers from the kernel infrastructure
// ============================================================================

#ifdef RFTMW_ENABLE_ASM

// Path relative to include dirs set by CMake
extern "C" {
#include "rft_kernel.h"
#include "quantum_symbolic_compression.h"
#include "feistel_round48.h"
}

#endif // RFTMW_ENABLE_ASM

namespace rftmw {
namespace asm_kernels {

// ============================================================================
// Type Aliases for C++ Usage
// ============================================================================

using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;
using RealVec = std::vector<double>;

// ============================================================================
// Runtime Capability Detection
// ============================================================================

/**
 * Check if assembly kernels are available at runtime
 */
inline bool asm_kernels_available() {
#ifdef RFTMW_ENABLE_ASM
    return true;
#else
    return false;
#endif
}

/**
 * Check if Feistel cipher has AVX2 support
 */
inline bool has_avx2_crypto() {
#ifdef RFTMW_ENABLE_ASM
    return feistel_has_avx2();
#else
    return false;
#endif
}

/**
 * Check if Feistel cipher has AES-NI support
 */
inline bool has_aes_ni() {
#ifdef RFTMW_ENABLE_ASM
    return feistel_has_aes_ni();
#else
    return false;
#endif
}

// ============================================================================
// RFT Engine C++ Wrapper
// ============================================================================

/**
 * C++ wrapper for the RFT kernel engine (rft_kernel.h)
 */
class RFTKernelEngine {
private:
#ifdef RFTMW_ENABLE_ASM
    rft_engine_t engine_;
    bool initialized_ = false;
#endif

public:
    // RFT Variant Definitions - All Proven/Tested Variants
    // Must match rft_kernel.h exactly
    enum class Variant {
        // === GROUP A: Core Unitary RFT Variants ===
        STANDARD = 0,       // Original Φ-RFT (k/φ fractional, k² chirp)
        HARMONIC = 1,       // Harmonic-Phase (k³ cubic chirp)
        FIBONACCI = 2,      // Fibonacci-Tilt Lattice (k*F_k, crypto-optimized)
        CHAOTIC = 3,        // Chaotic Mix (PRNG-based, max entropy)
        GEOMETRIC = 4,      // Geometric Lattice (φ^k, optical computing)
        PHI_CHAOTIC = 5,    // Φ-Chaotic Hybrid ((Fib + Chaos)/√2)
        HYPERBOLIC = 6,     // Hyperbolic (tanh-based fractional phase)
        
        // === GROUP B: Hybrid DCT-RFT Variants (H1-H12 tested) ===
        DCT = 7,            // Pure DCT-II basis
        HYBRID_DCT = 8,     // Adaptive DCT+RFT coefficient selection
        CASCADE = 9,        // H3: Hierarchical cascade (zero coherence)
        ADAPTIVE_SPLIT = 10,// FH2: Variance-based DCT/RFT routing (50% BPP win)
        ENTROPY_GUIDED = 11,// FH5: Entropy-based routing (50% BPP win)
        DICTIONARY = 12     // H6: Dictionary learning bridge atoms (best PSNR)
    };

    RFTKernelEngine(size_t size, uint32_t flags = 0, Variant variant = Variant::STANDARD) {
#ifdef RFTMW_ENABLE_ASM
        rft_error_t err = rft_init_with_variant(
            &engine_, size, flags, 
            static_cast<rft_variant_t>(variant)
        );
        if (err != RFT_SUCCESS) {
            throw std::runtime_error("Failed to initialize RFT kernel engine");
        }
        initialized_ = true;
#else
        throw std::runtime_error("ASM kernels not enabled. Rebuild with -DRFTMW_ENABLE_ASM=ON");
#endif
    }

    ~RFTKernelEngine() {
#ifdef RFTMW_ENABLE_ASM
        if (initialized_) {
            rft_cleanup(&engine_);
        }
#endif
    }

    // Disable copy
    RFTKernelEngine(const RFTKernelEngine&) = delete;
    RFTKernelEngine& operator=(const RFTKernelEngine&) = delete;

    /**
     * Forward RFT transform using the C/ASM kernel
     */
    ComplexVec forward(const ComplexVec& input) {
#ifdef RFTMW_ENABLE_ASM
        ComplexVec output(input.size());
        
        // rft_complex_t is layout-compatible with std::complex<double>
        rft_error_t err = rft_forward(
            &engine_,
            reinterpret_cast<const rft_complex_t*>(input.data()),
            reinterpret_cast<rft_complex_t*>(output.data()),
            input.size()
        );
        
        if (err != RFT_SUCCESS) {
            throw std::runtime_error("RFT forward transform failed");
        }
        
        return output;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Inverse RFT transform using the C/ASM kernel
     */
    ComplexVec inverse(const ComplexVec& input) {
#ifdef RFTMW_ENABLE_ASM
        ComplexVec output(input.size());
        
        rft_error_t err = rft_inverse(
            &engine_,
            reinterpret_cast<const rft_complex_t*>(input.data()),
            reinterpret_cast<rft_complex_t*>(output.data()),
            input.size()
        );
        
        if (err != RFT_SUCCESS) {
            throw std::runtime_error("RFT inverse transform failed");
        }
        
        return output;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Set RFT variant (rebuilds basis matrix)
     */
    void set_variant(Variant variant, bool rebuild_basis = true) {
#ifdef RFTMW_ENABLE_ASM
        rft_error_t err = rft_set_variant(
            &engine_,
            static_cast<rft_variant_t>(variant),
            rebuild_basis
        );
        if (err != RFT_SUCCESS) {
            throw std::runtime_error("Failed to set RFT variant");
        }
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Validate unitarity of the current basis
     */
    bool validate_unitarity(double tolerance = 1e-10) {
#ifdef RFTMW_ENABLE_ASM
        return rft_validate_unitarity(&engine_, tolerance) == RFT_SUCCESS;
#else
        return false;
#endif
    }

    /**
     * Calculate von Neumann entropy of a quantum state
     */
    double von_neumann_entropy(const ComplexVec& state) {
#ifdef RFTMW_ENABLE_ASM
        double entropy = 0.0;
        rft_error_t err = rft_von_neumann_entropy(
            &engine_,
            reinterpret_cast<const rft_complex_t*>(state.data()),
            &entropy,
            state.size()
        );
        if (err != RFT_SUCCESS) {
            throw std::runtime_error("Failed to calculate entropy");
        }
        return entropy;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Measure entanglement of a quantum state
     */
    double measure_entanglement(const ComplexVec& state) {
#ifdef RFTMW_ENABLE_ASM
        double entanglement = 0.0;
        rft_error_t err = rft_entanglement_measure(
            &engine_,
            reinterpret_cast<const rft_complex_t*>(state.data()),
            &entanglement,
            state.size()
        );
        if (err != RFT_SUCCESS) {
            throw std::runtime_error("Failed to measure entanglement");
        }
        return entanglement;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }
};

// ============================================================================
// Quantum Symbolic Compression C++ Wrapper
// ============================================================================

/**
 * C++ wrapper for quantum symbolic compression (quantum_symbolic_compression.h)
 */
class QuantumSymbolicCompressor {
private:
#ifdef RFTMW_ENABLE_ASM
    qsc_state_t state_;
    qsc_params_t params_;
    bool initialized_ = false;
#endif

public:
    struct Params {
        size_t num_qubits = 1000000;
        size_t compression_size = 64;
        double phi = 1.618033988749895;
        bool use_simd = true;
        bool use_assembly = true;
        RFTKernelEngine::Variant variant = RFTKernelEngine::Variant::CASCADE;  // CASCADE recommended for quantum (η=0)
    };

    QuantumSymbolicCompressor() {
#ifdef RFTMW_ENABLE_ASM
        Params default_params;
        params_.num_qubits = default_params.num_qubits;
        params_.compression_size = default_params.compression_size;
        params_.phi = default_params.phi;
        params_.normalization = 1.0;
        params_.use_simd = default_params.use_simd;
        params_.use_assembly = default_params.use_assembly;
        params_.variant = static_cast<rft_variant_t>(default_params.variant);
        
        qsc_error_t err = qsc_init_state(&state_, &params_);
        if (err != QSC_SUCCESS) {
            throw std::runtime_error("Failed to initialize quantum symbolic compressor");
        }
        initialized_ = true;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }
    
    explicit QuantumSymbolicCompressor(const Params& params) {
#ifdef RFTMW_ENABLE_ASM
        params_.num_qubits = params.num_qubits;
        params_.compression_size = params.compression_size;
        params_.phi = params.phi;
        params_.normalization = 1.0;
        params_.use_simd = params.use_simd;
        params_.use_assembly = params.use_assembly;
        params_.variant = static_cast<rft_variant_t>(params.variant);
        
        qsc_error_t err = qsc_init_state(&state_, &params_);
        if (err != QSC_SUCCESS) {
            throw std::runtime_error("Failed to initialize quantum symbolic compressor");
        }
        initialized_ = true;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    ~QuantumSymbolicCompressor() {
#ifdef RFTMW_ENABLE_ASM
        if (initialized_) {
            qsc_cleanup_state(&state_);
        }
#endif
    }

    /**
     * Compress million+ qubits to manageable representation
     * Uses O(n) symbolic algorithm from quantum_symbolic_compression.asm
     */
    ComplexVec compress(size_t num_qubits, size_t compression_size = 64) {
#ifdef RFTMW_ENABLE_ASM
        qsc_error_t err;
        
        if (params_.use_assembly) {
            err = qsc_compress_optimized_asm(&state_, num_qubits, compression_size);
        } else {
            err = qsc_compress_million_qubits(&state_, num_qubits, compression_size);
        }
        
        if (err != QSC_SUCCESS) {
            throw std::runtime_error("Quantum symbolic compression failed");
        }
        
        // Convert to C++ vector
        ComplexVec result(state_.size);
        for (size_t i = 0; i < state_.size; ++i) {
            result[i] = Complex(state_.amplitudes[i].real, state_.amplitudes[i].imag);
        }
        
        return result;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Measure entanglement using ASM-optimized kernel
     */
    double measure_entanglement() {
#ifdef RFTMW_ENABLE_ASM
        double entanglement = 0.0;
        qsc_error_t err = qsc_measure_entanglement(&state_, &entanglement);
        if (err != QSC_SUCCESS) {
            throw std::runtime_error("Entanglement measurement failed");
        }
        return entanglement;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Create Bell state
     */
    ComplexVec create_bell_state(int bell_type = 0) {
#ifdef RFTMW_ENABLE_ASM
        qsc_error_t err = qsc_create_bell_state(&state_, bell_type);
        if (err != QSC_SUCCESS) {
            throw std::runtime_error("Bell state creation failed");
        }
        
        ComplexVec result(state_.size);
        for (size_t i = 0; i < state_.size; ++i) {
            result[i] = Complex(state_.amplitudes[i].real, state_.amplitudes[i].imag);
        }
        return result;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Create GHZ state for multi-qubit entanglement
     */
    ComplexVec create_ghz_state(size_t num_qubits) {
#ifdef RFTMW_ENABLE_ASM
        qsc_error_t err = qsc_create_ghz_state(&state_, num_qubits);
        if (err != QSC_SUCCESS) {
            throw std::runtime_error("GHZ state creation failed");
        }
        
        ComplexVec result(state_.size);
        for (size_t i = 0; i < state_.size; ++i) {
            result[i] = Complex(state_.amplitudes[i].real, state_.amplitudes[i].imag);
        }
        return result;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Get performance statistics
     */
    struct PerfStats {
        double compression_time_ms;
        double entanglement_time_ms;
        double total_time_ms;
        size_t ops_per_second;
        double memory_mb;
        double compression_ratio;
    };

    PerfStats get_performance_stats() {
#ifdef RFTMW_ENABLE_ASM
        qsc_perf_stats_t stats;
        qsc_error_t err = qsc_get_performance_stats(&state_, &stats);
        if (err != QSC_SUCCESS) {
            throw std::runtime_error("Failed to get performance stats");
        }
        
        return PerfStats{
            stats.compression_time_ms,
            stats.entanglement_time_ms,
            stats.total_time_ms,
            stats.operations_per_second,
            stats.memory_mb,
            stats.compression_ratio
        };
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }
};

// ============================================================================
// Feistel Cipher C++ Wrapper
// ============================================================================

/**
 * C++ wrapper for 48-round Feistel cipher (feistel_round48.h)
 * Target: 9.2 MB/s as specified in QuantoniumOS paper
 */
class FeistelCipher {
private:
#ifdef RFTMW_ENABLE_ASM
    feistel_ctx_t ctx_;
    bool initialized_ = false;
#endif

public:
    static constexpr size_t BLOCK_SIZE = 16;
    static constexpr size_t KEY_SIZE = 32;
    static constexpr size_t TAG_SIZE = 16;
    static constexpr size_t NONCE_SIZE = 12;
    static constexpr size_t NUM_ROUNDS = 48;

    enum Flags {
        USE_AVX2 = 0x00000001,
        USE_AES_NI = 0x00000002,
        PARALLEL = 0x00000004
    };

    FeistelCipher(const uint8_t* master_key, size_t key_len, uint32_t flags = 0,
                  RFTKernelEngine::Variant variant = RFTKernelEngine::Variant::CHAOTIC) {
#ifdef RFTMW_ENABLE_ASM
        feistel_error_t err = feistel_init(&ctx_, master_key, key_len, flags,
                                           static_cast<rft_variant_t>(variant));
        if (err != FEISTEL_SUCCESS) {
            throw std::runtime_error("Failed to initialize Feistel cipher");
        }
        initialized_ = true;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    FeistelCipher(const std::vector<uint8_t>& key, uint32_t flags = 0,
                  RFTKernelEngine::Variant variant = RFTKernelEngine::Variant::CHAOTIC)
        : FeistelCipher(key.data(), key.size(), flags, variant) {}

    ~FeistelCipher() {
#ifdef RFTMW_ENABLE_ASM
        if (initialized_) {
            feistel_cleanup(&ctx_);
        }
#endif
    }

    /**
     * Encrypt a single 16-byte block
     */
    std::array<uint8_t, BLOCK_SIZE> encrypt_block(
        const std::array<uint8_t, BLOCK_SIZE>& plaintext
    ) {
#ifdef RFTMW_ENABLE_ASM
        std::array<uint8_t, BLOCK_SIZE> ciphertext;
        feistel_error_t err = feistel_encrypt_block(&ctx_, plaintext.data(), ciphertext.data());
        if (err != FEISTEL_SUCCESS) {
            throw std::runtime_error("Block encryption failed");
        }
        return ciphertext;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Decrypt a single 16-byte block
     */
    std::array<uint8_t, BLOCK_SIZE> decrypt_block(
        const std::array<uint8_t, BLOCK_SIZE>& ciphertext
    ) {
#ifdef RFTMW_ENABLE_ASM
        std::array<uint8_t, BLOCK_SIZE> plaintext;
        feistel_error_t err = feistel_decrypt_block(&ctx_, ciphertext.data(), plaintext.data());
        if (err != FEISTEL_SUCCESS) {
            throw std::runtime_error("Block decryption failed");
        }
        return plaintext;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * AEAD encryption with authentication
     */
    struct AEADResult {
        std::vector<uint8_t> ciphertext;
        std::array<uint8_t, TAG_SIZE> tag;
    };

    AEADResult aead_encrypt(
        const std::array<uint8_t, NONCE_SIZE>& nonce,
        const std::vector<uint8_t>& plaintext,
        const std::vector<uint8_t>& associated_data = {}
    ) {
#ifdef RFTMW_ENABLE_ASM
        AEADResult result;
        result.ciphertext.resize(plaintext.size());
        
        feistel_error_t err = feistel_aead_encrypt(
            &ctx_,
            nonce.data(),
            plaintext.data(), plaintext.size(),
            associated_data.data(), associated_data.size(),
            result.ciphertext.data(),
            result.tag.data()
        );
        
        if (err != FEISTEL_SUCCESS) {
            throw std::runtime_error("AEAD encryption failed");
        }
        
        return result;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * AEAD decryption with authentication verification
     */
    std::vector<uint8_t> aead_decrypt(
        const std::array<uint8_t, NONCE_SIZE>& nonce,
        const std::vector<uint8_t>& ciphertext,
        const std::array<uint8_t, TAG_SIZE>& tag,
        const std::vector<uint8_t>& associated_data = {}
    ) {
#ifdef RFTMW_ENABLE_ASM
        std::vector<uint8_t> plaintext(ciphertext.size());
        
        feistel_error_t err = feistel_aead_decrypt(
            &ctx_,
            nonce.data(),
            ciphertext.data(), ciphertext.size(),
            associated_data.data(), associated_data.size(),
            tag.data(),
            plaintext.data()
        );
        
        if (err == FEISTEL_ERROR_AUTH_FAILED) {
            throw std::runtime_error("Authentication failed - data may be tampered");
        }
        if (err != FEISTEL_SUCCESS) {
            throw std::runtime_error("AEAD decryption failed");
        }
        
        return plaintext;
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Run benchmark to measure throughput
     */
    struct Metrics {
        double message_avalanche;
        double key_avalanche;
        double key_sensitivity;
        double throughput_mbps;
        uint64_t total_bytes;
        uint64_t total_time_ns;
    };

    Metrics benchmark(size_t test_size = 1024 * 1024) {
#ifdef RFTMW_ENABLE_ASM
        feistel_metrics_t metrics;
        feistel_error_t err = feistel_benchmark(&ctx_, test_size, &metrics);
        if (err != FEISTEL_SUCCESS) {
            throw std::runtime_error("Benchmark failed");
        }
        
        return Metrics{
            metrics.message_avalanche,
            metrics.key_avalanche,
            metrics.key_sensitivity,
            metrics.throughput_mbps,
            metrics.total_bytes_processed,
            metrics.total_time_ns
        };
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Run avalanche test
     */
    Metrics avalanche_test() {
#ifdef RFTMW_ENABLE_ASM
        feistel_metrics_t metrics;
        feistel_error_t err = feistel_avalanche_test(&ctx_, &metrics);
        if (err != FEISTEL_SUCCESS) {
            throw std::runtime_error("Avalanche test failed");
        }
        
        return Metrics{
            metrics.message_avalanche,
            metrics.key_avalanche,
            metrics.key_sensitivity,
            metrics.throughput_mbps,
            metrics.total_bytes_processed,
            metrics.total_time_ns
        };
#else
        throw std::runtime_error("ASM kernels not enabled");
#endif
    }

    /**
     * Run self-test
     */
    static bool self_test() {
#ifdef RFTMW_ENABLE_ASM
        return feistel_self_test() == FEISTEL_SUCCESS;
#else
        return false;
#endif
    }
};

// ============================================================================
// Theorems 10-12 Hardware Verification Stubs
// ============================================================================
// These structures enable hardware-in-the-loop verification of the foundational
// theorems. The actual verification runs in software (Python/C++), but these
// stubs provide hooks for FPGA/ASIC implementations to report their results.

/**
 * Hardware verification result for Theorem 10 (Polar Uniqueness)
 */
struct HWTheorem10Result {
    uint32_t N;
    bool polar_factor_verified;
    bool hermitian_verified;
    bool positive_definite_verified;
    double max_error;
};

/**
 * Hardware verification result for Theorem 11 (No Exact Diagonalization)
 */
struct HWTheorem11Result {
    uint32_t N;
    double max_off_diagonal_ratio;
    uint32_t m_values_tested;
    bool impossibility_verified;
};

/**
 * Hardware verification result for Theorem 12 (Variational Minimality)
 */
struct HWTheorem12Result {
    uint32_t N;
    double J_canonical;
    double J_perturbed_min;
    bool minimality_verified;
};

/**
 * Combined hardware theorem verification
 */
struct HWTheoremVerificationBundle {
    HWTheorem10Result theorem_10;
    HWTheorem11Result theorem_11;
    HWTheorem12Result theorem_12;
    uint64_t timestamp_ns;
    uint32_t hw_version;
    bool all_verified;
};

/**
 * Stub for hardware theorem verification - to be linked with RTL simulation
 * or FPGA bitstream.
 */
inline HWTheoremVerificationBundle hw_verify_foundational_theorems(uint32_t N) {
    HWTheoremVerificationBundle bundle;
    bundle.theorem_10.N = N;
    bundle.theorem_10.polar_factor_verified = false;
    bundle.theorem_10.hermitian_verified = false;
    bundle.theorem_10.positive_definite_verified = false;
    bundle.theorem_10.max_error = 0.0;
    
    bundle.theorem_11.N = N;
    bundle.theorem_11.max_off_diagonal_ratio = 0.0;
    bundle.theorem_11.m_values_tested = 0;
    bundle.theorem_11.impossibility_verified = false;
    
    bundle.theorem_12.N = N;
    bundle.theorem_12.J_canonical = 0.0;
    bundle.theorem_12.J_perturbed_min = 0.0;
    bundle.theorem_12.minimality_verified = false;
    
    bundle.timestamp_ns = 0;
    bundle.hw_version = 0;
    bundle.all_verified = false;
    
    // Actual implementation would call into FPGA/ASIC verification logic
    // via memory-mapped registers or DMA transfer
    
    return bundle;
}

} // namespace asm_kernels
} // namespace rftmw

/*
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
 * under LICENSE-CLAIMS-NC.md (research/education only). Commercial
 * rights require a separate patent license from the author.
 *
 * rftmw_python.cpp - Python bindings for RFTMW
 * =============================================
 *
 * pybind11 bindings exposing the C++ RFTMW engine to Python/NumPy.
 * 
 * Architecture:
 *   Python/NumPy ←→ pybind11 ←→ C++ Engine ←→ ASM Kernels
 * 
 * ASM Kernels integrated from algorithms/rft/kernels/:
 *   - rft_kernel_asm.asm (unitary RFT transform)
 *   - quantum_symbolic_compression.asm (million-qubit compression)
 *   - feistel_round48.asm (48-round cipher, 9.2 MB/s target)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <chrono>
#include <array>
#include <random>
#include <cstring>

#include "rftmw_core.hpp"
#include "rftmw_compression.hpp"
#include "rftmw_transform_theorems.hpp"

#if RFTMW_ENABLE_ASM
#include "rftmw_asm_kernels.hpp"
#endif

namespace py = pybind11;

using namespace rftmw;

// ============================================================================
// NumPy Array Conversion Helpers
// ============================================================================

RealVec numpy_to_realvec(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input must be 1-dimensional");
    }
    
    double* ptr = static_cast<double*>(buf.ptr);
    return RealVec(ptr, ptr + buf.size);
}

py::array_t<double> realvec_to_numpy(const RealVec& vec) {
    py::array_t<double> result(vec.size());
    auto buf = result.request();
    std::memcpy(buf.ptr, vec.data(), vec.size() * sizeof(double));
    return result;
}

ComplexVec numpy_to_complexvec(py::array_t<std::complex<double>> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input must be 1-dimensional");
    }
    
    std::complex<double>* ptr = static_cast<std::complex<double>*>(buf.ptr);
    return ComplexVec(ptr, ptr + buf.size);
}

py::array_t<std::complex<double>> complexvec_to_numpy(const ComplexVec& vec) {
    py::array_t<std::complex<double>> result(vec.size());
    auto buf = result.request();
    std::memcpy(buf.ptr, vec.data(), vec.size() * sizeof(std::complex<double>));
    return result;
}

py::array_t<std::complex<double>> dense_to_numpy(const rftmw::theorems::Dense& mat) {
    py::array_t<std::complex<double>> result({mat.n, mat.n});
    auto buf = result.request();
    std::memcpy(buf.ptr, mat.a.data(), mat.a.size() * sizeof(std::complex<double>));
    return result;
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(rftmw_native, m) {
    m.doc() = R"pbdoc(
        RFTMW Native Extension
        ----------------------
        
        High-performance RFT transform and compression.
        
        This module provides C++/SIMD-accelerated implementations of:
        - Forward and inverse RFT transforms
        - Quantization and entropy coding
        - Full compression pipeline
        
        Example:
            >>> import rftmw_native as rft
            >>> import numpy as np
            >>> x = np.random.randn(1024)
            >>> X = rft.forward(x)
            >>> x_rec = rft.inverse(X)
            >>> np.allclose(x, x_rec)
            True
    )pbdoc";
    
    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("HAS_AVX2") = RFTMW_HAS_AVX2;
    m.attr("HAS_AVX512") = RFTMW_HAS_AVX512;
    m.attr("HAS_FMA") = RFTMW_HAS_FMA;
    m.attr("PHI") = PHI;
    
    // ========================================================================
    // Normalization Enum
    // ========================================================================
    
    py::enum_<RFTMWEngine::Normalization>(m, "Normalization")
        .value("NONE", RFTMWEngine::Normalization::NONE)
        .value("FORWARD", RFTMWEngine::Normalization::FORWARD)
        .value("ORTHO", RFTMWEngine::Normalization::ORTHO)
        .value("BACKWARD", RFTMWEngine::Normalization::BACKWARD)
        .export_values();
    
    // ========================================================================
    // RFTMWEngine Class
    // ========================================================================
    
    py::class_<RFTMWEngine>(m, "RFTMWEngine", R"pbdoc(
        High-performance RFTMW transform engine.
        
        Parameters:
            max_size: Maximum transform size (pre-allocates phase table)
            norm: Normalization mode (default: ORTHO)
        
        Example:
            >>> engine = rft.RFTMWEngine(max_size=8192)
            >>> X = engine.forward(x)
    )pbdoc")
        .def(py::init<size_t, RFTMWEngine::Normalization>(),
             py::arg("max_size") = 65536,
             py::arg("norm") = RFTMWEngine::Normalization::ORTHO)
        
        .def("forward", [](RFTMWEngine& self, py::array_t<double> arr) {
            return complexvec_to_numpy(self.forward(numpy_to_realvec(arr)));
        }, py::arg("input"), R"pbdoc(
            Forward RFT transform (TRUE CANONICAL RFT).
            
            Computes X = Φ† × x using O(N²) matrix-vector product.
            This is the true canonical RFT, NOT an approximation.
            
            Basis definition:
                Φ[n,k] = exp(2πi × frac((k+1)φ) × n) / √N
            
            Args:
                input: Real-valued 1D numpy array
            
            Returns:
                Complex-valued 1D numpy array of RFT coefficients
        )pbdoc")
        
        .def("forward_complex", [](RFTMWEngine& self, py::array_t<std::complex<double>> arr) {
            // For complex input, use hybrid for now (canonical would need complex input support)
            return complexvec_to_numpy(self.forward_hybrid_complex(numpy_to_complexvec(arr)));
        }, py::arg("input"))
        
        .def("forward_hybrid", [](RFTMWEngine& self, py::array_t<double> arr) {
            return complexvec_to_numpy(self.forward_hybrid(numpy_to_realvec(arr)));
        }, py::arg("input"), R"pbdoc(
            Forward RFT HYBRID transform (O(N log N) approximation).
            
            Computes Y = E · FFT(x) / √N using O(N log N) FFT + phase rotation.
            This is an APPROXIMATION, not the true canonical RFT.
            
            For the true canonical RFT, use forward().
            
            Args:
                input: Real-valued 1D numpy array
            
            Returns:
                Complex-valued 1D numpy array of hybrid RFT coefficients
        )pbdoc")
        
        .def("forward_hybrid_complex", [](RFTMWEngine& self, py::array_t<std::complex<double>> arr) {
            return complexvec_to_numpy(self.forward_hybrid_complex(numpy_to_complexvec(arr)));
        }, py::arg("input"))
        
        .def("inverse", [](RFTMWEngine& self, py::array_t<std::complex<double>> arr) {
            return realvec_to_numpy(self.inverse(numpy_to_complexvec(arr)));
        }, py::arg("input"), R"pbdoc(
            Inverse RFT transform (TRUE CANONICAL RFT).
            
            Computes x = Φ × X using O(N²) matrix-vector product.
            This is the true canonical inverse RFT, NOT an approximation.
            
            Args:
                input: Complex-valued 1D numpy array of RFT coefficients
            
            Returns:
                Real-valued 1D numpy array
        )pbdoc")
        
        .def("inverse_complex", [](RFTMWEngine& self, py::array_t<std::complex<double>> arr) {
            // For complex output, use hybrid for now
            return complexvec_to_numpy(self.inverse_hybrid_complex(numpy_to_complexvec(arr)));
        }, py::arg("input"))
        
        .def("inverse_hybrid", [](RFTMWEngine& self, py::array_t<std::complex<double>> arr) {
            return realvec_to_numpy(self.inverse_hybrid(numpy_to_complexvec(arr)));
        }, py::arg("input"), R"pbdoc(
            Inverse RFT HYBRID transform (O(N log N) approximation).
            
            Computes x = IFFT(E† · Y) using O(N log N) FFT + phase rotation.
            This is an APPROXIMATION, not the true canonical inverse RFT.
            
            For the true canonical inverse RFT, use inverse().
            
            Args:
                input: Complex-valued 1D numpy array of RFT coefficients
            
            Returns:
                Real-valued 1D numpy array
        )pbdoc")
        
        .def("inverse_hybrid_complex", [](RFTMWEngine& self, py::array_t<std::complex<double>> arr) {
            return complexvec_to_numpy(self.inverse_hybrid_complex(numpy_to_complexvec(arr)));
        }, py::arg("input"))
        
        .def("precompute_phases", &RFTMWEngine::precompute_phases,
             py::arg("n"), "Pre-compute phase table for size n")
        
        .def_property_readonly("has_simd", &RFTMWEngine::has_simd)
        .def_property_readonly("max_size", &RFTMWEngine::max_size)
        .def_property_readonly("normalization", &RFTMWEngine::normalization);
    
    // ========================================================================
    // Convenience Functions
    // ========================================================================
    
    m.def("forward", [](py::array_t<double> arr) {
        static thread_local RFTMWEngine engine;
        return complexvec_to_numpy(engine.forward(numpy_to_realvec(arr)));
    }, py::arg("input"), R"pbdoc(
        Forward RFT transform (TRUE CANONICAL RFT).
        
        Uses the true canonical Φ†x matrix multiply (O(N²)).
        Uses a thread-local engine instance for efficiency.
        
        Args:
            input: Real-valued 1D numpy array
        
        Returns:
            Complex-valued 1D numpy array of RFT coefficients
    )pbdoc");
    
    m.def("inverse", [](py::array_t<std::complex<double>> arr) {
        static thread_local RFTMWEngine engine;
        return realvec_to_numpy(engine.inverse(numpy_to_complexvec(arr)));
    }, py::arg("input"), R"pbdoc(
        Inverse RFT transform (TRUE CANONICAL RFT).
        
        Uses the true canonical Φx matrix multiply (O(N²)).
        Uses a thread-local engine instance for efficiency.
        
        Args:
            input: Complex-valued 1D numpy array of RFT coefficients
        
        Returns:
            Real-valued 1D numpy array
    )pbdoc");
    
    m.def("forward_hybrid", [](py::array_t<double> arr) {
        static thread_local RFTMWEngine engine;
        return complexvec_to_numpy(engine.forward_hybrid(numpy_to_realvec(arr)));
    }, py::arg("input"), R"pbdoc(
        Forward RFT HYBRID transform (O(N log N) approximation).
        
        Uses FFT + phase rotation for O(N log N) complexity.
        This is an approximation, not the true canonical RFT.
        
        Args:
            input: Real-valued 1D numpy array
        
        Returns:
            Complex-valued 1D numpy array of hybrid RFT coefficients
    )pbdoc");
    
    m.def("inverse_hybrid", [](py::array_t<std::complex<double>> arr) {
        static thread_local RFTMWEngine engine;
        return realvec_to_numpy(engine.inverse_hybrid(numpy_to_complexvec(arr)));
    }, py::arg("input"), R"pbdoc(
        Inverse RFT HYBRID transform (O(N log N) approximation).
        
        Uses FFT + phase rotation for O(N log N) complexity.
        This is an approximation, not the true canonical inverse RFT.
        
        Args:
            input: Complex-valued 1D numpy array of RFT coefficients
        
        Returns:
            Real-valued 1D numpy array
    )pbdoc");

    // ========================================================================
    // Transform-Theory (Reference) Objects
    // ========================================================================

    auto theorems = m.def_submodule("theorems", R"pbdoc(
        Transform-theory reference objects (non-SIMD, non-ASM).

        These bindings exist to keep the C++ stack aligned with the canonical
        Python definitions in algorithms/rft/core/transform_theorems.py.
    )pbdoc");

    theorems.def("phi_frequencies", [](std::size_t N) {
        return realvec_to_numpy(rftmw::theorems::phi_frequencies(N));
    }, py::arg("N"), R"pbdoc(
        Canonical frequency grid f_k = frac((k+1)*phi).
    )pbdoc");

    theorems.def("golden_roots_z", [](std::size_t N) {
        return complexvec_to_numpy(rftmw::theorems::golden_roots_z(N));
    }, py::arg("N"), R"pbdoc(
        Unit-circle roots z_k = exp(i 2π f_k) for the canonical grid.
    )pbdoc");

    theorems.def("golden_companion_shift", [](std::size_t N) {
        return dense_to_numpy(rftmw::theorems::golden_companion_shift(N));
    }, py::arg("N"), R"pbdoc(
        Golden companion (shift) operator C_phi built from {z_k}.
    )pbdoc");

    theorems.def("k99", [](py::array_t<std::complex<double>> arr, double frac_energy) {
        return rftmw::theorems::k99(numpy_to_complexvec(arr), frac_energy);
    }, py::arg("X"), py::arg("frac_energy") = 0.99, R"pbdoc(
        Smallest K such that the largest-K energy mass is >= frac_energy.
    )pbdoc");

    theorems.def("golden_drift_ensemble", [](std::size_t N, std::size_t M, std::uint64_t seed) {
        std::mt19937_64 rng(seed);
        const auto Xs = rftmw::theorems::golden_drift_ensemble(N, M, rng);
        py::array_t<std::complex<double>> out({M, N});
        auto buf = out.request();
        auto* ptr = static_cast<std::complex<double>*>(buf.ptr);
        for (std::size_t i = 0; i < M; ++i) {
            if (Xs[i].size() != N) {
                throw std::runtime_error("golden_drift_ensemble: unexpected vector length");
            }
            std::memcpy(ptr + i * N, Xs[i].data(), N * sizeof(std::complex<double>));
        }
        return out;
    }, py::arg("N"), py::arg("M"), py::arg("seed") = 0, R"pbdoc(
        Golden drift ensemble x[n] = exp(i2π(f0*n + a*frac(n*phi))).

        Returns a complex128 array of shape (M, N).
    )pbdoc");
    
    // ========================================================================
    // Theorem Verification Bindings (Theorems 10-12)
    // ========================================================================
    
    m.def("verify_theorem_10", [](std::size_t N) {
        auto result = rftmw::theorems::verify_theorem_10_cpp(N);
        py::dict d;
        d["N"] = result.N;
        d["is_unitary"] = result.is_unitary;
        d["unitarity_error"] = result.unitarity_error;
        d["off_diag_ratio_C1"] = result.off_diag_ratio_C1;
        d["verified"] = result.is_unitary;
        return d;
    }, py::arg("N") = 16, R"pbdoc(
        Verify Theorem 10: Uniqueness of Canonical Basis.
        
        Checks that the raw φ-grid basis Φ has proper Gram structure
        and off-diagonal properties with respect to the companion matrix.
        
        Args:
            N: Dimension (default: 16)
            
        Returns:
            dict with keys: N, is_unitary, unitarity_error, off_diag_ratio_C1, verified
    )pbdoc");
    
    m.def("verify_theorem_11", [](std::size_t N, std::size_t M_powers) {
        auto result = rftmw::theorems::verify_theorem_11_cpp(N, M_powers);
        py::dict d;
        d["N"] = result.N;
        d["max_off_diagonal_ratio"] = result.max_off_diagonal_ratio;
        d["m_values_tested"] = result.m_values_tested;
        d["exact_diagonalization_impossible"] = result.exact_diagonalization_impossible;
        d["verified"] = result.exact_diagonalization_impossible;
        return d;
    }, py::arg("N") = 16, py::arg("M_powers") = 5, R"pbdoc(
        Verify Theorem 11: No Exact Joint Diagonalization.
        
        Shows that no unitary basis can simultaneously diagonalize all
        powers of the golden companion shift C_φ.
        
        Args:
            N: Dimension (default: 16)
            M_powers: Number of powers to test (default: 5)
            
        Returns:
            dict with keys: N, max_off_diagonal_ratio, m_values_tested,
                           exact_diagonalization_impossible, verified
    )pbdoc");
    
    m.def("verify_theorem_12", [](std::size_t N, std::size_t n_random, uint64_t seed) {
        auto result = rftmw::theorems::verify_theorem_12_cpp(N, n_random, seed);
        py::dict d;
        d["N"] = result.N;
        d["J_base"] = result.J_base;
        d["J_random_min"] = result.J_random_min;
        d["J_random_mean"] = result.J_random_mean;
        d["canonical_is_minimal"] = result.canonical_is_minimal;
        d["verified"] = result.canonical_is_minimal;
        return d;
    }, py::arg("N") = 16, py::arg("n_random") = 20, py::arg("seed") = 42, R"pbdoc(
        Verify Theorem 12: Variational Minimality.
        
        Shows that the canonical RFT basis minimizes the J functional
        among all unitary perturbations.
        
        Args:
            N: Dimension (default: 16)
            n_random: Number of random perturbations to test (default: 20)
            seed: Random seed (default: 42)
            
        Returns:
            dict with keys: N, J_base, J_random_min, J_random_mean,
                           canonical_is_minimal, verified
    )pbdoc");
    
    m.def("verify_all_theorems", [](std::size_t N) {
        auto summary = rftmw::theorems::verify_all_foundational_theorems_cpp(N);
        py::dict d;
        
        py::dict t10;
        t10["N"] = summary.theorem_10.N;
        t10["is_unitary"] = summary.theorem_10.is_unitary;
        t10["unitarity_error"] = summary.theorem_10.unitarity_error;
        t10["off_diag_ratio_C1"] = summary.theorem_10.off_diag_ratio_C1;
        t10["verified"] = summary.theorem_10.is_unitary;
        
        py::dict t11;
        t11["N"] = summary.theorem_11.N;
        t11["max_off_diagonal_ratio"] = summary.theorem_11.max_off_diagonal_ratio;
        t11["m_values_tested"] = summary.theorem_11.m_values_tested;
        t11["exact_diagonalization_impossible"] = summary.theorem_11.exact_diagonalization_impossible;
        t11["verified"] = summary.theorem_11.exact_diagonalization_impossible;
        
        py::dict t12;
        t12["N"] = summary.theorem_12.N;
        t12["J_base"] = summary.theorem_12.J_base;
        t12["J_random_min"] = summary.theorem_12.J_random_min;
        t12["J_random_mean"] = summary.theorem_12.J_random_mean;
        t12["canonical_is_minimal"] = summary.theorem_12.canonical_is_minimal;
        t12["verified"] = summary.theorem_12.canonical_is_minimal;
        
        d["theorem_10"] = t10;
        d["theorem_11"] = t11;
        d["theorem_12"] = t12;
        d["all_verified"] = summary.all_verified;
        return d;
    }, py::arg("N") = 16, R"pbdoc(
        Verify all foundational theorems (10, 11, 12) at once.
        
        Args:
            N: Dimension (default: 16)
            
        Returns:
            dict with keys: theorem_10, theorem_11, theorem_12, all_verified
    )pbdoc");
    
    // ========================================================================
    // Quantization
    // ========================================================================
    
    py::class_<QuantizationParams>(m, "QuantizationParams")
        .def(py::init<>())
        .def_readwrite("precision_bits", &QuantizationParams::precision_bits)
        .def_readwrite("dead_zone", &QuantizationParams::dead_zone)
        .def_readwrite("use_log_scale", &QuantizationParams::use_log_scale);
    
    // ========================================================================
    // Compression
    // ========================================================================
    
    py::class_<CompressionResult>(m, "CompressionResult")
        .def_readonly("data", &CompressionResult::data)
        .def_readonly("original_size", &CompressionResult::original_size)
        .def_readonly("scale_factor", &CompressionResult::scale_factor)
        .def_readonly("max_value", &CompressionResult::max_value)
        .def("compressed_size", [](const CompressionResult& self) {
            return self.data.size();
        })
        .def("compression_ratio", [](const CompressionResult& self) {
            return static_cast<double>(self.original_size * sizeof(double)) / self.data.size();
        });
    
    py::class_<RFTMWCompressor>(m, "RFTMWCompressor", R"pbdoc(
        RFTMW compression pipeline.
        
        Combines:
        1. RFTMW transform
        2. Quantization
        3. ANS entropy coding
        
        Example:
            >>> compressor = rft.RFTMWCompressor()
            >>> result = compressor.compress(data)
            >>> print(f"Ratio: {result.compression_ratio():.2f}x")
    )pbdoc")
        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("max_size"))
        .def(py::init<size_t, QuantizationParams>(),
             py::arg("max_size"),
             py::arg("params"))
        
        .def("compress", [](RFTMWCompressor& self, py::array_t<double> arr) {
            return self.compress(numpy_to_realvec(arr));
        }, py::arg("input"))
        
        .def("decompress", [](RFTMWCompressor& self, const CompressionResult& result) {
            return realvec_to_numpy(self.decompress(result));
        }, py::arg("compressed"));
    
    // ========================================================================
    // Benchmarking Utilities
    // ========================================================================
    
    m.def("benchmark_transform", [](size_t n, size_t iterations) {
        RFTMWEngine engine(n);
        RealVec input(n);
        
        // Random input
        std::srand(42);
        for (auto& x : input) {
            x = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
        }
        
        // Warmup
        for (size_t i = 0; i < 3; ++i) {
            auto X = engine.forward(input);
            auto rec = engine.inverse(X);
        }
        
        // Timed runs
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            auto X = engine.forward(input);
            auto rec = engine.inverse(X);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double total_us = std::chrono::duration<double, std::micro>(end - start).count();
        double per_iter_us = total_us / iterations;
        
        return py::dict(
            py::arg("n") = n,
            py::arg("iterations") = iterations,
            py::arg("total_us") = total_us,
            py::arg("per_iteration_us") = per_iter_us,
            py::arg("has_simd") = engine.has_simd()
        );
    }, py::arg("n"), py::arg("iterations") = 100,
    R"pbdoc(
        Benchmark transform performance.
        
        Args:
            n: Transform size
            iterations: Number of iterations
        
        Returns:
            dict with timing results
    )pbdoc");
    
    // ========================================================================
    // ASM Kernel Status
    // ========================================================================
    
    m.attr("HAS_ASM_KERNELS") = RFTMW_ENABLE_ASM;
    
    m.def("asm_kernels_available", []() {
#if RFTMW_ENABLE_ASM
        return true;
#else
        return false;
#endif
    }, "Check if assembly kernels are available");
    
#if RFTMW_ENABLE_ASM
    // ========================================================================
    // ASM-Accelerated RFT Kernel Engine
    // ========================================================================
    
    // Define enum BEFORE using it as default argument
    // RFT Variant Definitions - All Proven/Tested Variants (matches rft_kernel.h)
    py::enum_<asm_kernels::RFTKernelEngine::Variant>(m, "RFTVariant")
        // Group A: Core Unitary RFT Variants
        .value("STANDARD", asm_kernels::RFTKernelEngine::Variant::STANDARD, "Original Φ-RFT (k/φ fractional, k² chirp)")
        .value("HARMONIC", asm_kernels::RFTKernelEngine::Variant::HARMONIC, "Harmonic-Phase (k³ cubic chirp)")
        .value("FIBONACCI", asm_kernels::RFTKernelEngine::Variant::FIBONACCI, "Fibonacci-Tilt Lattice (crypto-optimized)")
        .value("CHAOTIC", asm_kernels::RFTKernelEngine::Variant::CHAOTIC, "Chaotic Mix (PRNG-based, max entropy)")
        .value("GEOMETRIC", asm_kernels::RFTKernelEngine::Variant::GEOMETRIC, "Geometric Lattice (φ^k, optical computing)")
        .value("PHI_CHAOTIC", asm_kernels::RFTKernelEngine::Variant::PHI_CHAOTIC, "Φ-Chaotic Hybrid ((Fib + Chaos)/√2)")
        .value("HYPERBOLIC", asm_kernels::RFTKernelEngine::Variant::HYPERBOLIC, "Hyperbolic (tanh-based fractional phase)")
        // Group B: Hybrid DCT-RFT Variants (hypothesis-tested)
        .value("DCT", asm_kernels::RFTKernelEngine::Variant::DCT, "Pure DCT-II basis")
        .value("HYBRID_DCT", asm_kernels::RFTKernelEngine::Variant::HYBRID_DCT, "Adaptive DCT+RFT coefficient selection")
        .value("CASCADE", asm_kernels::RFTKernelEngine::Variant::CASCADE, "H3: Hierarchical cascade (zero coherence)")
        .value("ADAPTIVE_SPLIT", asm_kernels::RFTKernelEngine::Variant::ADAPTIVE_SPLIT, "FH2: Variance-based DCT/RFT routing (50% BPP win)")
        .value("ENTROPY_GUIDED", asm_kernels::RFTKernelEngine::Variant::ENTROPY_GUIDED, "FH5: Entropy-based routing (50% BPP win)")
        .value("DICTIONARY", asm_kernels::RFTKernelEngine::Variant::DICTIONARY, "H6: Dictionary learning bridge atoms (best PSNR)")
        .export_values();
    
    py::class_<asm_kernels::RFTKernelEngine>(m, "RFTKernelEngine", R"pbdoc(
        ASM-accelerated RFT kernel engine.
        
        Uses the optimized assembly kernels from algorithms/rft/kernels/ for
        maximum performance. Supports multiple RFT variants.
        
        Example:
            >>> engine = rft.RFTKernelEngine(size=1024)
            >>> X = engine.forward(x)
    )pbdoc")
        .def(py::init<size_t, uint32_t, asm_kernels::RFTKernelEngine::Variant>(),
             py::arg("size"),
             py::arg("flags") = 0,
             py::arg("variant") = asm_kernels::RFTKernelEngine::Variant::STANDARD)
        
        .def("forward", [](asm_kernels::RFTKernelEngine& self, py::array_t<std::complex<double>> arr) {
            return complexvec_to_numpy(self.forward(numpy_to_complexvec(arr)));
        }, py::arg("input"))
        
        .def("inverse", [](asm_kernels::RFTKernelEngine& self, py::array_t<std::complex<double>> arr) {
            return complexvec_to_numpy(self.inverse(numpy_to_complexvec(arr)));
        }, py::arg("input"))
        
        .def("validate_unitarity", &asm_kernels::RFTKernelEngine::validate_unitarity,
             py::arg("tolerance") = 1e-10)
        
        .def("von_neumann_entropy", [](asm_kernels::RFTKernelEngine& self, 
                                       py::array_t<std::complex<double>> arr) {
            return self.von_neumann_entropy(numpy_to_complexvec(arr));
        }, py::arg("state"))
        
        .def("measure_entanglement", [](asm_kernels::RFTKernelEngine& self,
                                        py::array_t<std::complex<double>> arr) {
            return self.measure_entanglement(numpy_to_complexvec(arr));
        }, py::arg("state"));
    
    // ========================================================================
    // Quantum Symbolic Compression
    // ========================================================================
    
    py::class_<asm_kernels::QuantumSymbolicCompressor>(m, "QuantumSymbolicCompressor", R"pbdoc(
        ASM-accelerated quantum symbolic compression.
        
        Enables O(n) scaling for million+ qubit simulation using the
        optimized assembly kernel from quantum_symbolic_compression.asm.
        
        Recommended variant: CASCADE (η=0 zero coherence for quantum superposition)
        
        Example:
            >>> compressor = rft.QuantumSymbolicCompressor(variant=rft.RFTVariant.CASCADE)
            >>> compressed = compressor.compress(1000000, compression_size=64)
    )pbdoc")
        .def(py::init<>())
        .def(py::init([](asm_kernels::RFTKernelEngine::Variant variant) {
            asm_kernels::QuantumSymbolicCompressor::Params params;
            params.variant = variant;
            return new asm_kernels::QuantumSymbolicCompressor(params);
        }), py::arg("variant"))
        
        .def("compress", [](asm_kernels::QuantumSymbolicCompressor& self,
                           size_t num_qubits, size_t compression_size) {
            return complexvec_to_numpy(self.compress(num_qubits, compression_size));
        }, py::arg("num_qubits"), py::arg("compression_size") = 64)
        
        .def("measure_entanglement", &asm_kernels::QuantumSymbolicCompressor::measure_entanglement)
        
        .def("create_bell_state", [](asm_kernels::QuantumSymbolicCompressor& self, int bell_type) {
            return complexvec_to_numpy(self.create_bell_state(bell_type));
        }, py::arg("bell_type") = 0)
        
        .def("create_ghz_state", [](asm_kernels::QuantumSymbolicCompressor& self, size_t num_qubits) {
            return complexvec_to_numpy(self.create_ghz_state(num_qubits));
        }, py::arg("num_qubits"));
    
    // ========================================================================
    // Feistel Cipher (48-round, 9.2 MB/s target)
    // ========================================================================
    
    py::class_<asm_kernels::FeistelCipher>(m, "FeistelCipher", R"pbdoc(
        ASM-accelerated 48-round Feistel cipher.
        
        High-performance cipher using optimized assembly from feistel_round48.asm.
        Target throughput: 9.2 MB/s as specified in QuantoniumOS paper.
        
        Recommended variant: CHAOTIC (maximum entropy diffusion for security)
        
        Features:
        - 48-round Feistel network with 128-bit blocks
        - AES S-box substitution with AVX2
        - MixColumns diffusion with vectorization
        - AEAD authenticated encryption
        
        Example:
            >>> key = bytes(32)  # 256-bit key
            >>> cipher = rft.FeistelCipher(key, variant=rft.RFTVariant.CHAOTIC)
            >>> encrypted = cipher.encrypt_block(plaintext)
    )pbdoc")
        .def(py::init([](py::bytes key, uint32_t flags, asm_kernels::RFTKernelEngine::Variant variant) {
            std::string key_str = key;
            return new asm_kernels::FeistelCipher(
                reinterpret_cast<const uint8_t*>(key_str.data()),
                key_str.size(),
                flags,
                variant
            );
        }), py::arg("key"), py::arg("flags") = 0, py::arg("variant") = asm_kernels::RFTKernelEngine::Variant::CHAOTIC)
        .def(py::init([](py::bytes key) {
            std::string key_str = key;
            return new asm_kernels::FeistelCipher(
                reinterpret_cast<const uint8_t*>(key_str.data()),
                key_str.size(),
                0,
                asm_kernels::RFTKernelEngine::Variant::CHAOTIC
            );
        }), py::arg("key"))
        
        .def("encrypt_block", [](asm_kernels::FeistelCipher& self, py::bytes plaintext) {
            std::string pt = plaintext;
            if (pt.size() != 16) {
                throw std::invalid_argument("Block must be 16 bytes");
            }
            std::array<uint8_t, 16> pt_arr, ct_arr;
            std::memcpy(pt_arr.data(), pt.data(), 16);
            ct_arr = self.encrypt_block(pt_arr);
            return py::bytes(reinterpret_cast<const char*>(ct_arr.data()), 16);
        }, py::arg("plaintext"))
        
        .def("decrypt_block", [](asm_kernels::FeistelCipher& self, py::bytes ciphertext) {
            std::string ct = ciphertext;
            if (ct.size() != 16) {
                throw std::invalid_argument("Block must be 16 bytes");
            }
            std::array<uint8_t, 16> ct_arr, pt_arr;
            std::memcpy(ct_arr.data(), ct.data(), 16);
            pt_arr = self.decrypt_block(ct_arr);
            return py::bytes(reinterpret_cast<const char*>(pt_arr.data()), 16);
        }, py::arg("ciphertext"))
        
        .def("benchmark", [](asm_kernels::FeistelCipher& self, size_t test_size) {
            auto metrics = self.benchmark(test_size);
            return py::dict(
                py::arg("message_avalanche") = metrics.message_avalanche,
                py::arg("key_avalanche") = metrics.key_avalanche,
                py::arg("key_sensitivity") = metrics.key_sensitivity,
                py::arg("throughput_mbps") = metrics.throughput_mbps,
                py::arg("total_bytes") = metrics.total_bytes,
                py::arg("total_time_ns") = metrics.total_time_ns
            );
        }, py::arg("test_size") = 1024 * 1024)
        
        .def("avalanche_test", [](asm_kernels::FeistelCipher& self) {
            auto metrics = self.avalanche_test();
            return py::dict(
                py::arg("message_avalanche") = metrics.message_avalanche,
                py::arg("key_avalanche") = metrics.key_avalanche
            );
        })
        
        .def_static("self_test", &asm_kernels::FeistelCipher::self_test);
    
    // ASM capability detection
    m.def("has_avx2_crypto", &asm_kernels::has_avx2_crypto,
          "Check if Feistel cipher has AVX2 acceleration");
    m.def("has_aes_ni", &asm_kernels::has_aes_ni,
          "Check if Feistel cipher has AES-NI acceleration");
    
#endif // RFTMW_ENABLE_ASM
}

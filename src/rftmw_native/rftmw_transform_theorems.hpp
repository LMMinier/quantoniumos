/*
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2026 Luis M. Minier / quantoniumos
 *
 * rftmw_transform_theorems.hpp
 * ============================
 *
 * Reference (non-SIMD, non-ASM) constructions for the repo's transform-theory
 * theorem objects.
 *
 * This header is intentionally simple and dependency-free (no Eigen).
 * It exists to keep the C++ stack aligned with the canonical Python
 * definitions used by:
 *   - algorithms/rft/core/transform_theorems.py
 *   - tests/proofs/test_rft_transform_theorems.py
 *
 * Canonical frequency grid:
 *   f_k = frac((k+1) * φ)
 *   z_k = exp(i 2π f_k)
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rftmw::theorems {

using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;
using RealVec = std::vector<double>;

constexpr double PHI = 1.6180339887498948482045868343656;
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

inline double frac(double x) {
    return x - std::floor(x);
}

inline RealVec phi_frequencies(std::size_t N) {
    RealVec f;
    f.reserve(N);
    for (std::size_t k = 0; k < N; ++k) {
        f.push_back(frac((static_cast<double>(k) + 1.0) * PHI));
    }
    return f;
}

inline ComplexVec golden_roots_z(std::size_t N) {
    const RealVec f = phi_frequencies(N);
    ComplexVec z;
    z.reserve(N);
    for (double fk : f) {
        const double ang = TWO_PI * fk;
        z.emplace_back(std::cos(ang), std::sin(ang));
    }
    return z;
}

// Row-major dense matrix helper
struct Dense {
    std::size_t n = 0;
    ComplexVec a;

    explicit Dense(std::size_t n_) : n(n_), a(n_ * n_, Complex(0.0, 0.0)) {}

    Complex& operator()(std::size_t i, std::size_t j) { return a[i * n + j]; }
    const Complex& operator()(std::size_t i, std::size_t j) const { return a[i * n + j]; }
};

inline Dense eye(std::size_t N) {
    Dense Id(N);
    for (std::size_t i = 0; i < N; ++i) {
        Id(i, i) = Complex(1.0, 0.0);
    }
    return Id;
}

inline Dense matmul(const Dense& A, const Dense& B) {
    if (A.n != B.n) {
        throw std::invalid_argument("matmul: dimension mismatch");
    }
    const std::size_t N = A.n;
    Dense C(N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t k = 0; k < N; ++k) {
            const Complex aik = A(i, k);
            if (aik == Complex(0.0, 0.0)) {
                continue;
            }
            for (std::size_t j = 0; j < N; ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    return C;
}

inline Dense matpow(Dense base, std::size_t p) {
    const std::size_t N = base.n;
    Dense out = eye(N);
    while (p > 0) {
        if (p & 1U) {
            out = matmul(out, base);
        }
        p >>= 1U;
        if (p) {
            base = matmul(base, base);
        }
    }
    return out;
}

// Companion matrix from monic polynomial coefficients.
// coeffs: [1, a_{N-1}, ..., a0]
inline Dense companion_from_poly_coeffs(const std::vector<Complex>& coeffs) {
    if (coeffs.size() < 2 || coeffs.front() == Complex(0.0, 0.0)) {
        throw std::invalid_argument("companion_from_poly_coeffs: invalid coeffs");
    }
    const std::size_t N = coeffs.size() - 1;
    Dense C(N);
    for (std::size_t i = 0; i + 1 < N; ++i) {
        C(i, i + 1) = Complex(1.0, 0.0);
    }
    // Last row = [-a0, -a1, ..., -a_{N-1}]
    for (std::size_t j = 0; j < N; ++j) {
        C(N - 1, j) = -coeffs[N - j];
    }
    return C;
}

// Build monic polynomial coefficients from roots:
// p(z) = Π (z - r_i) = z^N + a_{N-1} z^{N-1} + ... + a0
inline std::vector<Complex> poly_from_roots(const ComplexVec& roots) {
    std::vector<Complex> coeffs{Complex(1.0, 0.0)};
    for (const Complex& r : roots) {
        std::vector<Complex> next(coeffs.size() + 1, Complex(0.0, 0.0));
        for (std::size_t i = 0; i < coeffs.size(); ++i) {
            next[i] += coeffs[i];
            next[i + 1] += -r * coeffs[i];
        }
        coeffs.swap(next);
    }
    return coeffs;
}

inline Dense companion_matrix_from_roots(const ComplexVec& roots) {
    return companion_from_poly_coeffs(poly_from_roots(roots));
}

inline Dense golden_companion_shift(std::size_t N) {
    return companion_matrix_from_roots(golden_roots_z(N));
}

inline Dense golden_filter_operator(const Dense& C, const ComplexVec& h) {
    if (C.n != h.size()) {
        throw std::invalid_argument("golden_filter_operator: dimension mismatch");
    }
    const std::size_t N = C.n;
    Dense H(N);
    for (std::size_t m = 0; m < N; ++m) {
        Dense Cm = matpow(C, m);
        const Complex hm = h[m];
        for (std::size_t i = 0; i < N * N; ++i) {
            H.a[i] += hm * Cm.a[i];
        }
    }
    return H;
}

inline int k99(const ComplexVec& X, double frac_energy = 0.99) {
    if (X.empty()) {
        return 0;
    }
    std::vector<double> p;
    p.reserve(X.size());
    double sum = 0.0;
    for (const Complex& v : X) {
        const double e = std::norm(v);
        p.push_back(e);
        sum += e;
    }
    if (sum <= 0.0) {
        return static_cast<int>(X.size());
    }
    for (double& v : p) {
        v /= sum;
    }
    std::sort(p.begin(), p.end(), std::greater<double>());
    double c = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        c += p[i];
        if (c >= frac_energy) {
            return static_cast<int>(i + 1);
        }
    }
    return static_cast<int>(p.size());
}

inline std::vector<ComplexVec> golden_drift_ensemble(std::size_t N, std::size_t M, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::uniform_real_distribution<double> unifA(-1.0, 1.0);

    std::vector<double> frac_n;
    frac_n.reserve(N);
    for (std::size_t n = 0; n < N; ++n) {
        frac_n.push_back(frac(static_cast<double>(n) * PHI));
    }

    std::vector<ComplexVec> out;
    out.reserve(M);
    for (std::size_t i = 0; i < M; ++i) {
        const double f0 = unif01(rng);
        const double a = unifA(rng);
        ComplexVec x;
        x.reserve(N);
        for (std::size_t n = 0; n < N; ++n) {
            const double ang = TWO_PI * (f0 * static_cast<double>(n) + a * frac_n[n]);
            x.emplace_back(std::cos(ang), std::sin(ang));
        }
        out.push_back(std::move(x));
    }
    return out;
}

}  // namespace rftmw::theorems

# /hardware/rtl/

## ARCHITECTURAL FEASIBILITY STUDY — RTL INDEX

Mandatory labels:
- **No silicon fabricated**
- **Performance based on simulation/synthesis only**
- **Not a claim of optimality**

This directory indexes RTL/TL-Verilog sources.

Primary sources currently live in `hardware/`:
- `hardware/rftpu_architecture.tlv`
- `hardware/rftpu_architecture.sv`
- `hardware/rftpu_architecture_gen.sv`
- `hardware/rft_middleware_engine.sv`
- `hardware/fpga_top.sv`
- `hardware/*.vh`

## Foundational Theorem Support (Theorems 10-12)

The hardware implementations are designed to maintain the mathematical properties
established by Theorems 10-12 in `THEOREMS_RFT_IRONCLAD.md`:

### Theorem 10: Polar Uniqueness
The RTL implements the canonical RFT basis U = Φ(Φ†Φ)^{-1/2} which is the
**unique** unitary factor in the polar decomposition. Hardware verification
points:
- Fixed-point Gram matrix computation preserves positive definiteness
- Square root inverse uses Newton-Raphson with convergence guarantees
- Unitarity error bounded by fixed-point precision (configurable via `DATA_WIDTH`)

### Theorem 11: No Exact Diagonalization  
The golden companion shift operator C_φ cannot be exactly diagonalized by any
unitary. Hardware implications:
- Off-diagonal residuals are **inherent**, not implementation errors
- Pipeline stages must preserve non-zero off-diagonal structure
- Test vectors verify residual bounds match software reference

### Theorem 12: Variational Minimality
The canonical basis minimizes J(U) = Σ 2^{-m} ||off(U† C_φ^m U)||_F² over its
natural class. Hardware verification:
- J functional can be computed in hardware for validation
- Perturbation tests verify no basis achieves lower J
- Optimality margin provides robustness against quantization

## Hardware Verification Interface

The C++ header `src/rftmw_native/rftmw_asm_kernels.hpp` defines structures for
hardware-in-the-loop theorem verification:

```cpp
struct HWTheorem10Result {
    uint32_t N;
    bool polar_factor_verified;
    bool hermitian_verified;
    bool positive_definite_verified;
    double max_error;
};

struct HWTheorem11Result {
    uint32_t N;
    double max_off_diagonal_ratio;
    uint32_t m_values_tested;
    bool impossibility_verified;
};

struct HWTheorem12Result {
    uint32_t N;
    double J_canonical;
    double J_perturbed_min;
    bool minimality_verified;
};
```

These can be populated via memory-mapped registers or DMA from FPGA/ASIC
implementations to verify mathematical properties in silicon.

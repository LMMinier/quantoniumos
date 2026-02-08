/* SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
 */

/**
 * RFT-SIS: Experimental Lattice-Style Cryptographic Hash Implementation
 * =====================================================================
 * 
 * This implements a lattice-style hash function combining:
 * 1. φ-RFT (Golden Ratio Recursive Fourier Transform) for diffusion
 * 2. SIS-style compression over Z_q
 * 
 * Notes:
 * - No formal security reduction to SIS/LWE is claimed.
 * - This is a research playground; do not treat as a PQC primitive.
 * - Matrix A is sampled uniformly in Z_q via rejection sampling.
 */

#include "rft_sis.h"
#include "sha256_portable.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

// ============================================================================
// Constants
// ============================================================================

static const double PHI = 1.6180339887498948482;
static const double TWO_PI = 6.283185307179586476;

// Fibonacci sequence mod 256 for RFT phase mixing
static const uint8_t FIB_MOD256[32] = {
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 121, 98, 219,
    61, 24, 85, 109, 194, 47, 241, 32, 17, 49, 66, 115, 181, 40, 221, 5
};

// ============================================================================
// Internal Helper Functions
// ============================================================================

/**
 * Deterministic PRNG for lattice matrix generation
 * Uses SHA-256 in counter mode
 */

typedef struct {
    uint8_t seed_block[32];
    uint32_t counter;
    uint8_t block[32];
    size_t offset;
} rft_prng_state_t;

static void prng_refill(rft_prng_state_t* state) {
    uint8_t block[36];
    memcpy(block, state->seed_block, 32);
    memcpy(block + 32, &state->counter, 4);
    sha256_hash(block, 36, state->block);
    state->offset = 0;
    state->counter++;
}

static void prng_init(rft_prng_state_t* state, const uint8_t* seed, size_t seed_len) {
    sha256_hash(seed, seed_len, state->seed_block);
    state->counter = 0;
    state->offset = 32;
}

static uint32_t prng_next_u32(rft_prng_state_t* state) {
    if (state->offset + 4 > 32) {
        prng_refill(state);
    }
    uint32_t val = ((uint32_t)state->block[state->offset]) |
                   ((uint32_t)state->block[state->offset + 1] << 8) |
                   ((uint32_t)state->block[state->offset + 2] << 16) |
                   ((uint32_t)state->block[state->offset + 3] << 24);
    state->offset += 4;
    return val;
}

/**
 * Generate uniform random element in Z_q via rejection sampling
 */
static int32_t sample_zq_uniform(rft_prng_state_t* state) {
    const uint64_t limit = (uint64_t)UINT32_MAX + 1ULL;
    const uint64_t bound = limit - (limit % RFT_SIS_Q);
    uint32_t val;
    do {
        val = prng_next_u32(state);
    } while ((uint64_t)val >= bound);
    return (int32_t)(val % RFT_SIS_Q);
}

/**
 * Apply φ-RFT phase rotation to input vector
 * This provides golden-ratio based diffusion
 */
static void apply_rft_phases(const double* phases, const double* input,
                            double* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        // Phase rotation: output[i] = input[i] * exp(i * phases[i])
        // For real input, this becomes mixing with sin/cos
        double phase = phases[i];
        double c = cos(phase);
        double s = sin(phase);
        
        // Mix with neighboring elements for diffusion
        size_t prev = (i == 0) ? n - 1 : i - 1;
        size_t next = (i == n - 1) ? 0 : i + 1;
        
        output[i] = input[i] * c + 
                   input[prev] * s * PHI +
                   input[next] * s / PHI;
    }
}

/**
 * Quantize float vector to short integers in [-β, β]
 */
static void quantize_to_sis(const double* input, int32_t* output, size_t n) {
    // Find max for normalization
    double max_val = 1e-15;
    for (size_t i = 0; i < n; i++) {
        double abs_val = fabs(input[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    
    // Scale and quantize
    double scale = (RFT_SIS_BETA * 0.95) / max_val;
    for (size_t i = 0; i < n; i++) {
        int32_t val = (int32_t)round(input[i] * scale);
        // Clamp to [-β, β]
        if (val > RFT_SIS_BETA) val = RFT_SIS_BETA;
        if (val < -RFT_SIS_BETA) val = -RFT_SIS_BETA;
        output[i] = val;
    }
}

/**
 * Compute SIS: y = A·s mod q
 */
static void compute_sis(const rft_sis_ctx_t* ctx, const int32_t* s,
                       int32_t* y) {
    for (size_t i = 0; i < RFT_SIS_M; i++) {
        int64_t sum = 0;
        for (size_t j = 0; j < RFT_SIS_N; j++) {
            sum += (int64_t)ctx->A[i][j] * (int64_t)s[j];
        }
        // Reduce mod q, centered representation
        y[i] = (int32_t)(sum % RFT_SIS_Q);
        if (y[i] > RFT_SIS_Q / 2) y[i] -= RFT_SIS_Q;
        if (y[i] < -RFT_SIS_Q / 2) y[i] += RFT_SIS_Q;
    }
}

// ============================================================================
// Public API Implementation
// ============================================================================

rft_sis_error_t rft_sis_init(rft_sis_ctx_t* ctx, const uint8_t* seed, rft_variant_t variant) {
    if (!ctx) return RFT_SIS_ERROR_INVALID_PARAM;
    
    memset(ctx, 0, sizeof(rft_sis_ctx_t));
    ctx->variant = variant;  // Store variant for RFT phase generation
    
    // Use default seed if none provided
    const uint8_t default_seed[] = "RFT-SIS-v1.0-QuantoniumOS-2025";
    if (!seed) {
        seed = default_seed;
    }
    
    // Generate lattice matrix A from seed (uniform in Z_q)
    rft_prng_state_t prng;
    prng_init(&prng, seed, strlen((const char*)seed));
    for (size_t i = 0; i < RFT_SIS_M; i++) {
        for (size_t j = 0; j < RFT_SIS_N; j++) {
            ctx->A[i][j] = sample_zq_uniform(&prng);
        }
    }
    
    // Pre-compute RFT phases using golden ratio
    for (size_t k = 0; k < RFT_SIS_N; k++) {
        // Canonical RFT phase: θ_k = 2π * frac((k+1)·φ)
        double freq = (double)(k + 1) * PHI;
        double frac = freq - floor(freq);
        ctx->rft_phases[k] = TWO_PI * frac;
    }
    
    ctx->initialized = true;
    return RFT_SIS_SUCCESS;
}

rft_sis_error_t rft_sis_cleanup(rft_sis_ctx_t* ctx) {
    if (!ctx) return RFT_SIS_ERROR_INVALID_PARAM;
    
    // Secure cleanup
    memset(ctx, 0, sizeof(rft_sis_ctx_t));
    return RFT_SIS_SUCCESS;
}

rft_sis_error_t rft_sis_hash(const rft_sis_ctx_t* ctx,
                            const uint8_t* input, size_t input_len,
                            uint8_t* output) {
    if (!ctx || !ctx->initialized || !output) {
        return RFT_SIS_ERROR_INVALID_PARAM;
    }
    if (!input && input_len > 0) {
        return RFT_SIS_ERROR_INVALID_PARAM;
    }
    
    // Step 1: Expand input to n-dimensional vector using SHA-256
    double expanded[RFT_SIS_N];
    memset(expanded, 0, sizeof(expanded));
    
    // Hash input to get initial expansion
    uint8_t hash_chain[32];
    
    if (input_len > 0) {
        sha256_hash(input, input_len, hash_chain);
    } else {
        // Hash empty string
        sha256_hash((const uint8_t*)"", 0, hash_chain);
    }
    
    // Expand to full dimension using hash chain
    size_t dim_idx = 0;
    uint8_t chain_input[64];
    memcpy(chain_input, hash_chain, 32);
    if (input_len > 0 && input_len <= 32) {
        memcpy(chain_input + 32, input, input_len);
    }
    
    while (dim_idx < RFT_SIS_N) {
        // Convert hash bytes to floats
        for (size_t i = 0; i < 32 && dim_idx < RFT_SIS_N; i += 4, dim_idx++) {
            uint32_t val = ((uint32_t)hash_chain[i]) |
                          ((uint32_t)hash_chain[i+1] << 8) |
                          ((uint32_t)hash_chain[i+2] << 16) |
                          ((uint32_t)hash_chain[i+3] << 24);
            expanded[dim_idx] = ((double)val / 4294967296.0) * 2.0 - 1.0;
        }
        
        // Chain hash for more expansion
        sha256_hash(chain_input, 64, hash_chain);
        memcpy(chain_input, hash_chain, 32);
    }
    
    // Step 2: Apply φ-RFT transformation
    double rft_output[RFT_SIS_N];
    apply_rft_phases(ctx->rft_phases, expanded, rft_output, RFT_SIS_N);
    
    // Apply second round of RFT for better diffusion
    double rft_output2[RFT_SIS_N];
    apply_rft_phases(ctx->rft_phases, rft_output, rft_output2, RFT_SIS_N);
    
    // Step 3: Quantize to short integer vector
    int32_t s[RFT_SIS_N];
    quantize_to_sis(rft_output2, s, RFT_SIS_N);
    
    // Step 4: Compute SIS: y = A·s mod q
    int32_t y[RFT_SIS_M];
    compute_sis(ctx, s, y);
    
    // Step 5: Final hash of lattice point
    // Combine y with domain separator
    size_t final_input_len = sizeof(y) + 12;
    uint8_t* final_input = (uint8_t*)malloc(final_input_len);
    if (!final_input) return RFT_SIS_ERROR_INIT_FAILED;
    
    memcpy(final_input, y, sizeof(y));
    memcpy(final_input + sizeof(y), "RFT-SIS-v1.0", 12);
    sha256_hash(final_input, final_input_len, output);
    free(final_input);
    
    return RFT_SIS_SUCCESS;
}

rft_sis_error_t rft_sis_kdf(const rft_sis_ctx_t* ctx,
                           const uint8_t* master_key, size_t master_key_len,
                           const uint8_t* info, size_t info_len,
                           uint8_t* output, size_t output_len) {
    if (!ctx || !ctx->initialized || !master_key || !output) {
        return RFT_SIS_ERROR_INVALID_PARAM;
    }
    
    // Combine master key with info for domain separation
    size_t combined_len = master_key_len + info_len;
    uint8_t* combined = (uint8_t*)malloc(combined_len);
    if (!combined) return RFT_SIS_ERROR_INIT_FAILED;
    
    memcpy(combined, master_key, master_key_len);
    if (info && info_len > 0) {
        memcpy(combined + master_key_len, info, info_len);
    }
    
    // Generate output using RFT-SIS hash in counter mode
    uint8_t counter[4] = {0, 0, 0, 0};
    uint8_t block[RFT_SIS_HASH_SIZE];
    size_t offset = 0;
    
    while (offset < output_len) {
        // Hash: combined || counter
        size_t block_input_len = combined_len + 4;
        uint8_t* block_input = (uint8_t*)malloc(block_input_len);
        if (!block_input) {
            free(combined);
            return RFT_SIS_ERROR_INIT_FAILED;
        }
        
        memcpy(block_input, combined, combined_len);
        memcpy(block_input + combined_len, counter, 4);
        
        rft_sis_error_t err = rft_sis_hash(ctx, block_input, block_input_len, block);
        free(block_input);
        
        if (err != RFT_SIS_SUCCESS) {
            free(combined);
            return err;
        }
        
        size_t copy_len = (output_len - offset < RFT_SIS_HASH_SIZE) ? 
                         (output_len - offset) : RFT_SIS_HASH_SIZE;
        memcpy(output + offset, block, copy_len);
        offset += copy_len;
        
        // Increment counter
        for (int i = 0; i < 4 && ++counter[i] == 0; i++);
    }
    
    free(combined);
    return RFT_SIS_SUCCESS;
}

rft_sis_error_t rft_sis_mac(const rft_sis_ctx_t* ctx,
                           const uint8_t* key, size_t key_len,
                           const uint8_t* data, size_t data_len,
                           uint8_t* tag) {
    if (!ctx || !ctx->initialized || !key || !tag) {
        return RFT_SIS_ERROR_INVALID_PARAM;
    }
    
    // HMAC-like construction: H(K || H(K || data))
    // Inner hash
    size_t inner_len = key_len + data_len;
    uint8_t* inner_input = (uint8_t*)malloc(inner_len);
    if (!inner_input) return RFT_SIS_ERROR_INIT_FAILED;
    
    memcpy(inner_input, key, key_len);
    if (data && data_len > 0) {
        memcpy(inner_input + key_len, data, data_len);
    }
    
    uint8_t inner_hash[RFT_SIS_HASH_SIZE];
    rft_sis_error_t err = rft_sis_hash(ctx, inner_input, inner_len, inner_hash);
    free(inner_input);
    
    if (err != RFT_SIS_SUCCESS) return err;
    
    // Outer hash
    size_t outer_len = key_len + RFT_SIS_HASH_SIZE;
    uint8_t* outer_input = (uint8_t*)malloc(outer_len);
    if (!outer_input) return RFT_SIS_ERROR_INIT_FAILED;
    
    memcpy(outer_input, key, key_len);
    memcpy(outer_input + key_len, inner_hash, RFT_SIS_HASH_SIZE);
    
    err = rft_sis_hash(ctx, outer_input, outer_len, tag);
    free(outer_input);
    
    return err;
}

rft_sis_error_t rft_sis_mac_verify(const rft_sis_ctx_t* ctx,
                                  const uint8_t* key, size_t key_len,
                                  const uint8_t* data, size_t data_len,
                                  const uint8_t* tag) {
    uint8_t computed_tag[RFT_SIS_HASH_SIZE];
    
    rft_sis_error_t err = rft_sis_mac(ctx, key, key_len, data, data_len, computed_tag);
    if (err != RFT_SIS_SUCCESS) return err;
    
    // Constant-time comparison
    uint8_t diff = 0;
    for (size_t i = 0; i < RFT_SIS_HASH_SIZE; i++) {
        diff |= computed_tag[i] ^ tag[i];
    }
    
    return (diff == 0) ? RFT_SIS_SUCCESS : RFT_SIS_ERROR_HASH_FAILED;
}

rft_sis_error_t rft_sis_avalanche_test(const rft_sis_ctx_t* ctx,
                                       rft_sis_metrics_t* metrics) {
    if (!ctx || !ctx->initialized || !metrics) {
        return RFT_SIS_ERROR_INVALID_PARAM;
    }
    
    memset(metrics, 0, sizeof(rft_sis_metrics_t));
    
    // Test with small input changes
    uint8_t input1[32] = "Test input for avalanche test!!";
    uint8_t input2[32] = "Test input for avalanche test!\"";  // One bit different
    
    uint8_t hash1[RFT_SIS_HASH_SIZE];
    uint8_t hash2[RFT_SIS_HASH_SIZE];
    
    rft_sis_hash(ctx, input1, 32, hash1);
    rft_sis_hash(ctx, input2, 32, hash2);
    
    // Count differing bits
    size_t bits_changed = 0;
    for (size_t i = 0; i < RFT_SIS_HASH_SIZE; i++) {
        uint8_t xor_val = hash1[i] ^ hash2[i];
        while (xor_val) {
            bits_changed += xor_val & 1;
            xor_val >>= 1;
        }
    }
    
    metrics->bits_changed = bits_changed;
    metrics->total_bits = RFT_SIS_HASH_SIZE * 8;
    metrics->avalanche_ratio = (double)bits_changed / metrics->total_bits;
    
    return RFT_SIS_SUCCESS;
}

const char* rft_sis_version(void) {
    return "RFT-SIS v1.0 (Experimental Lattice Hash)";
}

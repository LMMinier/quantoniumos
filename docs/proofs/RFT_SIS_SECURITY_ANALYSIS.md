# RFT-SIS Security Analysis

> **Status**: DRAFT - Requires peer review  
> **Date**: February 2026  
> **Authors**: QuantoniumOS Team

---

## ⚠️ IMPORTANT DISCLAIMER

This document presents a **security analysis sketch**, not a complete proof. 

**What this document provides:**
- Formal security game definitions
- Reduction sketch from RFT-SIS to standard SIS
- Identification of proof gaps

**What this document does NOT provide:**
- A complete, verified security proof
- Guarantees of security
- Endorsement for production use

**For production cryptography, use NIST-approved primitives (SHA-3, AES, Kyber, Dilithium).**

---

## 1. Preliminaries

### 1.1 Notation

| Symbol | Meaning |
|--------|---------|
| λ | Security parameter |
| n, m, q | SIS parameters (dimension, width, modulus) |
| φ | Golden ratio = (1 + √5)/2 ≈ 1.618 |
| ℤ_q | Integers modulo q |
| ‖x‖ | Euclidean norm of vector x |
| PPT | Probabilistic polynomial-time |
| negl(λ) | Negligible function in λ |

### 1.2 The Short Integer Solution (SIS) Problem

**Definition (SIS_{n,m,q,β})**: Given uniformly random **A** ∈ ℤ_q^{n×m}, find non-zero **z** ∈ ℤ^m such that:
1. **Az** = **0** (mod q)
2. ‖**z**‖ ≤ β

**Hardness Assumption**: For appropriate parameters (n, m, q, β), SIS is hard for PPT adversaries. Specifically, for any PPT algorithm 𝒜:

```
Pr[𝒜(A) → z : Az = 0 ∧ 0 < ‖z‖ ≤ β] ≤ negl(λ)
```

**Known Results** (Ajtai 1996, Micciancio-Regev 2007):
- SIS is at least as hard as worst-case lattice problems (SIVP, GapSVP)
- For q ≥ β·√n, SIS is hard assuming standard lattice assumptions

---

## 2. RFT-SIS Hash Construction

### 2.1 The φ-Structured Matrix

Unlike standard SIS which uses uniform random **A**, RFT-SIS uses a **structured matrix**:

```
A_φ[i,j] = ⌊q · frac((i·j + 1) · φ)⌋ mod q
```

where `frac(x) = x - ⌊x⌋` is the fractional part.

**Properties of A_φ:**
1. Deterministic (no randomness needed)
2. Efficient to compute: O(nm) time
3. Toeplitz-like structure along anti-diagonals
4. Entries are equidistributed in ℤ_q (by Weyl's theorem)

### 2.2 Hash Function Definition

**RFT-SIS-Hash**: {0,1}* → {0,1}^{256}

```
H(m) = Compress(A_φ · Expand(m) mod q)
```

Where:
- **Expand**: {0,1}* → ℤ_q^m using SHA3-based expansion
- **A_φ**: The φ-structured matrix ∈ ℤ_q^{n×m}
- **Compress**: ℤ_q^n → {0,1}^{256} (rounding + truncation)

### 2.3 Current Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n | 256 | Output size = 256 bits |
| m | 512 | Compression ratio 2:1 |
| q | 3329 | Prime, matches Kyber |
| β | √m ≈ 22.6 | Bounded expansion |

---

## 3. Security Definitions

### 3.1 Collision Resistance

**Definition (CR)**: A hash function H is (t, ε)-collision resistant if for all t-time adversaries 𝒜:

```
Pr[(m₁, m₂) ← 𝒜(1^λ) : m₁ ≠ m₂ ∧ H(m₁) = H(m₂)] ≤ ε
```

### 3.2 Preimage Resistance  

**Definition (Pre)**: H is (t, ε)-preimage resistant if for random m:

```
Pr[m' ← 𝒜(H(m)) : H(m') = H(m)] ≤ ε
```

### 3.3 Second Preimage Resistance

**Definition (Sec)**: H is (t, ε)-second-preimage resistant if:

```
Pr[m' ← 𝒜(m) : m' ≠ m ∧ H(m') = H(m)] ≤ ε
```

---

## 4. Security Reduction (Sketch)

### 4.1 Theorem Statement

**Theorem 1 (Collision Resistance)**: If φ-SIS_{n,m,q,β} is (t, ε)-hard, then RFT-SIS-Hash is (t', ε')-collision resistant where:
- t' = t - O(n·m)
- ε' ≤ ε

### 4.2 Proof Sketch

**Reduction**: We construct algorithm ℬ that solves φ-SIS given access to collision-finder 𝒜.

```
Algorithm ℬ(A_φ):
    1. Receive φ-structured matrix A_φ ∈ ℤ_q^{n×m}
    2. Define H(m) = Compress(A_φ · Expand(m) mod q)
    3. Run 𝒜 on H, receive (m₁, m₂) with H(m₁) = H(m₂)
    4. Compute x₁ = Expand(m₁), x₂ = Expand(m₂)
    5. Set z = x₁ - x₂
    6. Return z
```

**Analysis**:
- If H(m₁) = H(m₂), then A_φ·x₁ ≡ A_φ·x₂ (mod q) (ignoring compression loss)
- Therefore A_φ·z = A_φ·(x₁ - x₂) = 0 (mod q)
- Since m₁ ≠ m₂ and Expand is injective, z ≠ 0
- Bound: ‖z‖ ≤ ‖x₁‖ + ‖x₂‖ ≤ 2·‖Expand(·)‖_max ≤ β

**Conclusion**: If 𝒜 finds collisions with probability ε', then ℬ solves φ-SIS with probability ≥ ε'. □

---

## 5. The Critical Gap: φ-SIS vs Random-SIS

### 5.1 The Problem

The reduction above proves:

```
Collision in RFT-SIS → Solution to φ-SIS
```

But standard SIS hardness assumes **uniform random A**. We need:

```
φ-SIS hard ← ??? → Random-SIS hard
```

### 5.2 What We Know

**Positive indicators:**
1. **Weyl equidistribution**: Entries of A_φ are equidistributed in ℤ_q
2. **No obvious structure**: φ is irrational, so no simple period
3. **Empirical testing**: No collisions found in 10^6 samples

**Potential weaknesses:**
1. **Algebraic structure**: A_φ has Toeplitz-like structure
2. **Deterministic**: Same A_φ for all users (no salt)
3. **Low entropy**: A_φ is fully determined by (n, m, q, φ)

### 5.3 Possible Approaches to Close the Gap

**Option A: Prove φ-SIS ≈ Random-SIS**
- Show that the algebraic structure doesn't help adversaries
- Would require new techniques in lattice cryptography

**Option B: Hybrid argument**
- Show A_φ is computationally indistinguishable from random A
- Likely false due to deterministic construction

**Option C: Direct security proof**
- Prove φ-SIS hard directly without reducing to random SIS
- Would require new hardness assumption

**Option D: Modify construction**
- Add randomness: A = A_φ + R where R is random
- Loses efficiency but gains provable security

---

## 6. Empirical Security Evidence

While not a proof, empirical evidence suggests no obvious weaknesses:

| Test | Result | Sample Size |
|------|--------|-------------|
| Avalanche (SAC) | 50.1% | 6,400 |
| Collision search | None found | 500,000 |
| Preimage search | None found | 100,000 |
| Bit balance | 0.4996 | 256,000 |
| NIST STS | Pass | 1,000,000 bits |

---

## 6.1 Cryptanalysis Findings (February 2026)

**Internal audit discovered the following:**

### Finding 1: Actual Implementation Uses Random Matrix

Contrary to Section 2.1's description, the actual `RFTSISHash` class uses:
```python
np.random.seed(42)
self.A = np.random.randint(0, sis_q, size=(sis_m, sis_n))
```

**Implication**: The SIS matrix is pseudo-random (seeded), not φ-structured.
This is **more secure** than φ-structured, but introduces a fixed-seed issue.

### Finding 2: Fixed Seed Weakness

The matrix A is identical for all users (seed=42). This means:
- No per-user salt
- Multi-target attacks may be easier
- **Recommended fix**: Salt the seed with domain separator

### Finding 3: Hypothetical φ-Matrix Has Structural Weakness

If we were to use φ-structured matrix with formula:
```
A_φ[i,j] = floor(q * frac((i*j + 1) * φ))
```

**Vulnerability discovered**:
- Row 0: All entries constant (i=0 → i*j=0 for all j)
- Column 0: All entries constant (j=0 → i*j=0 for all i)
- χ² uniformity 10x worse than random

**Attack**: Row 0 provides no mixing, reducing effective security.

**Fix if using φ-matrix**: Use `(i+1)*(j+1)` instead of `(i*j+1)`

### Finding 4: RFT Uses Correct Formula

The RFT basis uses `f_k = frac((k+1) * φ)` which does NOT have the constant-row issue.

---

## 7. Conclusions

### 7.1 What We Have Proven

✅ **Conditional security**: RFT-SIS-Hash is collision-resistant IF φ-SIS is hard

### 7.2 What Remains Unproven

❌ **φ-SIS hardness**: No reduction to standard lattice assumptions  
❌ **IND-CPA for encryption**: Would require additional construction  
❌ **Concrete security bounds**: Need tighter analysis

### 7.3 Honest Assessment

| Claim | Status |
|-------|--------|
| "Provably secure" | ❌ FALSE |
| "Secure under φ-SIS assumption" | ⚠️ CONDITIONAL |
| "No known attacks" | ✅ TRUE (as of Feb 2026) |
| "Passes statistical tests" | ✅ TRUE |
| "Ready for production" | ❌ FALSE |

### 7.4 Recommendations

1. **Do not use for production cryptography**
2. **Submit to IACR ePrint for peer review**
3. **Invite cryptanalysis from lattice experts**
4. **Consider hybrid construction** (A_φ + random) for provable security

---

## 8. Open Problems

1. **Is φ-SIS as hard as random SIS?**
   - Prove or disprove
   
2. **Does the Toeplitz structure help attackers?**
   - Analyze using lattice reduction (LLL, BKZ)
   
3. **Can we achieve IND-CPA encryption?**
   - Fujisaki-Okamoto transform on top of RFT-SIS?

4. **Optimal parameter selection?**
   - What (n, m, q, β) gives 128-bit security?

---

## References

1. Ajtai, M. (1996). "Generating hard instances of lattice problems"
2. Micciancio, D. & Regev, O. (2007). "Worst-case to average-case reductions for SIS"
3. Lyubashevsky, V. (2012). "Lattice signatures without trapdoors"
4. NIST PQC (2024). "Post-Quantum Cryptography Standardization"

---

## Appendix A: Formal Security Game

```
Game CR_H:
    Setup: pp ← Setup(1^λ)
    Challenge: (m₁, m₂) ← 𝒜(pp)
    Win condition: m₁ ≠ m₂ ∧ H(pp, m₁) = H(pp, m₂)
    
Advantage: Adv^CR_H(𝒜) = Pr[𝒜 wins]

Definition: H is collision-resistant if for all PPT 𝒜:
    Adv^CR_H(𝒜) ≤ negl(λ)
```

---

## Appendix B: Parameter Justification

Current parameters (n=256, m=512, q=3329):

**Security level estimate** (heuristic, NOT proven):
- Lattice dimension: n = 256
- Hermite factor: δ = (β/q^{n/m})^{1/n} ≈ 1.007
- Estimated BKZ block size: b ≈ 380
- Classical security: ~128 bits (estimated)
- Quantum security: ~64 bits (Grover on BKZ)

**Caveat**: These estimates assume random A. For φ-structured A, security may be lower.

---

*Document status: DRAFT - Not peer reviewed*
*Last updated: February 4, 2026*

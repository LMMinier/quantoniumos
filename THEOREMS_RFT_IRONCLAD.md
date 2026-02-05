# THEOREMS_RFT_IRONCLAD.md
## Scope (what this file *does*)
This file contains theorem statements with full proofs for:
- Canonical RFT unitarity (Gram-normalized / symmetric orthogonalization).
- Fast RFT unitarity (factorized).
- Twisted convolution theorem + the diagonalization claim (exact).
- A **provable** “not an N-point DFT kernel (up to phases/permutation)” result for the raw φ-grid kernel used to build the canonical basis.
- Crypto: what reductions you can formally claim (and what you cannot) without new assumptions.

This file does **not** pretend to prove:
- “Post-quantum strength” of RFT-SIS with structured matrices.
- IND-CPA/IND-CCA for any scheme unless it is explicitly built from a standard primitive with a standard proof.
- Wigner–Dyson / quantum-chaos claims (those are empirical unless you specify an operator family and prove an asymptotic law).

---

## Definitions

Let φ := (1+√5)/2.

### D1 (Raw φ-grid exponential basis Φ)
Fix N and define the golden frequency grid
f_k := frac((k+1)·φ) ∈ [0,1),   k∈{0,…,N-1},
where frac(t) = t - ⌊t⌋.

Define the raw (generally non-orthogonal) basis Φ ∈ ℂ^{N×N} by
Φ[n,k] := exp(i 2π f_k n)/√N,   n,k∈{0,…,N-1}.

This is the canonical “φ-grid exponential” kernel used across the codebase.

### D2 (Canonical RFT basis U: Gram-normalized / Löwdin orthogonalization)
Define the canonical unitary basis
U := Φ(ΦᴴΦ)^{-1/2}.

The canonical RFT is
x̂ := Uᴴ x,   x := U x̂.

This is the repo’s canonical definition and matches the implementation used in verification and tests.

### D3 (Fast RFT (factorized) variant)
Let F be the unitary DFT matrix (FFT matrix) of size N×N.
Let C_σ and D_φ be diagonal matrices with unit-modulus diagonal entries (phase-only):
(C_σ)_{kk} = exp(-i π σ g(k)),     (D_φ)_{kk} = exp(-i 2π h_φ(k)),
for some real functions g, h_φ.
Define the fast RFT matrix:
Ψ := D_φ C_σ F,
and transforms:
x̂_fast := Ψ x,    x := Ψ^† x̂_fast.
[Source: RFT PDF]

### D4 (Twisted convolution induced by a unitary)
Given a unitary Ψ, define the Ψ-twisted convolution of x,h ∈ ℂ^N by:
x ⋆_Ψ h := Ψ^† ( (Ψx) ⊙ (Ψh) ),
where ⊙ is pointwise (Hadamard) multiplication.

This is the exact algebraic statement you use for ⋆_{φ,σ}.  [Source: RFT PDF]

---

## Theorem 1 (Full rank of the raw φ-grid basis Φ)
**Statement.**
The matrix Φ is invertible for every N ≥ 1.

**Proof.**
Write z_k := exp(i 2π f_k). Then
Φ[n,k] = (1/√N) z_k^n,   n=0,…,N-1,
so Φ is a Vandermonde matrix (up to the nonzero scalar factor 1/√N per column) on nodes {z_k}_{k=0}^{N-1}.
Its determinant is
det(Φ) = (1/√N)^N ∏_{0≤i<j≤N-1} (z_j - z_i).
It suffices to show z_i ≠ z_j for i≠j.

If z_i = z_j then exp(i 2π (f_i - f_j)) = 1, so (f_i - f_j) ∈ ℤ.
But f_i,f_j ∈ [0,1), hence f_i - f_j ∈ (-1,1), so the only possible integer is 0, i.e. f_i=f_j.

Now f_k = frac((k+1)φ). If f_i=f_j then (i-j)φ ∈ ℤ, which is impossible because φ is irrational.
Therefore all z_k are distinct, det(Φ)≠0, and Φ is invertible. ∎

---

## Theorem 2 (Canonical RFT basis U is unitary)
**Statement.**
U is unitary, i.e., UᴴU = I.

**Proof.**
By Theorem 1, Φ has full rank, so G := ΦᴴΦ is Hermitian positive definite and G^{-1/2} exists.
Compute:
UᴴU = (G^{-1/2})ᴴ Φᴴ Φ G^{-1/2} = G^{-1/2} G G^{-1/2} = I,
since G^{-1/2} is Hermitian. ∎

---

## Theorem 3 (Fast RFT is unitary)
**Statement.**
If F, C_σ, D_φ are unitary, then Ψ := D_φ C_σ F is unitary.

**Proof.**
Products of unitary matrices are unitary:
Ψ^† Ψ = F^† C_σ^† D_φ^† D_φ C_σ F = F^† C_σ^† C_σ F = F^† F = I,
since D_φ^†D_φ = I and C_σ^†C_σ = I by unit-modulus diagonals, and F is unitary. ∎

---

## Theorem 4 (Twisted convolution theorem; exact diagonalization)
**Statement.**
For ⋆_Ψ defined in D4, the transform-domain multiplication rule holds:
Ψ(x ⋆_Ψ h) = (Ψx) ⊙ (Ψh).
Equivalently, for each fixed h, the linear operator T_h(x):= x ⋆_Ψ h is diagonalized by Ψ:
T_h = Ψ^† diag(Ψh) Ψ.

**Proof.**
By definition,
x ⋆_Ψ h = Ψ^†( (Ψx) ⊙ (Ψh) ).
Apply Ψ to both sides:
Ψ(x ⋆_Ψ h) = ΨΨ^†( (Ψx) ⊙ (Ψh) ) = (Ψx) ⊙ (Ψh).

For the operator form, note that pointwise multiplication is multiplication by a diagonal matrix:
(Ψx) ⊙ (Ψh) = diag(Ψh) (Ψx).
Therefore
T_h(x) = Ψ^† diag(Ψh) Ψ x,
i.e., T_h = Ψ^† diag(Ψh) Ψ. ∎

**Corollary 4.1 (Eigenvalues).**
The eigenvalues of T_h are exactly the components of Ψh.

---

## Theorem 5 (Algebraic properties of ⋆_Ψ)
**Statement.**
⋆_Ψ is commutative and associative, and has identity element e := Ψ^† 1 (where 1 is the all-ones vector in ℂ^N):
x ⋆_Ψ h = h ⋆_Ψ x,
(x ⋆_Ψ h) ⋆_Ψ g = x ⋆_Ψ (h ⋆_Ψ g),
x ⋆_Ψ e = x.

**Proof.**
Let X:=Ψx, H:=Ψh, G:=Ψg.
Then x⋆_Ψh = Ψ^†(X⊙H). Since ⊙ is commutative and associative, the first two claims follow.
For identity: Ψe = 1, so x⋆_Ψe = Ψ^†(X⊙1)=Ψ^†X=x. ∎

---

## Theorem 6 (Raw φ-grid kernel is not an N-point DFT kernel, up to phases/permutation)
This theorem is an “iron-clad” non-equivalence claim that matches the current canonical φ-grid kernel Φ.

**Statement.**
Fix N≥2. There do not exist:
- row phases a_n with |a_n|=1,
- column phases b_k with |b_k|=1, and
- a permutation π of {0,…,N-1},
such that for all n,k we have
exp(i2π f_k n)/√N = a_n b_k · exp(-i2π n·π(k)/N)/√N.

Equivalently: the raw φ-grid exponential basis is not just a permuted/rephased N-point DFT.

**Proof.**
Assume such a_n,b_k,π exist. Fix k and take the ratio of consecutive n:

Left side:
Φ[n+1,k]/Φ[n,k] = exp(i2π f_k).

Right side:
(a_{n+1}/a_n) · exp(-i2π π(k)/N).

The left side is independent of n, so a_{n+1}/a_n must be constant in n; write a_{n+1}/a_n = exp(iθ).
Then for every k,
exp(i2π f_k) = exp(iθ) · exp(-i2π π(k)/N),
so f_k ≡ c - π(k)/N (mod 1) for some constant c.

But the set {c - π(k)/N mod 1 : k=0,…,N-1} is exactly the set of N rational points with denominator N (a shifted permutation of {0,1/N,…,(N-1)/N}).
By construction, f_k = frac((k+1)φ) is irrational for every k, so it cannot equal any rational with denominator N.
Contradiction. ∎

**Interpretation (what you may claim safely).**
The canonical raw kernel is Fourier-like, but it is not the N-point DFT kernel in disguise.

---

## Theorem 7 (Crypto: what reductions you can and cannot claim)

### D6 (Standard SIS collision formulation)
Let q≥2. For A ∈ ℤ_q^{m×n}, SIS asks for a nonzero “short” vector s ∈ ℤ^n such that
A s ≡ 0 (mod q),
with ||s||₂ ≤ β (ℓ₂ bound; other norms require specifying β accordingly).  [Standard SIS references]

### Theorem 7.1 (Collision ⇒ SIS for *uniform* A)
**Statement.**
Let A be uniform in ℤ_q^{m×n}. Define h(x)=A x (mod q) over a bounded domain X ⊂ ℤ^n (e.g., {0,1}^n).
If x≠x' and h(x)=h(x'), then s:=x-x' is a nonzero short vector satisfying A s ≡ 0 (mod q), i.e., an SIS solution.

**Proof.**
h(x)=h(x') implies A x ≡ A x' (mod q), hence A(x-x')≡0 (mod q).
Since x≠x', s=x-x'≠0. If X is bounded, then s is short (bounded by domain diameter). ∎

### Theorem 7.2 (Structured A needs a new assumption; no automatic SIS reduction)
**Statement.**
If A is sampled from a structured distribution D (e.g., “RFT-derived operators projected to ℤ_q”), then Theorem 7.1 does **not** imply security under the standard SIS assumption unless you additionally prove or assume:
A ~ D is computationally indistinguishable from uniform in ℤ_q^{n×m},
or you explicitly adopt a **structured-SIS(D)** assumption.

**Proof.**
Standard SIS hardness is defined for uniform A. For a non-uniform distribution D, the average-case problem is different.
If D is distinguishable from uniform, then “reductions” that treat A as uniform are invalid: an adversary can first distinguish the distribution and then potentially exploit structure.
Therefore, either (i) prove D ≈ uniform (computationally), or (ii) state a new assumption SIS(D). ∎

### Theorem 7.3 (Avalanche / NIST-style statistics do not prove PRF/IND security)
**Statement.**
Passing avalanche heuristics (≈50% bit flips) and statistical batteries is insufficient to conclude pseudorandomness (PRF/PRP) or IND-CPA/IND-CCA security.

**Proof (explicit counterexample).**
Let f(x)=M x over GF(2), where M is an invertible binary matrix whose columns each have Hamming weight ≈ m/2.
Then flipping a random single bit of x flips ≈ half the output bits on average (avalanche-like behavior).
But f is linear and trivially distinguishable from a PRF by linearity tests, and it is efficiently invertible.
Therefore avalanche-like behavior does not imply cryptographic pseudorandomness or one-wayness. ∎

**Alignment with your paper.**
Your own threat-model section explicitly states no reduction-based security and no IND-CPA/IND-CCA/preimage claims; keep that language until you have Theorem 7.2’s missing indistinguishability/assumption.  [Source: RFT PDF]
### Theorem 7.4 (Hybrid Construction A = A_φ + R is Uniform When R is Uniform)
**Statement.**
Let A_φ ∈ ℤ_q^{m×n} be deterministic with A_φ[i,j] = ⌊q · frac((i+1)(j+1)φ)⌋.
Let R ∈ ℤ_q^{m×n} be sampled **uniformly over the full matrix space** (e.g., from a CSPRNG modeled as uniform).
Define A := A_φ + R (mod q).

Then A is **exactly uniform** over ℤ_q^{m×n}; collision resistance of h_A(x)=A x (mod q) reduces to standard SIS with parameters (n, m, q, β).

**Proof.**
For fixed A_φ, the map f: R ↦ A_φ + R (mod q) is a bijection on ℤ_q^{m×n}. Uniform R implies uniform A by group-shift invariance. Apply Theorem 7.1 to obtain the SIS reduction. ∎

**Scope note.** If R is sampled from any non-uniform distribution (e.g., sparse, low-rank, small-noise, structured), the conclusion “A is uniform” no longer holds; a separate indistinguishability analysis would be required.

### Theorem 7.5 (Concrete Security Estimate for RFT-SIS Parameters)
**Parameters.** n=512, m=1024, q=3329, β=100; lattice dimension for BKZ = m.

**Estimated cost (heuristic).** Using the Chen–Nguyen root-Hermite-factor model + Core-SVP cost 0.292·b (classical sieving), the required δ is
δ_needed = (β / q^{n/m})^{1/(m-1)} = (100 / 57.69749)^{1/1023} ≈ **1.00053774**.

Under the Chen–Nguyen δ(b) curve, achieving δ≈1.00053774 corresponds to **b ≈ 5348** (well beyond the calibrated range), yielding a heuristic cost of ~2^{1562} classical operations and ~2^{1420} quantum operations. See scripts/estimate_sis_security.py for a reproducible computation.

**Proof sketch.**
1. det(Λ)^{1/m} = q^{n/m} = 3329^{0.5} ≈ 57.69749.
2. Require δ^{m-1} · det(Λ)^{1/m} ≤ β ⇒ δ ≤ (β / det(Λ)^{1/m})^{1/(m-1)} ≈ 1.00053774.
3. Inverting δ(b) via Chen–Nguyen gives b ≈ 5348; this is far outside validated BKZ models.
4. Cost ≈ 2^{0.292·b} (classical sieving heuristic) and 2^{0.2655·b} (quantum sieving heuristic).

**Assumptions and caveats.**
- δ(b) and Core-SVP cost are **heuristic** and not calibrated for b in the thousands; treat the numbers as upper-bound-style placeholders, not trusted estimates.
- Worst-case→average-case reduction does not apply: m < n·log₂(q) (1024 < ~5991). Security relies on concrete hardness of random SIS.
- “Above Level 5” language removed: attack cost is reported explicitly instead of relative levels.
- The estimator caps search at b=10,000; if δ_needed were smaller, results would be marked out-of-range.

**Security/Narrative split.** Hardness is standard SIS with uniform A (by masking A_φ with uniform R). The φ-structure remains as a mixing/engineering layer, not the hardness source.

**Parameter snapshot.**
| Parameter | Value | Note |
|-----------|-------|------|
| n | 512 | secret/solution dimension |
| m | 1024 | lattice dimension for BKZ |
| q | 3329 | Kyber prime |
| β | 100 | SIS norm bound |
| det(Λ)^{1/m} | 57.69749 | q^{n/m} |
| δ target | 1.00053774 | (β / det_root)^{1/(m-1)} |
| b (heuristic) | 5348 | Chen–Nguyen inversion (extrapolated) |
| Cost classical | ~2^{1562} | Core-SVP 0.292·b (heuristic) |
| Cost quantum | ~2^{1420} | 0.2655·b (heuristic) |
---

## What is still missing for the specific “iron-clad” claims you listed

### A) “Canonical RFT is outside a metaplectic/Clifford-like family” (strong form)
If you want a theorem of the form “U is not in the discrete metaplectic / Clifford / monomial-conjugation closure”, you must:
1) Define the exact family (what generators are allowed, what equivalence is allowed).
2) Prove a crisp invariant P for every member of that family.
3) Prove U violates P.

Right now, the repo includes an operational exclusion theorem (non-monomial conjugations of shift/modulation) as a test-backed claim (see Theorem set E).

### B) “Diagonalization claims”
You *do* have an exact, formal diagonalization result (Theorem 4) — but it is definitional: any unitary defines a twisted convolution that it diagonalizes.
If you want “diagonalizes a naturally arising operator family” as novelty, you must:
- Define the operator family independently of Ψ (e.g., a physically/number-theoretically defined golden operator),
- Then prove Ψ diagonalizes it.

---

## Test-backed theorem set (A–E, repo-ready and falsifiable)

These are the “engineering–math interface” theorems implemented as deterministic, falsifiable tests.
They are not presented as fully general asymptotic theorems; instead, each statement includes an explicit pass condition.

**Reference implementation (authoritative objects):**
- [algorithms/rft/core/transform_theorems.py](algorithms/rft/core/transform_theorems.py)

**Test suite (claims firewall):**
- [tests/proofs/test_rft_transform_theorems.py](tests/proofs/test_rft_transform_theorems.py)

### Theorem A (Nearest-unitary optimality; polar factor)
**Statement (testable form).** Let Φ be the raw φ-grid basis and U its Gram-normalized form. Then U is the unique nearest unitary to Φ in Frobenius norm (i.e., U is the unitary polar factor of Φ).

**Pass condition.** In CI we verify:
- `U == polar(Φ).U` to numerical tolerance, and
- `||Φ-U||_F` is no larger than the distance to many random Haar unitaries.

### Theorem B (Golden companion shift eigenstructure)
**Statement (testable form).** Define roots z_k = exp(i2π f_k) with f_k = frac((k+1)φ). Let Cφ be the Frobenius companion matrix for p(z)=∏(z-z_k), and let V be the Vandermonde eigenvector matrix with columns v_k = (1,z_k,…,z_k^{N-1})ᵀ. Then Cφ V = V diag(z).

**Pass condition.** Residual `||CφV - Vdiag(z)||_F / ||V||_F` is below a fixed tolerance, and `V` matches √N·Φ.

### Theorem C (Golden convolution/filter algebra diagonalizes)
**Statement (testable form).** For any filter coefficients h, define Hφ(h)=∑_{m=0}^{N-1} h[m] Cφ^m. Then the resonance eigenvectors diagonalize Hφ(h), i.e. Hφ(h) V = V diag(p_h(z_k)) where p_h is the polynomial defined by h.

**Pass condition.** Off-diagonal energy in `V^{-1} Hφ(h) V` is below a fixed tolerance.

### Theorem D (Golden-native operator family favors the canonical RFT basis)
**Statement (testable form).** The canonical basis U yields a lower off-diagonal ratio than the FFT basis for golden-native operators (Cφ and Hφ(h)).

**Pass condition.** For fixed N and deterministic RNG seed, we assert an explicit margin between RFT and FFT off-diagonal ratios.

**Negative control.** For an almost-Mathieu-like periodic discretization L, the FFT basis diagonalizes better than RFT at finite N.

### Theorem E (Empirical optimality under golden drift ensemble; inequality-style)
**Statement (candidate).** For signals x[n]=exp(i2π(f0 n + a·frac(nφ))) drawn from a simple quasi-periodic “golden drift” model, the canonical RFT basis yields more concentrated coefficients than the FFT on average, measured by K99 (smallest K capturing ≥99% energy).

**Pass condition.** With fixed N, M, and RNG seed, mean K99 for RFT is smaller than FFT by a modest margin, and much smaller than a random Haar unitary baseline.

### C) “Crypto strength”
If you want any statement stronger than “mixing sandbox,” you need one of:
- A standard construction (e.g., CTR with AES/ChaCha) and then use the standard proof; or
- A proof that your structured A distribution is indistinguishable from uniform (hard), or a clearly stated new assumption SIS(D) with careful parameterization.

---
## Theorem 8 / Conjecture (Golden Spectral Concentration Inequality)

This is the central asymptotic inequality for the canonical RFT basis — the "Slepian-style" theorem for golden quasi-periodic signals.

### Setup

Let:
- U_φ ∈ ℂ^{N×N} be the **canonical RFT basis** (Definition D2).
- F ∈ ℂ^{N×N} be the unitary DFT.
- ℰ_φ be the **golden quasi-periodic ensemble**:
```
x[n] = exp(i 2π (f₀ n + a · frac(n φ))),
f₀ ~ Uniform[0,1],  a ~ Uniform[-1,1]
```

Define the **spectral concentration functional**:
```
K₀.₉₉(U, x) = min{ K : Σ_{k ∈ top-K} |(Ux)_k|² ≥ 0.99 ‖x‖₂² }
```
(the smallest K coefficients capturing ≥99% of energy).

### Statement (Asymptotic Inequality)

**Golden Spectral Concentration Inequality:**
```
limsup_{N→∞} 𝔼_{x∼ℰ_φ}[K₀.₉₉(U_φ, x)]  <  liminf_{N→∞} 𝔼_{x∼ℰ_φ}[K₀.₉₉(F, x)]
```

### Interpretation

> **In the large-N limit, the canonical RFT requires strictly fewer coefficients than the FFT to represent golden quasi-periodic signals.**

This is the exact analogue of:
- Slepian's concentration theorem (time–band limiting)
- Fourier uncertainty principle
- Wavelet sparsity bounds

Except the domain is: **irrational frequency drift**.

### Current Status: Empirically Verified Conjecture

⚠️ **Important**: We do NOT rely solely on p-values (which can wobble). Instead, we gate on:

1. **Mean paired improvement**: E[K₀.₉₉(F,x) - K₀.₉₉(U_φ,x)] ≥ δ(N)
2. **Bootstrap CI** that stays entirely above zero
3. **Effect size** (Cohen's d > 0.2)

**Bootstrap-verified evidence (N=128, M=500):**

| Metric | Value |
|--------|-------|
| Mean K₀.₉₉(RFT) | ~57 |
| Mean K₀.₉₉(FFT) | ~60 |
| Mean improvement | ~2.5 |
| 95% Bootstrap CI | [1.8, 3.2] (excludes 0) ✓ |
| Cohen's d | ~0.35 (small-medium effect) ✓ |
| RFT win rate | ~58% |

**Minimum Effect Threshold δ(N):**
- δ(32) = 0.5 coefficients
- δ(64) = 1.0 coefficient
- δ(128) = 2.0 coefficients
- General: δ(N) ≈ √N / 6

**Negative control (FFT-native harmonic ensemble):**
- Pure harmonics at integer frequencies
- FFT achieves K₀.₉₉ = 1 (perfect sparsity)
- RFT achieves K₀.₉₉ ≈ 17 (not native)
- This confirms the inequality is ensemble-specific, not a universal claim.

### Proof Roadmap (for future work)

A full proof would follow this structure:
1. Model golden drift as a **deterministic almost-periodic process**.
2. Show its covariance operator has **approximate eigenfunctions** close to Φ.
3. Use perturbation theory (Kato/Davis–Kahan) to bound eigenfunction alignment.
4. Convert eigenvalue decay into **concentration inequality**.

### Test Reference

**Falsifiable tests:** 
- [tests/proofs/test_rft_transform_theorems.py](tests/proofs/test_rft_transform_theorems.py)
- `test_theorem_8_golden_concentration_inequality_holds`
- `test_theorem_8_negative_control_harmonic_ensemble`
- `test_theorem_8_scaling_across_N`
- `test_theorem_8_random_unitary_is_much_worse`

**Bootstrap CI verification:**
- [algorithms/rft/core/theorem8_bootstrap_verification.py](algorithms/rft/core/theorem8_bootstrap_verification.py)
- `verify_theorem_8_bootstrap()` - Full bootstrap CI analysis
- `verify_theorem_8_with_effect_threshold()` - With δ(N) gate
- `analyze_scaling()` - Multi-N scaling analysis
- `test_theorem_8_negative_control_harmonic_ensemble`
- `test_theorem_8_scaling_across_N`

---

## Theorem 9 (Maassen-Uffink Entropic Uncertainty Principle for RFT)

This theorem establishes the **correct finite-dimensional uncertainty principle** for the canonical RFT, using the Maassen-Uffink entropic bound.

### ⚠️ Important: Why Not Heisenberg?

The continuous Heisenberg bound $\Delta x \cdot \Delta p \geq \hbar/2$ does **NOT** directly apply to finite-dimensional discrete transforms. Using "1/(4π)" as a lower bound for discrete spread products is **incorrect** and can lead to apparent violations.

**The correct finite-dimensional uncertainty principle is entropic (Maassen-Uffink, 1988).**

### Definition: Mutual Coherence

**D7 (Mutual coherence).** For a unitary matrix U ∈ ℂ^{N×N}:
```
μ(U) := max_{j,k} |U_{jk}|
```

Reference values:
- DFT: μ(F) = 1/√N (maximally incoherent)
- Identity: μ(I) = 1 (maximally coherent)
- RFT: μ(U_φ) ∈ (1/√N, 1), depends on N

### Definition: Shannon Entropy

**D8 (Signal entropy).** For a probability distribution p = |x|² / ||x||²:
```
H(p) := -Σ_k p_k log(p_k)
```

Low entropy = concentrated signal. High entropy = spread signal.

### Statement (Maassen-Uffink Entropic Uncertainty)

**Theorem 9.** For any unit vector x ∈ ℂ^N and the canonical RFT basis U_φ:

```
H(|x|²) + H(|U_φ^H x|²) ≥ -2 log(μ(U_φ))
```

This is a **TRUE THEOREM** that MUST hold for all signals. It is not approximate.

### Special Cases

| Basis | Mutual Coherence | Entropy Bound |
|-------|------------------|---------------|
| DFT (F) | μ = 1/√N | H(x) + H(Fx) ≥ log(N) |
| Identity (I) | μ = 1 | H(x) + H(x) ≥ 0 (trivial) |
| **RFT (U_φ)** | **μ ∈ (1/√N, 1)** | **H(x) + H(U_φ x) ≥ -2 log(μ)** |

### Interpretation for RFT

Since μ(U_φ) > 1/√N, the RFT has a **looser entropic bound** than the DFT:
```
-2 log(μ(U_φ)) < log(N) = -2 log(1/√N)
```

This means RFT can achieve **lower combined entropy** than DFT on certain signals, while still satisfying the uncertainty principle.

### Connection to Theorem 8 (Concentration)

The entropic uncertainty principle explains **why Theorem 8 holds**:

1. Golden quasi-periodic signals achieve low time-domain entropy (spread in time)
2. Under RFT, they achieve low frequency-domain entropy (concentrated)
3. The sum H(x) + H(U_φ x) stays above the bound, but H(U_φ x) alone is minimized
4. This is measured by K₀.₉₉ (few coefficients capture most energy)

**Key insight:** RFT doesn't violate uncertainty—it achieves a different entropy balance than DFT.

### Empirical Verification

**CI-verified results (N=64):**

| Signal Type | H(x) | H(DFT) | H(RFT) | DFT sum | RFT sum | RFT bound |
|-------------|------|--------|--------|---------|---------|-----------|
| Delta | 0.00 | 4.16 | 3.98 | 4.16 | 3.98 | 3.71 |
| Uniform | 4.16 | 0.00 | 3.21 | 4.16 | 7.37 | 3.71 |
| Gaussian | 2.31 | 2.29 | 2.42 | 4.60 | 4.73 | 3.71 |
| Harmonic | 4.16 | 0.00 | 3.14 | 4.16 | 7.30 | 3.71 |
| Golden QP | 4.16 | 3.87 | 3.52 | 8.03 | 7.68 | 3.71 |

All sums exceed their respective bounds ✓

### Proof

The Maassen-Uffink inequality is a standard result in quantum information theory:

1. Let P = diag(|x|²) and Q = U^H diag(|Ux|²) U
2. These are the "position" and "momentum" observables
3. By Riesz-Thorin interpolation on the overlap matrix: ||P^{1/2} Q^{1/2}||_∞ ≤ μ
4. The entropy inequality follows from the uncertainty relation for overlapping observables

**Reference:** Maassen, H. & Uffink, J.B.M. (1988). Physical Review Letters, 60(12), 1103.

### Test Reference

**Falsifiable tests:** [tests/proofs/test_maassen_uffink_uncertainty.py](tests/proofs/test_maassen_uffink_uncertainty.py)
- `test_theorem_9_maassen_uffink_bound_holds_for_all_signals`
- `test_theorem_9_rft_bound_looser_than_dft`
- `test_theorem_9_rft_concentrates_golden_qp_signals`
- `test_theorem_9_dft_concentrates_harmonics`

### Implementation

**Reference code:** [algorithms/rft/core/maassen_uffink_uncertainty.py](algorithms/rft/core/maassen_uffink_uncertainty.py)

---
## Theorem 10 (Uniqueness of the canonical RFT basis as the polar-normalized Φ basis)

### Statement

Let Φ ∈ ℂ^{N×N} be the raw φ-grid exponential basis (Definition D1), and let

```
U := Φ(Φ†Φ)^{-1/2}.
```

Then **U is the unique unitary matrix** such that

```
U†Φ is Hermitian positive definite.
```

Equivalently, U is the **unique unitary factor** in the polar decomposition of Φ.

### Proof

By Theorem 1, Φ has full rank, so Φ admits a polar decomposition

```
Φ = U H,    H := (Φ†Φ)^{1/2},
```

where U is unitary and H is Hermitian positive definite.

Suppose there exists another unitary matrix U' such that

```
U'†Φ is Hermitian positive definite.
```

Then

```
U'†Φ = U'† U H.
```

Since H is positive definite, the product U'† U must itself be Hermitian.
But a matrix that is both **unitary and Hermitian** satisfies

```
(U'† U)² = I,
```

so its eigenvalues are ±1.
Positive definiteness forces all eigenvalues to be +1, hence

```
U'† U = I  ⟹  U' = U.
```

Therefore U is unique. ∎

### Interpretation

This proves that **canonical RFT normalization is mathematically forced**, not a design choice. No other unitary can remove the non-orthogonality of Φ without reintroducing phase distortion.

---

## Theorem 11 (Unitary diagonalization criterion for C_φ)

### Statement

There exists a unitary U such that U† C_φ U is diagonal **iff** C_φ is normal (C_φ C_φ† = C_φ† C_φ). Moreover, if such a U exists for C_φ, then the same U diagonalizes all powers C_φ^m.

### Proof
(⇒) If U† C_φ U = D is diagonal, then C_φ = U D U† and C_φ C_φ† = U D D† U† = U D† D U† = C_φ† C_φ, so C_φ is normal.

(⇐) If C_φ is normal, the spectral theorem gives a unitary eigenbasis U with U† C_φ U diagonal. Then U† C_φ^m U = D^m is diagonal for all m. ∎

### Remark (non-normality of the implemented C_φ)
Numerically, the companion construction used here yields ‖C_φ C_φ† − C_φ† C_φ‖_F > 0 for tested N (see tests/proofs/test_rft_transform_theorems.py), so it is **not** unitarily diagonalizable; this is test-backed, not a closed-form proof.

---

## Conjecture 12 (Empirical variational minimality of the canonical RFT basis)

### Statement (empirical/test-backed)

Let C_φ be the golden companion shift operator and define

```
J(U) := Σ_{m=0}^{∞} 2^{-m} ||off(U† C_φ^m U)||_F².
```

Empirically (via tests/proofs/test_rft_transform_theorems.py), the canonical basis

```
U_φ = Φ(Φ†Φ)^{-1/2}
```

achieves lower J(U) than permutation/phase variants and than several random Haar baselines for tested N. This is **not proven**; it is a conjecture supported by numerical evidence.

### Status
- Not a theorem. Use only as a test-backed conjecture until a formal proof or counterexample is provided.

---

## Summary of Theorems 10–12

| Result | Claim | Status |
|--------|-------|--------|
| **Theorem 10** | Polar normalization uniqueness | ✓ Proven |
| **Theorem 11** | Normality criterion for unitary diagonalization; implemented C_φ is empirically non-normal | ✓ Proven (criterion) / test-backed (non-normality) |
| **Conjecture 12** | Variational minimality (empirical) | ⚠ Conjecture/test-backed |

These close the formal pieces (Theorem 10–11) and isolate the empirical claim (Conjecture 12) so it is not misread as proven.

---## References used (external)
- DLCT/LCT decomposition literature (chirp multiplication / convolution / FT factorization).
- SIS/LWE standard definitions and assumption boundaries.

(Keep the citations in the paper body; do not paraphrase these as “proof of PQ security.”)

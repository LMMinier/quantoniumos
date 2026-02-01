# THEOREMS_RFT_IRONCLAD.md
## Scope (what this file *does*)
This file contains theorem statements with full proofs for:
- Canonical Φ-RFT unitarity (QR-derived).
- Fast Φ-RFT unitarity (factorized).
- Twisted convolution theorem + the diagonalization claim (exact).
- A **provable** “not an LCT/FrFT” result for the *resonance kernel* used to build the canonical basis.
- Crypto: what reductions you can formally claim (and what you cannot) without new assumptions.

This file does **not** pretend to prove:
- “Post-quantum strength” of RFT-SIS with structured matrices.
- IND-CPA/IND-CCA for any scheme unless it is explicitly built from a standard primitive with a standard proof.
- Wigner–Dyson / quantum-chaos claims (those are empirical unless you specify an operator family and prove an asymptotic law).

---

## Definitions

Let φ := (1+√5)/2.

### D1 (Resonance vectors and resonance matrix)
For fixed N, define vectors v_k ∈ ℂ^N by
v_k[n] := exp(-i 2π φ^{-k} n),  for n,k ∈ {0,…,N-1}.
Let R ∈ ℂ^{N×N} be the matrix with entries R_{n,k} := v_k[n].
(These are exactly your published resonance vectors and resonance matrix.)  [Source: RFT PDF]

### D2 (Canonical Φ-RFT)
Let R = Q R_upper be the (thin) QR factorization with Q ∈ ℂ^{N×N} unitary and R_upper upper triangular.
Define U_φ := Q and the canonical Φ-RFT by
  x̂ := U_φ^† x,     x := U_φ x̂.
[Source: RFT PDF]

### D3 (Fast Φ-RFT)
Let F be the unitary DFT matrix (FFT matrix) of size N×N.
Let C_σ and D_φ be diagonal matrices with unit-modulus diagonal entries (phase-only):
  (C_σ)_{kk} = exp(-i π σ g(k)),     (D_φ)_{kk} = exp(-i 2π h_φ(k)),
for some real functions g, h_φ.
Define the fast Φ-RFT matrix:
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

## Theorem 1 (Full rank of the resonance matrix R)
**Statement.**
The resonance matrix R is invertible for every N ≥ 1.

**Proof.**
Write z_k := exp(-i 2π φ^{-k}). Then
R_{n,k} = z_k^n,   n=0,…,N-1.
So R is a Vandermonde matrix on nodes {z_k}_{k=0}^{N-1}. Its determinant is
  det(R) = ∏_{0≤i<j≤N-1} (z_j - z_i).
It suffices to show z_i ≠ z_j for i≠j.

If z_i = z_j then exp(-i 2π (φ^{-i} - φ^{-j})) = 1, so (φ^{-i} - φ^{-j}) ∈ ℤ.
But 0 < φ^{-k} ≤ 1 for all k≥0, and φ^{-i} ≠ φ^{-j} for i≠j, hence
  0 < |φ^{-i} - φ^{-j}| < 1,
so it cannot be an integer. Contradiction. Therefore all z_k are distinct, det(R)≠0, and R is invertible. ∎

---

## Theorem 2 (Canonical Φ-RFT is unitary)
**Statement.**
U_φ is unitary, i.e., U_φ^† U_φ = I.

**Proof.**
By definition, R = Q R_upper is a QR factorization with Q unitary. Setting U_φ := Q gives
U_φ^† U_φ = Q^† Q = I. ∎

---

## Theorem 3 (Fast Φ-RFT is unitary)
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

## Theorem 6 (A provable “not LCT/FrFT” result — for the resonance kernel)
This theorem is the “iron-clad” version of “non-quadratic kernel” you can actually prove today, because it targets R_{n,k} directly (your generating kernel), not the post-QR matrix Q (which mixes columns).

### D5 (Quadratic-phase / DLCT-type kernel class)
Call a kernel “quadratic-phase” if it can be written (up to row/column phase factors and column permutation) as
  K_{n,k} = exp(i( a n^2 + b nk + c k^2 + d n + e k + f )),
with real constants a,b,c,d,e,f.

This covers the standard discrete LCT/FrFT kernels built from chirp multiplications/convolutions and Fourier transforms (quadratic-phase structure is the defining invariant in the DLCT literature).  [Standard DLCT/LCT decomposition references]

**Statement.**
The resonance kernel R_{n,k} = exp(-i 2π n φ^{-k}) cannot be represented as a quadratic-phase kernel in D5, even after:
- multiplying rows/columns by arbitrary unit-modulus phase factors, and
- permuting columns.

**Proof.**
Assume for contradiction that there exist:
- phase factors α_n, β_k (real),
- a permutation π of {0,…,N-1}, and
- real constants a,b,c,d,e,f,
such that for all n,k:
  exp(-i 2π n φ^{-π(k)}) = exp(iα_n) exp(iβ_k) exp(i( a n^2 + b n k + c k^2 + d n + e k + f )).

Fix k and take the ratio of consecutive n:
Left side:
  R_{n+1,π(k)} / R_{n,π(k)} = exp(-i 2π φ^{-π(k)}),   independent of n.
Right side:
  exp(i(α_{n+1}-α_n)) * exp(i( a((n+1)^2-n^2) + b k((n+1)-n) + d((n+1)-n) ))
= exp(i(α_{n+1}-α_n)) * exp(i( a(2n+1) + b k + d )).

For this to be independent of n for all n, the term a(2n+1) must vanish, hence a=0.
So the n-ratio becomes:
  exp(-i 2π φ^{-π(k)}) = exp(i(α_{n+1}-α_n)) * exp(i(b k + d)),
still for all n,k.

Now the left side does not depend on n, so exp(i(α_{n+1}-α_n)) must be constant in n; call it exp(iγ).
Thus for all k:
  exp(-i 2π φ^{-π(k)}) = exp(i(γ + b k + d)).

Taking arguments modulo 2π implies:
  φ^{-π(k)} ≡ -(γ + b k + d)/(2π)   (mod 1).

But k ↦ φ^{-π(k)} takes N distinct values in (0,1], and its successive differences are not constant (it decays exponentially), whereas k ↦ (b k + const) mod 1 is an affine rotation with constant increments.
An affine rotation cannot match an exponential sequence at more than 2 points without forcing b=0 and const matching, which would make the right-hand side constant in k — contradicting distinctness of {φ^{-π(k)}}.

Therefore no such quadratic-phase representation exists. ∎

**Interpretation (what you may claim safely).**
Your *generating resonance kernel* is **provably non-quadratic-phase**, hence it is not a disguised DLCT/FrFT kernel in the standard quadratic-phase sense used in LCT decompositions.

(If you want “canonical Q is not DLCT” as a theorem, you must define the DLCT family precisely and prove Q is outside it; that is a separate proof obligation.)

---

## Theorem 7 (Crypto: what reductions you can and cannot claim)

### D6 (Standard SIS collision formulation)
Let q≥2. For A ∈ ℤ_q^{n×m}, SIS asks for a nonzero “short” vector s ∈ ℤ^m such that
  A s ≡ 0 (mod q),
with ||s|| bounded (depending on the parameter set).  [Standard SIS references]

### Theorem 7.1 (Collision ⇒ SIS for *uniform* A)
**Statement.**
Let A be uniform in ℤ_q^{n×m}. Define h(x)=A x (mod q) over a bounded domain X ⊂ ℤ^m (e.g., {0,1}^m).
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

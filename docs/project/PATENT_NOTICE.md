# PATENT NOTICE - QuantoniumOS

**Version 2.0 — February 2026**

---

## USPTO Application 19/169,399

**Title:** Hybrid Computational Framework for Quantum and Resonance Simulation  
**Filing Date:** April 3, 2025  
**First Named Inventor:** Luis Michael Minier  
**Confirmation No.:** 6802  
**Status:** Pending

---

## Patent Claims Summary

| Claim | Title | Description | Primary Implementation |
|-------|-------|-------------|------------------------|
| **1** | Symbolic Resonance Fourier Transform Engine | Golden-ratio frequency grid (φ-grid) with Gram-normalized unitary basis | `canonical_true_rft.py`, `resonant_fourier_transform.py` |
| **2** | Resonance-Based Cryptographic Subsystem | RFT-SIS hash function, Feistel cipher with RFT mixing | `rft_sis_hash_v31.py`, `enhanced_cipher.py` |
| **3** | Geometric Structures for Cryptographic Waveform Hashing | Topological waveform embedding and geometric hash functions | `geometric_waveform_hash.py` |
| **4** | Hybrid Mode Integration | DCT-RFT cascade codecs with zero-coherence routing | `h3_arft_cascade.py`, hybrid codecs |

**Full Specifications:** [docs/patent/USPTO_ALGORITHM_SPECIFICATIONS.md](../patent/USPTO_ALGORITHM_SPECIFICATIONS.md)

---

## What This Patent Covers

### Claim 1: Resonance Fourier Transform (RFT)

The patented method includes:
- **φ-Grid Construction:** Frequency grid based on golden ratio: $f_k = \text{frac}((k+1)\phi)$
- **Gram Normalization:** Unitary basis via $\widetilde{\Phi} = \Phi(\Phi^H\Phi)^{-1/2}$
- **Golden Spectral Properties:** Theorems 1-9 as documented in THEOREMS_RFT_IRONCLAD.md
- **Hardware Implementation:** RFTPU architecture for silicon implementation

### Claim 2: RFT-Based Cryptography

The patented method includes:
- **RFT-SIS Hash:** Short Integer Solution lattice hash using RFT basis
- **Feistel-48 Cipher:** 48-round Feistel network with RFT mixing function
- **φ-Phase Signatures:** Golden-ratio phase-based digital signatures

### Claim 3: Geometric Waveform Hashing

The patented method includes:
- **Topological Embedding:** Waveform-to-manifold mapping
- **Geometric Hash Functions:** Hash functions based on geometric invariants
- **Collision Resistance:** Geometric structure for collision resistance

### Claim 4: Hybrid Transform Codecs

The patented method includes:
- **Zero-Coherence Routing:** Automatic DCT/RFT basis selection
- **Cascade Architecture:** Multi-stage transform decomposition
- **Adaptive Quantization:** Signal-dependent coefficient quantization

---

## License Grant for Open-Source Files

### AGPL-Licensed Files (NOT in CLAIMS_PRACTICING_FILES.txt)

For files distributed under AGPL-3.0-or-later, patent rights are granted per AGPL Section 11:

> "Each contributor grants you a non-exclusive, worldwide, royalty-free patent license under the contributor's essential patent claims, to make, use, sell, offer for sale, import and otherwise run, modify and propagate the contents of its contributor version."

**This means:** AGPL files may be used commercially with AGPL compliance. Patent license follows the code.

### Research-Licensed Files (IN CLAIMS_PRACTICING_FILES.txt)

For files under LICENSE-CLAIMS-NC.md:

| Use Case | Patent License Granted |
|----------|----------------------|
| Academic research | ✅ Yes (non-commercial) |
| Education | ✅ Yes (non-commercial) |
| Peer review | ✅ Yes (non-commercial) |
| Benchmarking | ✅ Yes (non-commercial) |
| **Commercial use** | ❌ **No — requires separate license** |

---

## Prohibited Activities (Without Commercial License)

The following activities **require a commercial patent license**:

### Software
- Integrating RFT algorithms into commercial products
- Offering RFT transforms as a service (SaaS/PaaS)
- Building competitive products using RFT methods
- Reselling or sublicensing RFT implementations

### Hardware
- Manufacturing ASICs implementing RFTPU architecture
- Producing FPGAs with RFT accelerator IP
- Selling hardware products using RFT methods

### Derivative Works
- Creating "clean room" implementations of patented methods
- Developing equivalent algorithms based on patent disclosures
- Building products that practice patent claims through any implementation

---

## Research & Verification Rights

### Explicitly Permitted (No Commercial License Required)

This patent notice **does not restrict**:

1. **Scientific Verification**
   - Running tests to verify mathematical claims
   - Reproducing published benchmarks
   - Publishing verification reports

2. **Academic Publication**
   - Citing and discussing the patented methods
   - Publishing research using the software
   - Including negative or critical findings

3. **Security Research**
   - Auditing cryptographic implementations
   - Responsible vulnerability disclosure
   - Publishing security analyses

4. **Patent Challenges**
   - Filing prior art submissions
   - Participating in patent examination
   - Challenging patent validity through proper legal channels

---

## Commercial Licensing

### How to Obtain a Commercial License

For commercial use of patent-protected methods, contact:

**Luis M. Minier**  
📧 **Email:** luisminier79@gmail.com  
🔗 **GitHub:** [github.com/LMMinier/quantoniumos/issues](https://github.com/LMMinier/quantoniumos/issues)

### License Tiers

| Tier | Criteria | Includes |
|------|----------|----------|
| **Startup** | < $1M ARR, < 10 employees | Software use, limited hardware |
| **Enterprise** | > $1M ARR | Full software and hardware rights |
| **Academic Commercial** | University spin-offs | Technology transfer support |
| **Hardware OEM** | ASIC/FPGA manufacturing | Silicon rights, IP blocks |

### What's Included in Commercial License

- Patent license for USPTO 19/169,399
- Rights to derivative works
- Technical support (tier-dependent)
- Updates and improvements
- Hardware manufacturing rights (OEM tier)

---

## Trademark Notice

The following marks are trademarks of Luis M. Minier:

| Mark | Status |
|------|--------|
| **QuantoniumOS** | ™ Common Law |
| **RFTPU** | ™ Common Law |
| **Φ-RFT** | ™ Common Law |
| **Resonant Fourier Transform** | ™ Common Law |

Use of these marks requires separate written permission and is **NOT** included in any software license.

---

## Related Applications

This patent application may be related to:
- Continuation applications (if filed)
- Divisional applications (if filed)
- Foreign counterpart applications (if filed)
- Any patent that issues from the above

Commercial licensees receive rights to all related applications.

---

## Contact Information

**Inventor/Licensor:** Luis M. Minier  
📧 **Email:** luisminier79@gmail.com  
🔗 **GitHub:** [github.com/LMMinier](https://github.com/LMMinier)  
📍 **Repository:** [github.com/LMMinier/quantoniumos](https://github.com/LMMinier/quantoniumos)

---

## Legal Disclaimer

This notice is provided for informational purposes. It does not constitute legal advice. The scope of patent claims is determined by the patent claims as issued (or as pending), not by this notice. Consult a patent attorney for specific legal questions.

---

**Last Updated:** February 3, 2026  
**Version:** 2.0

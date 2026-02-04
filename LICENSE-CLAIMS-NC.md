# PATENT-PRACTICING FILES LICENSE (LICENSE-CLAIMS-NC)

**Version 2.0 — February 2026**

This license applies ONLY to the files explicitly listed in `docs/project/CLAIMS_PRACTICING_FILES.txt`. These files implement algorithms claimed in **U.S. Patent Application 19/169,399** ("Hybrid Computational Framework for Quantum and Resonance Simulation").

> **Key Claimed Algorithm**: Hybrid FFT/RFT Transform  
> Formula: $Y = E \odot \text{FFT}(x) / \sqrt{N}$ where $E[k] = e^{i \cdot 2\pi \cdot \text{frac}((k+1)\phi)}$  
> See: `docs/HYBRID_FFT_RFT_ALGORITHM.md`

---

## PREAMBLE

This dual-license structure is designed to:
1. **Protect IP:** Ensure commercial exploitation requires explicit authorization
2. **Enable Research:** Allow unrestricted academic verification and reproducibility
3. **Support Open Science:** Maintain transparency for peer review and validation

---

## 1. DEFINITIONS

- **"Covered Files"**: Files listed in `docs/project/CLAIMS_PRACTICING_FILES.txt`
- **"Patent Claims"**: Methods described in USPTO Application 19/169,399
- **"Non-Commercial Use"**: Use that does not generate revenue, competitive advantage, or commercial products
- **"Research Use"**: Scientific inquiry, validation, benchmarking, and peer review
- **"Commercial Use"**: Any use intended for profit, competitive advantage, or integration into commercial products/services
- **"Derivative Work"**: Any work based upon or incorporating the Covered Files

---

## 2. GRANT OF LICENSE (Non-Commercial Research)

Subject to the terms of this License, **Luis M. Minier** grants you a worldwide, royalty-free, non-exclusive, non-transferable license to:

### 2.1 Permitted Non-Commercial Uses

| Use Case | Permitted | Notes |
|----------|-----------|-------|
| **Academic Research** | ✅ YES | Validate claims, reproduce benchmarks, publish findings |
| **Peer Review** | ✅ YES | Verify correctness for journal/conference review |
| **Education** | ✅ YES | Teaching, coursework, student projects |
| **Personal Learning** | ✅ YES | Non-profit hobbyist experimentation |
| **Open-Source Contribution** | ✅ YES | Bug fixes, documentation, tests (upstream only) |
| **Benchmarking** | ✅ YES | Compare performance for research papers |
| **Security Audits** | ✅ YES | Identify vulnerabilities (responsible disclosure) |

### 2.2 Rights Granted for Research

You may:
- **Copy** the Covered Files for research purposes
- **Modify** the Covered Files to test hypotheses or validate claims
- **Run** the Covered Files to reproduce experimental results
- **Publish** findings, benchmarks, and analyses derived from running the software
- **Share** the unmodified Covered Files with attribution for collaborative research
- **Create** derivative works for non-commercial research only

---

## 3. RESTRICTIONS (Commercial Use Prohibited Without License)

### 3.1 Prohibited Commercial Uses

You may **NOT**, without a separate commercial license:

| Prohibited Use | Description |
|----------------|-------------|
| **Product Integration** | Incorporate algorithms into paid software/hardware |
| **Service Provision** | Use algorithms to process data for paying customers |
| **Competitive Analysis** | Use to build competing commercial products |
| **Resale/Sublicensing** | Sell, license, or sublicense the Covered Files |
| **Patent Circumvention** | Create "clean room" implementations of claimed methods |
| **Hardware Manufacturing** | Fabricate ASICs/FPGAs implementing claimed methods |
| **Cloud Services** | Offer RFT transforms as a service (SaaS/PaaS) |

### 3.2 Patent Rights Reserved

**NO PATENT LICENSE IS GRANTED** for commercial purposes. The algorithms in Covered Files are subject to:
- **USPTO Application 19/169,399** (pending)
- Any continuation, divisional, or foreign counterpart applications

Commercial practice of the claimed methods **requires a separate written patent license**.

### 3.3 Trademark Restrictions

The names "QuantoniumOS," "RFTPU," "Φ-RFT," and associated logos are trademarks of Luis M. Minier and are **NOT licensed** under this agreement.

---

## 4. ATTRIBUTION REQUIREMENTS

All permitted uses must include:

### 4.1 Source Code Attribution
```
This software includes code from QuantoniumOS (https://github.com/LMMinier/quantoniumos)
Licensed under LICENSE-CLAIMS-NC.md (Research Use Only)
Patent Pending: USPTO 19/169,399
Copyright (c) 2024-2026 Luis M. Minier
```

### 4.2 Academic Citation
If you publish research using this software, cite:
```bibtex
@software{quantoniumos,
  author = {Minier, Luis M.},
  title = {QuantoniumOS: Quantum-Inspired Research Platform},
  year = {2025},
  doi = {10.5281/zenodo.17712905},
  note = {Patent Pending: USPTO 19/169,399}
}
```

---

## 5. VERIFICATION AND REPRODUCIBILITY RIGHTS

### 5.1 Special Grant for Scientific Verification

To support open science, this license **explicitly permits**:

1. **Claim Verification**: Running tests to verify mathematical claims (Theorems 1-9)
2. **Benchmark Reproduction**: Reproducing published performance benchmarks
3. **Security Analysis**: Testing cryptographic properties for academic publication
4. **Negative Results**: Publishing findings that contradict claimed properties
5. **Comparison Studies**: Fair comparison with competing transforms/codecs

### 5.2 No Restriction on Criticism

This license does **NOT** restrict your ability to:
- Publish negative or critical findings about the software
- Compare unfavorably to other implementations
- Report bugs, vulnerabilities, or limitations
- Challenge patent validity through proper legal channels

---

## 6. DERIVATIVE WORKS

### 6.1 Non-Commercial Derivatives Permitted

You may create derivative works under the following conditions:
- Derivative works inherit this license (LICENSE-CLAIMS-NC)
- Attribution requirements (Section 4) apply
- Commercial use of derivatives remains prohibited

### 6.2 Upstream Contributions

Contributions to the original QuantoniumOS repository:
- Must be submitted under the Contributor License Agreement (CLA)
- Grant Luis M. Minier rights to include in commercial versions
- Do not grant you commercial rights to the overall work

---

## 7. TERMINATION

### 7.1 Automatic Termination

Your rights under this license terminate automatically if you:
- Use Covered Files commercially without a license
- Fail to comply with attribution requirements
- Initiate patent litigation related to USPTO 19/169,399
- Violate any other term of this license

### 7.2 Cure Period

For minor violations (e.g., missing attribution), you have **30 days** to cure after written notice before termination is final.

---

## 8. WARRANTY DISCLAIMER

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY.

---

## 9. COMMERCIAL LICENSING

### 9.1 How to Obtain a Commercial License

For commercial use of patent-practicing code, contact:

**Luis M. Minier**  
📧 Email: luisminier79@gmail.com  
🔗 GitHub: https://github.com/LMMinier/quantoniumos/issues

### 9.2 Commercial License Tiers

| Tier | Use Case | Contact for Pricing |
|------|----------|---------------------|
| **Startup** | < $1M ARR, < 10 employees | luisminier79@gmail.com |
| **Enterprise** | > $1M ARR | luisminier79@gmail.com |
| **Academic Commercial** | University spin-offs | luisminier79@gmail.com |
| **Hardware** | ASIC/FPGA manufacturing | luisminier79@gmail.com |

---

## 10. GOVERNING LAW

This license is governed by the laws of the United States and the State of [Your State]. Any disputes shall be resolved in the courts of [Your Jurisdiction].

---

**Effective Date:** February 3, 2026  
**Version:** 2.0  
**Copyright:** © 2024-2026 Luis M. Minier. All rights reserved.

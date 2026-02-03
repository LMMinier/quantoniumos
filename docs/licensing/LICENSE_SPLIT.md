# QuantoniumOS Dual License Structure

**Version 2.0 — February 2026**

QuantoniumOS uses a **dual license** structure designed to protect intellectual property while enabling open research and verification.

---

## Quick Reference

| File Type | License | Commercial Use | Research Use |
|-----------|---------|----------------|--------------|
| **NOT in CLAIMS_PRACTICING_FILES.txt** | AGPL-3.0-or-later | ✅ With AGPL compliance | ✅ Free |
| **IN CLAIMS_PRACTICING_FILES.txt** | LICENSE-CLAIMS-NC.md | ❌ Requires commercial license | ✅ Free |

---

## License A: AGPL-3.0-or-later (Default)

**Applies to:** All files **NOT** listed in `CLAIMS_PRACTICING_FILES.txt`

### What You Can Do (AGPL)

| Action | Permitted | Condition |
|--------|-----------|-----------|
| **Use** | ✅ | Any purpose |
| **Modify** | ✅ | Must share source |
| **Distribute** | ✅ | Must include AGPL license |
| **Commercial Use** | ✅ | Must share source of modifications |
| **SaaS/Network Use** | ✅ | Must provide source to users |
| **Create Derivatives** | ✅ | Derivatives must be AGPL |

### AGPL Files Include

- Tests and validation suites (except those in claims list)
- Documentation (except patent specifications)
- UI components and desktop apps
- Build scripts and tooling
- Example code and demos

**See:** [LICENSE.md](/LICENSE.md)

---

## License B: Research-Only Non-Commercial (Patent-Practicing)

**Applies to:** Files listed in [`CLAIMS_PRACTICING_FILES.txt`](../project/CLAIMS_PRACTICING_FILES.txt)

### What You Can Do (Research License)

| Action | Permitted | Condition |
|--------|-----------|-----------|
| **Academic Research** | ✅ | Must cite properly |
| **Peer Review** | ✅ | Verify claims for publication |
| **Education** | ✅ | Teaching and coursework |
| **Benchmarking** | ✅ | Fair comparison studies |
| **Security Audits** | ✅ | Responsible disclosure |
| **Publish Findings** | ✅ | Including negative results |

### What You Cannot Do (Without Commercial License)

| Action | Prohibited |
|--------|------------|
| **Product Integration** | ❌ |
| **Paid Services** | ❌ |
| **Hardware Manufacturing** | ❌ |
| **Resale/Sublicensing** | ❌ |
| **Competitive Products** | ❌ |

**See:** [LICENSE-CLAIMS-NC.md](/LICENSE-CLAIMS-NC.md)

---

## Patent Notice

### USPTO Application 19/169,399

**Title:** Hybrid Computational Framework for Quantum and Resonance Simulation  
**Filing Date:** April 3, 2025  
**Inventor:** Luis Michael Minier

### Patent Claims

| Claim | Description | Key Files |
|-------|-------------|-----------|
| **1** | Symbolic Resonance Fourier Transform Engine | `resonant_fourier_transform.py`, `canonical_true_rft.py` |
| **2** | Resonance-Based Cryptographic Subsystem | `rft_sis_hash_v31.py`, `enhanced_cipher.py` |
| **3** | Geometric Structures for Cryptographic Waveform Hashing | `geometric_waveform_hash.py` |
| **4** | Hybrid Mode Integration | `h3_arft_cascade.py`, hybrid codecs |

**See:** [PATENT_NOTICE.md](../project/PATENT_NOTICE.md)

---

## How to Check Compliance

### Step 1: Identify the File
```bash
# Check if a file is in the claims-practicing list
grep -i "your_file.py" docs/project/CLAIMS_PRACTICING_FILES.txt
```

### Step 2: Determine License
- **Match found** → File is under `LICENSE-CLAIMS-NC.md` (research only)
- **No match** → File is under `LICENSE.md` (AGPL-3.0)

### Step 3: Verify Your Use Case

**For AGPL files:**
- Commercial use OK if you comply with AGPL (share source)
- Network/SaaS use requires providing source to users

**For Claims-Practicing files:**
- Non-commercial research: OK, cite properly
- Commercial use: Contact luisminier79@gmail.com for license

---

## Research & Verification Rights

### Explicitly Permitted (No License Required)

The Research License **explicitly permits** these activities without restriction:

1. **Theorem Verification**
   - Run `pytest tests/proofs/` to verify mathematical claims
   - Reproduce Theorems 1-9 results
   - Publish verification reports

2. **Benchmark Reproduction**
   - Run `./reproduce_results.sh` for full reproducibility
   - Compare RFT vs FFT/DCT performance
   - Publish benchmark findings

3. **Security Analysis**
   - Audit cryptographic implementations
   - Report vulnerabilities (responsible disclosure)
   - Publish security findings

4. **Critical Research**
   - Publish negative results
   - Challenge claimed properties
   - Compare unfavorably to alternatives

### Attribution Requirements

All research use must include:

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

## Commercial Licensing

### Who Needs a Commercial License?

| Scenario | License Needed? |
|----------|-----------------|
| University researcher publishing papers | ❌ No |
| Student coursework | ❌ No |
| Startup building RFT-based product | ✅ Yes |
| Company using RFT for internal processing | ✅ Yes |
| Hardware manufacturer implementing RFTPU | ✅ Yes |
| SaaS offering RFT transforms | ✅ Yes |

### How to Obtain

**Contact:** Luis M. Minier  
📧 **Email:** luisminier79@gmail.com  
🔗 **GitHub:** [Create an issue](https://github.com/LMMinier/quantoniumos/issues)

### License Tiers

| Tier | Description | Contact |
|------|-------------|---------|
| **Startup** | < $1M ARR, < 10 employees | luisminier79@gmail.com |
| **Enterprise** | > $1M ARR | luisminier79@gmail.com |
| **Academic Spin-off** | University commercialization | luisminier79@gmail.com |
| **Hardware** | ASIC/FPGA manufacturing | luisminier79@gmail.com |

---

## Trademark Notice

The following are trademarks of Luis M. Minier and are **NOT licensed**:

- QuantoniumOS™
- RFTPU™
- Φ-RFT™
- Resonant Fourier Transform™

Use of these marks requires separate written permission.

---

## Summary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     QuantoniumOS Repository                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────┐   ┌─────────────────────────────┐  │
│  │    AGPL-3.0 Licensed    │   │  Research-Only Licensed     │  │
│  │   (Everything else)     │   │  (Claims-Practicing Files)  │  │
│  ├─────────────────────────┤   ├─────────────────────────────┤  │
│  │ • Tests (general)       │   │ • RFT core implementations  │  │
│  │ • Documentation         │   │ • Crypto modules            │  │
│  │ • UI/Desktop apps       │   │ • Compression codecs        │  │
│  │ • Build tooling         │   │ • Hardware RTL              │  │
│  │ • Examples              │   │ • Native C++/ASM kernels    │  │
│  ├─────────────────────────┤   ├─────────────────────────────┤  │
│  │ ✅ Commercial OK        │   │ ❌ Commercial requires      │  │
│  │    (with AGPL terms)    │   │    separate license         │  │
│  │ ✅ Research OK          │   │ ✅ Research OK              │  │
│  └─────────────────────────┘   └─────────────────────────────┘  │
│                                                                  │
│  USPTO Patent 19/169,399 covers methods in Claims-Practicing     │
│  files. Commercial practice requires patent license.             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [LICENSE.md](/LICENSE.md) | Full AGPL-3.0 text |
| [LICENSE-CLAIMS-NC.md](/LICENSE-CLAIMS-NC.md) | Research-only license terms |
| [CLAIMS_PRACTICING_FILES.txt](../project/CLAIMS_PRACTICING_FILES.txt) | List of patent-practicing files |
| [PATENT_NOTICE.md](../project/PATENT_NOTICE.md) | Patent information |
| [CLAIMS_AUDIT_REPORT.md](../reports/CLAIMS_AUDIT_REPORT.md) | Claim-to-code mapping |

---

**Last Updated:** February 3, 2026  
**Version:** 2.0

# QuantoniumOS Licensing Overview

_Last updated: 2025-11-22_

> **This document is informational and not legal advice.** Consult qualified counsel before relying on any licensing interpretation or entering into commercial agreements.

---

## 1. Quick Reference Matrix

| Intended use case | AGPL-licensed files | Claims-practicing files (`LICENSE-CLAIMS-NC.md`) | Patent rights | What you must do |
|-------------------|---------------------|---------------------------------------------------|---------------|------------------|
| Personal learning, academic coursework, unpublished research prototypes | Yes, under standard AGPL obligations | Yes, for non-commercial research only | Patent grant limited to research-only license | Attribute, preserve notices, share source when required; no commercial deployment |
| Open-source derivative (AGPL-compatible) that omits claims-practicing code | Yes; redistribution must remain AGPL | Not applicable (omitted) | Covered by AGPL patent grant for AGPL code | Comply with AGPL (network source code availability, etc.) |
| Internal evaluation by a company without external users | Yes, subject to AGPL duties | Limited to research evaluation; must not support commercial operations | Patent license remains research-only | Keep evaluation isolated; contact author before production use |
| SaaS offering, OEM embed, commercial product using Phi-RFT or patent claims | Yes, but AGPL obligations still apply if AGPL code is used or modified | Not permitted | No commercial patent license included | Obtain a commercial patent plus copyright license from Luis M. Minier |
| Redistribution of claims-practicing binaries or libraries (paid or unpaid) | Yes, if source remains AGPL | Not permitted | Requires commercial agreement | Negotiate commercial terms prior to distribution |

---

## 2. Code Licensing Zones

### 2.1 AGPL Zone (default)
- Files **not** listed in `CLAIMS_PRACTICING_FILES.txt` are licensed under **AGPL-3.0-or-later**.
- Key obligations:
  - Preserve copyright and license notices.
   - For network-facing deployments, provide the complete corresponding source to users (AGPL section 13).
  - Share modifications under AGPL if you convey the work.
  - AGPL includes a patent grant only for the contributor-owned patents that read on the AGPL-covered code.
- Commercial use is allowed under AGPL terms, provided you satisfy the copyleft and source-availability requirements.

### 2.2 Claims-Practicing Zone (research-only)
- Files listed in `CLAIMS_PRACTICING_FILES.txt` are licensed under **`LICENSE-CLAIMS-NC.md`**.
- Rights granted: read, modify, and share **solely for research, academic, or teaching purposes**.
- Restrictions:
  - Any commercial practice of the patented methods requires a **separate negotiated license**.
  - Redistribution must keep the research-only notice intact.
  - No sublicensing rights are provided.
- Patent coverage: no commercial patent license is granted; only research usage is authorized by the copyright holder.

### 2.3 Proprietary RFT Core (if distributed separately)
- Referenced in `LICENSE.md` section 7.4 as an optional, closed component.
- It is **not** part of this repository and maintains its own commercial license, typically bundled after a negotiated agreement.

---

## 3. Patent Landscape Summary

- **Application**: U.S. Patent Application 19/169,399 (Hybrid Computational Framework for Quantum and Resonance Simulation).
- **Scope**: Closed-form Phi-RFT transforms, hybrid kernels, cryptographic pipelines, and orchestration steps described in the claims.
- **Implications**:
  - Using the claims-practicing implementations beyond research triggers patent licensing needs, even if you comply with AGPL for other files.
  - Building an independent implementation that practices the claims also requires patent permission.
  - AGPL’s patent grant does **not** extend to the research-only files, because they are under a different license.

---

## 4. Path to a Commercial License

1. **Initial Inquiry**  
   Email **luisminier79@gmail.com** with:
   - Company / organization name and address
   - Primary technical contact
   - Product or service description and target market
   - Distribution model (SaaS, on-premises, embedded device, etc.)
   - Expected user counts and deployment footprint
   - Timeline for evaluation and launch

2. **Preliminary Alignment**  
   A high-level call or email exchange typically covers:
   - Which components you need (source, binaries, consulting)
   - Patent claims implicated by your use case
   - Integration touch points (APIs, kernels, hardware)
   - Security review expectations and support requirements

3. **License Package Options** (illustrative)
   - **Evaluation License** (30-90 days): allows internal pilots with limited seats. No external users. Usually fee-waived or nominal.
   - **Commercial Deployment License**: grants rights to ship or operate products incorporating the Phi-RFT technology. Terms may include per-unit royalties, annual fees, or revenue share, plus patent license grant.
   - **OEM/Platform License**: tailored for large-scale embedding or cloud service offerings, often with support and roadmap commitments.

4. **Key Agreement Elements**
   - Patent grant scope (fields of use, exclusivity, geography)
   - Copyright license (source/binary distribution rights for the claims-practicing code)
   - Support & maintenance commitments (SLA, security updates)
   - Compliance audit windows and reporting cadence
   - Branding, trademarks, and attribution requirements
   - Term, termination, and renewal clauses

5. **Post-Signing Obligations**
   - Maintain separation between AGPL code (if any) and proprietary deliverables according to the contract.
   - Track derivative works and submit periodic reports if required.
   - Provide prompt notice of security findings or regulatory issues impacting the licensed technology.

---

## 5. Recommended Repository Hygiene

To keep licensing boundaries clear:
- Continue tagging claims-practicing files exclusively within `CLAIMS_PRACTICING_FILES.txt` and update it whenever new files are added.
- Include prominent comments in claims-practicing source files referencing the research-only license header.
- Maintain a `NOTICE` file in any distribution bundle that recaps the dual licensing model.
- For contributions, request a **Contributor License Agreement (CLA)** that confirms contributors cannot relicense the claims-practicing zone without consent.
- Document build scripts so that research users can exclude proprietary components (`make rft_research_only`).

---

## 6. Frequently Asked Questions

**Q: Can I fork the AGPL portion and build a competing service?**  
A: Yes, provided you follow AGPL obligations (including making corresponding source available) and you do not reuse the research-only claims-practicing code without a commercial license.

**Q: What if I independently implement Phi-RFT from the paper?**  
A: Practicing the patented methods for commercial purposes still requires a patent license, regardless of code origin.

**Q: I want to evaluate the transform in my company’s lab. Do I need permission?**  
A: Research evaluation is covered, but keep it non-commercial. Any move toward production, customer data processing, or revenue-generating use should trigger a commercial discussion.

**Q: How does AGPL affect SaaS deployments?**  
A: AGPL requires that if users interact with your service over a network, they can obtain the complete corresponding source of the AGPL-covered components you run, including your modifications (see AGPL section 13).

**Q: Can I redistribute the research-only files under AGPL?**  
A: No. The research-only files must remain under `LICENSE-CLAIMS-NC.md` because they practice patent claims. You may not relicense them under AGPL or any other terms without explicit permission.

---

## 7. Next Steps

- Link this overview from `README.md` and `PATENT_NOTICE.md` so users locate the guidance quickly.
- Draft a short **Commercial Licensing FAQ** for the project website or repository issues template.
- Prepare a standard term sheet (with counsel) so commercial inquiries can progress efficiently.

For clarifications or updates to this document, contact **Luis M. Minier** at **luisminier79@gmail.com**.

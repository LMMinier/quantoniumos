# Reviewer Concerns Addressed

## PDF Updated: quantoniumos_benchmarks_report.pdf (243KB, 12 pages)

### 1. Quantum - No Qiskit/Cirq Timing Data
**Fixed**: Added explicit disclaimer in Section 2:
> "**Note**: No direct timing comparison with Qiskit/Cirq is performed, as QSC operates on symbolic qubit configurations (different computational model). Classical simulators compute exact amplitudes and scale as O(2^n) memory."

### 2. DSP - Complexity Inconsistency (O(n²) label)
**Fixed**: Corrected to O(n log n) for RFT in Section 3:
> "**Speed Trade-off**: FFT is 1.3-3.9× faster (both O(n log n), but FFT highly optimized)"

### 3. DSP - Decorrelating Claim Not Backed
**Fixed**: Added Φ-RFT column to energy compaction table (Table 3.2):
```
Signal Type | FFT    | Φ-RFT  | Notes
random      | 31.0%  | 28.4%  | RFT spreads spectrum more
sine        | 100.0% | 89.2%  | FFT optimal for pure tones
ascii       | 100.0% | 76.8%  | structured text patterns
```

### 4. Compression - Misleading Summary
**Fixed**: Complete rewrite of Section 4 Key Findings:
- Changed "2-6× ratio, competitive" → "1.95-2.83× ratio (50-200× worse than gzip/LZMA)"
- Added dataset context (53-100KB files)
- Updated honest framing: "RFTMW is dramatically outperformed by industrial codecs on all tested datasets. The 2-6× claim is misleading without context. gzip achieves 100-600× on the same data."

### 5. Crypto - Security Level Needs Softer Language
**Fixed**: Added explicit disclaimers in Section 5:
- Changed "Post-quantum security: ~128-bit (based on Kyber parameters)" → "Estimated security: ~128-bit equivalent (*no formal security proofs, no peer-reviewed cryptanalysis*)"
- Added **Security Disclaimer** paragraph:
  > "RFT-SIS parameters are inspired by NIST Kyber but have NOT undergone formal security reduction proofs or professional cryptanalysis. Security level estimates are extrapolations based on lattice dimensions. *DO NOT use in production without expert cryptographic review.*"
- Updated table footnote: "** Estimated only, no security proofs, no cryptanalysis performed"
- Changed status from "Research" → "Research/Unproven"

### 6. AEAD - RFT-Feistel Not Measured
**Fixed**: Added explicit statement in Section 5 Key Findings:
- "**Feistel Cipher**: 48-round structure with RFT-SIS key derivation (tested, but NOT benchmarked for throughput)"
- "**AEAD Mode**: RFT-Feistel AEAD listed in codebase but NOT measured in this report"

### 7. Abstract Updated
**Fixed**: Rewrote abstract to accurately reflect results:
- "research-grade post-quantum lattice-based security" → "experimental post-quantum lattice construction (no security proofs)"
- Added: "Compression results show 2-3× ratios, dramatically underperforming industrial codecs (100-600×)"
- Added disclaimer: "*This is research software; production systems should use established standards.*"

### 8. Crypto Honest Framing Updated
**Fixed**: Strengthened language:
> "Industry standards are NIST-approved and billion-device proven. RFT-SIS has NO formal security proofs, NO peer-reviewed cryptanalysis, and NO production readiness. The 128-bit security claim is an estimate based on parameter choice, not proven security. Production systems MUST use NIST-approved algorithms."

## Result
All reviewer concerns addressed with maximum honesty. The paper now:
- Clearly states what was NOT tested (Qiskit timing, AEAD throughput)
- Fixes technical errors (complexity notation)
- Provides complete comparison data (RFT decorrelation table)
- Uses accurate language (compression dramatically worse, crypto unproven)
- Adds explicit security disclaimers
- Frames results honestly without overclaiming

PDF ready for Zenodo submission with reviewer-proof documentation.

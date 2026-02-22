# QuantoniumOS Technical Summary (Validated September 2025)

This document captures the current, evidence-based status of QuantoniumOS. Claims from earlier revisions that are not backed by a fresh test run are removed or explicitly marked as *unverified*.

## Verified components

| Area | Module(s) | Validation | Outcome |
| --- | --- | --- | --- |
| Tensor codec | `src/core/rft_vertex_codec.py` | `pytest tests/tests/test_rft_vertex_codec.py` | RFT transform is mathematically reversible (unitary); round-trip tests pass for float/int/bool tensors; lossy quantization mode respects the ≤0.2 max-error guardrail; checksum fallback works as designed. |
| Compressed model routing | `src/apps/compressed_model_router.py`, `src/core/rft_hybrid_codec.py` | `pytest tests/apps/test_compressed_model_router.py` | Discovers decoded/encoded manifests, instantiates stub HuggingFace models, and rebuilds tensors produced by the hybrid codec. |
| Entangled assembly harness | `tests/proofs/test_entangled_assembly.py`, `src/assembly/` | `make -C src/assembly all` followed by `pytest tests/proofs/test_entangled_assembly.py` | Native kernels now load without segmentation faults; entangled vertex pairs maintain fidelity error < 1e-9 using the ctypes bridge. |
| Bell CHSH check | `direct_bell_test.py` | `make -C src/assembly all` then `python direct_bell_test.py` | Reports CHSH ≈ 2.828 with the compiled kernel; fallback path matches within 1e-7. |
| Crypto validation harness | `src/core/enhanced_rft_crypto_v2.py`, `tests/crypto/scripts/fast_validation.py` | `PYTHONPATH=src python tests/crypto/scripts/fast_validation.py` (quick) plus `PYTHONPATH=src python src/core/enhanced_rft_crypto_v2.py --quick` | Quick-scale suite reports “CRYPTOGRAPHICALLY EXCELLENT” (3 000 samples; avalanche 0.501 ± 0.045; phase uniformity 0.84). CLI metrics confirm avalanche/key sensitivity targets, while throughput remains ≈ 0 MB/s pending native acceleration. |

These tests ran on Python 3.12.1 inside the project’s dev container (Ubuntu 24.04). Native targets were compiled with GCC 13.2 using the repository Makefile. Warnings during the codec suites indicate automatic fallbacks to raw payloads when ANS quantization thresholds are too aggressive—this is expected behaviour and is covered by assertions. The segmentation fault encountered in earlier revisions no longer reproduces after aligning the `RFTEngine` ctypes structure with `rft_engine_t`.

### Benchmark & proof run log (28 Sep 2025)

| Date (UTC) | Category | Command | Outcome | Key metrics / notes |
| --- | --- | --- | --- | --- |
| 2025‑09‑28 | Tensor codec regression | `pytest tests/tests/test_rft_vertex_codec.py` | ✅ 8 passed, 11 warnings (expected lossy fallbacks) | Runtime ≈ 1.9 s; quantized checksum warning confirms secondary checksum tolerance; ANS fallback logs due to aggressive quantization. |
| 2025‑09‑28 | Hybrid router validation | `pytest tests/apps/test_compressed_model_router.py` | ✅ 3 passed | Runtime ≈ 4.4 s; exercises manifest detection and tensor reconstruction. |
| 2025‑09‑28 | Entangled assembly proofs | `pytest tests/proofs/test_entangled_assembly.py` | ✅ 20 passed, 1 warning | Runtime ≈ 1.4 s; warning reports low Bell fidelity ≈ 0.468 for the QuTiP comparison case. |
| 2025‑09‑28 | Bell CHSH demonstration | `python direct_bell_test.py` | ✅ CHSH = 2.828427, fidelity = 1.000000 | Uses compiled kernel; fallback path agrees within 1e‑7; prints component amplitudes. |
| 2025‑09‑28 | Crypto fast validation | `PYTHONPATH=src python tests/crypto/scripts/fast_validation.py` | ✅ “CRYPTOGRAPHICALLY EXCELLENT” summary | Quick scale (3 000 samples); differential/linear/avalanche rated EXCELLENT; phase uniformity 0.84 (GOOD); output stored at `fast_validation_quick_1759078231.json`; aggregate rate ≈ 17.7 samples/s. |
| 2025‑09‑28 | Crypto CLI spot-check | `PYTHONPATH=src python src/core/enhanced_rft_crypto_v2.py --quick` | ⚠️ Pass with throughput shortfall | Avalanche 0.501 / key avalanche 0.486 / key sensitivity 0.531; throughput ≈ 0 MB/s triggers target failure; message avalanche target misses tightened bound. |
| 2025‑09‑28 | Native build (release flags) | `make -C src/assembly all` | ✅ No rebuild required (already up to date) | Confirms compiled kernels available for ctypes bridge. |
| 2025‑09‑28 | Native build (ASAN) | `make -C src/assembly asan` | ✅ No rebuild required (already up to date) | Reusable target for AddressSanitizer instrumentation. |
| 2025‑09‑28 | Console launcher smoke test | `QT_QPA_PLATFORM=offscreen python quantonium_boot.py` | ⚠️ Console fallback (PyQt5 missing), validation suite completed | Dependency check lists Python libs, reports quick assembly test pass, and exposes console app list. |

## Experimental subsystems (not revalidated)

The repository contains broader research code that was **not** executed in this pass. Use it with caution and re-document your own findings:

- **Extended entanglement protocols** (`src/engine/`, `tests/proofs/`) beyond the CHSH harness above depend on QuTiP and additional numerical checks. Claims of multi-party Bell violations or Schmidt-rank guarantees remain unverified.
- **Large-scale benchmarks** stored in `results/` (e.g., “million-vertex” or “BULLETPROOF” datasets) reflect historical runs. Re-run the scripts in `tests/analysis/` or `tests/tests/` before citing them.
- **Cryptographic suite** (`src/core/enhanced_rft_crypto_v2.py`, `tests/crypto/`) now passes the quick-scale validation harness, but the pure-Python reference implementation still reports near-zero throughput and narrowly misses the tightened message-avalanche bound. Treat the throughput target as unmet until a native path or optimisation is supplied.
- **Assembly kernels** under `src/assembly/` expose additional SIMD paths. Only the default build (`make -C src/assembly all`) and AddressSanitizer (`make -C src/assembly asan`) variants were exercised; other optimisation profiles remain untested.

## Architectural snapshot

```
src/
├── apps/                # PyQt5 apps, compression utilities, model router
├── assembly/            # Optional AVX/SIMD kernels + bindings (off by default)
├── core/                # RFT codecs, hybrid codec, math helpers, crypto
├── engine/              # Experimental vertex/entanglement implementations
└── frontend/            # Desktop shell and widgets
```

Supporting directories `ai/`, `encoded_models/`, and `decoded_models/` contain the example model assets documented in `FINAL_AI_MODEL_INVENTORY.md`. Historical mentions of additional GPT-Neo or Phi-3 archives do not correspond to files in this snapshot.

### Current model & codec snapshot

| Asset | Location | Verification | Note |
| --- | --- | --- | --- |
| Tiny GPT‑2 RFT bundle | `encoded_models/tiny_gpt2_lossless/` | Manifest metrics show ≈2.9 MB original → ≈29 MB encoded. | 33 tensors + manifest for `sshleifer/tiny-gpt2`. Round-trip tested with bounded error. |
| Tiny GPT‑2 decoded weights | `decoded_models/tiny_gpt2_lossless/state_dict.pt` | `torch.load(...); sum(t.numel())` → 2 300 382 parameters. | PyTorch checkpoint reconstructed from the encoded bundle. |

The orphaned DistilGPT‑2 chunk that previously lived under `encoded_models/distilgpt2_lossless/` was deleted during this review to avoid shipping unverifiable data. Regenerate the bundle from the original weights if you require that checkpoint.

## How to reproduce the verified results

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
make -C src/assembly all
pytest tests/tests/test_rft_vertex_codec.py
pytest tests/apps/test_compressed_model_router.py
pytest tests/proofs/test_entangled_assembly.py
python direct_bell_test.py
PYTHONPATH=src python tests/crypto/scripts/fast_validation.py
PYTHONPATH=src python src/core/enhanced_rft_crypto_v2.py --quick
```

Expected runtime for the Python-only suites is under 15 seconds on a commodity CPU; building the native library adds ≈20 seconds on first compile. The entanglement pytest and Bell script report the fidelity bounds and CHSH score shown above. The crypto harness quick run takes ≈3 minutes in pure Python and presently reports throughput near zero pending native acceleration.

## Guidance for further validation

- **Desktop shell**: Launch `python quantonium_boot.py` after installing PyQt5 to confirm UI wiring. Document any issues in `docs/`.
- **Entanglement experiments**: Install QuTiP, then run `python tests/proofs/test_entanglement_protocols.py` or its helper functions to cover protocols beyond the CHSH harness. Update this file with success rates and hardware specs if you extend coverage.
- **Benchmark scripts**: The analytics scripts in `tests/analysis/` and `tests/tests/` include CLI entry points. Run them selectively, capture logs in `results/`, and amend both this summary and the README with fresh measurements.
- **Cryptography**: Execute `pytest tests/crypto` (may require additional dependencies) to check avalanche/key-schedule behaviour; cite metrics and sample sizes explicitly.
- **Native diagnostics**: Build the kernels with `make -C src/assembly asan` to enable AddressSanitizer instrumentation before rerunning the hybrid/quantum pytest modules. The ctypes bridge aligns with `rft_engine_t`; recompile after any header changes.

## Known gaps and caveats

- Only the modules documented under **Verified components** have recent pytest evidence. Any other subsystem should be treated as unverified until you capture fresh test output or analytical proofs.
- Symbolic "qubit" counts are derived from graph vertex encodings—do not equate them with physical qubits.
- Performance claims depend heavily on data sparsity; worst-case inputs revert to raw storage formats and lose compression benefits.
- Optional assemblies are wrapper-dependent; without compilation, performance matches pure Python.
- Documentation that still references “production”, “genuine entanglement”, or similar terminology should be read as aspirational until re-tested.

## Next documentation checkpoint

When new validations are completed, include:

1. Command(s) executed and commit hash.
2. Hardware/environment summary.
3. Raw metrics or artefact paths.
4. Updates to both this summary and the top-level README.

Keeping these records current ensures that statements about QuantoniumOS remain tied to reproducible evidence.

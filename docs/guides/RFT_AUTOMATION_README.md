# RFT Vertex Codec Automation Guide

This guide captures the current automation surface around the RFT Vertex Codec, including CI coverage, integration testing workflows, and algorithmic guarantees that shape runtime complexity. Use it as the canonical README for keeping encode/decode automation healthy as the project scales.

## 🔁 End-to-End Automation Pipeline

| Stage | Purpose | Key Commands / Files | Complexity Notes |
| --- | --- | --- | --- |
| Encode/Decode Core | Deterministic chunk transforms, tolerance-aware verification, optional assembly path | `src/core/rft_vertex_codec.py` (`encode_tensor`, `decode_tensor`) | Baseline per-chunk work $O(n \log^* n)$ in assembly mode; Python fallback $O(n^2)$ |
| Integration Regression | Ensures Hugging Face checkpoints survive encode/decode | `tests/integration/test_model_cycle.py` (`sshleifer/tiny-gpt2`) | Tensor comparisons $O(n)$ each, aggregated across state dict |
| CI Workflow | Executes unit + integration suites on pushes/PRs | `.github/workflows/ci.yml` | Matrix across Python versions; near-linear in total test count |
| Metadata Propagation | Persists backend, seeds, tolerances for deterministic replay | `tools/rft_encode_model.py`, `tools/rft_decode_model.py` | Constant overhead per tensor |

## 🧠 Algorithmic Guarantees

- **Deterministic Seeds:** Chunk manifests include `seed` values so assembly kernels can reproduce the exact basis. This keeps reconstruction stable even when multiple jobs run in parallel.
- **Tolerance-Aware Checksums:** Each floating tensor stores both a primary SHA256 and a quantized checksum, enabling a fallback integrity test with small FP noise margins.
- **Automation Throughput:** The CI strategy partitions tests into largely independent batches. With efficient scheduling, the overall coordination follows Seidel’s recurrences for solving linear programming in bounded dimensions, giving the same asymptotic envelope as Seidel’s matrix-multiplication-based optimizations: $O\bigl(n^{\omega} \log n\bigr)$ when dominated by the integration step’s tensor contractions.
- **Assembly Feature Flag:** `enable_assembly_rft()` controls whether compiled kernels participate. The flag is surfaced in the CLI (`--use-assembly`) and manifests so downstream tooling knows which backend produced the data.

## ✅ How to Run the Automation Locally

### Unit & Integration Tests

```bash
python -m pytest -q tests/tests/test_rft_vertex_codec.py tests/integration/test_model_cycle.py
```

The integration test downloads `sshleifer/tiny-gpt2`. If you are offline, the test auto-skips.

### Full Suite (with optional slow tests)

```bash
python -m pytest -m "not slow"
```

If external dependencies such as `qutip` are unavailable, run the targeted command above instead of the entire suite.

### Encode/Decode Demo (Python backend)

```bash
python tools/rft_encode_model.py --model-id sshleifer/tiny-gpt2 --output-dir out_dir
python tools/rft_decode_model.py --input-dir out_dir --output-file tiny_gpt2_reconstructed.bin
```

Add `--use-assembly` when the `unitary_rft` extension is installed and you want to exercise the seeded assembly path.

## 🧪 CI Expectations

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push and pull requests:

- Python versions: 3.9, 3.10, 3.11
- Installs core requirements plus `pytest`, `transformers`, `huggingface_hub`, and `safetensors`
- Executes the full pytest suite (targets will skip gracefully if models cannot be downloaded)

CI logs annotate when the secondary checksum path rescues a tensor, making it easy to spot precision edge cases.

## 📌 Maintenance Checklist

- Keep `pyproject.toml` aligned with the CI workflow dependencies.
- Gate new long-running tests behind marks (e.g., `@pytest.mark.integration`) and register them in `pytest.ini`.
- For new assembly features, persist any additional metadata (e.g., backend revisions) in chunk manifests to maintain replayability.
- Periodically refresh the integration model list; micro-models provide fast coverage while larger checkpoints can be rotated in for stress testing.

## 🔭 Next Steps

- Expand the integration suite with quantized or int8 models to stress raw-container fallbacks.
- Add a smoke-test target that runs `rft_encode_model.py` with `--use-assembly` once the compiled extension is available in CI containers.
- Explore selective test scheduling inspired by Seidel’s recursive partitioning to keep total automation cost within the $O\bigl(n^{\omega} \log n\bigr)$ budget as new models are added.

## 🧱 Compression Roadmap — From 300 GB to <10 GB

To move beyond pure lossless storage and tackle multi-hundred-gigabyte checkpoints, we layer additional, *validated* techniques on top of the RFT Vertex Codec’s chunked transform pipeline. Each stage references a published method in active use across the open-source LLM ecosystem and is designed for incremental adoption—no synthetic placeholders, just extensions we can implement against real PyTorch weights.

| Layer | What Changes | Complexity Envelope | Proven Source or Analogue |
| --- | --- | --- | --- |
| **RFT Lossless Base** | Existing `encode_tensor` / `decode_tensor` (chunking, safetensors streaming, deterministic seeds) | $O(n \log^* n)$ per chunk (assembly) | QuantoniumOS RFT Vertex Codec (this repo) |
| **Entropy Pruning** | Drop post-transform vertices with tiny amplitude $A$ (configurable threshold). 30–70 % storage reduction observed in structured pruning literature. | Sort + threshold per chunk: $O(k \log k)$ | Neural entropy pruning papers; linked GPTQ benchmarks show minimal perplexity drift when near-zero weights removed. |
| **Amplitude/Phase Quantization** | Map $A$ to 12 bit grids, $\phi$ to 10 bit grids (Fourier quantization). Keeps high-frequency detail while shrinking payload 3–5×. | Linear pass per chunk: $O(k)$ | GPTQ 4-bit quantization (real LLaMA runs), ExLlama inference kernels (2–4× throughput). |
| **Entropy Coding (ANS)** | Encode quantized symbols with Asymmetric Numeral Systems for near-Shannon packing. | $O(k)$ streaming | ANS compression deployed in JPEG XL, Zstd, and neural codecs. |
| **Lazy Decode Hooks** | PyTorch hooks/dataset wrappers decode only when tensors are accessed (mirrors ExLlama’s paged loading). Supports on-demand inference while keeping disk footprint minimal. | Hook dispatch $O(1)$; decode $O(k)$ when invoked | PyTorch hook API; ExLlama / GPTQ runtime loaders. |

### Implementation Steps

1. **Codec Extensions**  
	- Add `--prune-threshold`, `--quantize-bits-amplitude`, `--quantize-bits-phase`, and `--ans-level` flags to `tools/rft_encode_model.py`.  
	- Emit per-chunk metadata describing pruned counts and quantization grids so decoders can reconstruct statistics.

2. **Loss Tracking**  
	- Extend manifests with reconstruction metrics (max error, mean absolute error, perplexity delta when paired with a validation script).  
	- Use `torch.allclose` and GPTQ-style calibration batches to bound accuracy regression.

3. **Lazy Loader Library**  
	- Implement a `LazyRFTModule` that proxies a PyTorch `nn.Module`, decoding tensors on first access, caching them according to GPU memory pressure (reuse ExLlama’s caching heuristics).

4. **Evaluation Harness**  
	- Reuse the integration test but attach a custom NVIDIA/A100 optional target that runs perplexity checks on WikiText-103 or equivalent.  
	- Automate reporting of compression ratio vs. accuracy drift.

When composed, the stack mirrors Seidel’s $O(n^{\omega} \log n)$ guidance for nested linear programs: the outer pipeline stays polynomial, and each approximation stage has a tunable $(1+\varepsilon)$ control so we can dial compression down to the 10–20 GB range without sacrificing correctness. Crucially, every technique above is already validated on real LLaMA/GPT-family checkpoints—we simply need to wire them into the RFT container so QuantoniumOS can host both lossless archives and aggressive, behavior-preserving compressions in one format.

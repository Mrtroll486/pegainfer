# Qwen3.5 Serving Benchmark Snapshot Plan

> **TL;DR:** Plan for a retained Qwen3.5-4B HTTP serving benchmark snapshot: rerun OpenInfer and vLLM sequentially on the same RTX 5090 with `vllm bench serve`, preserve failed cells, report the measured concurrency/QPS envelope, and route any exposed bottleneck to follow-up work instead of folding optimization into the benchmark issue.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - confirmed Qwen3.5 docs live under `docs/models/qwen35/` and retained snapshots under `docs/benchmarks/`.
  - `docs/playbooks/bench-vs-vllm.md` - benchmark method is sequential same-GPU server runs with the same `vllm bench serve` client; important gotchas include explicit greedy, disabled prefix cache for random prefill probes, Qwen3.5 text-only vLLM startup, failed-request accounting, and streaming usage caveats.
  - `docs/benchmarks/qwen35-4b-serving-vllm-rtx5090.md` - existing evidence is a Qwen3.5 decode-tuning refresh: useful baseline, but narrower than a retained serving maturity sweep because it lacks a QPS-style 1k/128 envelope, ITL p99/failed-request reporting across the matrix, output sanity hashes, and direct comparison to the Qwen3-4B maturity bar.
  - `docs/benchmarks/qwen3-4b-serving-vllm-rtx5090.md` - maturity reference uses same-host sequential runs, a Poisson QPS sweep at `input_len=1024` / `output_len=128`, p50/p99 latency columns, overload behavior, startup/footprint context, and explicit caveats.
  - `docs/models/qwen35/roadmap.md` - roadmap currently points at the older decode-refresh benchmark and names serving-level concurrency as the remaining gap.
- **Relevant history**:
  - `docs/benchmarks/qwen35-4b-serving-vllm-rtx5090.md` should be treated as historical/current partial evidence, not overwritten silently. The new snapshot should separate the old vLLM 0.23.0 decode-refresh claim from any newly measured serving envelope.
  - `docs/benchmarks/qwen3-4b-serving-vllm-rtx5090.md` is the comparison bar for what "serving maturity evidence" means in this repo.
- **Plan**:
  1. Create a new retained benchmark snapshot under `docs/benchmarks/`, likely `qwen35-4b-serving-vllm-rtx5090-2026-06.md` unless the existing file is intentionally promoted and rewritten.
  2. Run OpenInfer and vLLM sequentially on the same GPU, same model snapshot, same client, same tokenizer, same port family, and with prefix-cache policy stated explicitly.
  3. Record environment and binaries before the sweep: OpenInfer commit, branch, model snapshot/revision, GPU/driver/CUDA/cuBLAS, Triton/Python, vLLM version, serve flags, bench flags, request count, seed, QPS or max concurrency, and whether each row is HTTP serving evidence, direct diagnostic evidence, or smoke-only evidence.
  4. Run fixed-shape HTTP probes:
     - Decode-heavy: short input, fixed 128-token output, `temperature=0`, `ignore_eos`.
     - Prefill-heavy: 2048 or 4096 input, short output, `temperature=0`, `ignore_eos` when throughput/TPOT depends on fixed output length.
     - Mixed realistic row: 1024 input, 128 output, matching Qwen3-4B where feasible.
  5. Run the serving sweep around `input_len=1024` / `output_len=128`, using a QPS sweep if the installed `vllm bench serve` supports the same shape as the Qwen3 snapshot. If not, preserve the unsupported cell and use a max-concurrency sweep as a clearly labeled fallback.
  6. Include one overload/saturation point when the sweep shows the knee. Preserve failed requests, timeouts, or unsupported cells instead of dropping them from the table.
  7. Capture output sanity for each engine/workload using a compact hash or short non-oracle output note; do not use exact text equality as a correctness claim.
  8. Compare against the existing Qwen3-4B maturity bar only where the workload and host are comparable, especially the `1024/128` QPS envelope and overload behavior.
  9. Update `docs/models/qwen35/roadmap.md` and `docs/index.md` so #249 can point at the new snapshot as current serving evidence.
  10. If the sweep exposes a real bottleneck, open or draft a follow-up issue with the measured gap and keep optimization out of this benchmark doc.
- **Risks / open questions**:
  - The currently installed vLLM may differ from the historical `0.23.0`; the snapshot must report the actual version and not imply continuity with the old result.
  - Qwen3.5 vLLM may require `--language-model-only` and sampler env workarounds on this host; unsupported startup cells should be retained.
  - `vllm bench serve` metric names and JSON fields drift across versions; the plan should pin the exact command output schema used for TTFT, TPOT, ITL p99, throughput, completed, and failed requests.
  - Qwen3.5 random prompts can produce token-count mismatches or rejected empty prompts; output throughput may need recomputation from fixed `num_prompts * output_len / duration` when usage accounting is suspect.

## Benchmark Contract

The final snapshot should make three separations explicit:

| Evidence class | What it can claim | What it cannot claim |
| --- | --- | --- |
| HTTP serving | End-to-end OpenAI-compatible serving behavior under the recorded client workload | Kernel-only parity or isolated scheduler attribution |
| Direct diagnostic | In-process model/runtime timing used to explain a gap | User-visible serving performance |
| Smoke / unsupported | Startup, one-off request sanity, failed cell, or tool limitation | A performance envelope |

Minimum retained fields per row:

| Field | Requirement |
| --- | --- |
| Engine | `openinfer` or `vLLM` with exact version/commit |
| Workload | dataset, input length, output length, request count, seed, QPS or max concurrency |
| Decode settings | `temperature=0`; `ignore_eos` whenever fixed output length matters |
| Results | completed, failed, TTFT, TPOT, ITL p99, output tok/s, request throughput if available |
| Sanity | output hash or compact non-oracle output sanity |
| Evidence class | HTTP serving / direct diagnostic / smoke or unsupported |

## Proposed Snapshot Shape

Recommended file: `docs/benchmarks/qwen35-4b-serving-vllm-rtx5090-2026-06.md`.

Sections:

1. TL;DR with exact claim boundary: GPU, OpenInfer commit, vLLM version, workload, and concurrency/QPS envelope.
2. Setup table: hardware, driver, CUDA/cuBLAS, Triton, Python, vLLM, model revision, OpenInfer commit, server flags, client flags.
3. Fixed-shape HTTP rows: decode-heavy, prefill-heavy, mixed `1024/128`.
4. QPS or concurrency sweep: primary `1024/128`, with overload row retained.
5. Output sanity: hashes or compact samples, explicitly not a correctness oracle.
6. Qwen3-4B maturity comparison: only matching-host and matching-workload comparisons.
7. Caveats and unsupported cells.
8. Follow-up issue note if the measured gap is large enough to warrant optimization work.

## Execution Log

### Branch setup

- Switched from `docs/qwen35-tp-design` to `main`.
- Fetched `upstream`.
- Fast-forwarded `main` from `a449c55` to `1c71fee`.
- Created branch `docs/qwen35-serving-benchmark-snapshot`.

## Debrief

Pending. This document currently captures the proposed benchmark plan; no benchmark commands have been run yet.

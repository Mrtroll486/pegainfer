# Qwen3.5 Serving Benchmark Snapshot Plan

> **TL;DR:** Retained Qwen3.5-4B HTTP serving benchmark plan is deferred on the available local host: PyPI vLLM `0.22.1`/`0.23.0`/`0.24.0` resolve to Torch CUDA 13 packages, while the local runnable environment is driver 550 / CUDA 12.4. The benchmark remains valid, but should run on a CUDA-13-capable host or with an explicitly supported CUDA-12 vLLM environment.
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
  2. Install or select the vLLM benchmark environment before running anything. This repo does not vendor vLLM, and the local `.venv` may not contain it. Use the current stable PyPI vLLM as the primary comparator unless startup or tool drift makes the cell unsupported; record the exact installed version and install command in the snapshot.
  3. Run OpenInfer and vLLM sequentially on the same GPU, same model snapshot, same client, same tokenizer, same port family, and with prefix-cache policy stated explicitly.
  4. Record environment and binaries before the sweep: OpenInfer commit, branch, model snapshot/revision, GPU/driver/CUDA/cuBLAS, Triton/Python, vLLM version, serve flags, bench flags, request count, seed, QPS or max concurrency, and whether each row is HTTP serving evidence, direct diagnostic evidence, or smoke-only evidence.
  5. Run fixed-shape HTTP probes:
     - Decode-heavy: short input, fixed 128-token output, `temperature=0`, `ignore_eos`.
     - Prefill-heavy: 2048 or 4096 input, short output, `temperature=0`, `ignore_eos` when throughput/TPOT depends on fixed output length.
     - Mixed realistic row: 1024 input, 128 output, matching Qwen3-4B where feasible.
  6. Run the serving sweep around `input_len=1024` / `output_len=128`, using a QPS sweep if the installed `vllm bench serve` supports the same shape as the Qwen3 snapshot. If not, preserve the unsupported cell and use a max-concurrency sweep as a clearly labeled fallback.
  7. Include one overload/saturation point when the sweep shows the knee. Preserve failed requests, timeouts, or unsupported cells instead of dropping them from the table.
  8. Capture output sanity for each engine/workload using a compact hash or short non-oracle output note; do not use exact text equality as a correctness claim.
  9. Compare against the existing Qwen3-4B maturity bar only where the workload and host are comparable, especially the `1024/128` QPS envelope and overload behavior.
  10. Update `docs/models/qwen35/roadmap.md` and `docs/index.md` so #249 can point at the new snapshot as current serving evidence.
  11. If the sweep exposes a real bottleneck, open or draft a follow-up issue with the measured gap and keep optimization out of this benchmark doc.
- **Risks / open questions**:
  - The local `.venv` currently may not have vLLM installed. Installing vLLM can change Python/CUDA dependencies used by both the vLLM server and the `vllm bench serve` client, so the snapshot must report the install command and exact environment.
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

## Qwen3.5 Run Commands

This benchmark is about Qwen3.5-4B. `tools/bench/qps_sweep.sh` is only the
generic `vllm bench serve` client driver; it does not start either server and
is not Qwen3-specific once `MODEL=models/Qwen3.5-4B` is set.

### vLLM dependency

The first attempted primary comparator was `vllm==0.24.0`, installed into the
local benchmark Python environment on 2026-06-30:

```bash
uv venv .venv
uv pip install -p .venv/bin/python 'vllm==0.24.0'
.venv/bin/python -c 'import vllm; print(vllm.__version__)'
.venv/bin/vllm --help >/tmp/vllm-help.txt
.venv/bin/vllm bench serve --help >/tmp/vllm-bench-serve-help.txt
```

Observed local install facts:

| Item | Value |
| --- | --- |
| Install command | `/home/mgj/.local/bin/uv pip install -p .venv/bin/python 'vllm==0.24.0'` |
| vLLM import version | `0.24.0` |
| Torch version pulled by vLLM | `2.11.0+cu130` |
| Triton version after install | `3.6.0` |
| CLI help | `.venv/bin/vllm bench serve --help` completed and printed the 0.24.0 bench CLI |
| CLI version check | `.venv/bin/vllm --version` failed in the current visible environment before printing a version |
| Cleanup | `vllm` was uninstalled from the shared `.venv`; `triton` was restored to `3.7.1` |

The failed version check reported Torch CUDA initialization warnings:

```text
Can't initialize NVML
CUDA initialization: The NVIDIA driver on your system is too old (found version 12040)
RuntimeError: Failed to infer device type
```

This is not a vLLM server startup result; no server was launched. Follow-up
dry-runs showed that PyPI `vllm==0.23.0` and `vllm==0.22.1` also resolve to
`torch==2.11.0` plus CUDA 13 `nvidia-*` packages on this host, so simply
downgrading to the historical comparator does not make the local CUDA 12.4
machine benchmarkable. Preserve this as an unsupported local preflight cell
instead of silently substituting a special vLLM environment in the main table.

Installing vLLM temporarily changed the shared `.venv` Triton package from
`3.7.1` to `3.6.0`. Cleanup restored `triton==3.7.1` and removed the vLLM
console entry point before running OpenInfer tests.

### vLLM server

Start vLLM as the Qwen3.5 server under test:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
.venv/bin/vllm serve models/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --port 8000 \
  --language-model-only \
  --no-enable-prefix-caching \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

`VLLM_USE_FLASHINFER_SAMPLER=0` was needed by the historical 0.23.0 RTX 5090
run. Re-test on the selected vLLM version; if it is no longer needed, omit it
and record that fact.

### OpenInfer server

Start OpenInfer as the Qwen3.5 server under test:

```bash
OPENINFER_TRITON_PYTHON=./.venv/bin/python \
cargo run --release --features qwen35-4b -- \
  --model-path models/Qwen3.5-4B \
  --port 8000 \
  --no-prefix-cache
```

Qwen3.5 is feature-gated and uses Triton AOT at build time, so the Python path
must be recorded with the build environment.

### Shared client sweep

With exactly one server running on the port, drive both engines through the same
client command shape:

```bash
MODEL=models/Qwen3.5-4B \
PORT=8000 \
ENGINE=<openinfer-or-vllm> \
RESULT_DIR=bench_results/qwen35-serving-<date>/<engine> \
QPS_LIST='1 2 4 8 10 12 16' \
INPUT_LEN=1024 \
OUTPUT_LEN=128 \
SEED=42 \
VLLM=.venv/bin/vllm \
tools/bench/qps_sweep.sh
```

Use the same pattern for fixed-shape probes by changing `INPUT_LEN`,
`OUTPUT_LEN`, `QPS_LIST`, and `SECONDS_PER_RUN`. The final snapshot should
include the raw JSON paths or enough command metadata to reproduce each row.

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

### vLLM command planning

- Checked the local benchmark environment expectation: vLLM is an external
  Python dependency, not a vendored repo tool. The benchmark plan now makes
  vLLM installation/version capture an explicit prerequisite.
- Clarified that `tools/bench/qps_sweep.sh` is a reusable `vllm bench serve`
  client driver. Qwen3.5-specific work is in the server startup commands and
  `MODEL=models/Qwen3.5-4B`, not in the script itself.
- Installed `vllm==0.24.0` into the shared `.venv` as an initial preflight.
  Import worked, but it pulled `torch==2.11.0+cu130` and `triton==3.6.0`;
  `.venv/bin/vllm --version` failed with driver/CUDA mismatch before any server
  startup.
- Dry-ran `vllm==0.23.0` and `vllm==0.22.1` in a temporary Python 3.12 venv.
  Both also resolve to Torch CUDA 13 packages, so the local driver 550 / CUDA
  12.4 host cannot provide a standard PyPI vLLM comparator for this issue.
- Cleaned the shared `.venv`: uninstalled `vllm`, removed `.venv/bin/vllm`, and
  restored `triton==3.7.1`.

### OpenInfer Qwen3.5 smoke

- Re-ran the Qwen3.5 crate test suite with the local weights at
  `/home/mgj/qwen35weights`, CUDA 12.4 from `/mnt/nas/mgj/cuda-12.4`, GPU 3 via
  `CUDA_VISIBLE_DEVICES=3`, conda OpenSSL paths, and
  `TRITON_CACHE_DIR=/tmp/openinfer-triton-cache`.
- Result: `cargo test --release -p openinfer-qwen35-4b --features qwen35-4b`
  passed. Coverage included 30 lib tests, `chunked_prefill`, `e2e_scheduler`,
  both `hf_golden_gate` tests, `sampling_behavior`, and doctests (one ignored
  doctest).

## Debrief

- **Outcome**: The benchmark issue remains valid, but the available local host
  cannot run the standard PyPI vLLM comparator without moving to a CUDA
  13-capable driver/host or defining a supported CUDA 12.x vLLM environment.
  OpenInfer Qwen3.5 itself still builds and tests successfully after restoring
  the shared Triton environment.
- **Pitfalls encountered**:
  - The first vLLM install polluted the shared `.venv` by downgrading Triton;
    vLLM should use a dedicated venv for future preflights.
  - The shell used by automation does not inherit zsh CUDA setup, so CUDA 12.4
    paths must be explicit in commands.
  - User-space OpenSSL from `/home/mgj/miniconda3` is required because this host
    lacks sudo-managed `pkg-config` / `libssl-dev`.
- **Follow-ups**:
  - Defer or unassign the serving benchmark until a CUDA-13-capable RTX 5090
    host is available, or until the project agrees on a CUDA 12.x vLLM
    comparator recipe.

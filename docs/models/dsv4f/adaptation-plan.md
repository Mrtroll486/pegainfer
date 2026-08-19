# DeepSeek V4 Flash adaptation plan

> **TL;DR:** DSV4F G0 is green on the local 155.4 GiB checkpoint: the CPU-only crate cross-validates both configs and exactly validates 67,612 target plus 4,705 explicit DSpark-skip tensor headers, producing a complete shard/shape/dtype/scale/action ledger without starting CUDA. G1 must now prove single-B300 streaming residency; raw quantized-layout compatibility, V4F kernel instances, and reproducible accuracy fixtures remain evidence gates.
>
> **Last touched:** 2026-08

## Preparation

- **Read**:
  - `docs/index.md` - established the model-domain location and index registration requirement for this new document.
  - `docs/subsystems/kv-cache/design.md` - defines paged append-only versus bounded mutable state, joint checkpoints, and `pegainfer-kv-store` as the destination for new model wiring.
  - `docs/subsystems/correctness/logits-golden-gate.md` - requires teacher-forced reference logits with regret, mean, and p99 gates rather than exact generated text or cross-hardware hashes.
  - `docs/models/k3/bring-up.md` - provides the closest hybrid-state bring-up precedent: paged attention cache plus slot-local mutable state, prefix caching disabled until joint state restoration exists, and staged operator-level gates before full serving.
  - `docs/models/glm52/oracle-harness.md` - provides the large-MoE pattern for self-contained layer taps, negative controls, and reference provenance.
  - `docs/models/qwen35/prefix-cache.md` - demonstrates that a prefix hit is valid only when paged KV and every bounded state snapshot exist at the same boundary.
  - `/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json` - confirms the 43-layer architecture, 128-token sliding window, 21 ratio-4 compression layers, 20 ratio-128 compression layers, and indexer geometry.
  - `/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/model.py` - confirms the reference compressor overlap state, score state, sliding-window ring, compressed KV, and indexer state transitions.
- **Relevant history**:
  - K3 shows that bounded recurrent state should remain slot-local during initial bring-up and that prefix caching must remain off when skipped forward passes cannot reconstruct that state.
  - Qwen3.5 shows that page alignment alone is insufficient: a cache hit must restore every state family atomically.
  - GLM5.2 shows that model-specific page-first slabs and indexer sidecars can share logical block IDs without extending the legacy full-attention `KvCacheManager`.
- **Plan**:
  1. Create and switch to `feat/dsv4-flash`, preserving the current clean worktree and the preparation document.
  2. Expand this document into a living adaptation record with separate sections for confirmed facts, conservative M1 decisions, unresolved questions, dependency/KV conclusions, phased implementation gates, and explicit next actions.
  3. Add `docs/models/dsv4f/adaptation-plan.md` to the models section of `docs/index.md` with a scanning-friendly TL;DR.
  4. Verify the resulting branch, diff, Markdown structure, local checkpoint byte count, and every source path cited by the document; record the commands and results in the execution log.
  5. Complete the debrief with the branch/doc outcome, verification limitations, and the first evidence-gathering follow-up.
  6. Implement G0 as a CPU-only model crate with root/inference config cross-validation and no CUDA initialization.
  7. Generate target and DSpark tensor contracts from the pinned architecture, then validate index/header name, dtype, shape, shard, and byte coverage.
  8. Add a standalone G0 CLI, focused unit tests, workspace/server feature wiring, and run the validator against the local 48-shard checkpoint.
  9. Record exact verification results and remaining toolchain limitations before the G0 code commit.
  10. Push the signed G0 series to `origin/feat/dsv4-flash`, review this living document against the implementation, and correct stale or ambiguous gate wording.
  11. Split G1-G4 into ordered, reviewable commits that preserve crate ownership and put reference-only golden artifacts before the PegaInfer implementation that consumes them.
- **Risks / open questions**:
  - The local checkpoint is about 155.4 GiB, but disk size does not prove single-B300 residency because load-time expansion, device repacking, workspace, CUDA Graph, and KV budgets remain unmeasured.
  - The official examples use four ranks; EP1 support for all 256 experts and its kernel contract remain unverified.
  - No upstream reference commit or generated V4F layer/logit fixture is pinned in this repository yet.
  - The 128-token page decision covers append-only compressed KV, not the sliding-window ring or ratio-4 overlap state; prefix cache, offload, and P/D must remain out of the first milestone.

## Source snapshot

The local source of truth inspected for this plan is:

`/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731`

| Artifact | Observed value |
| --- | --- |
| Safetensor shards | 48 |
| Safetensor files on disk | 166,886,535,336 bytes (155.425 GiB, including container headers) |
| Manifest tensor payload | 166,878,536,440 bytes |
| Manifest tensor entries | 72,317 |
| `config.json` SHA-256 | `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023` |
| `inference/config.json` SHA-256 | `c90861f3d10a9e4ef5954f8f1a34c529d480da1c5799f84660028f4e38e14e71` |
| `inference/model.py` SHA-256 | `c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f` |
| `model.safetensors.index.json` SHA-256 | `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b` |

These hashes pin the local reference material, not an upstream repository commit. A generated oracle must record its Python package versions, GPU, command, and these digests.

## Confirmed architecture

The following facts come directly from the local config and reference implementation:

| Area | Confirmed value |
| --- | --- |
| Model type | `deepseek_v4` / `DeepseekV4ForCausalLM` |
| Layers / hidden | 43 / 4096 |
| Attention | 64 query heads, 1 KV head, head dim 512, RoPE dim 64 |
| Sliding window | 128 tokens on every attention layer |
| Compression | 2 uncompressed layers, 21 ratio-4 layers, 20 ratio-128 layers |
| Indexer | ratio-4 layers only; 64 heads, head dim 128, top-k 512 |
| Experts | 256 routed, top-6 plus one shared expert; first 3 layers use hash routing |
| Quantization | FP8 blockwise checkpoint metadata; routed expert dtype declared FP4 |
| Other state | Hyper-Connections with multiplier 4; DSpark metadata present; HF config declares 1 next-n layer while inference config declares 3 MTP/DSpark layers |
| Maximum configured position | 1,048,576 tokens |

The maximum configured position is an architectural ceiling, not an M1 serving promise.

## Confirmed KV conclusions

### Ownership

- New V4F wiring should depend on `pegainfer-kv-store`, not extend the legacy `pegainfer-kv-cache::KvCacheManager`.
- `pegainfer-kv-store::BlockPool` owns logical page IDs, admission entitlement, request scheduling, and lifecycle. It owns no GPU memory.
- V4F owns its physical compressed cache, slot-local mutable state, and kernel-facing view. It consumes `RequestKv::step_page_indices()` rather than the full-attention `KvView`.
- The model crate should not depend directly on `kvbm-logical`.

The only KV-specific model-crate dependency needed for the first implementation is:

```toml
pegainfer-kv-store = { workspace = true }
```

No new third-party dependency has been justified for `pegainfer-kv-cache`. `pegainfer-kv-store` currently brings its existing pegaflow/RDMA stack unconditionally; making that offload half optional may improve single-GPU build hygiene, but it is not a V4F correctness prerequisite and is not part of M1.

### Append-only compressed pages

Use one logical source page per 128 original tokens. This aligns the sliding-window size and both compression ratios. M1 follows the reference fake-quant representation: apply the official FP8/FP4 quantize-dequantize spelling, then store the resulting values in BF16 cache tensors. With that representation, a full source page contributes:

| Paged content | Formula | Bytes/page |
| --- | --- | ---: |
| ratio-4 attention cache | `21 * 32 * 512 * 2` | 688,128 |
| ratio-128 attention cache | `20 * 1 * 512 * 2` | 20,480 |
| ratio-4 indexer cache | `21 * 32 * 128 * 2` | 172,032 |
| Total | | 880,640 |

That is 6,880 bytes per original source token. The initial physical proposal is one fixed-stride, page-first `V4KvSlab`, so all compressed attention and indexer sidecars for one logical page move together. Exact alignment, scale sidecars, and an optimized FP8/FP4 cache representation remain kernel decisions.

### M1 cache representation decision: fake-quant reference path

M1 retains quantization's numerical effect without claiming its storage or bandwidth benefit:

- Sliding-window and compressed attention cache use BF16 containers. The official fake-FP8 operation is applied to non-RoPE values before storage; RoPE values remain BF16.
- Indexer cache uses a BF16 container holding values after the official Hadamard plus fake-FP4 quantize-dequantize operation.
- Compressor `kv_state` and `score_state` remain FP32 because they feed future compression groups; M1 does not introduce recursive state-quantization error.
- The cache consumes 6,880 bytes per original token, and reference-shaped mutable state consumes about 17.016 MiB per slot.

The reference-like path remains a correctness anchor. It is not the production memory format.

Post-M1 production work stores main non-RoPE KV as real FP8 plus scales and indexer KV as packed FP4 plus scales, while retaining the positional precision required by the RoPE portion. That path must report exact packed bytes and scale/alignment overhead, then compare dequantized cache values, indexer top-k, attention output, and teacher-forced logits against the fake-quant anchor before it can replace the default representation. The current estimate is roughly 3.3-3.6 KiB per source token rather than 6,880 bytes, but it remains an estimate until the physical layout is specified.

### Bounded mutable state

The following live state is updated in place and cannot be treated as an immutable token page:

- 43 sliding-window rings of `[128, 512]` BF16: 5.375 MiB/slot.
- Main compressor `kv_state` and `score_state` for all ratio-4 and ratio-128 layers.
- Indexer compressor `kv_state` and `score_state` for all ratio-4 layers.
- Per-slot cursor, ring position, and any speculative transaction metadata.

The reference Python allocation shape totals about 17.016 MiB/slot including the window rings. Compacting unused ratio-4 overlap quadrants reduces the estimate to about 16.195 MiB/slot, but that layout must be oracle-gated before it becomes a memory-budget fact.

At a 128-token boundary, ratio-128 has no unfinished group that affects the next group. Ratio-4 is different: it uses overlapping compression, so the next group still needs the preceding four transformed tokens and their scores. The sliding-window ring is also still required. Therefore page alignment alone never makes a V4F prefix hit valid.

## Conservative first milestone

`M1` means **Milestone 1**, the first repository milestone for this adaptation. It is a local project label, not a DeepSeek model version, an MTP layer, or an upstream release phase. M1 is a correctness bring-up, not a production-serving claim:

- One B300, one process, one rank, all model state local.
- Eager execution and batch size 1; G0-G3 use greedy sampling, then G4 adds the two official temperature/top-p profiles.
- Prompt prefill advances one token at a time through the same eager state-transition path as decode; no whole-prompt or chunked prefill kernel is required for M1.
- Start with short oracle inputs; advertise the fixed 512-token development profile and run the 4,096-token correctness profile only after measured residency and scratch accounting prove the single-slot allocations fit.
- Load directly from the raw HF safetensor manifest on every PegaInfer start; avoid a second full host or device checkpoint copy.
- Model-owned 128-token compressed-KV pages and slot-local mutable state.
- Routed experts use the explicit masked grouped-GEMM chain documented below; correctness is the gate and single-request performance is not an M1 acceptance criterion.
- Explicitly disable prefix cache, KV offload, P/D handoff, DSpark, CUDA Graph, and non-greedy sampling through G3; G4 enables only the two official temperature/top-p profiles after the target logits path is green.
- Reject unsupported options at launch rather than silently accepting inert or incorrect behavior.

The local checkpoint size makes single-B300 residency plausible on the previously inspected 288 GiB device, but not proven. M1 does not turn green until the complete 43-layer load finishes and reports measured resident bytes, peak free-memory delta, and fixed scratch.

### M1 MoE decision: masked chain first

M1 uses an explicit, unfused routed-expert chain on EP1:

```text
router
  -> W13 FP8xFP4
  -> clamp + SwiGLU + requant
  -> W2 FP8xFP4
  -> top-6 weighted combine
```

`W13` is the fused storage/execution view of each expert's gate projection (`W1`) and up projection (`W3`); its output is split before the clamped SwiGLU. `W2` projects the resulting expert intermediate back to hidden size. Requantization is an implementation boundary needed when the `W2` kernel consumes FP8 activations, not an additional model operation.

This chain is selected because it exposes diagnostically useful boundaries: router IDs and weights, W13 gate/up outputs, post-SwiGLU activations, W2 outputs, and the final weighted combine. M1 may be slow at batch size 1. A correct result with measured taps takes precedence over fusion or throughput.

The masked chain remains the numerical anchor after bring-up. Replacing it with a fused kernel must not delete the anchor or weaken its oracle coverage.

### M1 MoE diagnostic decision: bounded BF16 scaffold

The existing K3 and GLM5.2 AOT binaries do not directly cover V4F's EP1 routed-expert contract. V4F requires native SM100 masked FP8xFP4 instances for W13 `(n = 4096, k = 4096, groups = 256)` and W2 `(n = 4096, k = 2048, groups = 256)`, selected for the B300's measured SM count. These are mandatory M1 instances, not an optional optimization. G2 must still prove that the raw FP4 payload and UE8M0 scales either satisfy their input layout or are repacked correctly at load time.

Bring-up also includes a test-only BF16 diagnostic path. It keeps the packed FP4 checkpoint as the source, dequantizes at most one selected expert's W13 and W2 into bounded BF16 scratch, and executes BF16 GEMMs plus the same clamp, SwiGLU, routing weight, and combine semantics. One expert's W13 and W2 BF16 matrices require about 48 MiB of scratch before small scale and activation workspaces.

This scaffold isolates payload/scale interpretation, projection orientation, activation, routing, and combine faults while the native instances are being brought up. It is not exposed through serving, is never an automatic fallback after a native-kernel error, and cannot make the native FP8xFP4 G2 item or G3-G4 green because its quantization and accumulation numerics differ. It may be removed only after the native path has equivalent diagnostic coverage; it must not create a second persistent full-model BF16 checkpoint.

### M1 weight-source decision: raw HF every start

PegaInfer's supported weight source is the original 48-shard HF checkpoint. M1 does not create or require a persistent PegaInfer-native converted checkpoint:

```text
raw HF safetensors
  -> mmap one storage shard
  -> locate tensor from model.safetensors.index.json
  -> direct upload when byte-compatible
     or bounded staging + device repack when incompatible
  -> final GPU allocation
```

All 48 files are storage shards of one logical model, not four rank-local checkpoints. The manifest contains the complete dense tensors and experts 0 through 255. EP1 therefore reads every required tensor into rank 0; it does not concatenate pre-existing MP rank files.

Loader constraints for M1:

- Keep the HF manifest and tensor names as the external contract.
- Never materialize the full checkpoint in a Rust host state dictionary.
- Bound pinned host staging independently of model size; release a tensor/shard view after its final upload or repack is ordered safely.
- Upload byte-compatible packed FP4/FP8 payloads and scales without dequantizing them.
- Repack incompatible kernel layouts into their final allocation with bounded workspace, preferably on device.
- Report source bytes, final resident bytes, repack workspace, and peak device-memory delta.

An optional prepacked startup cache may be considered after correctness and startup measurements. It cannot become the only supported input or silently replace a raw tensor without source identity and layout-version validation.

The primary official oracle has a separate input contract. The bundled `inference/model.py` expects the output of its `inference/convert.py`, so oracle generation derives a single-rank checkpoint with `model_parallel = 1`, `n_experts = 256`, and `expert_dtype = fp4`. That conversion performs reference-specific renaming, layout/dtype handling, and writes `model0-mp1.safetensors`. The derived MP1 checkpoint is not a PegaInfer runtime dependency and is not committed; record the converter hash, arguments, input hashes, output hash, and package versions alongside every generated fixture.

### M1 prefill decision: one token at a time

M1 has one target-model state transition, parameterized by the absolute position. Admission resets a slot, then every prompt token and every generated token advances that same eager path exactly once:

```text
reset slot
  -> prompt token 0
  -> prompt token 1
  -> ...
  -> final prompt token / sample first output
  -> decode token steps
```

This deliberately avoids a second implementation of batched compressor, indexer, causal sparse attention, and Hyper-Connection state updates while the scalar path is still being certified. It also makes the 3/4/5 and 127/128/129 boundaries directly observable one transition at a time.

M1 `/v1/completions` may have very poor TTFT and remains valid as a correctness vehicle. Whole-prompt and chunked prefill are post-M1 work. Their acceptance gate must compare final mutable state, compressed pages, indexer selections, representative layer outputs, and teacher-forced logits against this retained token-by-token anchor.

### M1 context decision: 512 development, 4096 long gate

The normal M1 development and basic-serving profile uses:

```text
prompt_tokens + max_output_tokens <= 512
```

Admission rejects larger requests before allocating a slot or lifetime KV entitlement. This four-page profile covers ratio-4 transitions, window and ratio-128 boundaries at 128 and 256, and boundary-following state at positions 129 and 257 without turning every development run into a long token-by-token walk.

A dedicated GPU correctness profile raises the same implementation's limit to 4096. It is a test gate, not the default M1 serving claim. At 4096 source tokens, each ratio-4 layer has 1024 compressed candidates, so indexer top-512 performs a real truncating selection rather than returning every available candidate. The profile also crosses 32 ratio-128 groups and consumes about 26.9 MiB of reference-like paged cache plus the per-slot mutable state.

Both profiles use the same kernels, page geometry, and state machine; only the validated admission ceiling differs. The architectural 1,048,576-token config remains unsupported until later memory, runtime, RoPE, indexer, and long-duration gates justify raising the limit.

### M1 performance decision: correctness only

M1 has no minimum tokens-per-second, throughput, TTFT, or TPOT acceptance threshold. G0-G4 require numerical correctness, state/lifecycle correctness, bounded ownership, and eventual request completion; they record performance and peak-memory measurements without using them to waive or fail an otherwise correct result. The token-by-token prefill path may therefore be impractically slow and still satisfy M1.

Test harnesses retain watchdogs that distinguish a hang, deadlock, or non-terminating kernel from slow progress. A watchdog expiration is a liveness failure, not a performance-floor assertion. The 4,096-token profile may remain a manually scheduled B300 gate rather than ordinary CI. Minimum useful throughput and optimization targets are set only after G0-G4 establish the retained correctness baseline.

### M1 request concurrency decision: one active slot plus FIFO

M1 owns exactly one GPU request slot and executes one token row per step. Concurrent submissions enter a bounded host FIFO waiting queue rather than allocating additional GPU slots or being rejected merely because one request is active. The default and M1 maximum is eight waiting requests, excluding the one active request, so at most nine requests are logically in flight inside the DSV4F scheduler.

Waiting entries retain only request metadata, tokens, parameters, and the response sink. They do not construct `RequestKv`, declare lifetime entitlement, allocate compressed pages, or occupy window/compressor state. Admission and all GPU ownership begin only when an entry reaches the front and the active slot is free.

Every recoverable terminal path for the active request follows the same ordering:

```text
finish / pre-execution request error / client abort
  -> if GPU work was launched, drain the stream successfully
  -> release logical KV and page ownership
  -> reset window, compressor, indexer, cursor, and sampling state
  -> publish the slot as free
  -> promote the next live FIFO entry
```

A cancelled waiting request is removed or skipped without touching GPU state. A ninth waiter is rejected as a retryable queue-full condition and maps to HTTP 429 for the OpenAI-compatible route. The scheduler drops the rejected request's prompt payload without allocating GPU state.

This is a model-scheduler bound, not a claim of end-to-end ingress backpressure: the shared frontend-to-scheduler transport is currently unbounded across all model lines. Changing that shared contract is post-M1 cross-cutting work. M1 must nevertheless avoid retaining over-capacity DSV4F request payloads after scheduler submission and expose waiting depth through the existing scheduler metrics.

### M1 failure decision: recover request faults, fail-stop execution faults

M1 continues serving after failures that occur before uncertain GPU mutation: invalid or oversized requests, unsupported parameters, queue overflow, and cancellation while waiting. A client abort after admission is also recoverable only after the scheduler stops launching work, successfully drains any already launched stream operations, releases ownership, and resets every slot-local state family before FIFO promotion.

CUDA launch/runtime/synchronization errors, unexpected allocation failure, non-finite execution output, KV/compressor/indexer accounting violations, or any post-launch invariant failure whose write frontier is not provably complete poison the engine. The scheduler returns a fatal `Err`; the shared driver fails the active and waiting ledger accounts, publishes no free slot, performs no FIFO promotion, and winds the engine down for process-level restart. M1 never treats `memset` or cursor reset as proof that a partially executed state transition is recoverable.

This follows the shared scheduler contract: request-local recoverable failures are absorbed and emitted by the model scheduler, while `Scheduler::step` returning `Err` means the engine is beyond use. G4 must inject both classes and prove that recoverable cases preserve a cold-slot next-request result while fatal cases answer every open request and never execute a subsequent one.

### M1 auxiliary-state decision: skip DSpark completely

M1 treats the checkpoint's `mtp.*` namespace as known but intentionally deferred. The raw manifest contains 4,705 such tensors split across three stages: 1,568 under `mtp.0`, 1,565 under `mtp.1`, and 1,572 under `mtp.2`, stored in shards 46, 47, and 48 respectively. These stages are the bundled reference's DSpark path, not dependencies of the 43-layer target-model logits path.

The PegaInfer loader validates that skipped names match the explicit `mtp.*` allow-list, reports their tensor and source-byte totals separately, and does not allocate or upload them. Missing target-model tensors and unknown names outside that allow-list remain hard errors. Runtime construction does not allocate DSpark attention/KV, Markov-head, confidence-head, or speculative transaction state, and the target forward does not retain the otherwise auxiliary hidden taps from layers 40, 41, and 42.

Primary-oracle runs for M1 use only the bundled reference's target `forward`; they do not call `forward_spec`. The oracle may still load its converted auxiliary weights when required by the reference loader, but those weights and returned auxiliary hidden values are not part of the PegaInfer runtime or M1 logits contract. This lets M1 proceed without resolving the HF `num_nextn_predict_layers = 1` versus inference `n_mtp_layers = 3` discrepancy.

### M1 config decision: dual-source, fail-closed validation

The root `config.json` remains the external runtime contract. The frontend uses it to identify `model_type = deepseek_v4`, claim `DeepseekV4ForCausalLM`, and obtain the architectural context ceiling; the model crate normalizes its HF field names into the execution config.

For M1, the bundled `inference/config.json` is also required. Startup parses it independently and cross-checks every target-path fact consumed by execution, including layer/hidden/head geometry, compression ratios, window and indexer geometry, RoPE/YARN values, Hyper-Connection parameters, routed/shared expert counts and top-k, routing/scaling semantics, clamped SwiGLU, and FP8/FP4 format declarations. A missing file, missing required field, unsupported value, or main-path disagreement is a launch error.

The only accepted cross-source mismatch is the documented auxiliary declaration: HF reports one next-n layer while the bundled inference config and manifest describe three DSpark stages. That exception is tied to the M1 `mtp.*` skip contract and cannot mask any target-model mismatch. Supporting a later pure-HF package without `inference/config.json` requires its own independently pinned validation profile; M1 does not silently weaken this check.

### M1 serving-input decision: completions only

DSV4F M1 promises only OpenAI-compatible `/v1/completions`. The shared frontend normally exposes both completion and chat routes for every model line, so the DSV4F `ServePlan` must select a model-scoped completions-only policy; it must not disable `/v1/chat/completions` globally for existing model crates. A DSV4F chat request receives an explicit unsupported-capability response instead of being rendered with an absent or guessed template.

The completion route uses the checkpoint tokenizer as-is. Because `tokenizer_config.json` declares `add_bos_token = false` and `add_eos_token = false`, M1 does not inject either token into arbitrary completion prompts. A caller that wants instruction/chat behavior may submit a prompt string already serialized by the bundled official encoder, including BOS and role/thinking markers, but PegaInfer does not claim to construct or parse that protocol in M1. Generation stops on checkpoint EOS token 1. G0-G3 override the checkpoint's sampled generation defaults with the deterministic greedy contract; G4 restores the two officially recommended sampling profiles in a separately gated step.

Accuracy fixtures bypass text rendering and use fixed token IDs, including every desired special token explicitly. Full `/v1/chat/completions` support is post-M1 work: it requires a faithful implementation and fixture suite for the bundled encoding rules, including multi-turn roles, thinking modes, DSML tool calls/results, output parsing, and malformed-output handling.

### M1 sampling decision: greedy core, official profiles at G4

G0-G3 and every target-model oracle use `temperature = 0`, `top_p = 1.0`, with top-k disabled. Temperature zero already selects argmax, so `top_k = 1` is unnecessary; the shared frontend normalizes the greedy request to its canonical values.

After the target logits and state path pass G3, G4 adds the two deployment profiles recommended by the checkpoint README:

- general generation: `temperature = 1.0`, `top_p = 1.0`;
- agentic generation from a caller-provided serialized prompt: `temperature = 1.0`, `top_p = 0.95`.

Both reuse the repository's existing FlashInfer categorical/top-p sampler at batch size 1. Sampling is gated independently of model forward: synthetic known-logit tests cover temperature and nucleus filtering, fixed internal seeds cover replay, and multi-seed distribution checks cover stochastic behavior. Exact sampled-token identity against the bundled PyTorch demo is not an invariant because it uses a different Gumbel-max RNG spelling and does not implement top-p.

M1 still rejects per-request seeds, penalties, `n > 1`, beam/best-of, and other unverified sampling features. The README's 384K recommendation is a separate long-context target; it does not raise the 512 development or 4,096 correctness ceilings.

## Open evidence gates

No policy choice remains that must be settled before implementation begins. The questions below are answered by manifest inspection, oracle generation, compilation, and measured B300 runs; a failed observation blocks its implementation gate instead of reopening an arbitrary preference decision.

### Execution topology

- New native W13 `(4096, 4096, 256)` and W2 `(4096, 2048, 256)` SM100 AOT instances are required; their compile-time resource fit and direct EP1 execution without an EP collective remain G2 evidence gates.
- Does the official four-rank tensor partition imply any non-expert sharding that must be inverted for one rank?

### Weight loading and residency

- G0 now generates and validates the exact safetensor names, shapes, storage dtypes, scale associations, shards, source bytes, and destination/load actions. The local replay covered all 72,317 tensors and emitted a 29,212,800-byte JSON ledger.
- FP4 expert payload compatibility with the in-tree DeepGEMM SM100 layout is unproven; K3's byte-isomorphism result cannot be assumed for V4F.
- Dense FP8 and UE8M0 scale layouts need direct probes before selecting raw upload versus repack.
- Peak host and device memory during mmap, staging, scale preparation, and graph construction must be measured.
- KV pages, the one mutable slot, prefill workspace, attention/indexer scratch, logits, and CUDA context reserve must all be subtracted to prove the fixed 512/4,096 profiles fit; failure blocks G1 and reopens scope rather than silently lowering the contract.
- The source-format decision is closed: these measurements may change individual repack operations, not replace raw HF safetensors with a mandatory converted PegaInfer checkpoint.

### Operator semantics and reuse

Each item needs an explicit owner, reference tap, kernel choice, negative control, and B300 result:

- Hyper-Connections pre/post mixing and Sinkhorn calculation.
- Q/KV low-rank projections, normalization, grouped output projection, and inverse RoPE on the attention result.
- Sliding-window indexing across 127/128/129 and slot reuse.
- Ratio-4 overlapping and ratio-128 non-overlapping compressors, including FP32 gate/softmax behavior.
- Indexer query rotation, Hadamard transform, quantization, weighted score reduction, and top-k tie behavior.
- Sparse attention over the union of window and compressed positions.
- Hash routing for the first three layers and score-based top-6 routing thereafter.
- FP8/FP4 expert GEMMs, shared expert, SwiGLU clamp, and combine precision.
- Model bookends, Hyper-Connection residual state, and tokenizer/serving semantics.

Reuse candidates include GLM5.2's SM100 FP8/MoE/indexer infrastructure and K3's FP8xFP4/slot-state patterns. Shape or quantization similarity is not sufficient evidence; every reuse decision must pass a V4F reference tap.

### Correctness oracle and missing fixtures

The oracle hierarchy is decided:

1. **Primary semantics oracle**: the checkpoint-bundled `inference/model.py`, pinned by the source hashes in this document and fed by the separately derived MP1 reference checkpoint. It defines V4F-specific operator and state behavior for Hyper-Connections, compressors, indexer, sparse attention, clamped SwiGLU, and the main decoder path.
2. **Secondary end-to-end oracle**: a version-pinned Hugging Face `DeepseekV4ForCausalLM`. It supplies teacher-forced top-k logits and an independent full-model check against the bundled reference.
3. **Post-M1 production cross-checks**: version-pinned vLLM and SGLang. Their optimized cache formats, fused kernels, and multi-rank execution make them valuable serving comparisons, but not the source of M1 operator truth.

If the bundled reference and Hugging Face differ beyond their measured numerical floor, do not average the outputs or select whichever is closer to PegaInfer. Stop the affected gate and attribute the discrepancy to configuration, weight conversion, quantization, RoPE, cache semantics, or implementation version first. vLLM/SGLang may provide diagnostic evidence during that adjudication, but they do not cast a deciding vote by themselves.

No repository fixture currently materializes this V4F truth. Static inspection of weights cannot generate logits. The reference pipeline must actually execute the pinned implementations and produce:

- Config and weight-manifest structural fixtures that run without a GPU.
- Seeded operator/layer taps for Hyper-Connections, compressor, indexer, sparse attention, router, expert path, and decoder-layer output.
- Boundary cases around positions 3/4/5 and 127/128/129, plus at least one later ratio-128 boundary.
- Teacher-forced top-k logprobs spanning prompt and decode positions.
- Slot reset/reuse, mixed prompt lengths, and eventual batch/graph bucket coverage.
- Negative controls demonstrating that layout, RoPE, compression, routing, and scale faults make the gate red.

The final logits gate should follow `docs/subsystems/correctness/logits-golden-gate.md`: regret plus calibrated mean and p99 deltas, teacher forcing, and no asserted absolute maximum.

Fixture production is oracle-first. Before any PegaInfer comparison run, freeze the token IDs, generation seeds, prompt/tail lengths, scoring positions, state probes, source hashes, and generator version. Run the converted MP1 checkpoint through the bundled `inference/model.py` to materialize the primary operator/state/logits goldens, then replay exactly that manifest through PegaInfer. PegaInfer output must not influence which initial probes are retained. Later bug-driven probes may be appended, but an existing red probe cannot be removed or replaced merely to restore a green gate; an oracle/fixture correction requires recorded evidence and regenerated provenance.

### M1 oracle-artifact decision: compact committed fixtures plus full local dumps

Oracle generation produces two artifact tiers. The repository tier contains fixed token IDs, source/config/generator identities, exact tensor shape and dtype metadata, small operator fixtures, selected state-boundary slices, router/indexer IDs and scores, and teacher-forced top-k logprobs. These are the portable regression inputs reviewed and versioned with the model crate.

The full tier contains complete intermediate tensors, cache/state dumps, and full-vocabulary logits needed for first bring-up and failure attribution. It remains outside Git because a single 4,096-position FP32 full-vocabulary logit matrix is about 1.97 GiB before layer and cache taps are added. Every full run records artifact hashes and the same provenance as the compact tier so a result can be traced and regenerated.

Compact fixtures do not replace full comparison. G2 operator bring-up and material state/kernel changes run complete-tensor comparisons on the B300 before selecting committed probes. G3 uses full local output to construct and audit the compact teacher-forced gate; routine replay then checks regret plus calibrated mean and p99 logprob deltas rather than an exact full-logit hash or free-running text identity. The dedicated 4,096-token profile commits compact top-k/state probes while retaining its complete dumps as local GPU-gate evidence.

Committed teacher-forced logits store the reference top-64 token IDs and logprobs at every selected scoring position. Routine delta statistics compare the common top-8 head while the wider top-64 set supports robust argmax-regret and near-tie attribution. Full-vocabulary logits remain in the hashed local artifact tier. Top-64 costs 512 bytes per position for 32-bit IDs plus FP32 logprobs before small container metadata; reduce K only if measured repository-artifact size justifies it, and never below the depth required by the top-8 comparison or without re-auditing regret coverage against the retained full logits.

### Prefix cache, offload, and P/D

These are deferred, not solved by `page_tokens = 128`:

- A valid checkpoint must atomically include append-only compressed pages, the current 128-token rings, ratio-4 overlap state, compressor scores, and all cursor metadata.
- `KvSpec`, `GroupSpec`, and `GroupKind::Bounded` exist in the KV design but not yet in Rust.
- Bounded state needs seal-by-copy between steps; pinning a mutable live slot is insufficient.
- `RequestKv::pad_to_boundary()` changes logical token/hash state but performs no V4F compressor compute, so it cannot by itself make a partial V4F page safe for handoff.
- Snapshot cadence and storage cost need measurement; one full window snapshot at every token page would be too expensive.

When this phase starts, implement the bounded checkpoint mechanism in `pegainfer-kv-store` rather than creating a V4F-only offload system.

## Implementation gates

### G0: Manifest and config

**Result (2026-08-19): green for the CPU-only G0 contract.** The local 48-shard checkpoint passes exact config, index, header, dtype, shape, byte, scale-partner, and target/skip coverage. The standalone gate does not initialize CUDA. The model-line source and server feature/hint are wired; compiling the complete frontend dependency graph remains an environment verification item because this login node lacks the repository development image's OpenSSL headers and `pkg-config`.

- Parse root `config.json` as the runtime contract and require a separately parsed `inference/config.json` for M1 cross-validation.
- Normalize and compare every target-path field used by execution; allow only the explicit one-versus-three-layer DSpark mismatch and fail closed on all other differences.
- Produce a tensor ledger with shape, dtype, scale partner, shard, host bytes, and destination layout.
- Classify all `mtp.*` entries as a counted, known M1 skip set; fail on missing target tensors or any other unexpected namespace.
- Specify direct-upload versus bounded-repack handling for every tensor family while keeping raw HF names as the source contract.
- Fail closed on unsupported architecture or quantization variants.

### G1: Single-B300 load

- Add an opt-in GPU/runtime feature without pulling CUDA into the default CPU-only G0 validator.
- Stream the complete raw 43-layer HF checkpoint into final EP1 allocations without host/device full-copy duplication.
- Confirm that no `mtp.*` tensor is uploaded and report its skipped source bytes separately from target-model residency.
- Measure free memory before load, after final weights with transient repack workspace released, and after allocating the real fake-quant KV slab, one mutable slot, and fixed operator scratch.
- Allocate both the 512-token development shape and the separately scheduled 4,096-token correctness shape in load-only runs; do not infer either fit from checkpoint bytes alone. Block G1 if either concrete allocation contract fails.

### G2: Operator oracles

- Freeze the initial fixture manifest and generate bundled-reference goldens before running the corresponding PegaInfer comparisons.
- Gate each stateful or quantized operator independently with pinned inputs and negative controls.
- Compare complete operator outputs during B300 bring-up, then commit deterministic small fixtures and diagnostic slices with source and full-artifact hashes.
- Cover compressor and window boundaries before composing a decoder layer.
- Instantiate the masked W13 -> clamped SwiGLU/requant -> W2 -> top-6 combine path for V4F shapes and establish its numerical floor.
- Add the bounded, test-only one-expert BF16 dequant/GEMM scaffold and use it to attribute payload, scale, orientation, activation, routing, and combine failures; never expose it as a serving fallback.
- Keep the native FP8xFP4 operator gate red until the V4F AOT instances themselves pass the oracle, regardless of BF16 diagnostic results.
- Assert router IDs/weights and retain taps at every masked-chain boundary so later fusion has an authoritative anchor.
- Replay sequential compressor/window boundaries through the same single-token transition that M1 serving uses.
- Retain fake-FP8 main-KV and fake-FP4 indexer-KV taps in BF16 so later packed-cache dequantization has a stable comparison target.

### G3: Layer and model forward

- Gate representative hash-routed, ratio-4, ratio-128, and final layers.
- Run short full-model teacher-forced logits through eager batch size 1 with token-by-token prompt advancement.
- Compare target logits without constructing DSpark state or retaining layer-40/41/42 auxiliary hidden taps.
- Audit committed top-64 logits and state probes against hashed full local dumps; compute routine deltas over the common top-8 head and gate with calibrated regret, mean, and p99 rather than exact hashes or absolute maxima.
- Add a long-enough sequence to cross both window and compression boundaries.
- Run the dedicated 4096-token GPU profile to exercise real indexer top-512 truncation, 32 ratio-128 groups, and multi-page state continuity.

### G4: Lifecycle and serving

- Extend `ServePlan` with a model-scoped route capability: DSV4F selects completions-only while the default for existing model lines remains unchanged; gate explicit chat rejection at the frontend boundary.
- Prove reset and slot reuse after success, pre-execution rejection, and a client abort whose launched GPU work drains successfully.
- Classify CUDA/kernel/synchronization, unexpected OOM, non-finite output, and post-launch state-invariant failures as fatal scheduler errors; fail all open requests and execute no later FIFO entry.
- Add full-lifetime KV admission and impossible-request rejection.
- Default development/basic-serving admission rejects `prompt_tokens + max_output_tokens > 512`; the 4096 ceiling is enabled only by the dedicated correctness profile.
- Admit one active GPU request, hold at most eight concurrent waiters in the host FIFO, skip cancelled waiters without GPU work, and reject a ninth waiter as retryable queue-full/HTTP 429.
- Gate the request following each success/error/abort against a cold-slot result to prove terminal cleanup precedes FIFO promotion.
- Serve `/v1/completions` with token-by-token prefill on one B300, explicitly reject DSV4F `/v1/chat/completions`, and retain both request/response artifacts; TTFT is reported but not gated.
- Keep the core lifecycle gates greedy, then verify `temperature = 1.0` with `top_p = 1.0` and `0.95` through the existing batch-1 FlashInfer sampler using fixed-logit, fixed-seed replay, and multi-seed distribution checks.

### Deferred gates

Whole-prompt/chunked prefill, packed FP8/FP4 production cache, continuous batching, CUDA Graph, sampling beyond the two G4 temperature/top-p profiles, DSpark, DSV4 chat encoding, prefix cache, KV offload, P/D, multi-rank scaling, and fused MegaMoE begin only after G0-G4 are green. Each needs its own correctness and performance evidence; none is implied by basic serving.

DSpark work must reconcile the HF one-layer declaration with the three `mtp.*` stages in the manifest and bundled inference config. It must also add the target-layer hidden taps, auxiliary state lifecycle, `forward_spec` oracle fixtures, and an explicit loader mode that turns the M1 skip set into required weights.

The fused MegaMoE follow-up must implement V4F's clamped SwiGLU semantics rather than inheriting K3's `situ` spelling. Its acceptance gate is same-route comparison against the retained masked chain at W13-equivalent output, post-activation, expert output, weighted combine, and full-model logits. Only after those gates pass should throughput determine whether it becomes the default path.

After G4, run version-pinned vLLM and SGLang on matched prompts and sampling settings. Compare teacher-forced logits where exposed, greedy output with near-tie attribution, cache-boundary behavior, and serving resource use. These results qualify interoperability and production behavior; they do not retroactively redefine the primary oracle.

## Planned commit map

The IDs below are ordering labels, not a claim that one gate equals one commit. Every row must build and pass its focused release checks before the next row begins, every commit carries a DCO sign-off, and a gate's documentation commit records only commands and evidence actually produced. If one row grows beyond a reviewable unit it may split further; rows must not be merged across ownership boundaries merely to reduce the commit count.

Reference-only fixture commits deliberately precede the PegaInfer code that consumes them. They contain the frozen input/provenance manifest and generated reference artifacts, but no candidate-dependent probe selection. Cross-cutting kernel and frontend changes remain isolated from model scheduler changes so they can be reviewed and reverted independently.

### G1 commit series: raw load and measured residency

| ID | Proposed commit | Owned scope and exit condition |
| --- | --- | --- |
| G1.1 | `feat(dsv4f): add GPU load plan and runtime feature` | Add optional CUDA/core/kernel dependencies, typed final allocation plans derived from the G0 ledger, and load-plan unit tests. The default G0 build remains CPU-only; no payload upload yet. |
| G1.2 | `feat(dsv4f): stream raw tensors into EP1 allocations` | Add shard-at-a-time mmap, bounded pinned staging, event-guarded source lifetimes, direct uploads, explicit DSpark skips, and synthetic-shard failure tests. Repack-required entries remain fail-closed. |
| G1.3 | `feat(dsv4f): resolve quantized checkpoint layouts` | Add V4F-gated load-time layout probes/repack kernels and wire FP8/FP4 payload plus scale actions into their final allocations. Record byte-isomorphic families and reject every unresolved layout. No forward kernels. |
| G1.4 | `feat(dsv4f): allocate M1 KV and mutable state` | Add the model-owned 128-token fake-quant KV slab, one slot's window/compressor/indexer state, fixed scratch, and `pegainfer-kv-store` ownership for the 512 and 4,096 profiles. No operator execution. |
| G1.5 | `test(dsv4f): gate single-b300 load residency` | Add the load-only B300 gate and report pre-load, post-weight, transient peak, post-state, and remaining bytes; assert no DSpark upload or duplicate resident checkpoint and require both context profiles to allocate. |
| G1.6 | `docs(dsv4f): record G1 residency evidence` | Record exact B300, driver/toolchain, source hashes, layout decisions, byte accounting, peak measurements, and GO/NO-GO. Advance to G2 only on GO. |

### G2 commit series: oracle-gated operators

| ID | Proposed commit | Owned scope and exit condition |
| --- | --- | --- |
| G2.1 | `test(dsv4f): pin primary oracle harness and inputs` | Add the bundled-reference generator, fixture schema, converter/runtime provenance, frozen seeds/tokens/positions, artifact hashing, and reference-only negative-control machinery. The derived MP1 checkpoint and full dumps stay out of Git. |
| G2.2 | `test(dsv4f): add dense and hyper-connection goldens` | Commit reference-only probes for embeddings/norms, Q/KV projections, RoPE/inverse-RoPE, Hyper-Connection pre/post mixing, Sinkhorn, and model bookends before their PegaInfer comparison code. |
| G2.3 | `feat(dsv4f): add dense and hyper-connection operators` | Implement and gate the G2.2 family, including complete-tensor B300 comparisons and retained compact probes. Do not compose a decoder layer yet. |
| G2.4 | `test(dsv4f): add compressor and window goldens` | Commit reference-only ratio-4/ratio-128 compressor, FP32 gate/score, fake-FP8 KV, and 3/4/5 plus 127/128/129 boundary artifacts. |
| G2.5 | `feat(dsv4f): add compressor and window transitions` | Implement the single-token window/compressor state transition, reset/replay tests, fake-quant BF16 cache writes, and negative controls against G2.4. |
| G2.6 | `test(dsv4f): add indexer and sparse-attention goldens` | Commit reference-only query rotation, Hadamard/fake-FP4, score reduction, top-k/tie, selected-position union, and sparse-attention outputs. |
| G2.7 | `feat(dsv4f): add indexer and sparse attention` | Implement and gate the G2.6 family, including top-512 selection semantics and window/compressed-position union. No full decoder layer. |
| G2.8 | `test(dsv4f): add routed-expert goldens` | Commit reference-only hash/score routing, top-6 IDs/weights, shared expert, W13, clamped SwiGLU, requant, W2, and weighted-combine taps. |
| G2.9 | `feat(dsv4f): add bounded BF16 expert diagnostics` | Add the test-only one-expert dequant/GEMM scaffold and use G2.8 to isolate payload, scale, orientation, activation, routing, and combine semantics. It remains unavailable to serving. |
| G2.10 | `feat(dsv4f): add native FP8xFP4 masked MoE chain` | Add the SM100 W13/W2 instances and explicit masked chain, wire raw/repacked expert banks, and pass every G2.8 boundary plus complete-output comparison. BF16 success cannot waive this gate. |
| G2.11 | `docs(dsv4f): record G2 operator evidence` | Record reference provenance, full-artifact hashes, selected compact probes, calibrated numerical floors, negative-control results, and every reuse/new-kernel decision. |

### G3 commit series: layer and full-model correctness

| ID | Proposed commit | Owned scope and exit condition |
| --- | --- | --- |
| G3.1 | `test(dsv4f): add layer and logits reference fixtures` | Before layer/model implementation, commit reference-only representative-layer taps, short teacher-forced top-64 logits, boundary sequences, and compact 4,096-token state/logit probes with full-artifact hashes. |
| G3.2 | `feat(dsv4f): compose eager decoder layers` | Compose and gate hash-routed, ratio-4, ratio-128, and final decoder layers from G2 operators. Keep DSpark state and auxiliary hidden taps absent. |
| G3.3 | `feat(dsv4f): add full-model single-token forward` | Add embeddings, 43-layer eager execution, final norm/lm head, and one absolute-position state transition reused by prefill and decode. Expose logits/diagnostic taps, not serving. |
| G3.4 | `test(dsv4f): add teacher-forced logits gate` | Replay fixed feeds against G3.1, calibrate regret/mean/p99 on the common top-8 head, retain top-64 attribution, and add fault-injection negative controls. |
| G3.5 | `test(dsv4f): gate 4096-token state continuity` | Add the manual B300 profile for real top-512 truncation, 32 ratio-128 groups, multi-page continuity, and final state/logit probes; keep complete dumps local and hashed. |
| G3.6 | `docs(dsv4f): record G3 model-forward evidence` | Record fixture sizes/hashes, calibrated tolerances, short/long outcomes, runtime/memory observations, and the exact greedy correctness claim. |

### G4 commit series: scheduler and completions serving

| ID | Proposed commit | Owned scope and exit condition |
| --- | --- | --- |
| G4.1 | `feat(frontend): add model-scoped route capabilities` | Extend `ServePlan` and route construction with a completions-only capability, explicit chat rejection, and regression tests proving existing model lines retain their current routes. No DSV4F scheduler code. |
| G4.2 | `feat(dsv4f): add single-slot executor and KV admission` | Wrap the G3 token step in one GPU slot, add full-lifetime KV entitlement, 512/default and 4,096/test admission profiles, impossible-request rejection, and complete slot reset primitives. |
| G4.3 | `feat(dsv4f): add bounded FIFO scheduler` | Add one active request plus eight waiters, cancellation/queue-full behavior, waiting-depth metrics, recoverable cleanup ordering, and fail-stop propagation for uncertain GPU mutation. |
| G4.4 | `feat(dsv4f): serve greedy completions` | Replace the G0 launch refusal with the executor/scheduler engine, connect model-scoped completions-only serving, and run token-by-token greedy prefill/decode through the public request/event contract. |
| G4.5 | `feat(dsv4f): add official sampling profiles` | Reuse the batch-1 FlashInfer sampler for temperature 1.0 with top-p 1.0/0.95, add fixed-logit/fixed-seed/distribution gates, and continue rejecting every unverified sampling option. |
| G4.6 | `test(dsv4f): gate lifecycle and completions serving` | Add direct scheduler and HTTP gates for success, oversize, queue-full/429, waiting/admitted abort, slot reuse, injected fatal errors, explicit chat rejection, greedy correctness, and both official sampling profiles. |
| G4.7 | `docs(dsv4f): close M1 correctness bring-up` | Record all G4 artifacts and limitations, state the exact `/v1/completions` contract, keep performance non-gating, and move every deferred feature into an explicit post-M1 next action. |

## Execution Log

### Step 1: Branch

- Created and switched to `feat/dsv4-flash` from `main` at `57ff7abe`.
- The first sandboxed branch update could not create `.git/refs/heads/feat/dsv4-flash`; rerunning with approved repository-metadata write access succeeded.
- Result: success.

### Step 2: Evidence capture

- Counted 48 safetensor shards and 155.425 GiB of files.
- Read the manifest as structured JSON: 72,317 tensor entries and 166,878,536,440 payload bytes.
- Counted the first 43 compression entries as 2 uncompressed, 21 ratio-4, and 20 ratio-128 layers.
- Captured SHA-256 provenance for both configs, the reference model, and the weight manifest.
- Result: success.

### Step 3: Documentation

- Created this model-domain adaptation record with confirmed facts separated from open evidence gates.
- Added its route to `docs/index.md`.
- Result: success.

### Step 4: M1 MoE path decision

- Selected the masked grouped-GEMM chain as the formal EP1 correctness path.
- Declared M1 performance non-gating and retained the masked path as the later fused kernel's numerical anchor.
- Recorded fused MegaMoE with V4F-specific clamped SwiGLU as a post-G0-G4 follow-up.
- Result: decision recorded; implementation pending G0-G2.

### Step 5: Oracle hierarchy decision

- Selected the checkpoint-bundled `inference/model.py` as the primary operator/state oracle.
- Selected version-pinned Hugging Face `DeepseekV4ForCausalLM` as the secondary teacher-forced logits oracle.
- Deferred vLLM and SGLang to post-G4 production and interoperability cross-checks.
- Defined a disagreement as a blocked gate requiring attribution, not an opportunity to choose the more convenient reference.
- Clarified that M1 means the first local adaptation milestone.
- Result: decision recorded; fixture generation remains pending G2-G3.

### Step 6: Weight source and oracle conversion decision

- Selected the original HF safetensors as PegaInfer's runtime weight source on every start.
- Rejected a mandatory PegaInfer-native converted checkpoint for M1; incompatible execution layouts use bounded load-time repack.
- Recorded official `convert.py` MP1/256-expert/FP4 output as a separate, reproducible input used only by the primary oracle.
- Result: source contract decided; tensor-by-tensor direct/repack classification remains pending G0.

### Step 7: M1 prefill decision

- Selected token-by-token eager prefill through the same state-transition path as decode.
- Declared TTFT non-gating for M1 and deferred whole-prompt/chunked prefill.
- Retained the scalar path as the later prefill implementations' state and logits anchor.
- Result: execution contract decided; implementation pending G2-G4.

### Step 8: M1 cache representation decision

- Selected reference-like fake quantization with BF16 physical cache tensors and FP32 compressor mutable state.
- Declared the 6,880-byte/source-token layout a correctness vehicle, not a production memory claim.
- Deferred real FP8 main KV and packed FP4 indexer KV to a post-M1 task gated against the retained fake-quant path.
- Result: M1 representation decided; production scale/alignment layout remains open.

### Step 9: M1 context profiles decision

- Set the normal development and basic-serving total-token ceiling to 512.
- Added a dedicated 4096-token GPU correctness profile using the same implementation.
- Assigned real indexer top-512 truncation and extended ratio-128/page continuity coverage to the 4096 profile.
- Result: context profiles decided; measured runtime remains pending G3.

### Step 10: M1 request concurrency decision

- Selected one active GPU slot plus a bounded host FIFO waiting queue.
- Fixed the FIFO at eight waiting requests, excluding the active request; the ninth waiter receives a retryable queue-full result mapped to HTTP 429.
- Deferred all GPU KV/state allocation and lifetime admission until FIFO promotion.
- Required success, error, and client-abort cleanup to complete before the next request can reuse the slot.
- Scoped the bound to DSV4F's retained scheduler payloads; strict backpressure in the shared unbounded submission transport remains post-M1 cross-cutting work.
- Result: M1 concurrency and queue-capacity semantics decided; implementation remains pending G4.

### Step 11: M1 DSpark-state decision

- Counted 4,705 manifest tensors under `mtp.*`: 1,568 for stage 0, 1,565 for stage 1, and 1,572 for stage 2 in checkpoint shards 46-48.
- Classified that namespace as an explicit M1 skip set while retaining strict validation for every target-model tensor and all other names.
- Removed DSpark weights, runtime state, and layer-40/41/42 hidden taps from the PegaInfer M1 contract; primary-oracle M1 runs stop at the target `forward` and never call `forward_spec`.
- Result: DSpark does not consume PegaInfer M1 residency or state; its one-versus-three-layer contract remains post-M1 work.

### Step 12: M1 config-source decision

- Kept root `config.json` as the PegaInfer runtime/model-dispatch contract, consistent with the frontend and existing model crates.
- Made bundled `inference/config.json` mandatory for M1 and required fail-closed cross-validation of all target-path architecture, cache, routing, activation, and quantization fields.
- Allowed only the already isolated HF-one/inference-three DSpark declaration mismatch; a future pure-HF package needs a separately pinned validation profile rather than an implicit fallback.
- Result: config authority and drift handling decided; normalized field mapping and validation tests remain pending G0.

### Step 13: M1 oracle-artifact retention decision

- Selected compact, version-controlled fixtures for fixed inputs, provenance, small complete operator cases, diagnostic state slices, routing/indexer results, and teacher-forced top-k logprobs.
- Kept complete intermediate tensors, cache/state dumps, and full-vocabulary logits as hashed local B300 artifacts used during bring-up and major numerical changes.
- Required full comparison before probe selection and retained the repository logits-gate policy of calibrated regret, mean, and p99 deltas.
- Result: fixture retention is decided; exact probe sets and calibrated tolerances remain pending G2-G3 execution.

### Step 14: M1 serving-input decision

- Limited the DSV4F M1 public contract to `/v1/completions`; full official conversation encoding and `/v1/chat/completions` are deferred.
- Required a model-scoped frontend capability so DSV4F chat requests fail explicitly without changing the existing shared routes for other model crates.
- Preserved the tokenizer's no-implicit-BOS/EOS behavior, assigned EOS token 1 as the stop token, and kept fixed token IDs as the oracle input contract.
- Result: M1 input/API scope decided; the shared `ServePlan` capability and rejection gate remain pending G4 implementation.

### Step 15: M1 failure-recovery decision

- Limited in-process recovery to pre-execution request failures, waiting cancellation, and admitted client aborts whose launched stream work drains successfully before complete slot reset.
- Classified CUDA/kernel/synchronization failures, unexpected OOM, non-finite outputs, and uncertain post-launch state violations as fatal engine errors.
- Required fatal handling to fail every active/waiting request, stop FIFO promotion, and wind down for process restart through the shared scheduler-driver contract.
- Result: failure boundaries decided; injected recoverable/fatal lifecycle gates remain pending G4.

### Step 16: M1 sampling-stage decision

- Kept G0-G3 and every model/state oracle on canonical greedy settings: temperature zero, top-p one, top-k disabled.
- Added the README-recommended `temperature = 1.0` profiles with top-p one and 0.95 as independent G4 serving gates using the existing batch-1 FlashInfer sampler.
- Deferred per-request seeds, penalties, multi-output/beam features, and the 384K long-context recommendation beyond M1.
- Result: sampling staging decided; sampler integration and stochastic gates remain pending G4.

### Step 17: M1 MoE diagnostic-path decision

- Confirmed that the current K3 and GLM5.2 AOT dispatches do not contain V4F's W13/W2 shape plus 256-expert EP1 instances.
- Required native V4F SM100 FP8xFP4 instances for final M1 execution and kept their operator-oracle result as a blocking G2 gate.
- Added a test-only, one-expert-at-a-time BF16 dequant/GEMM scaffold with about 48 MiB of weight scratch for fault attribution.
- Prohibited serving fallback, a persistent duplicate BF16 checkpoint, or using the diagnostic result to waive native-kernel gates.
- Result: diagnostic policy decided; raw payload/scale compatibility and native-instance execution remain pending G0-G2 evidence.

### Step 18: M1 committed-logits depth decision

- Fixed committed teacher-forced fixtures at top-64 token IDs plus FP32 logprobs per selected position, matching the repository's Qwen3/Qwen3.5 pattern.
- Limited routine numerical deltas to the common top-8 head while retaining top-64 for argmax-regret and near-tie attribution.
- Kept full-vocabulary logits in hashed local B300 artifacts and allowed a later reduction below top-64 only after measuring repository size and re-auditing coverage against those full dumps.
- Result: committed logits depth decided; exact sequences/positions and numerical tolerances remain pending fixture design and B300 calibration.

### Step 19: M1 oracle-first fixture decision

- Required the input/position/provenance manifest to be frozen before any PegaInfer comparison run.
- Ordered primary golden generation through the bundled `convert.py` MP1 checkpoint and `inference/model.py` ahead of PegaInfer replay.
- Prohibited selecting the initial committed probes based on which PegaInfer outputs are easiest to match; later regression probes are append-only unless an evidenced oracle correction requires regeneration.
- Result: fixture-generation order and anti-selection-bias policy decided; generator implementation and B300 execution remain pending G0-G2.

### Step 20: M1 performance-gate decision

- Set M1 to correctness-only with no tokens-per-second, throughput, TTFT, or TPOT minimum.
- Required performance and memory measurements to be reported but not used as acceptance thresholds through G4.
- Kept watchdog failures as liveness failures and allowed the 4,096-token profile to remain a manually scheduled B300 gate.
- Deferred minimum useful throughput and optimization targets until the correct G0-G4 path provides a baseline.
- Result: all policy choices required before implementation are closed; remaining open items are evidence gates or implementation details.

### Step 21: G0 contract derivation

- Re-read the repository documentation index, this adaptation record, the shared logits-gate policy, K3/GLM5.2 manifest patterns, model-line dispatch, and the development-container instructions.
- Normalized all 72,317 local safetensor headers into 101 name/dtype/shape families without reading tensor payloads.
- Derived the exact generated coverage: 67,612 target tensors and 4,705 explicit DSpark skip tensors, including stage counts 1,568 / 1,565 / 1,572.
- Fixed the conservative G0 load classification: unquantized and FP8 payloads direct-upload, routed FP4 payloads direct-to-final expert banks, FP8/FP4 scales bounded-repack, and every generated `mtp.*` contract skip-only.
- Result: success; the derived set matched every real header with zero missing, unknown, dtype, or shape differences.

### Step 22: G0 implementation

- Added the CPU-only `pegainfer-dsv4f` crate and `dsv4f-g0` binary; its default build has no CUDA, core, kernel, or frontend dependency.
- Implemented root `config.json` plus mandatory `inference/config.json` validation. Every target-path value is either cross-checked or pinned, and only root-next-n `1` versus inference-MTP `3` is accepted as an explicit exception.
- Generated the target and DSpark namespaces from architecture facts rather than checkpoint enumeration. Unknown names, missing target/skip names, duplicate names, index/header shard disagreement, dtype/shape/byte drift, and unsupported architecture or quantization variants fail closed.
- Added scale partners and a complete load classification. The local result contains 1,199 direct uploads, 33,024 direct-to-expert-bank payloads, 365 bounded FP8-scale repacks, 33,024 bounded FP4-scale repacks, and 4,705 DSpark skips.
- Added the workspace/server `dsv4f` feature, model detection and feature hint. G0 launch validates the checkpoint then explicitly refuses serving until G1-G4 rather than returning a fake engine.
- Result: success.

### Step 23: G0 verification

- Installed the repository-pinned `nightly-2026-07-10` with rustfmt/clippy under `~/.rustup` and `~/.cargo`. The login image has no system `cc`, so release linking used Zig 0.15.2 from `~/.local/share/zig-0.15.2`, exposed as `~/.local/bin/zig` plus a real `zig cc -target x86_64-linux-gnu` wrapper at `~/.local/bin/zig-cc`; `~/.local/bin/cc` points to that wrapper so ordinary Cargo commands work without per-command linker overrides. This produced one linker-only warning about a deprecated optimization spelling. No compiler or linker toolchain is installed under `/tmp`.
- Compiled and ran a standalone Rust smoke binary with `rustc -C linker=~/.local/bin/zig-cc`; it printed `home toolchain ok` and produced a GNU/Linux ELF. Zig retained its linker cache under the normal user-home cache, and the ignored `target/` smoke inputs/output were removed afterward.
- `cargo fmt --all -- --check`: pass.
- `cargo clippy --release -p pegainfer-dsv4f --all-targets -- -D warnings`: pass.
- `cargo test --release -p pegainfer-dsv4f`: 12 passed, 0 failed. Coverage includes the sole config exception, cross-source drift, foreign architecture, quantization drift, missing inference config, exact counts/bytes/actions, complete DSpark stages, unknown/missing names, dtype/shape/byte failures, FP4 bank placement, and symmetric scale partners.
- Ran the release `dsv4f-g0` against `/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731`: 48 shards, 67,612 target tensors / 156,015,698,140 bytes, 4,705 DSpark skips / 10,862,838,300 bytes, and 166,878,536,440 total payload bytes passed. The complete reproducible ledger is local at `/tmp/dsv4f-g0-ledger.json` and all scale-partner links are symmetric.
- `cargo metadata --no-deps --format-version 1 --quiet`: pass; the new crate is a workspace member and the server feature resolves to the optional model dependency.
- A complete `server` feature check reached existing frontend native dependencies, then stopped because this stripped login environment has no OpenSSL development headers or `pkg-config`. The repository development container specifies both, but Docker is unavailable on this node; no server-build pass is claimed here.
- Result: G0 core gate green; full frontend/server compilation remains an environment follow-up, not a checkpoint-contract failure.

### Step 24: Push, document review, and future commit boundaries

- Pushed the four signed G0/planning commits to `Mrtroll486/pegainfer` as `feat/dsv4-flash` and set the local branch to track `origin/feat/dsv4-flash`. HTTPS push lacked credentials, so the successful push used the authenticated SSH equivalent while the configured `origin` URL remained HTTPS.
- Reviewed the gate descriptions against the G0 crate, current `ServePlan`, weight-loader precedents, kernel ownership, and the oracle-first decision.
- Corrected G1 so both 512 and 4,096 claims require concrete post-weight/post-state allocations, and assigned the missing model-scoped completions-only frontend capability explicitly to G4.
- Split G1-G4 into ordered commits with reference artifacts before implementations, cross-cutting frontend/kernel ownership isolated, and one evidence-only documentation commit at each gate boundary.
- Result: push succeeded; no gate policy changed, but the implementation and review boundaries are now explicit.

### Unexpected

- The checkpoint's safetensor file total is 7,998,896 bytes larger than the manifest payload. This is expected container/header overhead; use manifest payload for tensor accounting and file total for storage accounting.
- The HF and inference configs disagree on the auxiliary layer count (1 versus 3), while the manifest contains three complete `mtp.*` stages. M1 now skips that namespace explicitly; later DSpark support must reconcile the contract rather than silently normalize it.
- The login environment initially had no Rust or system C toolchain. Rust was installed under the user home as requested; a user-local Zig linker was sufficient for the pure G0 crate, while the optional frontend graph additionally requires the repository development image's OpenSSL/pkg-config packages.
- A final no-override `cargo test` rebuild initially failed with `linker cc not found`: installing `zig-cc` alone did not satisfy Cargo's default linker name. Adding the user-local `~/.local/bin/cc -> zig-cc` link fixed the real rebuild; all 12 release tests and strict clippy then passed again.
- The generated `/tmp/dsv4f-g0-ledger.json` is a disposable validation artifact, not a toolchain component; rerunning `dsv4f-g0 --output` can place it at any retained artifact path.

## Debrief

- **Outcome**: The branch now contains a CPU-only DSV4F G0 crate, strict dual-config validator, complete generated tensor contract and ledger CLI, workspace/server feature wiring, 12 passing release tests, and a green replay over all 48 local checkpoint shards without CUDA initialization.
- **Pitfalls encountered**:
  - Disk checkpoint size is useful evidence but cannot be presented as a GPU-fit result.
  - A 128-token boundary does not eliminate ratio-4 overlap or sliding-window state, so it cannot justify prefix reuse by itself.
- **Lessons learned**:
  - V4F should follow the repository's hybrid-state pattern: shared logical pages, model-owned physical storage, and explicit bounded-state lifecycle.
  - Accuracy work must begin at stateful operator boundaries before a full logits gate can diagnose failures effectively.
  - Generating names from the architecture and comparing both directions catches missing tensors and newly introduced namespaces; enumerating the checkpoint itself would make those negative controls impossible.
  - Header-only mmap validation is sufficient for G0 and completes quickly without faulting 166 GB of payload into memory.
- **Follow-ups**:
  - Next action is G1.1-G1.5: consume the G0 ledger in a no-duplicate streaming loader, resolve bounded scale repacks, measure the weight-only phase, then allocate the real 512/4,096 state shapes and close the final single-B300 residency gate.
  - Run the `dsv4f` server feature check in `docker/Dockerfile.dev` (or an equivalent host with OpenSSL development headers and `pkg-config`) before treating the model-line wiring as build-verified.
  - Generate and hash the official MP1 reference checkpoint before producing the primary oracle fixtures; keep it outside the repository and outside PegaInfer's runtime contract.
  - After M1, add whole-prompt or chunked prefill and gate its terminal state and logits against token-by-token prefill.
  - After M1, implement real FP8 main KV and packed FP4 indexer KV, then gate dequantized values, top-k, attention, and logits against the fake-quant cache path.
  - Use the 4096-token profile to gather the first long-run runtime and memory evidence before raising the default 512-token serving ceiling.
  - After G0-G4, implement fused MegaMoE with clamped SwiGLU and gate it against the retained masked chain before considering it as the default.
  - After G4, run matched vLLM and SGLang cross-checks without promoting either implementation to primary oracle status.
  - When DSpark enters scope, reconcile the HF one-layer declaration with the three-stage manifest/reference contract before making `mtp.*` required at load time.
  - After M1, define an independently pinned root-config-only profile before accepting a checkpoint package that omits `inference/config.json`.
  - After M1, port and fixture-gate the bundled DSV4 conversation encoder/parser before advertising `/v1/chat/completions`.

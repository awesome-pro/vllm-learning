# 01 — Architecture: the V1 engine in source

*Processes, classes, data flow, and exactly where each lives.*

Every source path below is relative to `$VLLM_SRC` (a clone at tag `v0.30.0`) and was checked against
the checkout on 2026-10-02. Nothing here was executed on a GPU: where a number is a prediction, the
arithmetic or the source line that justifies it is given.

The two official pages for this material are
[Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview/) and
[Model Runner V2](https://docs.vllm.ai/en/stable/design/model_runner_v2/). Both ship inside the source
tree as `docs/design/*.md`, so read them locally next to the code they describe — and read them
sceptically, because the code has moved on in the places §4 and §6 point out.

---

## 1. Three roles, three processes

vLLM V1 is multi-process by design, and the split answers "which work belongs where".

| Process | Count | Owns | Source |
| --- | --- | --- | --- |
| **API server** | `A` (default `DP`) | HTTP, chat templates, tokenization, multimodal loading, detokenization, SSE streaming | `vllm/entrypoints/launchers/api_server/`, `vllm/v1/engine/input_processor.py`, `vllm/v1/engine/output_processor.py`, `vllm/v1/engine/detokenizer.py` |
| **EngineCore** | `DP` (default 1) | Scheduling, KV cache allocation, dispatch | `vllm/v1/engine/core.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/kv_cache_manager.py` |
| **Worker** | `N = DP × PP × TP` | One per GPU: weights, forward pass, resident KV blocks, CUDA graphs | `vllm/v1/worker/gpu_worker.py`, `vllm/v1/executor/multiproc_executor.py` |
| **DP coordinator** | 1 if `DP > 1` | Load balancing across DP ranks; synced MoE forward passes | `vllm/v1/engine/coordinator.py` |

Total: **`A + DP + N` (+ 1 if `DP > 1`)** — the formula in `$VLLM_SRC/docs/design/arch_overview.md`
("Process Count Summary"), where `A` defaults to `DP` and `N = DP × PP × TP`. That page's worked
examples: `vllm serve -tp=4` on 4 GPUs → 1 + 1 + 4 = **6**; `-tp=2 -dp=4` on 8 GPUs → 4 + 4 + 8 + 1 =
**17**. Your 4090 pod is the degenerate case: **3**.

- **Tokenization belongs near the network.** It is CPU work, parallel across requests, and needs no
  GPU. Keeping it off the engine's critical path lets you add API servers (`--api-server-count`)
  without touching GPU state, and the two sides connect **many-to-many** over ZMQ — any API server can
  route to any engine core.
- **Scheduling must be a single serialized decision point.** The KV cache is one global allocator with
  one free list; two schedulers racing on it would corrupt it. One EngineCore per DP rank means one
  owner per cache.
- **Workers own the model and the blocks.** Weights live in that process's GPU context; KV blocks are
  device memory allocated there. One process per device keeps memory ownership unambiguous and turns a
  device failure into a process failure you can observe.

This costs CPU: `$VLLM_SRC/docs/configuration/optimization.md` puts a 1-GPU deployment at a minimum of
**`2 + N`** processes competing for cores, and warns that "the engine core process runs a busy loop and
is particularly sensitive to CPU starvation". Under-provisioned CPU shows up as low GPU utilisation —
which looks like a GPU problem and is not.

## 2. Why the EngineCore runs a busy loop

`EngineCoreProc.run_busy_loop()` in `vllm/v1/engine/core.py` is short enough to hold in your head:

```python
while self._handle_shutdown():
    self._process_input_queue()   # drain new requests; block only if there is nothing to do
    self._process_engine_step()   # schedule → execute → publish outputs
```

`_process_input_queue()` blocks *only* while `not self.has_work()` — nothing running, nothing queued.
The moment there is work it drains the socket non-blockingly and steps. That is the design: **a loop
that never blocks while there is work to do.**

Why not an event loop? Because one step is microseconds to milliseconds of GPU work, and the CPU must
stay **ahead** of the GPU. If the engine core blocked on socket readiness, every decoding step would
inherit a syscall round-trip and an event-loop wakeup before the scheduler could produce the next
`SchedulerOutput` — and at a few hundred decode steps per second, that is a visible fraction of the
step. Blocking is how you starve a GPU that is 3 ms behind you.

V1 pushes this further with **async scheduling**: `AsyncScheduler`
(`vllm/v1/core/sched/async_scheduler.py`, selected in `SchedulerConfig.get_scheduler_cls`) prepares step
*N+1* while the GPU executes step *N*. `SchedulerConfig.async_scheduling` defaults to `None`, meaning
auto-enabled (`vllm/config/scheduler.py`). The cost is real: the engine core will consume a whole
physical core, yielding in `_process_engine_step()` only in the rare "no model executed but requests
remain" case.

## 3. The ZMQ hop, and what actually crosses it

Both client classes live in `vllm/v1/engine/core_client.py`: `SyncMPClient` for offline `LLM`,
`AsyncMPClient` for the server, plus DP variants. The send path is `_send_input()`:

```python
msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
self.input_socket.send_multipart(msg, copy=False)
```

Three frames: which engine, a one-byte request type, and a msgspec-msgpack-encoded payload.
`EngineCoreRequestType` values are declared as raw bytes (`ADD = b"\x00"`), so the type needs no
encoding step of its own.

Now look at what the payload *is*. `EngineCoreRequest` (`vllm/v1/engine/__init__.py`) is a
`msgspec.Struct` whose fields include `prompt_token_ids: list[int]`, `sampling_params`, `mm_features`,
`arrival_time`, `lora_request`. **There is no prompt string.** Text dies in the API server process;
token ids and typed parameters cross the wire in both directions — `EngineCoreOutput.new_token_ids`
comes back the same way, and `IncrementalDetokenizer` (`vllm/v1/engine/detokenizer.py`) turns it into
text again in the API server process. The engine core has no tokenizer at all.

To debug in one process instead of three, set `VLLM_ENABLE_V1_MULTIPROCESSING=0` (`vllm/envs.py`): the
client becomes `InprocClient` and the hop disappears. msgpack rather than pickle because the schema is
explicit (decode is typed via `MsgpackDecoder`) and large tensor payloads can ride out-of-band without
a copy.

## 4. One request, end to end

Follow a `POST /v1/chat/completions` through the real code. Open each file as you go.

| # | Hop | File |
| --- | --- | --- |
| 1 | HTTP route → handler | `vllm/entrypoints/launchers/api_server/routers.py`, `vllm/entrypoints/openai/chat_completion/api_router.py` → `vllm/entrypoints/openai/chat_completion/serving.py` |
| 2 | Render chat template, tokenize, validate params, resolve multimodal inputs → `EngineCoreRequest` | `vllm/v1/engine/input_processor.py` (`InputProcessor.process_inputs`), via `vllm/renderers/` |
| 3 | Register a result queue; hand the request to the engine core | `vllm/v1/engine/async_llm.py` (`AsyncLLM.add_request` → `engine_core.add_request_async`) |
| 4 | msgpack + ZMQ; `EngineCore.add_request` → `Scheduler.add_request` onto the waiting queue | `vllm/v1/engine/core_client.py`, `vllm/v1/engine/core.py`, `vllm/v1/core/sched/scheduler.py` |
| 5 | `Scheduler.schedule()` → `SchedulerOutput`: a token budget per request plus the KV blocks it may use (prefix-cache lookup first) | `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/sched/output.py` |
| 6 | Dispatch to the worker; build tensors; forward pass; sample | `vllm/v1/executor/multiproc_executor.py`, `vllm/v1/worker/gpu_worker.py`, `vllm/v1/worker/gpu/model_runner.py` (§6), `vllm/v1/sample/` |
| 7 | `Scheduler.update_from_output()`: append tokens, mark finished sequences, free/publish blocks | `vllm/v1/core/sched/scheduler.py` |
| 8 | `EngineCoreOutputs` back over ZMQ; the API server's background task pulls them | `vllm/v1/engine/async_llm.py` (`_run_output_handler`) |
| 9 | Detokenize incrementally; apply stop strings; build `RequestOutput` | `vllm/v1/engine/output_processor.py` (`OutputProcessor.process_outputs`), `vllm/v1/engine/detokenizer.py` |
| 10 | Yield the delta to the route's async generator → SSE `text/event-stream` chunk | `vllm/entrypoints/openai/chat_completion/api_router.py` |

Three things to notice. **Steps 5–7 repeat once per generated token** — a 500-token answer makes 500
trips round that loop, and every optimisation in vLLM makes those trips cheaper or fuller. **Steps 2–3
and 8–10 are in a different process from 4–7**, so anything you measure end-to-end includes both hops.
And **the docs are staler than the source**: the official Architecture Overview page sends you to
`vllm/entrypoints/openai/api_server.py`, which in v0.30.0 is a **59-line deprecation shim** that
re-exports from `vllm/entrypoints/launchers/api_server/`. The same page's "LLM Engine" section still
describes `LLMEngine`/`AsyncLLMEngine` and links `vllm/engine/llm_engine.py` — now six lines:
`LLMEngine = vllm.v1.engine.llm_engine.LLMEngine`. Read `vllm/v1/engine/` and
`vllm/entrypoints/launchers/`, not the page.

## 5. Two entrypoints, and a 4× difference you can see

| | Offline | Online |
| --- | --- | --- |
| You write | `from vllm import LLM` — `vllm/entrypoints/llm.py` | `vllm serve <model>` — `vllm/entrypoints/cli/main.py` → `vllm/entrypoints/cli/serve.py` |
| Engine class | `LLMEngine` — `vllm/v1/engine/llm_engine.py` | `AsyncLLM` — `vllm/v1/engine/async_llm.py` |
| `UsageContext` | `LLM_CLASS` | `OPENAI_API_SERVER` |

Both funnel into `EngineArgs.create_engine_config(usage_context=…)` →
`_set_default_max_num_seqs_and_batched_tokens_args()` → **`get_batch_defaults(world_size)`** in
`vllm/engine/arg_utils.py`. Read it top to bottom; it is about thirty lines:

1. It queries `current_platform.get_device_total_memory()` and `get_device_name().lower()` inside a
   `try/except` that falls back to `0` and `""` when the importing process has no GPU.
2. Three branches:
   - `>= 160 GiB` (B200/B300): 16384 batched tokens for **both** contexts, 1024 sequences.
   - `>= 70 GiB` **and `"a100" not in device_name`** (H100/H200): 16384 offline, 8192 serve; 1024
     sequences. The A100 exclusion is deliberate — the comment cites PR #17885, where a large batch
     budget *reduced* A100 throughput.
   - everything else, **including your 4090 and the A100**: 8192 offline, **2048 serve**; 256
     sequences.
3. It returns two dicts keyed by `UsageContext` (`vllm/usage/usage_lib.py`). TPU and CPU override
   afterwards.

Back in the caller the defaults apply only where the user left the flag `None`, then
`--performance-mode throughput` doubles both, and with `--no-enable-chunked-prefill` the budget is
floored at `max_model_len`. So on the 4090:

```
LLM(model="Qwen/Qwen3-8B")    → max_num_batched_tokens = 8192, max_num_seqs = 256
vllm serve Qwen/Qwen3-8B      → max_num_batched_tokens = 2048, max_num_seqs = 256
```

Same model, same card, **4× the per-step token budget** offline — and the serve value governs
everything you measure from Stage 1 onward. It is not a bug. It is the engine trading throughput for
latency: a smaller step means more scheduling points per second, so a new request does not sit behind a
large prefill. What the source *does not* tell you is the reasoning — the branch above it is labelled
only `# TODO(woosuk): Tune the default values for other hardware.` The numbers are in the code; the
intent is not. Stage 5 makes you measure it instead of guessing.

```bash
source scripts/env.sh
less +/"def get_batch_defaults" "$VLLM_SRC/vllm/engine/arg_utils.py"   # then follow the caller
grep -n "class UsageContext" -A 8 "$VLLM_SRC/vllm/usage/usage_lib.py"
```

Then start the server and find the effective values in the startup config dump — that value is what
the scheduler uses.

## 6. Model Runner V2 (MRV2)

**What it is.** The model-execution layer *inside* V1: it prepares per-step input tensors, runs the
forward pass, manages CUDA graphs, and samples. It is **not** a "V2 engine". `V1` names the engine
generation; `MRV1`/`MRV2` name two implementations of the runner within it.

**It is the default.** Since **v0.29.0**, MRV2 is used for all models unless something disqualifies it.
The decision is `VllmConfig.use_v2_model_runner` in `vllm/config/vllm.py`, in this order: HiSparse or
watermarking force V2; else the env var `VLLM_USE_V2_MODEL_RUNNER` (`vllm/envs.py`, tri-state, default
`None` = auto) if set; else ROCm architecture defaults; else no Triton → MRV1; else any unsupported
feature → MRV1.

**The file tell.** MRV1 is one flat module; MRV2 is a package.

| | Path | Size at v0.30.0 |
| --- | --- | --- |
| MRV1 (deprecated) | `vllm/v1/worker/gpu_model_runner.py` | 7,664 lines, one file |
| MRV2 (default) | `vllm/v1/worker/gpu/model_runner.py` + siblings | 84 `.py` files, ≈ 23,457 lines; `model_runner.py` itself 2,345 |

Siblings worth opening: `attn_utils.py`, `buffer_utils.py`, `cudagraph_utils.py`, `dp_utils.py`,
`cp_utils.py`, `eplb_utils.py`, `input_batch.py`, `warmup.py`, `sample/`.

**What still falls back to MRV1 (v0.30.0).** `_get_v2_model_runner_unsupported_features()` in
`vllm/config/vllm.py` returns the list that forces MRV1: stock torch.compile; sequence parallelism;
pipeline parallelism with `external_launcher`; the speculative methods `ngram`, `ngram_gpu`,
`draft_model`, `suffix`, `medusa`, `mlp_speculator`, `custom_class`; `parallel_drafting` for EAGLE;
anything in the dual-batch-overlap list when `use_ubatching`; elastic expert parallelism; custom logits
processors (including `vllm.logits_processors` entry-point plugins); and mamba cache mode `all`. On
ROCm, `DeepseekV32ForCausalLM`, `DeepseekV4ForCausalLM` and `GlmMoeDsaForCausalLM` default to MRV1 as a
speed preference. **This list shrinks fast** — on `main` (ahead of the tag), stock torch.compile,
elastic EP, custom logits processors, mamba `all` and `draft_model` have already been removed. Re-read
the function at your tag; MRV1's removal is targeted at v0.32.

**Why MRV2 exists.** `$VLLM_SRC/docs/design/model_runner_v2.md` is candid that MRV1 accumulated
"fundamental design mistakes and significant technical debt". The concrete ones:

- MRV1's *persistent batch* used its long-lived state tensors **directly as model inputs**, imposing
  strict layout and ordering: joining or finishing a request meant reordering rows tensor-wide, and a
  redundant backup copy (`CachedRequestState`) existed because a row could be overwritten while its
  request was still live. MRV2 pre-allocates a fixed `max_num_reqs`-row state tensor, gives each request
  a permanent row, gathers per-step inputs from it, and treats preemption as completion — so
  `CachedRequestState` is gone.
- MRV1 was written before async scheduling and retrofitted it with an **async barrier**. MRV2 assumes a
  CUDA stream with no CPU synchronisation points and removes the race structurally: keep the persistent
  CPU state *unpinned* and copy to a temporary pinned buffer, so the CPU never writes memory the GPU is
  reading. Large CPU→GPU copies become `StagedWriteTensor` — base tensor on the GPU, row diffs staged on
  the CPU, packed, copied once, applied by one kernel.
- Preparatory and sampling work moves onto the GPU. Input metadata (`input_ids`, `positions`,
  `query_start_loc`, `seq_lens`) is built in Triton — cheaper than a Python loop, and correct under
  speculation, where the GPU knows values the CPU does not yet — with UVA letting kernels read large
  CPU-resident tensors directly. Sampling gets a Gumbel-max Triton kernel that avoids materialising a
  softmax, and top-k logprobs are computed by selecting tokens *before* computing logprobs. CUDA graphs
  become explicit via a `CUDAGraphManager`, and `dummy_run` stops being overloaded for profiling,
  capture and warmup at once.

**Honesty about its state.** The design doc calls MRV2 "not yet feature-complete, not rigorously
tested"; `vllm/v1/worker/gpu/README.md` still opens with `# [Experimental] Model Runner V2`. Both
coexist with it being the default since v0.29.0 — and the official design page never says "this is the
default". That fact lives in `vllm/config/vllm.py` and the release notes.

## 7. The scheduler, briefly

Stage 2 and Stage 5 go deep; this is only the shape.

- `vllm/v1/core/sched/interface.py` — `SchedulerInterface`, the contract. `scheduler.py` implements it;
  `async_scheduler.py` subclasses it. `schedule()`'s docstring is the best three paragraphs in the
  codebase: "the scheduling decision is made at the iteration level… the scheduler produces a
  dictionary of `{req_id: num_tokens}`".
- The rhythm is two calls: `schedule()` → the worker executes → `update_from_output()`.
- Queues: `self.waiting` (a `RequestQueue`, either `FCFSRequestQueue` or `PriorityRequestQueue` —
  `vllm/v1/core/sched/request_queue.py`, selected by `--scheduling-policy`), `self.running:
  list[Request]`, plus a separate KV-holding waiting queue for requests parked on a KV connector.
- `SchedulerOutput` (`vllm/v1/core/sched/output.py`) carries `scheduled_new_reqs` (each
  `NewRequestData` with its `block_ids` and `num_computed_tokens`), `scheduled_cached_reqs`,
  `num_scheduled_tokens` (the per-request budget), `total_num_scheduled_tokens`,
  `scheduled_spec_decode_tokens`, `num_common_prefix_blocks`, `finished_req_ids`, `preempted_req_ids`,
  `has_structured_output_requests`, and optional KV-connector metadata.
- **Admission control.** The queue-limit flags `--max-num-queued-reqs` / `--max-num-queued-tokens` are
  new in **v0.29** (PR #49445); the module that makes them cheap landed in **v0.30.0** (PR #54746).
  `vllm/v1/engine/admission_control.py` holds `SharedAdmissionStats`, a lock-free counter array in
  shared memory with one cache-line-sized slot per API server process (so writers never invalidate each
  other's cache lines), aggregated by readers. `AsyncLLM.check_admission()` uses it to reject with
  **HTTP 503** once the in-flight request count or the prefill backlog (a TTFT QoS valve) is exceeded.
  Unlike `--max-num-seqs`, these are enforced in the API server and count across every DP rank it
  routes to.

## 8. Where the time goes for one request

| Phase | Process | Side | Appears as |
| --- | --- | --- | --- |
| HTTP parse, chat template, **tokenize** | API server | CPU | TTFT |
| msgpack + ZMQ hop | API server → EngineCore | CPU/IPC | TTFT (microseconds) |
| **Queue wait** in `self.waiting` | EngineCore | CPU | TTFT; grows with load, not prompt length |
| **Prefill** forward pass | Worker | **GPU, compute-bound** | TTFT; scales with prompt tokens |
| Sampling the first token | Worker | GPU | TTFT |
| Each **decode step**, ×N tokens | Worker | **GPU, bandwidth-bound** | inter-token latency; roughly flat per token |
| Scheduling + input prep, per step | EngineCore + Worker | CPU | inter-token latency — visible at small batch (what CUDA graphs and MRV2 attack) |
| **Detokenize** + SSE write | API server | CPU | inter-token latency at high concurrency |

TTFT is everything above the decode rows; inter-token latency is the last three. The trap is that a
CPU-side stall in the API server and a GPU-side stall in the worker both reach the client as latency.
Separating them is what `/metrics` is for (Stage 7).

## 9. How to see this yourself

With the server running (`bash scripts/serve.sh`, its own terminal), in a second terminal:

```bash
source scripts/env.sh
nvidia-smi                                                    # GPU + the compute-process table
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
ps -eo pid,ppid,etime,args | grep 'VLLM::' | grep -v grep      # the three processes
pgrep -af python
```

vLLM renames its own processes via `set_process_title()` (`vllm/utils/system_utils.py`), which formats
the title as `{VLLM_PROCESS_NAME_PREFIX}::{name}` — the prefix defaults to `VLLM` (`vllm/envs.py`). The
callers give you the naming scheme:

| Caller | Title |
| --- | --- |
| `vllm/v1/utils.py` | `VLLM::APIServer_<index>` |
| `vllm/v1/engine/core.py` | `VLLM::EngineCore` (or `EngineCore_DP<rank>`) |
| `vllm/v1/executor/multiproc_executor.py` | `VLLM::Worker` (+ `_DP<r>`, `_PP<r>`, `_TP<r>`, `_EP<r>` suffixes when those sizes exceed 1) |

**What you should observe on the 4090** — a prediction from the source, not a recording: exactly
**three** Python processes, `VLLM::APIServer_0`, `VLLM::EngineCore`, `VLLM::Worker`, which is
`A + DP + N = 1 + 1 + 1`. Predict **4** for `-tp=2` on the 2×4090 pod, and **6** for the 4-GPU `-tp=4`
example in `arch_overview.md`. Restarting with `--api-server-count 2` should show four processes for the
same single GPU.

Two details worth pausing on. `nvidia-smi`'s compute-process table lists **one** PID: the
`VLLM::Worker`. The `VLLM::EngineCore` PID is absent because it never touches the GPU — it only sends
it work over ZMQ and collects token ids. That is the process split made visible. And the worker's
parent PID is the API server, which is why killing the server tears the tree down.

**If `ps` shows no `VLLM::` titles**, `setproctitle` is missing (the import in `set_process_title()` is
wrapped in `try/except ImportError`) or `ps` truncated the command line. Fall back to `pgrep -af python`
and match PIDs against `nvidia-smi`. This is Stage 1's Experiment A; the memory number in `nvidia-smi`
staying flat while you generate is the other half of it.

---

## Checkpoint

✅ Why is tokenization in the API server but scheduling in the EngineCore?

✅ Why does the EngineCore run a busy loop instead of being event-driven, and what does async scheduling
add?

✅ How many OS processes does a 1-GPU `vllm serve` have, what is each, and what would `-tp=4` make it?

✅ What actually crosses the ZMQ hop in each direction, and where does text become token ids and back?

✅ On a 4090, what `--max-num-batched-tokens` do `LLM()` and `vllm serve` each default to, and which
function decides it? Give the file-level tell between MRV1 and MRV2.

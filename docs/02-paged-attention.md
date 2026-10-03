# 02 — PagedAttention, the KV cache, and where the memory goes

The constraint that decides how many requests your server holds at once is **KV cache memory**. Not
weights, not FLOPS. This doc covers why it is stored in blocks, who allocates it, how blocks are shared
and reclaimed, and how to read the number vLLM prints at startup as arithmetic.

Checked against the **v0.30.0** source tree on 2026-10-02. Source paths are relative to `$VLLM_SRC`.
Nothing here was run on a GPU; every number is read out of a config file or derived with the arithmetic
shown beside it.

> **Two doc warnings.** `$VLLM_SRC/docs/design/paged_attention.md` is still marked *"historical document
> based on the original paper… It no longer describes the code used in vLLM today."* Read it for the
> kernel's memory layout only. And the official optimization page calls the KV-memory knob
> `--kv-cache-memory`, which is not the flag's name — it only *appears* to work, and §9 explains why.

---

## 1. The problem: a cache sized for the worst case

A naive engine gives each sequence **one contiguous tensor sized for `max_model_len`**, because you
cannot know in advance how long the output will be. Three kinds of waste follow:

| Waste | Cause |
| --- | --- |
| **Internal fragmentation** | It reserved `max_model_len` slots and used 200 |
| **Reservation waste** | Slots held for output tokens that may never be generated |
| **External fragmentation** | Free space breaks into runs, none big enough to reuse |

Work it for the card you rent: RTX 4090, 24 GiB, `$MODEL_BIG` = `Qwen3-8B`,
`--gpu-memory-utilization 0.90`. From §4, KV bytes/token is **144 KiB** and the slab is about
**4.8 GiB ≈ 34,950 tokens** (§8). With `--max-model-len 32768`:

```
one worst-case reservation = 32,768 tokens × 144 KiB = 4.50 GiB
KV slab                    = 4.80 GiB
sequences that fit         = 1
stranded                   = 0.30 GiB — no second 4.50 GiB run exists
```

One sequence. A 200-token prompt inside it holds 4.50 GiB to store `200 × 144 KiB = 28.1 MiB` — **0.6 %
utilisation**. The leftover 0.30 GiB cannot serve a second request even if that request is 100 tokens,
because the allocator demands one contiguous run: external fragmentation. The PagedAttention paper
measured 60–80 % of KV memory wasted in the systems it compared against. KV memory limits batch size;
batch size produces throughput; every wasted byte is throughput you never get.

**The fix is virtual memory for KV.** Split the cache into fixed-size **blocks** (16 tokens by default,
§3) and give each sequence a **block table** mapping logical block index → physical block id. Blocks need
not be contiguous; only a sequence's last block is partly empty. Same card, same 200-token requests:

```
blocks in the slab    = 4.80 GiB / 2.25 MiB = 2,184 blocks
blocks for 200 tokens = ceil(200 / 16)      = 13 blocks (208 slots) = 29.25 MiB
internal waste        = 8 slots × 144 KiB   = 1.125 MiB (3.8 %)
sequences that fit    = 2,184 / 13          = 168
```

168 sequences instead of 1, from the same memory. That ratio — not the kernel — is why vLLM exists.

## 2. Blocks and block tables: virtual memory for attention

| Operating system | vLLM | Source |
| --- | --- | --- |
| Page | **Block** — 16 token slots for one KV group | `KVCacheBlock` in `vllm/v1/core/kv_cache_utils.py` |
| Page table | **Block table** — per-sequence array of physical block ids | `BlockTable` in `vllm/v1/worker/block_table.py` |
| Page-frame allocator | **Block pool** — flat block array plus a free queue | `BlockPool` in `vllm/v1/core/block_pool.py` |
| Page cache / reclaim | **Prefix cache** — hash-indexed blocks, LRU eviction | §6, §7 |

```
sequence "req-7":   logical block   0   1   2   3
                    physical id     5   2   9   4      ← non-contiguous, by design
```

**Indirection is what makes growth cheap.** Appending a block means "give me one more and write its id
at the end of the row." Nothing is copied, nothing is reallocated, no existing block moves. A contiguous
allocator has the opposite property: it must know the final size up front, and growth past the
reservation copies the whole cache. vLLM's block table is deliberately *append-only* (the in-tree design
doc says so), so a running sequence never rewrites history — which is what lets the pool hand out any
free block, in any order, without consulting the sequences already holding blocks.

The kernel side is dumb by design. `BlockTable` owns a `torch.int32` buffer of shape
`[max_num_reqs, max_num_blocks_per_req]` and a separate `torch.int64` `slot_mapping` buffer of
`max_num_batched_tokens` entries. Python fills the host array; `commit_block_table(num_reqs)` copies it
to the GPU once per step. The kernel reads that tensor and gathers K/V from wherever the blocks live —
it never learns that positions are logically contiguous. Because indirection is the *kernel's contract*,
the manager may allocate lazily, share, evict and remap. Two features fall out: **sharing** (two
sequences with the same prefix point at the same physical blocks; `BlockPool.is_block_writable()` is the
copy-on-write guard, true only when `ref_cnt == 1` **and** the block has no hash yet) and **cheap
reclaim** (freeing decrements refcounts; it never moves bytes).

## 3. Why 16 tokens per block

`CacheConfig.DEFAULT_BLOCK_SIZE: ClassVar[int] = 16` (`vllm/config/cache.py`). It trades three costs:

| Smaller block (8) | Larger block (32, 64, 128) |
| --- | --- |
| Less internal fragmentation in the last block | More waste per partly-filled sequence |
| More block-table entries per sequence → longer rows | Shorter rows, fewer entries per step |
| More indirection in the attention kernel | Better attention-kernel throughput |

16 is also the **floor the kernels support**. A backend declares `get_supported_kernel_block_sizes()`;
the Triton attention backend returns `[MultipleOf(16)]` (`vllm/v1/attention/backends/triton_attn.py`),
and `AttentionBackend.supports_block_size()` accepts any multiple of that
(`vllm/v1/attention/backend.py`). Ask for a size the selected backend cannot do and
`get_preferred_block_size()` silently picks the smallest it can. So 16 is the finest unit the mainstream
path runs, which makes it the natural default.

**When you would change it.** Raise `--block-size` for long sequences with huge block-table rows (128k
context, many concurrent requests), or when a backend prefers a bigger page. Go below 16 only if your
backend explicitly supports it. Note that **prefix-cache granularity follows the block size**: with
32-token blocks, a shared prefix shorter than 32 tokens is never worth reusing. A new block size also
recompiles the attention kernels.

## 4. Where the bytes actually go

```
KV bytes/token = 2 (K and V) × num_layers × num_kv_heads × head_dim × dtype_bytes
```

Worked for the model ladder (field values from each model's `config.json`):

| Model | Layers | KV heads | head_dim | bf16 | fp8 | Block (16 tok, bf16) |
| --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3-0.6B` (`$MODEL_TINY`) | 28 | 8 | 128 | **112 KiB** | 56 KiB | 1.75 MiB |
| `Qwen/Qwen3-4B` (`$MODEL_MID`) | 36 | 8 | 128 | **144 KiB** | 72 KiB | 2.25 MiB |
| `Qwen/Qwen3-8B` (`$MODEL_BIG`) | 36 | 8 | 128 | **144 KiB** | 72 KiB | 2.25 MiB |

The 8B arithmetic written out: `2 × 36 × 8 × 128 × 2 = 147,456 bytes = 144 KiB` exactly. At fp8 the last
factor becomes 1, giving 72 KiB. Anchors: 144 KiB/token is **≈137 GiB per million cached tokens**; 112
KiB/token is ≈107 GiB per million. One full 40,960-token sequence at 144 KiB/token costs 5.63 GiB.

**GQA is why this is affordable at all.** The 4B and 8B have 32 query heads but only **8 KV heads** — a
4:1 ratio. K and V are stored per *KV* head, so the cache is 4× smaller than a non-grouped model of the
same depth would need. That one choice is the biggest reason modern 8B models are servable on 24 GiB.

**One slab, allocated up front.** vLLM does not grow the KV cache as requests arrive. It loads weights,
runs `profile_run` on a dummy max-size batch to measure peak activation memory, computes
`requested_memory = ceil(total_device_memory × gpu_memory_utilization)` (`vllm/v1/worker/utils.py`),
then `available_kv_cache_memory_bytes = requested_memory − non_kv_cache_memory − cudagraph_estimate`
(`vllm/v1/worker/gpu_worker.py`). Dividing by the page size gives `num_blocks` (`get_kv_cache_configs` in
`vllm/v1/core/kv_cache_utils.py`), and each worker allocates the full cache tensors in
`initialize_kv_cache()` (`vllm/v1/worker/gpu_model_runner.py` for MRV1,
`vllm/v1/worker/gpu/model_runner.py` for MRV2). So **"how many tokens can I cache" is fixed before the
first request arrives**, and `nvidia-smi` will not move once the server is up — Stage 1's experiment, and
this is why.

## 5. The classes, and what each is actually responsible for

| File (`$VLLM_SRC/…`) | Class(es) | Responsibility |
| --- | --- | --- |
| `vllm/v1/core/block_pool.py` | `BlockPool`, `BlockHashToBlockMap` | Owns the block array and free queue; allocates, frees, re-pins, hashes, evicts. |
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager`, `KVCacheBlocks` | Per-request API (`get_computed_blocks`, `allocate_slots`, `cache_blocks`, `free`) and the request→blocks map. Delegates policy to a coordinator. |
| `vllm/v1/core/kv_cache_coordinator.py` | `KVCacheCoordinator` (ABC), `KVCacheCoordinatorNoPrefixCache`, `UnitaryKVCacheCoordinator`, `HybridKVCacheCoordinator` | One allocation decision per step that satisfies every KV group. Creates and owns the `BlockPool`. |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `SingleTypeKVCacheManager` + `FullAttentionManager`, `SlidingWindowManager`, `ChunkedLocalAttentionManager`, `MambaManager`, `CrossAttentionManager`, `RSWAManager`, `CircularBufferManager`, … | One manager per *KV layout*, each with its own `find_longest_cache_hit()` and block arithmetic. |
| `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock`, `FreeKVCacheBlockQueue`, `hash_block_tokens()`, `update_kv_cache_capacity()` | Block metadata, the LRU free queue, the hash function, the startup log line. |
| `vllm/v1/worker/block_table.py` | `BlockTable`, `MultiGroupBlockTable` | The GPU-side `int32` block-id tensor and `int64` slot mapping. Data movement only. |

> **Two files, similar names.** This doc means `vllm/v1/worker/block_table.py` (singular `BlockTable`).
> The separate `vllm/v1/worker/gpu/block_table.py` defines `BlockTables` (plural), the Model Runner V2
> container for those tables plus its Triton kernels. MRV2 imports both. Cite the one you mean.

What the code says that the docs do not:

- **`BlockPool`** keeps one `KVCacheBlock` per slab position and holds a `null_block` out of the free
  queue as a placeholder (block id 0, `is_null=True`, never cached, never freed). `get_new_blocks(n)`
  pops the *head* of the free queue and raises if `n` exceeds the free count — it never searches the
  cache. `touch(blocks)` re-pins a prefix-cache hit: remove from the free queue if `ref_cnt == 0`, then
  increment. `FreeKVCacheBlockQueue` is a doubly linked list threaded through the blocks' own
  `prev_free_block`/`next_free_block` fields, so removal from the middle is O(1) and no Python object is
  allocated on the hot path.
- **`KVCacheManager`** owns no blocks. It keeps `request_id → (blocks per group)` and forwards every
  decision to `self.coordinator`. `allocate_slots()` returns `KVCacheBlocks` **or `None`** — that `None`
  is the entire back-pressure signal the scheduler gets (§7).
- **The coordinator exists because one manager cannot describe a hybrid model.** Sliding-window and
  full-attention layers need *different block counts for the same tokens*: full attention reserves every
  token, sliding window only the last *W*. Add Mamba layers (a recurrent state, not a growing cache),
  MLA layers (a compressed latent instead of per-head K/V), or chunked-local attention, and no single
  `block_size` or eviction schedule describes them. `HybridKVCacheCoordinator` composes one manager per
  group, aligns them to a common *scheduler block size* (the LCM of all group block sizes), and returns
  one decision per step (`$VLLM_SRC/docs/design/hybrid_kv_cache_manager.md`). A plain Qwen3 model is all
  full attention, so you get one group and the `UnitaryKVCacheCoordinator` — the hybrid machinery is
  inert, which is why you will not see it in your first logs.

## 6. Prefix caching, at block granularity

The default is on: `CacheConfig.enable_prefix_caching: bool = True` (`vllm/config/cache.py`). To turn it
off you now pass **`--no-enable-prefix-caching`** — the negation is generated automatically because the
field is a `bool` (`argparse.BooleanOptionalAction` in `vllm/engine/arg_utils.py`). Guides telling you to
*pass* `--enable-prefix-caching` describe a version where it was opt-in.

**What is hashed.** `hash_block_tokens()` in `vllm/v1/core/kv_cache_utils.py`:

```python
return BlockHash(hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys)))
```

Three components, each earning its place:

1. **The parent block's hash.** The root uses a fixed `NONE_HASH`. This chaining is the whole trick: a
   block's identity depends on *every token before it*, so block 3 of one prompt cannot be confused with
   block 3 of a different prompt sharing those 16 tokens. Reuse is only possible at the same prefix
   position.
2. **The block's own token ids** — belt and braces against collisions, and the reason a prompt differing
   in one early token loses *all* downstream reuse.
3. **Extra keys** — LoRA ids, multimodal input hashes, prompt-embedding digests, and per-request
   `cache_salt` values (`generate_block_hash_extra_keys()` and the `_gen_*_extra_hash_keys` helpers).
   This is what stops two tenants sharing blocks they should not.

Only **full** blocks are cached; a partly-filled tail is never inserted. The default algorithm is
`sha256` (`--prefix-caching-hash-algo`); the `xxhash` variants are faster but flagged in
`vllm/config/cache.py` as a collision and multi-tenant risk.

**Why content-addressed rather than address-based.** An address-based cache keys on *where* a block
lives; a content-addressed one keys on *what is in it*. Only the second answers "does some block
anywhere in the pool already hold the KV for these tokens?" — exactly what
`BlockPool.get_cached_block(hash, group_ids)` asks, reached via `KVCacheManager.get_computed_blocks()` →
`coordinator.find_longest_cache_hit()`. Address-based reuse would require the second request to land in
the same physical slots, which paging deliberately prevents. Content addressing also makes sharing
trivial: the hit *is* the block, so a second sequence points at it and `touch()`es it.

Blocks are inserted on the write path by `cache_full_blocks()`. Because the block table is append-only,
v1 can create *duplicate* blocks with the same hash, cleaned up when a request is freed —
`$VLLM_SRC/docs/design/prefix_caching.md` walks that through. Stage 4 and `labs/05_prefix_caching.py`
make you measure it: TTFT for a repeated prompt with and without the feature, then with one token
changed at the front.

## 7. Eviction, refcounts, and what happens when the pool empties

**Policy: LRU over unreferenced blocks.** The mechanism is the free queue's ordering, and
`BlockPool.free_blocks()` states both rules in code:

```python
if block.block_hash is None or not self.enable_caching:
    blocks_to_evict_first.append(block)   # "LIFO reuse of non-cached blocks for better GPU locality"
else:
    blocks_to_evict_last.append(block)    # "FIFO reuse of cached blocks for LRU eviction behavior"
```

Unhashed blocks are reused last-in-first-out (locality). Hashed blocks are appended in order, so the
oldest-touched sits at the head and goes first. **The access pattern this assumes** is temporal locality
over prefixes: recent prompts resemble upcoming ones (multi-turn chat, RAG over a shared document, a
fixed system prompt). With no prefix reuse at all, cached blocks are pure overhead — they pin memory
nothing will ask for. That is the case for `--no-enable-prefix-caching`.

**Refcounts make eviction safe.** A block with `ref_cnt > 0` is not in the free queue at all: `touch()`
removed it. So a block is evictable only when **no live sequence references it**, and eviction means
"reset the hash and hand the block to the next writer" (`_maybe_evict_cached_block()`), never a copy.
Because it is *cache* eviction and recomputation is always a valid fallback, dropping the wrong block
costs time and never correctness.

**When the pool empties**, two things happen. A **waiting** request gets `None` from `allocate_slots()`,
the scheduler `break`s out of its admission loop (`vllm/v1/core/sched/scheduler.py`), and the request
stays queued: no error, no rejection, just latency. (Admission control can turn that into an HTTP 503,
but only if you configure it — doc 04.) A **running** request that needs another block causes preemption
instead: the scheduler takes the lowest-priority running request — `self.running[-1]` under the default
`fcfs` policy, or the lowest-priority arrival under `priority` — calls `_preempt_request()`, frees its
blocks, sets `num_computed_tokens = 0`, clears its speculative tokens, and **prepends it to the waiting
queue**.

That last line is the important one: **V1's preemption is recompute.**
`$VLLM_SRC/docs/usage/v1_guide.md` lists *GPU ↔ CPU KV Cache Swapping* as **🔴 Removed** — "with the new
simplified core architecture, vLLM V1 no longer requires KV cache swapping to handle request
preemptions." There is no `PreemptionMode` enum anywhere in the V1 source; grep for it and you get
nothing. A preempted request re-runs its prefill from token zero when re-admitted: cheap because blocks
made it cheap, expensive if it happens often. Watch `vllm:num_preemptions`
(`vllm/v1/metrics/loggers.py`). Off-device KV in V1 is an explicit, opt-in *connector* feature, not a
transparent swap path: `--kv-offloading-size` and `--kv-offloading-backend` in `vllm/config/cache.py`,
plus Stage 10's disaggregated connectors. For the lab, `--num-gpu-blocks-override`
(`CacheConfig.num_gpu_blocks_override`) forces an artificially tiny pool — its docstring says what it is
for: *"Used for testing preemption."*

## 8. Reading the startup log as arithmetic

Three lines, and they are the most information-dense output vLLM produces:

```
INFO ... Available KV cache memory: 4.80 GiB                          # vllm/v1/worker/gpu_worker.py
INFO ... Chunked prefill is enabled with max_num_batched_tokens=2048. # vllm/config/scheduler.py
INFO ... GPU KV cache size: 34,952 tokens, Maximum concurrency for 8,192 tokens per request: 4.27x
```

The third is a **single merged `logger.info_once`** in `vllm/v1/core/kv_cache_utils.py`
(`update_kv_cache_capacity`). Format string:
`"%s KV cache size: %s tokens, Maximum concurrency for %s tokens per request: %.2fx"`, device prefix
`"GPU"` on CUDA, thousands separators on both counts. Guides quoting two separate lines are stale.

The recipe, as the source computes it:

```
requested_memory = ceil(total_device_memory × gpu_memory_utilization)      # vllm/v1/worker/utils.py
available_kv     = requested_memory − non_kv_cache_memory − cudagraph_est  # vllm/v1/worker/gpu_worker.py
N (tokens)       = available_kv / KV_bytes_per_token
num_blocks       = available_kv / (KV_bytes_per_token × block_size)
max_concurrency  = num_blocks / cdiv(max_model_len, block_size)
```

**Worked example — predicted, not measured — for `Qwen3-8B` on 24 GiB at util `0.90`:**

| Step | Arithmetic | Result |
| --- | --- | --- |
| Requested | `0.90 × 24 GiB` | 21.6 GiB |
| Less bf16 weights | `8.19e9 params × 2 B = 16.4 GB` | ≈15.3 GiB |
| Less peak activation, non-torch, CUDA graphs | measured by `profile_run`, not guessable | ≈1.5 GiB |
| **Available KV** | `21.6 − 15.3 − 1.5` | **≈4.8 GiB** |
| **Tokens** | `4.8 × 1,048,576 / 144` (KiB) | **≈34,950** |
| Blocks | `34,950 / 16` | ≈2,184 |
| Concurrency at 8,192 tokens | `34,950 / 8,192` | ≈4.27x |

**Mind the error bars.** The one term you cannot derive from a config file is `non_kv_cache_memory` —
vLLM *measures* it. Read "4.8" as decimal GB instead of GiB and the same line predicts ≈32,600 tokens: a
7 % swing from a units choice. **The engine's printed number is authoritative; the derivation is a
sanity check, and the gap between them is the lesson.** That is what Stage 3's Experiment C and
`labs/04_kv_cache_memory.py` are for.

What moves the printed `N`:

| Change | Effect |
| --- | --- |
| `--gpu-memory-utilization` 0.90 → 0.95 | Up, roughly linearly in the freed bytes |
| `--kv-cache-dtype fp8`, or fp8 weights | Up — fewer KV bytes, or fewer weight bytes before the slab is cut |
| `--max-model-len` | **No direct effect on `N`** — it moves *concurrency*, not tokens |
| `-O2` vs `--enforce-eager` | `-O2` subtracts the graph estimate, so eager can print a *larger* `N` |

## 9. `--kv-cache-dtype fp8` and `--kv-cache-memory-bytes`

**`--kv-cache-dtype fp8`** (`CacheConfig.cache_dtype`, default `"auto"`) halves KV bytes/token: 144 KiB →
72 KiB for the 8B, 112 KiB → 56 KiB for the 0.6B. Same slab, so ≈2× the tokens and ≈2× the concurrency.
What stays the same: layers, heads and blocks — every slot is still there, just narrower. What you give
up is precision in K and V, which you detect by comparing outputs, not by reading them. On Ada (the 4090)
fp8 is native.

**`--kv-cache-memory-bytes`** (`CacheConfig.kv_cache_memory_bytes`) states the slab size directly. Its
docstring: *"kv_cache_memory_bytes (when not-None) ignores gpu_memory_utilization."* It accepts
human-readable sizes (`5G`, `2G`) because the argparse type is `human_readable_int`. Setting it skips the
memory-profiling pass, which shortens startup; the trade-off, from
`$VLLM_SRC/docs/configuration/optimization.md`, is that a conservative value caps concurrency while an
optimistic one fails at allocation time. The value is only valid on the same GPU with the same free
memory.

> **Doc-versus-source flag, and a trap worth understanding.** The official optimization page, the in-tree
> `$VLLM_SRC/docs/configuration/optimization.md`, and vLLM's own out-of-memory advice text
> (`vllm/v1/worker/gpu_worker.py`) all spell this knob **`--kv-cache-memory`**. The flag vLLM actually
> registers is `--kv-cache-memory-bytes` (`vllm/engine/arg_utils.py`), so you would expect the short
> spelling to fail.
>
> It does not fail. **It works** — and the reason is worth internalising. argparse matches unambiguous
> prefixes of long options by default (`allow_abbrev=True`, which vLLM never overrides), so
> `--kv-cache-memory`, `--kv-cache-mem`, and even `--kv-cache-memo` all bind to
> `--kv-cache-memory-bytes`. Verified on vLLM 0.30.0: `--kv-cache-mem=abc` fails with *"argument
> --kv-cache-memory-bytes: Value abc cannot be converted to…"*, naming the flag it really bound to, and a
> `--kv-cache-memory=1000000` run reports `'kv_cache_memory_bytes': 1000000` in its startup banner.
>
> Treat prefix matching as a party trick, not an interface: it holds only while the prefix stays
> unambiguous, and vLLM already has five `--kv-cache-*` flags. `--kv-cache=1` fails with *"ambiguous
> option: --kv-cache=1 could match --kv-cache-memory-bytes, --kv-cache-dtype,
> --kv-cache-dtype-skip-layers, --kv-cache-metrics, --kv-cache-metrics-sample"*. Write the full
> `--kv-cache-memory-bytes` in anything you intend to keep.

---

## Read in this order

1. `$VLLM_SRC/docs/design/prefix_caching.md` — design intent, with the duplicate-block example
2. `$VLLM_SRC/docs/design/hybrid_kv_cache_manager.md` — why coordination is needed at all
3. `vllm/v1/core/block_pool.py` — `get_cached_block`, `cache_full_blocks`, `get_new_blocks`, `touch`,
   `free_blocks`, `_maybe_evict_cached_block`
4. `vllm/v1/core/kv_cache_coordinator.py` — the ABC plus `Unitary…` and `Hybrid…`
5. `vllm/v1/core/kv_cache_utils.py` — `hash_block_tokens`, `FreeKVCacheBlockQueue`
6. `vllm/v1/worker/block_table.py` — the tensor the kernel actually reads

**Labs:** `labs/04_kv_cache_memory.py` (predict, then measure) and `labs/05_prefix_caching.py` (the TTFT
experiment) — Stages 3 and 4 in `CURRICULUM.md`.

---

## ✅ Checkpoint

1. Why do 16-token blocks beat one contiguous per-sequence buffer sized for `max_model_len`? Give the
   block-count arithmetic for a 200-token request on a 24 GiB 4090 running `Qwen3-8B`.
2. A block table maps *what* to *what*, which class owns the physical blocks, and why does the attention
   kernel never need to know that different sequences have different lengths?
3. What exactly goes into a block's hash, and why must it be content-addressed rather than
   address-based?
4. What are the eviction policy and the refcount rule that makes it safe? What does V1 do instead of
   GPU↔CPU swap when the pool runs dry?
5. Starting from `--gpu-memory-utilization 0.90` on a 24 GiB card, predict the three startup numbers for
   `Qwen3-8B` — then say which one you would trust and why.

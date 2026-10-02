# notes/ — your lab records

This folder is deliberately empty. It is where *your* measurements go, and it is the most valuable
part of this project: the docs contain predictions, this contains evidence.

The rule that makes it work: **every number gets a date and a version.** vLLM ships about every two
weeks and defaults do change. A number without a version is a rumour, including your own.

## The record template

Copy this into `notes/stage-NN-<topic>.md` for each stage. A stage is not finished until its record
has numbers in it.

```markdown
# Stage NN — <title>

- date: 2026-10-02
- vLLM: 0.30.0            (python -c "import vllm; print(vllm.__version__)")
- torch: 2.13.0+cu130     (python -c "import torch; print(torch.__version__)")
- GPU: RTX 4090 24 GB, driver 580.xx
- model: Qwen/Qwen3-4B
- command: vllm serve Qwen/Qwen3-4B --gpu-memory-utilization 0.90 --max-model-len 8192

## Startup log (the numbers vLLM printed before any traffic)
- Available KV cache memory: __ GiB
- GPU KV cache size: __ tokens
- Maximum concurrency for ____ tokens per request: __x
- default max_num_batched_tokens / max_num_seqs: __ / __

## Measured
| what | value | how |
| --- | --- | --- |
| model load time | __ s | wall clock to first ready log line |
| TTFT (short prompt) | __ ms | lab 02 |
| inter-token latency | __ ms | lab 02 |
| throughput, 1 request | __ tok/s | lab 03 |
| throughput, 8 concurrent | __ tok/s | lab 03 |
| p50 / p99 latency | __ / __ ms | lab 08 |

## What surprised me
-

## What I predict for the next stage
-
```

## Why each field is there

| Field | What it protects you from |
| --- | --- |
| date + versions | Comparing an old number against a new default and drawing a false conclusion. |
| GPU + driver | "It was faster yesterday" — with a different card or driver. |
| exact command | Re-running with one flag different and not noticing. |
| startup figures | Guessing your capacity instead of reading it off the log. |
| predictions | The only way to find out whether you actually understood the previous stage. |

## The prediction habit

Before each stage's experiment, write down what you think will happen and roughly by how much.
Then run it. Being wrong is the highest-value outcome available here — it points at exactly the
mental model that needs fixing. Being right for the wrong reason is the thing to watch for: if the
numbers match but your explanation was different, dig until you know why.

## Useful one-liners

```bash
# what am I actually running?
python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# where did my disk go?
du -sh /workspace/* | sort -h

# what did the server print at startup? (in the server's terminal, or:)
grep -E "KV cache size|Maximum concurrency|Available KV cache|Chunked prefill" server.log
```

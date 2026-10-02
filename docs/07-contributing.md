# 07 — Contributing: from reading to a merged PR

Stage 11's companion. Everything here was checked against the **v0.30.0** source tree (upstream `main`
as of **2026-10-02**) and the repo's own contributing docs, which are the authority — where this file
and the checkout disagree, the checkout wins.

This is the stage where the project stops being a tour. The realistic outcomes, in increasing order of
effort: **run modified vLLM**, **run the test suite**, **reproduce a real bug and report it with
evidence**, **land a small fix**. All four are contributions. Only the last one is a PR.

---

## 1. Three ways to run modified vLLM

| Way | Setup | Use it when | You cannot |
| --- | --- | --- | --- |
| PyPI wheel in `/workspace/venv` | `uv pip install vllm --torch-backend=auto` (what [scripts/bootstrap.sh](../scripts/bootstrap.sh) does) | Learning the engine, running labs, measuring | Change vLLM — it is a sealed install of v0.30.0 |
| **Editable source install** (`uv pip install -e .`) | `git clone <your fork>`, `uv venv`, `VLLM_USE_PRECOMPILED=1 uv pip install -e .` | **Changing vLLM.** Python edits take effect on the next process start; no rebuild | Nothing relevant — this is the dev loop |
| Official image `vllm/vllm-openai:v0.30.0` | `docker run --runtime nvidia --gpus all … vllm/vllm-openai:v0.30.0` | A production-representative run: same artifact other people deploy | Edit source, run `pytest`, or use VS Code comfortably (its entrypoint *is* the API server) |

The editable install is the one that matters for this stage, and it has two sub-modes. **Precompiled**
(`VLLM_USE_PRECOMPILED=1`) downloads matching compiled artifacts and installs your tree editable —
minutes. **Full source build** is the documented three-step path (`$VLLM_SRC/docs/contributing/README.md`)
when you touched C++/CUDA:

```bash
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
grep -v '^torch==' requirements/build/cuda.txt | uv pip install -r -
uv pip install -e . --no-build-isolation
```

---

## 2. An editable install on the pod, in its own venv

Upstream's `AGENTS.md` is explicit: "Never use system `python3` or bare `pip`/`pip install`. All
Python commands must go through `uv` and `.venv/bin/python`." The documented environment is
`uv venv --python 3.12 --seed --managed-python` (`$VLLM_SRC/docs/getting_started/installation/python_env_setup.inc.md`),
and CI runs Python 3.12, so match it.

Keep this **separate from `/workspace/venv`**. That venv is what [scripts/env.sh](../scripts/env.sh)
activates and what [scripts/serve.sh](../scripts/serve.sh) runs; overwriting it with a development
build breaks every lab in the project.

```bash
cd /workspace
git clone git@github.com:awesome-pro/vllm.git src/vllm-dev   # or https://github.com/awesome-pro/vllm.git
cd src/vllm-dev
git remote add upstream https://github.com/vllm-project/vllm.git

uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

uv pip install -r requirements/lint.txt
pre-commit install

VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

Check what you are actually importing before you believe anything:

```bash
.venv/bin/python -c "import vllm; print(vllm.__version__, vllm.__file__)"
```

`vllm.__file__` must point inside `src/vllm-dev/vllm/`, not into `site-packages`. `HF_HOME` should
still be `/workspace/hf` so the model ladder is not re-downloaded.

One architecture fact worth carrying into every debugging session from here: **"V1" is the engine
generation, MRV1/MRV2 are the model-runner layers inside it.** MRV2 is the default; MRV1 is deprecated
and slated for removal in v0.32. The concrete tell is the module shape — MRV1 is the flat file
`vllm/v1/worker/gpu_model_runner.py`, MRV2 is the package `vllm/v1/worker/gpu/model_runner.py`. When a
feature "does not work" on `main`, check whether a fallback message pushed you onto MRV1
(`vllm/config/vllm.py`).

---

## 3. Building from source: what actually compiles, and what it costs

**What compiles.** The kernels in `csrc/` plus the FlashAttention/MoE extensions, producing
`_C.abi3.so`, `_moe_C.abi3.so`, `cumem_allocator.abi3.so`, `vllm-flash-attn`, and friends — see the
build-directory listing in `$VLLM_SRC/docs/contributing/incremental_build.md`. The Rust frontend is
separate: `./tools/build_rust.sh` (or `--debug` for a faster build). **Pure Python changes compile
nothing**; with an editable install they are live immediately.

**The knobs that matter.**

| Knob | Effect |
| --- | --- |
| `VLLM_USE_PRECOMPILED=1` | Skip compilation entirely; fetch prebuilt wheels matching your merge-base with `main` |
| `VLLM_TARGET_DEVICE` | Choose the backend (`cuda` is auto-detected; `empty` installs without compiling anything, for import-only dev — `setup.py`) |
| `MAX_JOBS` | Cap parallel compilation jobs (env var, not a build arg) — the fix for OOM during a build |
| `--build-arg max_jobs=N`, `--build-arg nvcc_threads=N` | The same controls for the Docker image (`docker/Dockerfile`) |
| `--build-arg torch_cuda_arch_list=""` | Delegate architecture selection to PyTorch instead of building for every GPU |
| `--build-arg VLLM_USE_PRECOMPILED=1`, `--build-arg CUDA_VERSION=…` | Precompiled image build; CUDA version selection |

For repeated kernel work, do not rebuild everything: the CMake workflow
(`python tools/generate_cmake_presets.py`, then `cmake --preset release` and
`cmake --build --preset release --target install`) rebuilds only what changed and installs the shared
libraries back into your editable tree. Install `ccache` — the preset sets it as the compiler
launcher.

**The cost, concretely.** Upstream's own worked example for an aarch64 image build is "~15 GB memory,
~1475 s / ~25 min, image size 6.93 GB" with `max_jobs=66`. *Prediction:* on the pod, a full `csrc/`
build is tens of minutes of wall clock, and disk is the real risk — the setup doc budgets ~1 GB for the
source clone, ~10 GB for a venv and ~3 GB for the uv cache, and a CMake build tree plus object files
adds several GB more. The container disk is 30 GB. Build under `/workspace`, keep the build tree out
of the 60 GB volume's remaining headroom (the model ladder alone is 24.2 GB), and check with
`du -sh /workspace/*` before you assume you have room.

Cost in money: every hour of compiling is $0.34 on the Community 4090. Cheap for one build, expensive
as a habit — a forgotten pod that compiles overnight costs ~$2.70 and produces nothing. Do a build
session, then **stop the pod**.

---

## 4. Running the test suite

**Find the test the way CI finds it.** Each `.buildkite/test_areas/*.yaml` job lists
`source_file_dependencies` — the paths that trigger it — and the `pytest` commands it runs. Search
those files for the path you changed and you have both the right test file and the command CI uses.
Example: the "V1 Core" job in `.buildkite/test_areas/misc.yaml` depends on `tests/v1/core` and runs
`pytest -v -s -m 'not cpu_test' v1/core`.

**A realistic first target.** `tests/v1/core/test_scheduler.py` exists in the checkout, contains 136
test functions, and builds `VllmConfig`/`SchedulerConfig` objects with mocks rather than launching a
model — so it is readable, fast, and a good first "did I break the scheduler?" check. Others in the
same area: `test_async_scheduler.py`, `test_prefix_caching.py`, `test_kv_cache_utils.py`,
`test_single_type_kv_cache_manager.py`.

```bash
uv pip install -r requirements/test/cuda.in            # AGENTS.md's test dependencies
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py -v
```

**Unit tests vs GPU tests.** The markers are declared in `pyproject.toml`
(`[tool.pytest.ini_options]`): `slow_test`, `skip_global_cleanup`, `core_model`, `hybrid_model`,
`cpu_model`, `cpu_test`, `split`, `distributed` ("run this test only in distributed GPU tests"), and
`optional` (skipped unless you pass `--optional`). Tests are organised by area, not by kind:
`tests/v1/core` and `tests/v1/engine` are mostly Python-level; `tests/v1/e2e`, `tests/v1/distributed`,
`tests/kernels`, and `tests/models` need real GPUs, and CI gives them device labels (`device: l4`,
`device: h200_18gb`, `num_devices: 2`).

**Reality on one 4090.** You cannot run the suite. The honest loop is: run the unit file that covers
your change, then one targeted e2e run on the pod with a small model, and say in the PR which you ran.

---

## 5. Finding something to work on

The repo keeps a job board in `$VLLM_SRC/docs/contributing/README.md`: **good first issues** (plus the
"selected onboarding tasks" project board) and **new model requests**. "Good first issue" means a
maintainer has agreed the scope is bounded and will help — not that it is trivial.

Before writing any code, run upstream's duplicate-work checks (they are mandatory, `AGENTS.md` §1):

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

Then check the bug still exists on `main` before you touch it:

```bash
cd ~/Desktop/vllm
git fetch upstream && git log --oneline upstream/main -5
git log --oneline upstream/main -- vllm/v1/core/sched/scheduler.py | head
```

**Reproducing and reporting is a contribution.** A report with the version, the exact command, the
model, the GPU, the full startup log and the error is worth more than a speculative patch. Minimum
evidence: `vllm --version` (or `uv pip show vllm`), the complete `vllm serve` command line, the model
id and `--max-model-len`, the GPU and driver, reproduction steps, and what you expected instead. Two
rules from `AGENTS.md` apply to everything you submit: no "pure agent" PRs — a human must be able to
defend every changed line — and no one-line busywork PRs; mechanical cleanups ride along with
substantive work or not at all.

---

## 6. The fork workflow with your real remotes

Your fork is `~/Desktop/vllm` (this Mac), on `main`:

```
origin    git@github.com:awesome-pro/vllm.git      ← your fork
upstream  https://github.com/vllm-project/vllm.git ← the project
```

```bash
cd ~/Desktop/vllm
git fetch upstream
git switch -c fix/scheduler-priority-eviction main
git rebase upstream/main                     # keep the branch on top of the moving target
git push -u origin fix/scheduler-priority-eviction
```

On the pod, work in the same clone shape (`origin` = fork, `upstream` = project) so `git fetch
upstream && git rebase upstream/main` behaves identically there. Branch names are yours; PR **titles**
are not — they must start with bracketed tags, e.g. `[Bugfix][Core] Fix priority handling` or
`[Perf][Kernel] …` (type tags `[Bugfix]`, `[Feature]`, `[Perf]`, `[Refactor]`, `[CI]`, `[Test]`,
`[Doc]`, `[Misc]`; scope tags `[Model]`, `[Core]`, `[Kernel]`, `[Attention]`, `[MoE]`, `[Spec Decode]`,
`[KV Offload]`, `[Hardware][Vendor]`, …).

**Commits.** The project uses the DCO: every commit needs a `Signed-off-by:` line, which
`git commit -s` adds. AI-assisted work needs a `Co-authored-by:` (or `Assisted-by:`) trailer *plus* a
disclosure in the PR description:

```text
Fix block accounting on preemption

Co-authored-by: <assistant name>
Signed-off-by: Your Name <your.email@example.com>
```

**Hooks and linting.** `uv pip install -r requirements/lint.txt && pre-commit install` makes the hooks
run on every commit. The ones you will meet: `ruff-check`, `ruff-format` (Python line length is 88),
`typos`, `clang-format`, `markdownlint-cli2`, `shellcheck`, `check-json`, `check-spdx-header`,
`check-filenames`, `signoff-commit`, `validate-config`, and `mypy-3.10`…`mypy-3.13` (CI-only stage —
`pre-commit run --hook-stage manual mypy-3.12`). `pre-commit run -a` runs everything; use it before
pushing. Code style is Google's, with Google-style docstrings (`Args:`/`Returns:`/`Raises:`), not
Sphinx fields.

**The PR.** Target `main`; the description must let a stranger reproduce you: what changed, why, the
commands you ran and their results, model-eval results if output or accuracy could move, a statement
that you checked for duplicates, and an AI-assistance note if applicable. Changes over ~500 LOC
(excluding kernels/data/config/tests) need an RFC issue first. Contributors without write access are
capped at 6 open PRs. CI does not start on its own — a trusted reviewer comments `/ci run`, and the
build is **refused unless your branch is zero commits behind** the target; `--allow-stale` exists but
is a knowing risk.

---

## 7. Where to add a feature

The payoff of the whole project: given an idea, this is where it goes and what the minimum surface is.

| Goal | Where | Minimum surface |
| --- | --- | --- |
| **New model** | `vllm/model_executor/models/`, then register in `_VLLM_MODELS` in `vllm/model_executor/models/registry.py` (alphabetical) and list it in `$VLLM_SRC/docs/models/supported_models.md` | Every module takes `prefix=`; `embed_input_ids()`; a flattened `forward(input_ids, positions, intermediate_tensors, inputs_embeds)`; `load_weights()`. Reuse `Attention`/MoE layers rather than reimplementing them (`$VLLM_SRC/docs/contributing/model/basic.md`, `registration.md`) |
| **TP/PP support for a model** | The model's HF config class | `base_model_tp_plan` (colwise/rowwise per weight) and `base_model_pp_plan` |
| **New attention backend** | Subclass `AttentionBackend` (`vllm/v1/attention/backend.py`), register in `vllm/v1/attention/backends/registry.py` | The three abstract methods `get_name()`, `get_impl_cls()`, `get_builder_cls()`, plus a metadata builder; override the `supports_*` / `validate_configuration` hooks that apply; out-of-tree backends go through `register_backend()` |
| **New logits processor** | `vllm/v1/sample/logits_processor/` (`builtin.py` has Min-P, logit-bias and min-tokens as worked examples) | Subclass `LogitsProcessor`: `__init__(vllm_config, device, is_pin_memory)`, `apply(logits)`, `is_argmax_invariant()`, `update_state(batch_update)`. Users supply their own with `--logits-processors` (`ModelConfig.logits_processors`) |
| **Out-of-tree hardware or feature** | A separate package using the plugin system (<https://docs.vllm.ai/en/stable/design/plugin_system/>, in-tree `$VLLM_SRC/docs/design/plugin_system.md`) | An `entry_points` group: `vllm.general_plugins` to register models, `vllm.platform_plugins` for a `Platform`, `vllm.io_processor_plugins`, `vllm.stat_logger_plugins`, or `vllm.endpoint_plugins` (opt-in). The hook must be re-entrant |
| **Custom kernel** | A `CustomOp` subclass in your layer file (`vllm/model_executor/custom_op.py`; <https://docs.vllm.ai/en/stable/design/custom_op/>) | `@CustomOp.register("name")`, a `forward_native()` fallback and the platform variant (`forward_cuda()` etc.); enable with `--compilation_config.custom_ops '["+name"]'` |

The boundary rule: put it **in-tree** if it belongs to the core engine and adds no dependency; build
it as a **plugin** if it is hardware-specific, opinionated, or you would rather not chase `main` every
two weeks.

---

## 8. The maintenance reality

Upstream ships roughly every two weeks, and `main` never stops moving. Two structural facts to plan
around:

- **The deprecation policy is a three-stage pipeline across minor releases**
  (<https://docs.vllm.ai/en/stable/contributing/deprecation_policy/>; in-tree
  `$VLLM_SRC/docs/contributing/deprecation_policy.md`): deprecated-but-on (with a stated removal
  version) → off by default and erroring unless re-enabled → removed. No removals in patch releases.
  Any flag you build on can be gone in two releases, with warning.
- **Features do get removed.** V0 is already gone ("We have fully deprecated V0",
  `$VLLM_SRC/docs/usage/v1_guide.md`), and MRV1 is slated for removal in v0.32 while still serving as
  the fallback for sequence parallelism, dual-batch overlap, some speculative-decoding methods, and
  other paths (`vllm/config/vllm.py`). A patch against a code path that is being deleted will not be
  reviewed — check `git log upstream/main` for the area first.

The durable strategy is therefore **small, single-concern, rebase-friendly PRs.** Rebase on
`upstream/main` before asking for review (CI enforces zero-commits-behind), and if a branch has drifted
for two weeks, rewrite it rather than rebasing: the conflicts mean the change was not small enough.

---

## 9. ✅ Checkpoint

Answer these out loud, from your own pod session:

1. Which install gives you a working `vllm serve` **and** an editable tree, and which file do you check
   to prove the import is coming from your clone?
2. What does `VLLM_USE_PRECOMPILED=1` skip, what does it fetch instead, and when is that the wrong
   choice?
3. Your change touches `vllm/v1/core/sched/scheduler.py`. Which test file do you run first, and which
   `.buildkite/test_areas/*.yaml` job tells you what CI will run?
4. Walk the fork workflow from a dirty `main` to an open PR: `git remote -v`, fetch, rebase, branch,
   `git commit -s`, pre-commit, push, PR title, and the two things the description must contain.
5. You want to add a feature. Give the decision rule between in-tree code, a `CustomOp`, and a plugin —
   and name the minimum surface for the one you chose.

---

## Where to go next

- Back to [`CURRICULUM.md`](../CURRICULUM.md) — the "how you know you are done" list.
- Upstream's own entry point: <https://docs.vllm.ai/en/stable/contributing/>.
- The in-tree guides you will actually use: `$VLLM_SRC/docs/contributing/README.md`,
  `$VLLM_SRC/docs/contributing/incremental_build.md`, `$VLLM_SRC/docs/contributing/model/basic.md`,
  `$VLLM_SRC/docs/contributing/deprecation_policy.md`, and `AGENTS.md`.

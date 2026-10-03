# 03 — Renting the pod (RunPod setup)

Everything here was checked against RunPod's docs and Docker Hub on **2026-10-02**. GPU prices and
image tags move; re-check the price before a long session.

Read this once end to end (~10 min), then use the [session checklist](#session-checklist) every
time after that.

---

## 1. The decision you must make first: storage

This is where the money is, and there is a trap in it.

> **Network volumes are Secure Cloud only.** (`docs.runpod.io/storage/network-volumes`:
> "Network volumes are only available for Pods in the Secure Cloud.")

So the cheapest GPU and the cheapest storage are **mutually exclusive**. Pick one of these three
plans:

| | Plan A — volume disk *(recommended)* | Plan B — network volume | Plan C — nothing |
| --- | --- | --- | --- |
| Cloud | **Community** | **Secure** | Community |
| Compute | RTX 4090 @ **$0.34/hr** | RTX 4090 @ **$0.74/hr** | RTX 4090 @ $0.34/hr |
| Storage | Volume disk, 60 GB | Network volume, 60 GB | Container disk only |
| Storage rate | $0.10/GB/mo running, **$0.20/GB/mo stopped** | **$0.07/GB/mo**, always | $0 |
| Survives *stop* | ✅ | ✅ | ❌ (wiped) |
| Survives *terminate* | ❌ | ✅ | ❌ |
| Portable to another pod | ❌ (glued to that pod) | ✅ (region-locked) | n/a |
| **39 lab-hours/month** | **≈ $25/mo** | **≈ $33/mo** | **≈ $13/mo** |

The 39-hour figure is 3 sessions/week × 3 hours. The arithmetic:

```
Plan A   compute 39 h × $0.34              = $13.26
         storage 60 GB running  39 h       ≈  $0.32
         storage 60 GB stopped 691 h       ≈ $11.36   →  ≈ $24.94/mo
Plan B   compute 39 h × $0.74              = $28.86
         storage 60 GB × $0.07             =  $4.20   →  ≈ $33.06/mo
Plan C   compute 39 h × $0.34              = $13.26   →  ≈ $13.26/mo
```

**Take Plan A.** It is ~$8/month cheaper than Plan B at this usage level, and the compute you get
for $0.34/hr is identical. Its two real costs:

1. **Never terminate the pod** — terminating destroys the volume disk. Use **Stop**.
2. **You cannot move the disk.** If Community capacity for a 4090 is gone when you want to resume,
   you wait or deploy elsewhere and lose `/workspace`. Mitigate by keeping code and notes in git
   (see §7) — models are always re-downloadable.

Choose **Plan B** instead if you would rather have terminate-survival and reserved capacity, and
do not mind paying 2.2× for compute. Choose **Plan C** if you are disciplined about git and want to
spend the least — it costs nothing but ~8 minutes of re-setup per session.

**How big?** 60 GB holds everything in this guide: the vLLM source clone (~1 GB), the venv
(~10 GB), the uv cache (~3 GB), and the model ladder (0.6B ≈ 1.5 GB + 4B ≈ 8 GB + 8B ≈ 16.4 GB).
Reserve 100 GB only if you plan to pull the 30B MoE's FP8 weights (~30 GB) for Stage 10.

---

## 2. Create the pod

Console → **Pods** → **Deploy**.

| Setting | Value | Why |
| --- | --- | --- |
| Cloud | **Community** (Plan A/C) or Secure (Plan B) | $0.34 vs $0.74 per hour |
| GPU | **RTX 4090, 24 GB** | Ada = FP8 W8A8 support; best value in the catalogue |
| Image | **`runpod/pytorch:1.4.0-cu1300-torch291-ubuntu2404`** | CUDA 13.0 + torch 2.9.1 + Ubuntu 24.04, ships SSH and JupyterLab. Matches the CUDA 13.0 build that vLLM 0.30.0's PyPI wheel actually is (see §3a). |
| Container disk | **30 GB** | Holds the image plus scratch. Wiped on stop. |
| Volume disk | **60 GB** (Plan A) | Everything important lives on `/workspace`. |
| Expose HTTP ports | **8888**, **8000** | 8888 = JupyterLab (RunPod starts it for you), 8000 = your vLLM server |
| Expose TCP port | **22** (optional) | Direct SSH. Needs a public IP; enables `rsync`/`scp`, which the proxy does not support. |
| Env var | `HF_HOME=/workspace/hf` | Keeps model weights on the persistent disk, not the container. |
| Env var | `HF_TOKEN=…` (optional) | Only needed for gated models. The Qwen3 ladder is not gated. |

Use the **RunPod PyTorch** template as the starting point and override its image tag with the value
above. The image string is the part that matters.

> **Your tag may be newer than the table's, and that is fine.** The template currently offers
> `runpod/pytorch:1.4.0-rc.164-cu1300-torch2130-ubuntu2404` (template `runpod-torch-v280`) — an `rc`
> build carrying torch 2.13.0 rather than 2.9.1. It works, because what has to match is the **CUDA major
> version (13.0)** and your driver: `bootstrap.sh` builds its own venv and installs its own torch, so the
> image's preinstalled torch is never imported by anything you run. An `rc` tag is just one more variable
> to eliminate if CUDA misbehaves (§6a), which is the only reason the pinned tag is preferred.

**Then verify you actually got the volume disk.** This takes ten seconds and is easy to miss, because
*Pod details* lists **Volume disk** on its own line, separately from *Container disk* — and `/workspace`
exists either way. From inside the pod:

```bash
df -Pk /workspace / | awk '{print $1, $6}'   # same device as / => no volume disk attached
```

If both lines name the same filesystem, `/workspace` is on the container disk and everything this guide
puts there is temporary. `scripts/bootstrap.sh` runs this same check in step 1 and warns you. **A volume
disk cannot be attached to a running pod**, so if you meant Plan A and skipped it, redeploy now — before
you download 9 GB of weights.

> **Why not the official `vllm/vllm-openai:v0.30.0` image?** It is excellent and it is what you would
> deploy in production, but its entrypoint *is* the API server: you get a working OpenAI endpoint
> and no comfortable way to edit source, run `pytest`, or use VS Code. This guide is about
> understanding and changing vLLM, so it starts from a dev image and installs vLLM into a venv you
> control. Lab 00 shows you how to run the official image later as a one-off comparison.

After **Deploy**, wait for the pod to reach *Running*, then open the **web terminal** (Connect →
Web Terminal). That is the fastest way in, and it is enough to run the bootstrap.

---

## 3. Bootstrap (once per pod, ~6 min)

In the web terminal:

```bash
cd /workspace
git clone --depth 1 --branch v0.30.0 https://github.com/vllm-project/vllm.git src/vllm
# copy this guide in (see §7 for the three ways), then:
bash /workspace/vlearning/scripts/bootstrap.sh
```

`scripts/bootstrap.sh` installs `uv`, creates `/workspace/venv` on the **persistent** disk, installs
`vllm` matching your driver's CUDA, writes the environment exports into `~/.bashrc`, clones the
source, pre-downloads the model ladder, and finishes with a GPU sanity check. It is idempotent.

### What survives a stop, and what does not

RunPod is explicit about this: the **container disk is "cleared when the Pod stops"**, while the
**volume disk at `/workspace` is "retained until the Pod is deleted"**. That has three consequences
this project is built around:

| Thing | Where it lives | Survives a stop? |
| --- | --- | --- |
| Model weights (`$HF_HOME`) | `/workspace/hf` | ✅ |
| The vLLM venv (`/workspace/venv`) | `/workspace` | ✅ |
| uv's managed Python + wheel cache | `/workspace/uv-python`, `/workspace/uv-cache` | ✅ (bootstrap puts them here deliberately) |
| The vLLM source clone | `/workspace/src/vllm` | ✅ |
| `~/.bashrc` exports, `uv` itself, `~/.ssh` | container disk | ❌ |

So after a **stop/start** (not a terminate), your first command each session is:

```bash
cd /workspace/vlearning && source scripts/env.sh
```

That one line re-establishes every path and activates the venv, and it does not depend on
`~/.bashrc` surviving. If something still looks broken, re-run `bash scripts/bootstrap.sh` — it will
restore `uv` and the shell exports without re-downloading models or reinstalling vLLM, because the
venv and the interpreter it points at are both on the volume.

Log out and back in (or `source ~/.bashrc`) after the first bootstrap so the environment is live,
then:

```bash
bash /workspace/vlearning/labs/00_verify_install.sh
```

If that prints a GPU name, a vLLM version, and a short completion, you are done. **Stage 0 is
finished.** Everything else is learning.

---

## 3a. CUDA 13.0 vs 12.9 — read this before your first `pip install`

This is the one place where vLLM's own documentation is actively misleading, so here is the
verified position (checked 2026-10-02, two independent ways):

- vLLM's in-tree install docs say *"vLLM's binaries are compiled with CUDA 12.9 … by default"*.
  **That is stale for the 0.30.0 PyPI wheel.** It still describes the nightly and source-build
  default (`VLLM_MAIN_CUDA_VERSION`).
- The actual evidence: the PyPI metadata for `vllm==0.30.0` requires
  `nvidia-cutlass-dsl[cu13]==4.7.1`, and a `uv pip install --dry-run` for Linux x86_64 with
  `--torch-backend=cu130` resolves `torch==2.13.0+cu130`, `nvidia-cuda-runtime==13.0.96`,
  `nvidia-cudnn-cu13==9.20.0.48`. **The wheel is a CUDA 13.0 build.**

The trap: `uv pip install vllm==0.30.0 --torch-backend=cu129` resolves *happily* — it just installs
`torch 2.13.0+cu129` next to a CUDA-13 vLLM. The resolver cannot see CUDA ABI, so you get a broken
stack that fails at runtime, not at install time. `bootstrap.sh` picks the coherent pair for you.

| Stack | Image | Install | Driver needed |
| --- | --- | --- | --- |
| **A (default)** | `runpod/pytorch:1.4.0-cu1300-torch291-ubuntu2404` | `uv pip install vllm==0.30.0 --torch-backend=cu130` | **R580+** |
| B (fallback) | `runpod/pytorch:1.4.0-cu1290-torch291-ubuntu2404` | the `vllm-0.30.0+cu129-…x86_64.whl` release asset, `--torch-backend=cu129` | R535+ |

`bootstrap.sh` reads the driver version and chooses A or B automatically (`TORCH_BACKEND=cu130` or
`cu129` overrides it). Note that CUDA forward-compatibility mode — which lets a new CUDA runtime run
on an older driver — covers only **select professional and datacenter GPUs**. A consumer RTX 4090
cannot use it, so on that card stack A genuinely requires a new driver; it is not a soft warning.

---

## 4. Connecting properly: SSH and VS Code

**Web terminal** — zero setup, works immediately. Fine for bootstrap, awkward for real editing.

**SSH via the proxy** — no public IP needed. From Connect → SSH:

```bash
ssh <pod-id>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

The proxy does **not** support `scp`/`sftp`. It is still fine for VS Code Remote-SSH.

**SSH direct (recommended if you enabled TCP 22)** — the pod's public IP and mapped port:

```bash
ssh root@<public-ip> -p <mapped-port>
```

This also gives you `rsync`/`scp`, which is how you move this guide onto the pod (§7).

### VS Code Remote-SSH

1. Local: `ssh-keygen -t ed25519` if you have no key, then paste the **public** key into the pod's
   `~/.ssh/authorized_keys` (RunPod also lets you register keys account-wide, which is easier).
2. Local `~/.ssh/config`:

```
Host runpod-4090
    HostName ssh.runpod.io
    User <pod-id>-<hash>
    IdentityFile ~/.ssh/id_ed25519
    # If you enabled TCP 22, use the direct form instead:
    # HostName <public-ip>
    # Port <mapped-port>
    # User root
```

3. VS Code → Remote-SSH → **Connect to Host…** → `runpod-4090`.
4. Install the Python extension **on the remote**, and point it at `/workspace/venv/bin/python`.
   Now `import vllm` resolves, and you can `Ctrl-click` from a lab straight into vLLM's source.
   That single navigation trick is most of what makes this project work.

---

## 5. Reaching the vLLM server

`scripts/serve.sh` binds `0.0.0.0:8000`, which RunPod proxies to:

```
https://<pod-id>-8000.proxy.runpod.net
```

Two gotchas that will otherwise cost you an hour:

- **Bind `0.0.0.0`, not `127.0.0.1`.** vLLM's default host is loopback; from the proxy that is a
  `502 Bad Gateway`.
- **The proxy has a ~100 s timeout** (Cloudflare, HTTP 524). Long non-streaming generations can hit
  it. Every lab that generates at length uses `stream: true`, which is good practice anyway.

For anything public, set an API key (`VLLM_API_KEY=…` in the environment, or `--api-key`). Note
that it only guards the OpenAI-style routes — it is not a substitute for keeping the URL private.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `502 Bad Gateway` from the proxy URL | Server bound to loopback, still loading the model, or wrong port | Check `--host 0.0.0.0`, watch the pod logs for the load line, confirm port 8000 is exposed |
| `CUDA out of memory` at startup | Model + KV does not fit, or `--gpu-memory-utilization` too high | Drop `--max-model-len`, drop to `$MODEL_MID`, or lower `--gpu-memory-utilization` to 0.85 |
| Startup hangs for minutes | `torch.compile` + CUDA graph capture at `-O2` | Expected on first run; for a fast loop use `-O0` or `--enforce-eager` while experimenting |
| `nvcc: not found` during a JIT compile | The image lacks the CUDA toolkit for that kernel | Prefer a prebuilt backend: `VLLM_ATTENTION_BACKEND=FLASH_ATTN` |
| `CUDA driver version is insufficient for CUDA runtime version`, or `libcudart.so.13: cannot open shared object file` | You paired a CUDA-13 vLLM with a CUDA-12 torch (or the image's CUDA does not match the wheel) | Re-run bootstrap (it detects and fixes this), or force it: `TORCH_BACKEND=cu129 bash scripts/bootstrap.sh`. See §3a. |
| `nvidia-smi` works and the device count says `1`, but Python raises `CUDA unknown error` or `Error 804: forward compatibility was attempted on non supported HW` | NVML can see the card; the CUDA **runtime** cannot create a context on it — on a GeForce pod this is almost always RunPod's forward-compatibility shim | §6a. Do **not** reinstall anything — the install is not the problem. |
| Disk full | Container disk filled with model weights | Set `HF_HOME=/workspace/hf` and re-pull; check `du -sh /workspace/*` |
| Pod will not restart on Community | No 4090 capacity in that datacenter right now | Wait, or deploy a new pod and re-run bootstrap (Plan A's known weakness) |
| Weights re-download every session | `HF_HOME` pointed at the container disk | It must be on `/workspace`; verify with `echo $HF_HOME` |
| `venv/bin/python: No such file or directory` after a restart | uv's managed interpreter was on the container disk, which RunPod clears on stop | Re-run `bash scripts/bootstrap.sh` (current version keeps the interpreter in `/workspace/uv-python`) |
| `uv: command not found` after a restart | Same cause — `uv` itself lives in the container | Re-run `bash scripts/bootstrap.sh`, or `source /workspace/vlearning/scripts/env.sh` if the venv is intact |
| My notes/edits are gone but the models are still there | You edited files outside `/workspace` | Only `/workspace` persists. Keep the guide and your `notes/` in `/workspace/vlearning` and push to git |
| Everything is slow and `nvidia-smi` shows no processes | You are on the *pod's* CPU, not the GPU — e.g. a CPU-only pip install | Re-run `labs/00_verify_install.sh`; confirm vLLM reports `device_type='cuda'` |

### 6a. `nvidia-smi` works but CUDA does not

This is the failure that looks most alarming and is least serious. From `scripts/bootstrap.sh` step 7:

```
nvml    : 1 device(s) visible to nvidia-smi/NVML
runtime : FAILED - RuntimeError: CUDA unknown error - this may be due to an
          incorrectly set up environment, e.g. changing env variable
          CUDA_VISIBLE_DEVICES after program start. Setting the available
          devices to be zero.
```

Read it as **two separate facts from two separate libraries**, because that is what it is:

- The device count is answered by **NVML**, the management library `nvidia-smi` also uses. It is working.
- `torch.cuda.init()` failed because **`cudaGetDeviceCount()` returned an error code** — raise site is
  `device_count_impl`, PyTorch's `c10/cuda/CUDAFunctions.cpp`, which is where the wording comes from.

The error code varies with how the driver handshake breaks, and it is the most informative part of the whole
message. Two turn up on RunPod pods:

| Code | Text | What it means |
| --- | --- | --- |
| 999 | `CUDA unknown error - this may be due to an incorrectly set up environment …` | A generic driver-side failure; the usual causes are the forward-compat shim below, or a host whose GPU attach is broken |
| **804** | `forward compatibility was attempted on non supported HW` | **Specific and good news**: the forward-compat shim is loaded and is the entire problem — fix it with fix 1 below and nothing else |

What you did **not** get is just as informative:

| Error you did not get | Cause it rules out |
| --- | --- |
| `The NVIDIA driver on your system is too old (found version …)` | a driver too old for CUDA 13 — yours is 580 and advertises CUDA 13.0 |
| `Found no NVIDIA driver on your system` | a missing driver library in the container |
| `No CUDA GPUs are available` / `device_count 0` | no GPU attached to the pod |
| `ImportError` on `import torch` | a broken install — **reinstalling will not help** |

Three fixes, cheapest first.

**1. The RunPod forward-compatibility shim — by far the most likely cause on a GeForce pod.** This one is
worth understanding properly, because the obvious diagnosis is wrong. RunPod injects
`/etc/ld.so.conf.d/00-compat-<uuid>.conf`, which puts the image's bundled driver ahead of the host driver in
the **dynamic-loader cache**:

```
libcuda.so.1 => /usr/local/cuda-13.0/compat/libcuda.so.580.178.04   <- wins
libcuda.so.1 => /lib/x86_64-linux-gnu/libcuda.so.580.173.02         <- the host driver
```

A user-space driver *newer* than the kernel driver puts CUDA into **forward compatibility** mode, which NVIDIA
supports only on datacenter GPUs. On a GeForce card every CUDA call then fails with error **804**:

```
Error 804: forward compatibility was attempted on non supported HW
```

Two details make this trap nasty. `nvidia-smi` keeps working, because it uses **NVML**, not `libcuda` — so the
card looks healthy. And `LD_LIBRARY_PATH` is usually **empty** on these pods, so nothing you unset there will
help; the shim is in `ld.so.conf`. Confirm and fix it with:

```bash
ldconfig -p | grep 'libcuda\.so'        # compat listed first => this is your bug
grep -rl /compat /etc/ld.so.conf.d/     # the conf file responsible

mkdir -p /root/disabled-ld-so-conf
mv "$(grep -rl /compat /etc/ld.so.conf.d/)" /root/disabled-ld-so-conf/
ldconfig
ldconfig -p | grep 'libcuda\.so'        # now only /lib/x86_64-linux-gnu, at the host's version
```

**`scripts/bootstrap.sh` does this for you** in step 1, automatically, whenever the GPU is a GeForce. It is
conditional on purpose: on a datacenter card forward compatibility *is* supported, and when the host driver
genuinely is older than the toolkit the shim is doing a necessary job, so removing it there could break CUDA
rather than fix it.

On the pod this was verified on, the numbers were: shim loaded (580.178.04) → `torch.cuda.init()` raised error
804; shim removed → `torch.cuda.init()` succeeded and reported `22.7 GiB free / 23.5 GiB total`, capability
`(8, 9)`. No env var, no reinstall, no redeploy.

**2. A glitched GPU attach.** Stop the pod and start it again — restarting the container is not always
enough, because the NVIDIA runtime hooks run at container start. Then re-run `bash scripts/bootstrap.sh`.
If it fails on a second host as well, the problem is not your pod.

**3. Do not fight CUDA 13 — skip it:**

```bash
TORCH_BACKEND=cu129 bash scripts/bootstrap.sh
```

That installs the published CUDA 12.9 wheel paired with a CUDA 12 torch, which runs on the same 580
driver. You lose nothing for this guide: CUDA 13 versus 12.9 changes no vLLM concept, no lab, and no
number you record. It is the same fallback §3a uses for drivers older than 580.

---

## 7. Getting this guide onto the pod (and your notes back)

This project is a **public** git repo: `awesome-pro/vllm-learning`. Public means the pod can clone it
with **no credentials at all**, as long as you use the **HTTPS** URL:

```bash
cd /workspace
git clone https://github.com/awesome-pro/vllm-learning.git vlearning
```

Name the directory `vlearning` — every command in this guide assumes `/workspace/vlearning`.

> **Do not use the `git@github.com:` (SSH) URL on a fresh pod.** SSH asks GitHub to authenticate *you*,
> and a new pod has no key registered, so it fails with `git@github.com: Permission denied (publickey)`
> before GitHub ever looks at the repository. That error says nothing about whether the repo exists.

**Pushing your notes back.** Reading needs no credentials; pushing needs write access — and there is a
trap. `~/.ssh` lives on the **container disk**, which RunPod clears on every stop, so a key you generate
there disappears. Keep it on the volume and tell git where to find it:

```bash
ssh-keygen -t ed25519 -f /workspace/.ssh/id_ed25519 -N "" -C "runpod"
cat /workspace/.ssh/id_ed25519.pub     # GitHub -> Settings -> SSH and GPG keys -> New SSH key
cd /workspace/vlearning
git remote set-url origin git@github.com:awesome-pro/vllm-learning.git
git config core.sshCommand "ssh -i /workspace/.ssh/id_ed25519 -o IdentitiesOnly=yes"
```

`core.sshCommand` is stored in `.git/config`, which is on `/workspace`, so it outlives the container disk
even though the key's usual home does not.

> **The repo is public, so your notes publish with it.** Never commit a RunPod API key, a HuggingFace
> token, or `VLLM_API_KEY`. If you would rather it were private, weigh the trade first: a private repo
> makes the pod's HTTPS clone demand a token, so you would need one of the credential setups above just
> to *read* the guide.

Two alternatives if you would rather not use git at all:

1. **rsync over direct SSH** (needs TCP 22 + public IP):

   ```bash
   rsync -avz --exclude '.git' -e "ssh -p <port>" \
     ~/Desktop/vlearning/ root@<public-ip>:/workspace/vlearning/
   ```

2. **`runpodctl send`** from a pod-side terminal, which prints a one-line receive command. Works over
   the web terminal when you have no direct SSH.

---

## 8. Session checklist

**Start (2 min):**

- [ ] Deploy/start the pod, wait for *Running*
- [ ] `cd /workspace/vlearning && source scripts/env.sh` — **required after every restart**, because
      RunPod clears the container disk (and therefore `~/.bashrc`) when a pod stops
- [ ] `bash labs/00_verify_install.sh` — driver, version, GPU, one generation
- [ ] Start the server if the stage needs it: `bash scripts/serve.sh` (own terminal)
- [ ] `nvidia-smi` once, so you know the baseline memory

**Work:**

- [ ] One stage. Read the source paths before running the lab.
- [ ] Record every number in `notes/` — including the version and the date.

**End (1 min):**

- [ ] Commit and push `notes/` (or copy it home)
- [ ] **Stop the pod** — Stop, never Terminate on Plan A
- [ ] Confirm on the Pods page that it says *Stopped*, and check your credit balance

> A 4090 left running while you sleep costs ~$2.70 by morning, and a 60 GB volume disk that you
> forgot about costs $12/month. The single most expensive habit in this project is a pod you did
> not stop.

### If you chose Plan C (no volume disk)

Then nothing above survives a stop, and the pod is disposable — there is no `/workspace` worth
protecting. Your session ritual becomes four commands, ~10 minutes including the model download:

```bash
cd /workspace
git clone https://github.com/awesome-pro/vllm-learning.git vlearning   # public: no credentials
bash vlearning/scripts/bootstrap.sh     # ~6 min install, then prefetches tiny+mid (~9 GB)
cd vlearning && source scripts/env.sh
bash labs/00_verify_install.sh
```

Two consequences worth internalising:

- **Keep your notes on the Mac, not the pod.** Every lab ends by printing a `RECORD:` block, which is
  designed to be pasted into `notes/` on your laptop and committed from there. This also sidesteps the
  one thing Plan C makes awkward: getting push credentials onto a pod that forgets `~/.ssh` along with
  everything else.
- **Stopping is free and loses nothing**, so stop aggressively. The "Stop, never Terminate" rule in §1
  exists to protect a volume disk; with no volume disk there is nothing to protect — but a 4090 still
  costs ~$0.34/hour while it idles.

The ritual's real cost is the download, roughly 2–4 minutes on RunPod's network. If you only need the
small model today, cut it with `LADDER=tiny bash vlearning/scripts/bootstrap.sh` (1.5 GB instead of 9 GB).

---

## 9. When to rent something else

Only Stage 10 needs a different card. Prices are Community, on-demand, per hour.

| Option | $/hr | What it unlocks |
| --- | --- | --- |
| **2× RTX 4090 (24 GB each)** | **$0.68** | Tensor/pipeline parallelism on a real 2-GPU engine; also the cheapest way to run the 30B MoE in FP8 with `-tp=2` (~15 GB/GPU) |
| 1× RTX 6000 Ada (48 GB) | $0.74 | The 30B MoE FP8 on one card, and FP8 support (Ada) — the best single-card Stage 10 |
| 1× A100 80 GB (PCIe) | $1.19 | The 30B MoE in **bf16** (needs ~61 GB) — but Ampere has **no FP8 W8A8**, so Stage 9's FP8 labs do not work here |
| 1× RTX A6000 (48 GB) | $0.33 | Cheapest 48 GB, same price as a 4090 — but Ampere, so again no FP8. Good for KV-cache and concurrency experiments at scale, not for Stage 9. |

The A6000 row is the useful lesson: **VRAM is not the only spec that matters.** A 48 GB Ampere card
at the same price as a 24 GB Ada card will lose on anything FP8, which is now most of vLLM's
fast paths. Precisely: the CUTLASS FP8 W8A8 kernels need SM ≥ 89 (Ada/Hopper) and block-FP8 needs
SM ≥ 90, so on an A100/A6000 an `--quantization fp8` run silently falls back to a **weight-only**
FP8 path — you keep the memory saving and lose the compute saving. "It ran" is not the same as
"it ran faster", and `labs/10` makes you measure the difference.

> **Multi-GPU note.** A 2-GPU pod gives you both cards in one container, which is what
> `--tensor-parallel-size 2` needs. Multi-*node* work (Instant Clusters) is aimed at 16–64 GPU jobs
> and is far outside this guide's budget.

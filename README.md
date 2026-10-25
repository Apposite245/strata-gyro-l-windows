# Gyro-L in Strata on Windows and NVIDIA

How to build and run AgentionAI's [Gyro-L](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) quant of
Qwen3.8-Flash-Next (125B MoE, 50.2 GiB of weights) in [Strata](https://github.com/Niko1221/Strata) on a Windows PC with an
RTX 5090: the build recipe, the checks, a prebuilt engine, measured numbers, and how it works inside.

Nothing here changes Strata's code. Gyro-L needs the "pair trellis" expert types (TQP5/6/7), which upstream Strata does
not have yet; AgentionAI has them in a Strata branch, [`upstream-0.2.4-agention`](https://gitlab.com/agentionai-group/strata/-/tree/upstream-0.2.4-agention).
That branch documents a Linux build and lists its CUDA kernels as not yet validated on NVIDIA. This repository is a
tested Windows build of it, with the checks run on an RTX 5090.

> Not affiliated with AgentionAI or Strata. One machine and short tests: read the numbers as one data point.

## Contents

- [Results](#results)
- [The one trap: use the GitLab fork](#the-one-trap-use-the-gitlab-fork)
- [Quick start with the prebuilt engine](#quick-start-with-the-prebuilt-engine)
- [Building from source](#building-from-source)
- [Preparing the model](#preparing-the-model)
- [Running it](#running-it)
- [How it works](#how-it-works)
- [Where the VRAM goes](#where-the-vram-goes)
- [Troubleshooting](#troubleshooting)
- [Credits and licenses](#credits-and-licenses)

## Results

Tested on: RTX 5090 32 GB, Ryzen 7 5800X3D, 80 GB DDR4-3600, Windows 11, NVIDIA driver 617.14, engine 0.1.49.1.

**Checks: all pass.**

- `tq_cpu_parity` (AVX2): 0 failures, including tqp6/tqp6, tqp5/tqp7 and tqp7/tqp5, plain and Hadamard-rotated.
- `native_expert_parity --synthetic` and `--synthetic-hadamard` on the GPU, same three pairs: 0 failures in all 6 runs.
  - The GPU dequantizer against ggml's `to_float`: 0 of 4,915,200 values differ in any bit, every run.
  - The GPU's decode-once and per-entry kernels: bitwise equal.
  - Against a float reference: GPU relative error 1.73e-2 to 2.21e-2, CPU 0.87e-2 to 1.29e-2 (the bound Strata uses for
    every expert type is 3e-2).
- `native_expert_parity --load-check` on the published file: 0 failures (48 expert layers, 45.12 GiB, all 303 tensors
  served from the file have a kernel, direct I/O of the n-gram table works).

**Start:** 45-56 s from launch to ready. The experts load into RAM at 6.4-6.7 GiB/s, and the GPU's expert cache holds
10,435 of the 24,576 experts (19.08 GiB).

**Speed:**

- Short, unrelated prompts right after a start: 46.6 and 58.5 tok/s decode, with 79-80% of the expert calls answered
  from VRAM (12-13% more over PCIe, the rest on the CPU).
- One long agentic coding session (OpenCode), once the cache had settled on the task:
  - 14,430-token prompt, none reused: prefill 3,376 tok/s, decode 85.3 tok/s, 92.1% VRAM hits
  - 82,873-token prompt, none reused: prefill 4,555 tok/s, decode 98.6 tok/s, 96.4% hits
  - 88,942 tokens, 84,232 reused: decode 103.9 tok/s, 98.0% hits
  - 95,399 tokens, 95,381 reused: decode 111.9 tok/s, 97.1% hits
  - 100,009 tokens, 99,893 reused: decode 109.0 tok/s, 95.2% hits

So on a 32 GB card it holds about 100 tok/s at 100K tokens of context once the expert cache has warmed to the task.
Short prompts on unrelated topics are the worst case.

**Answers** (single runs, temperature 0.6, thinking on; the AIME problems may be in the training data):

- 16 textbook questions (maths, physics, chemistry, biology, history, linguistics, CS): 16/16.
- [`tests/answer_check.py`](tests/answer_check.py), 10 AIME 2024/2025 problems and 6 multi-step problems: 16/16,
  22,636 tokens in total.

## The one trap: use the GitLab fork

Strata gets its prompt-path kernels for the trellis types from AgentionAI's llama.cpp fork (`-DSTRATA_GGML_DIR`).
The branch's `docs/GYRO.md` links to `github.com/agentionai/llama.cpp`. **That copy is older**: its ggml stops at
`GGML_TYPE_COUNT = 147` and has no TQP5/6/7.

A build against it finishes without an error but leaves out Gyro-L's types. The only sign is that `tq_cpu_parity` runs no
tqp cases at all. Build against **`gitlab.com/agentionai-group/llama.cpp`** (the model card's link): its ggml has
`TQP5 = 147`, `TQP6 = 148`, `TQP7 = 149`, `GGML_TYPE_COUNT = 150`. [`scripts/get-sources.ps1`](scripts/get-sources.ps1)
checks this.

## Quick start with the prebuilt engine

The [release](../../releases) has `strata.exe`, `strata_kq.dll`, the two parity tools and NVIDIA's cuBLAS DLLs, built for
**RTX 50 series (sm_120) only**. The server, the pack tools and the expert profile come from the source tree, so you
still need the checkout:

1. `.\scripts\get-sources.ps1` (Git needed)
2. Unzip the release into `strata\build\` so that `strata\build\strata.exe` exists
3. Install the [Visual C++ runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe) if you don't have it
4. Continue at [Preparing the model](#preparing-the-model)

Other cards (RTX 40 / 30) need a build from source with another `CUDA_ARCH`. That is untested here.

## Building from source

You need:

- **Git**, **Python 3.10+** (or [uv](https://github.com/astral-sh/uv))
- **CUDA Toolkit 13.x**. RTX 50 cards need 13: 12.8 crashes the prompt path on sm_120.
- **Visual Studio 2022 or 2026 Build Tools** with the C++ workload. CUDA 13.0-13.2 accept only the VS 2019/2022 compilers;
  the VS 2026 Build Tools include the VS 2022 (v143) toolset, which the script selects with `-vcvars_ver=14.44`.
- **CMake** and **Ninja** (both ship with the VS Build Tools)

Steps:

```powershell
.\scripts\get-sources.ps1          # strata\ and agention-llama.cpp\, pinned commits
.\scripts\build-strata.cmd         # edit the paths at its top first; about 10-20 minutes
```

The build uses the same CMake options as Strata's own `setup.py` (`-DSTRATA_ENABLE_CUDA=ON -DSTRATA_Q6K_EXPERTS=ON
-DSTRATA_MMQ_KQUANTS=ON`), plus `-DSTRATA_BUILD_TESTS=ON` for the parity tools, and builds three targets: `strata`,
`native_expert_parity`, `tq_cpu_parity`.

Pinned sources:

- Strata: `gitlab.com/agentionai-group/strata`, branch `upstream-0.2.4-agention`, commit `22dfa634`
- llama.cpp fork: `gitlab.com/agentionai-group/llama.cpp`, main, commit `9aa51ce4`

## Preparing the model

```powershell
.\scripts\prepare-gyro-l.ps1 -Data D:\strata-data
```

This:

1. Downloads `Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf` (108 GB) and `mmproj-F16.gguf` at revision `d149933`, and checks the
   SHA-256 (`5428a564...`). Put `-Data` on a fast NVMe drive: about half of the file is the n-gram table, read from
   disk for every token.
2. Makes a Python environment in `strata\.venv` from Strata's `requirements.txt`.
3. Builds the pack with `tools\iq_pack.py --compat-bf16` (about 30 s, CPU only): the dense weights, the tokenizer and
   the Hadamard and expert tables, about 1.4 GB. The experts stay in the GGUF.
4. Runs the checks from [Results](#results). The GPU ones load the card briefly; `-NoGpuCheck` skips them.

**The MTP draft head.** Strata drafts tokens with its own MTP head, made from the original Qwen3.8-Flash-Next and the
same for every Flash-Next file. This build reused one that Strata's setup had already made for another Flash-Next model.
To make one, `docs/GYRO.md` in the branch gives these steps (not run as part of this build):

```powershell
strata\.venv\Scripts\python.exe strata\tools\mtp_fetch.py fetch --out D:\strata-data\mtp
strata\.venv\Scripts\python.exe strata\tools\mtp_pack.py --src D:\strata-data\mtp --experts q2_0 --out D:\strata-data\mtp\mtp-q2_0.gguf
strata\.venv\Scripts\python.exe strata\tools\mtp_rt.py --gguf D:\strata-data\mtp\mtp-q2_0.gguf --out D:\strata-data\mtp\rt
copy strata\data\draft_vocab.bin D:\strata-data\mtp\rt\
```

## Running it

Copy [`configs/strata-gyro-l.example.json`](configs/strata-gyro-l.example.json) and
[`configs/run-gyro-l.example.bat`](configs/run-gyro-l.example.bat), fix the paths and start the `.bat`. The server
answers at `http://127.0.0.1:8098/v1` (OpenAI) and `/v1/messages` (Anthropic), with a chat page at
`http://127.0.0.1:8098/`.

The example uses:

| Flag | What it does |
|---|---|
| `--expert-cache auto` | sizes the GPU's expert cache to the free VRAM |
| `--prefill auto` | picks the prompt chunk size |
| `--spec 4 --spec-min-p 0.5 --mtp <dir>` | MTP draft of up to 4 tokens per step |
| `--max-context 131072` | the context window (the model is trained for 262,144; the branch reports Gyro-L loading at both) |
| `--kv int8 --kv-resident 32768` | int8 KV cache; 32K positions stay on the GPU, the rest is streamed from pinned RAM |
| `--vram-reserve-mib 2000` | VRAM left free for other programs |

**Pictures** are off in the example. They need `strata-vision.exe`, which this build does not make: take it from an
official Strata release (the `engine` folder) or build `tools\vision`, then add `--vision` and a `"vision"` block as in
Strata's own configs, with the `mmproj-F16.gguf` the script downloaded.

**Memory:** Gyro-L holds about 45 GiB of RAM (the experts, locked) plus the GPU while it runs. Stop it before other GPU
work.

## How it works

**The Gyro format** ([spec](https://gitlab.com/agentionai-group/agention-gyro/-/blob/main/SPEC.md), Apache 2.0):

- Weights are stored in blocks of 128 with one fp16 scale. Each block is a bitstream that walks a tail-biting
  bit-shift trellis with a 15-bit state.
- Each state is hashed into a fixed 2,048-point codebook, and a weight is the scale times a codebook value, so decoding
  is a table lookup and bit-exact across implementations.
- Gyro-S and Gyro-M use TQK6/TQK7 and TQ2_T: 32 steps of 4 weights. Gyro-L uses the pair trellis, TQP5/6/7: 64 steps
  of 2 weights, K = 5/6/7 bits per step, 42/50/58-byte blocks, 2.625/3.125/3.625 bits per weight. According to
  [this repack's model card](https://huggingface.co/virtueer/Qwen3.8-Flash-Next-Gyro-L-Basalt), the 48 expert layers
  are 8 x TQP5, 24 x TQP6 and 16 x TQP7.
- The routed experts are stored in a block-Hadamard-rotated basis (block 128, sign vectors in the metadata). The engine
  rotates the matmul input to match instead of unrotating the weights.
- The dense parts (attention, shared experts, router, embeddings) are ordinary GGUF types. About half of the 108 GB file
  is the n-gram (PLE) table.

**How Strata runs a model that does not fit the GPU:**

- **Dense weights on the GPU.** The attention and other non-expert weights (about 4.3 GiB) always sit in VRAM.
- **Experts in pinned RAM.** All 45.12 GiB of routed experts are read into one host arena, page-locked with
  `cudaHostRegister` so the GPU can read it over PCIe.
- **An expert cache in VRAM.** The free VRAM becomes slots for experts, pre-filled from an expert-use profile
  (`data/expert-profile.bin`): 10,435 slots here.
- **Cache misses** run on the CPU (AVX2 trellis kernels that decode a block once for all the tokens of a verify window)
  or are streamed over PCIe; the engine splits them by measured link speed.
- **Hit rate decides speed.** A long session on one task keeps hitting the same experts (95-98% here); short unrelated
  prompts fall to about 80%.
- **Decode kernels** read the codebook as int8 with one global scale, against q8_1 activations.
- **The prompt path** uses the fork's MMQ kernels for the trellis types when built against the fork. Strata's own
  overlay for upstream ggml covers TQ2_T/TQK6/TQK7 but not TQP, so a Gyro-L build without the fork falls back to a
  slower FP16 path. While reading a prompt the engine borrows expert-cache slots for its buffers (2,285 slots, 4.16 GiB
  here) and gives them back after.
- **The KV cache** is small: only some of this hybrid model's layers use classic attention. Here it is int8, with 32K
  positions resident on the GPU and the full cache (1.55 GiB at 131K) in pinned RAM, streamed as needed. At about 100K
  tokens, 97-99% of the KV reads still hit VRAM.
- **The n-gram table** stays on the SSD and is read with direct I/O, row by row, for every token.
- **MTP drafting** proposes up to 4 tokens per step and verifies them in one pass. Here 60-90% of drafts were accepted
  on the test prompts.

## Where the VRAM goes

From the engine's start-up log on the 32 GB card:

| What | VRAM |
|---|---:|
| Expert cache (10,435 experts) | 19.1 GiB |
| Dense weights (2,975 MiB of projections + 1,416 MiB of other pack weights) | ~4.3 GiB |
| MTP draft layer (835 MiB) + draft-head buffers (281 MiB) | ~1.1 GiB |
| KV cache, 32K resident positions | ~0.4 GiB |
| Kept free by `--vram-reserve-mib 2000` | ~1.4 GiB |
| Windows desktop and other apps, CUDA context, graphs, buffers | the rest |

The picture reader (`strata-vision`), when on, is a separate process with about 1 GiB more. Lowering the reserve or
moving pictures to the CPU gives the expert cache more slots, but with a 95%+ hit rate in long sessions that gains only a
few percent.

## Troubleshooting

- **`tq_cpu_parity` shows no `tqp` lines**: you built against the GitHub fork. Rerun `get-sources.ps1` (it uses
  GitLab), delete `strata\build` and build again.
- **`unsupported Microsoft Visual Studio version`**: CUDA 13.0-13.2 reject the VS 2026 compiler. Keep
  `-vcvars_ver=14.44` (the v143 toolset) in `build-strata.cmd`, or use VS 2022.
- **The build stops in `ptxas`**: CUDA 13.0 sometimes fails on a file it just wrote. The script retries once, as
  Strata's setup does; running it again continues where it stopped.
- **`large pages refused ... VirtualAlloc error 1314`** in the log: the account lacks the lock-pages privilege, and the
  engine falls back to 4 KB pages. Harmless.
- **Git changed files on checkout**: with `core.autocrlf=true`, Windows Git rewrites line endings, and Strata hashes its
  sources. `get-sources.ps1` sets `core.autocrlf=false` in both checkouts.

## Credits and licenses

- **Gyro format, encoder, trellis kernels and the Strata branch:** Laurent Zuijdwijk / AgentionAI.
  - The branch: [agentionai-group/strata](https://gitlab.com/agentionai-group/strata).
  - The format spec: [agention-gyro](https://gitlab.com/agentionai-group/agention-gyro), Apache 2.0.
  - The llama.cpp fork: [agentionai-group/llama.cpp](https://gitlab.com/agentionai-group/llama.cpp), MIT.
- **Strata:** [Niko1221/Strata](https://github.com/Niko1221/Strata), MIT ([license](licenses/Strata-MIT.txt)).
- **ggml / llama.cpp:** the ggml authors, MIT ([license](licenses/ggml-llama.cpp-MIT.txt)).
- **The format family:** QTIP (Tseng et al., 2024).
- **The model:** Qwen/Qwen3.8-Flash-Next, under the Qwen Community License 1.0. This repository does not contain or
  redistribute any model weights.
- **The release's `cublas64_13.dll` and `cublasLt64_13.dll`:** from NVIDIA's CUDA Toolkit 13.0, redistributed under its
  EULA.
- **The scripts and docs here:** MIT ([LICENSE](LICENSE)). Built and tested with help from an AI assistant (Claude).

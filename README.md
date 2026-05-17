# vllm-autoresearch

Autonomous research harness for finding optimal vLLM serving configurations
for a given model on a given box.

Same pattern as [autoresearch](https://github.com/karpathy/autoresearch): an AI
agent edits a config file, launches a short benchmark, measures metrics, keeps
or discards, and repeats overnight.

> **Writeup:** [BLOG.md](./BLOG.md) — what actually moves the needle when
> serving Llama 3.3 70B on two RTX 5090s.
> **Receipts:** [FINDINGS.md](./FINDINGS.md) — σ-quantified per-iter results
> and noise-floor numbers.

## Plain-English recommendation

If you just want to serve a 70B model on a pair of RTX 5090s and skip the
methodology, here's what the experiments say. Tested on both Llama 3.3 70B
and DeepSeek-R1-Distill-Llama-70B (AWQ-int4 for both).

**Use the defaults. Set one thing.** In `config.py`, change
`KV_CACHE_DTYPE` from `"auto"` to `"fp8"`. That single change improved every
workload profile on both models — by 1–11% on Llama 3.3, and on R1 it lifted
the reasoning-trace workload by about 16% (the biggest single-knob gain
anywhere in the audit). The mechanism is the same on both: halving the
attention-cache memory frees room for more concurrent requests and longer
outputs.

Every other "obvious" optimization I tried — bigger batches, more concurrent
sequences, more memory headroom, a different paged-attention block size — was
worse on both models. The vLLM V1 defaults are well-tuned across them.
Resist the urge to twiddle.

**For R1 specifically:** also raise `MAX_MODEL_LEN` from 8192 to 16384. R1
emits long `<think>...</think>` reasoning traces that need the room. Without
this, the model is KV-starved, queue lengths blow up, and what should be a
700 ms first-token latency becomes 25 seconds. (Llama 3.3 doesn't need this
since chat completions stay under 4k tokens.)

**Don't chase fancier quantization.** FP8 W8A8 weights don't fit on 2× 32 GB
(70 GB raw vs 64 GB total VRAM). NVFP4 (Blackwell-native fp4) does fit and is
about 11% faster on batch-serving workloads, but is slower on interactive
chat and is 55× noisier run-to-run than AWQ. Worth it only if your traffic
is mostly batch. For interactive or reasoning use, stay on AWQ.

**Don't use llama.cpp for this.** It's a wonderful project for laptops, but
for serving a 70B model on two real GPUs with concurrent users, vLLM is
28–742% faster depending on the workload.

**Don't trust vLLM tuning blog posts from 2023 or earlier.** vLLM was
substantially rewritten ("V1") and many famous knobs no longer exist or have
been turned into no-ops. If you're following someone else's guide, check
their flags still appear in `vllm serve --help` on your installed version.

**Don't reason about which knobs will flip on a different model.** I made
two predictions about which rejected tuning iterations might behave
differently on R1 ("decode-dominant workload", "lower concurrency"). Both
were wrong when measured. The methodology — variance probe + Pareto rule —
transfers across same-architecture models; specific "this should be safer
because X" reasoning doesn't.

**The single most useful thing this repo offers** isn't the champion config —
it's `variance_probe.py`, which measures how much the same config naturally
fluctuates between runs. Once you know your noise floor, most "tuning wins"
you read about online turn out to be inside the noise.

**Bottom line:** install vLLM 0.20+, serve an AWQ-int4 70B model, set
`KV_CACHE_DTYPE = "fp8"`, leave everything else at its default. For R1 also
set `MAX_MODEL_LEN = 16384`. That's the same setup I'd ship.

## What's different from autoresearch

|                  | autoresearch              | vllm-autoresearch                    |
|------------------|---------------------------|---------------------------------------|
| What runs        | A 5-min training of a tiny GPT | An ~8-min benchmark of vLLM serving |
| Metric           | val_bpb (lower better)    | per-profile score (higher better)     |
| Knobs            | model arch, optimizer     | vLLM flags, KV cache, scheduling      |
| Cost per iter    | ~5 min                    | ~6–10 min (vLLM startup is the tax)   |
| Model            | trained from scratch      | pretrained, you bring your own        |

## Repo layout

```
config.py              <- agent edits this; tunable + locked sections
launch_vllm.py         <- spawns vLLM, version-checks every CLI flag, waits for /health
workload/*.jsonl       <- four labelled workload profiles (synthetic for now)
benchmark.py           <- fires each profile, returns rich per-profile metrics
run.py                 <- the loop: launch -> bench -> kill -> log
results.tsv            <- commit | config_hash | scores | latencies | status | description
program.md             <- agent playbook (what's fair game, what's locked, how to back off)
```

The agent only ever modifies `config.py`.  Everything else is infrastructure.

## Setup

This harness uses the **`vllm-serve` conda env** (vLLM 0.20.x + PyTorch 2.11 +
CUDA 12.9).  `uv sync` is not used — Blackwell support depends on a
prebuilt wheel that already lives in the conda env.

```bash
# verify the env
conda run -n vllm-serve python -c "import vllm, torch; print(vllm.__version__, torch.cuda.device_count())"

# stock baseline (do this first, on a clean working tree)
conda run -n vllm-serve python run.py --baseline --description "stock" > run.log 2>&1

# subsequent iterations
conda run -n vllm-serve python run.py --description "raised gpu_mem to 0.90" > run.log 2>&1
```

## What the benchmark measures

Each iteration runs **four** workload profiles back-to-back:

| profile      | shape                                  | concurrency |
|--------------|----------------------------------------|-------------|
| interactive  | short Q&A, ~200–400 output tokens      | 16          |
| coding       | code generation, ~400–700 output tokens | 16         |
| batch        | tiny prompts/replies, classification-style | 64       |
| long_context | 4–7 k input tokens, short replies      | 4           |

For each profile, `benchmark.py` records:

- requests/s, output tok/s, total tok/s
- TTFT p50/p95/p99
- inter-token p50/p90/p95/p99
- end-to-end request latency p50/p95/p99
- completed / errored / timed_out
- a single score combining throughput with TTFT and inter-token penalties

We **deliberately do not collapse the four scores into one number** — that
would hide tradeoffs (e.g. configs that crush batch throughput but are
intolerable interactively).  The agent must Pareto-improve.

## Output format

`run.py` prints rich per-profile metrics plus a final JSON summary:

```
SUMMARY {"config_hash": "abc123…", "scores": {"interactive": 134.2, …}, "status": "ok", "synthetic": true}
```

…and appends a row to `results.tsv`.  Extract scores via:

```bash
tail -1 results.tsv
grep '^SUMMARY ' run.log | tail -1
```

## Synthetic-workload warning

Every prompt in `workload/*.jsonl` is currently marked `"synthetic": true`.
Use these runs for **relative ranking of configs only** — until real
anonymised production prompts replace them, absolute throughput/latency
numbers are not trustworthy.  The `synthetic` column in `results.tsv` is `1`
until that happens.

## Hardware assumptions

- ≥ 2 visible GPUs with enough VRAM for the chosen model (`TENSOR_PARALLEL_SIZE`
  set in `config.py`).
- No other vLLM/training processes on the same GPUs.

Tested on: 2× RTX 5090 (32 GB each, Blackwell), CUDA 12.9, vLLM 0.20.0,
Llama 3.3 70B AWQ.

# Findings — May 2026 audit

**Target stack.** Llama 3.3 70B Instruct served via vLLM 0.20.0 on 2× RTX 5090 (Blackwell, sm_120, 32 GB each), TP=2, OpenAI-compatible `/v1/completions`.

**Methodology.** Each candidate config is benchmarked across four profiles
(interactive, coding, batch, long_context). A 3-run variance probe on a
frozen committed config measures the noise floor; subsequent iterations
must beat that floor by ≥2σ on at least one profile and not regress any
profile by >2σ (Pareto rule).

## Champion: iter 1 (commit `eba2631`)

The shipped configuration. Single change from the original baseline:

- `KV_CACHE_DTYPE = "fp8"` (was `"auto"`)

| profile        | score (mean ± σ over 3 runs) | output tok/s | p99 TTFT | p99 ITL |
|----------------|------------------------------|--------------|----------|---------|
| interactive    | 350.70 ± 0.19                | 350          | 730 ms   | 66 ms   |
| coding         | 328.84 ± 0.39                | 329          | 970 ms   | 55 ms   |
| batch          | 327.12 ± 0.08                | 328          | 1606 ms  | 273 ms  |
| long_context   | 100.87 ± 0.80                | 101          | 18.3 s   | 56 ms   |

vs the original AWQ baseline (commit `a4faebf`, KV cache `auto`):
+0.59% interactive, +0.93% coding, +1.72% batch, +10.86% long_context.
All deltas are 7–22σ — firm signal.

## Failed iterations

| commit    | change                              | result                                |
|-----------|-------------------------------------|---------------------------------------|
| `2c60e07` | `MAX_NUM_BATCHED_TOKENS = 16384`    | batch +5.9σ, interactive **−7.2σ** ✗ |
| `60ac980` | `MAX_NUM_SEQS = 128`                | interactive **−13.8σ**, batch **−5.5σ** ✗ |
| `e789ef7` | `GPU_MEMORY_UTILIZATION = 0.92`     | first three profiles +2.3 to +6.8σ, **long_context −126σ catastrophic crash** ✗ |
| `a137ba9` | `MAX_NUM_PARTIAL_PREFILLS = 4`      | vLLM 0.20 V1 raises `NotImplementedError: Concurrent Partial Prefill is not supported` — flag exists but feature was not ported from V0 ✗ |
| `30fe7be` | `BLOCK_SIZE = 32` (up from 16)      | interactive **−54σ**, batch **−1441σ** (req/s 22.0 → 14.3), long_context flat ✗ |

Larger prefill chunks improved batch throughput at the cost of single-stream
inter-token latency. The 64-sequence concurrency ceiling was not the
bottleneck; raising it hurt both latency and throughput, indicating effective
batch is gated upstream (chunked-prefill / decode kernel sweet spot).

Iter 5b is the most surprising failure: block size *is* a high-leverage knob
(±35% on batch throughput), but the vLLM default of 16 already sits at the
optimum for our workload's prompt-size distribution. Moving to 32 cuts
page-table overhead but explodes internal fragmentation — with 64 concurrent
short-prompt batch requests, each one claiming a 32-slot block wastes more
KV than 16-slot blocks did, so usable concurrency drops and the scheduler
pre-empts. Moving to 8 would do the opposite. The default isn't arbitrary.

Iter 4 is the most informative failure: bumping `GPU_MEMORY_UTILIZATION` from
0.85 → 0.92 *did* lift interactive/coding/batch by small but real margins
(+2.3 to +6.8σ), but vLLM crashed during the long_context profile and the
benchmark logged ~195 k connection failures in 60 s. The 0.85 default is not
slack — it's load-bearing headroom for prefill activation buffers on 7 k-token
prompts. **Conclusion: 0.85 is the headroom frontier on 2× 32 GB; pushing
higher requires either smaller `MAX_MODEL_LEN` or accepting an OOM on long
prompts.**

## Cross-stack: weight quantization probes

### FP8 W8A8 — DNF

`RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` (70 GB raw weights) does not fit
on 2× 32 GB. CUDA OOM during weight load — both GPUs allocated >29.5 GB before
the first attention buffer. Physical ceiling, not a tuning issue.

### NVFP4 W4A16 (commit `42e6697`, branch `quant/nvfp4-weights`)

`RedHatAI/Llama-3.3-70B-Instruct-NVFP4` (43 GB on disk, fits comfortably).
3-run variance probe:

| profile      | mean ± σ        | CV%   | vs AWQ iter 1   |
|--------------|-----------------|-------|------------------|
| interactive  | 295.67 ± 8.15   | 2.75% | **−55.03** (regression, −6.7σ) |
| coding       | 291.33 ± 2.67   | 0.92% | **−37.51** (regression, −14σ) |
| batch        | 363.55 ± 0.88   | 0.24% | **+36.43** (**+41σ win**) |
| long_context | 82.81 ± 1.17    | 1.41% | **−18.06** (regression, −15σ) |

NVFP4 trades single-stream latency for high-concurrency throughput. Reading
the timing breakdown:

- **TTFT improved by 9–15% on every profile** — Blackwell sm_120 FP4 tensor
  cores deliver real compute gains at the prefill stage.
- **Inter-token latency worsened by 7–10% on latency-sensitive profiles** —
  decode is memory-bandwidth-bound; FP4 weights are the same bytes per
  parameter as AWQ int4, so no bandwidth gain, and the fp4 decode path in
  vLLM 0.20 is less optimized than marlin's int4 kernel.
- **Batch wins +11% on tok/s and completes ~13% more requests** (1580 vs 1387
  in 60 s) — at concurrency=64 the faster prefill compounds with throughput.

### Noise floor comparison

NVFP4 is **dramatically noisier** than the AWQ stack on every profile:

| profile      | AWQ iter 1 CV% | NVFP4 CV% | ratio |
|--------------|-----------------|-----------|-------|
| interactive  | 0.05%           | 2.75%     | **55×** |
| coding       | 0.12%           | 0.92%     | 8×    |
| batch        | 0.03%           | 0.24%     | 8×    |
| long_context | 0.80%           | 1.41%     | 2×    |

Probable causes: the fp4 autotuner picking different kernels per startup,
thermal/clock sensitivity in the new sm_120 path, and less-mature inductor
caching. Anyone tuning the fp4 stack must run multiple iterations to
distinguish signal — single-shot tuning is unreliable.

## Cross-backend: llama.cpp Q4_K_M (commit `0a1ac00`, branch `backend/llama-cpp`)

Single benchmark run, llama.cpp build 9127 (a9883db8e), Q4_K_M GGUF (40 GB),
`--flash-attn on`, `--cont-batching`, `--kv-unified` (shared KV pool, the
fairer paged-attention analog), `--parallel 64`, `--ctx-size 32768`. Same
OpenAI-compatible benchmark hits either stack.

**Output tok/s — vLLM wins every profile:**

| profile      | vLLM iter 1 | llama.cpp | vLLM advantage |
|--------------|-------------|-----------|----------------|
| interactive  | 350         | 273       | +28%           |
| coding       | 329         | 240       | +37%           |
| batch        | 328         | 172       | **+90%**       |
| long_context | 101         | 12        | **+742%**      |

**p99 TTFT — vLLM wins three of four:**

| profile      | vLLM      | llama.cpp | winner            |
|--------------|-----------|-----------|-------------------|
| interactive  | 730 ms    | 1820 ms   | vLLM 2.5×         |
| coding       | 970 ms    | 3213 ms   | vLLM 3.3×         |
| batch        | 1606 ms   | 8271 ms   | vLLM 5.1×         |
| long_context | 18332 ms  | 11954 ms  | **llama.cpp 1.5×** |

**p99 inter-token — vLLM wins all four, often by huge margins:**

| profile      | vLLM    | llama.cpp | winner       |
|--------------|---------|-----------|--------------|
| interactive  | 66 ms   | 394 ms    | vLLM 6×      |
| coding       | 55 ms   | 67 ms     | vLLM 1.2×    |
| batch        | 273 ms  | 1021 ms   | vLLM 3.7×    |
| long_context | 56 ms   | 6203 ms   | **vLLM 110×** |

**SLO-gated scores:** vLLM 350.70 / 328.84 / 327.12 / 100.87 vs llama.cpp
0.00 / 94.52 / 172.27 / 0.00. Two profiles get zeroed by SLO violations
(interactive p99 TTFT 1820 > 1000 ms and ITL 394 > 80 ms; long_context p99
ITL 6203 ms ≫ 100 ms target). Completed requests in 60 s: 100 / 51 / 1387
/ 80 (vLLM) vs 79 / 40 / 747 / 13 (llama.cpp) — batch lost half capacity,
long_context lost 84%.

**Lone llama.cpp win** is long-context TTFT (1.5× faster prefill), suggesting
the GGUF prefill kernel is genuinely competitive on prefill-dominant
workloads. But the kv-unified pool seems to thrash under 4–7 k prompts at
parallel=64 — decode collapses to p99 6203 ms ITL, killing the score.
Tunable in principle (lower parallel, larger ctx-size, or non-kv-unified
mode), but the dominant pattern across the other three profiles makes
further llama.cpp tuning low-priority.

**Conclusion: no reason to consider llama.cpp on this stack.** vLLM
AWQ + KV fp8 dominates GGUF Q4_K_M on every metric except a single
prefill-only TTFT data point that doesn't show up in the SLO-weighted score.

## What V1 doesn't expose

A separate finding worth surfacing: many of the historically-quoted vLLM
tuning levers do not exist in vLLM 0.20's V1 scheduler. Discovered during
iter 5: `--num-scheduler-steps` (multi-step decode, a big V0-era throughput
knob) has been removed entirely; `--max-num-partial-prefills > 1` is still
accepted as an argument but raises `NotImplementedError` on startup. Other
V0 knobs like `--swap-space` and `--scheduler-delay-factor` are silently
dropped by our launcher's version probe.

The realistic V1 tuning surface for our workload class is roughly:
`GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`, `MAX_MODEL_LEN`, `KV_CACHE_DTYPE`,
`BLOCK_SIZE`, `MAX_NUM_BATCHED_TOKENS`, and the chunked-prefill /
prefix-caching toggles. Five of those seven were tested; of those, the
defaults are already optimal on four (`MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`,
`MAX_NUM_BATCHED_TOKENS`, `BLOCK_SIZE`) and only one (`KV_CACHE_DTYPE`)
moved the needle.

## Recommendations

1. **Ship `eba2631` (AWQ + KV fp8)** as the production config — best SLO-
   weighted aggregate, latency-clean, deterministic.
2. **Document `42e6697` (NVFP4)** as an optional **batch-throughput profile**
   for users whose workload is dominated by high-concurrency serving and who
   can tolerate the latency regression.
3. **Do not pursue FP8 W8A8 on 70B** with this hardware — 64 GB total VRAM
   is below the floor. Would require 3× 5090s or H100/H200-class memory.
4. **For future tuning iterations, prefer changes to scheduling/KV/sampling
   over concurrency-ceiling bumps** — the two failed iters both lifted
   ceilings and regressed latency.

## Open questions

- Will vLLM ≥0.21 improve the NVFP4 decode kernel? If it closes the ITL gap
  while keeping the prefill win, NVFP4 becomes the champion stack outright.
- ~~Would `GPU_MEMORY_UTILIZATION = 0.92` gain meaningful headroom on AWQ
  trunk?~~ **Answered (iter 4, `e789ef7`):** no — pushing past 0.85 OOMs on
  long_context. The 0.85 default is the headroom frontier on 2× 32 GB.
- ~~Cross-backend comparison (llama.cpp Q4_K_M with `--kv-unified`).~~
  **Answered (`0a1ac00`):** vLLM AWQ wins every output-throughput,
  inter-token, and SLO-gated profile by 28–742%. Sole llama.cpp win is
  long-context p99 TTFT (1.5×), insufficient to flip the SLO-weighted
  aggregate.

## Known issues

- ~~Zombie vLLM workers escape teardown when run.py logs `status="crash"`.~~
  **Fixed in `launch_vllm.py` / `launch_llama_cpp.py`** — the original
  teardown looked up the pgid lazily via `os.getpgid(proc.pid)`, which
  raises `ProcessLookupError` once the leader is reaped (exactly the case
  on crash). The bare `except Exception` swallowed it and workers
  survived. The fix caches the pgid synchronously at Popen time
  (`pgid = proc.pid`, valid because `preexec_fn=os.setsid`), polls the
  group with `os.killpg(pgid, 0)` to confirm it has emptied, and
  unconditionally escalates to SIGKILL after the grace period.

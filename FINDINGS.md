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

Larger prefill chunks improved batch throughput at the cost of single-stream
inter-token latency. The 64-sequence concurrency ceiling was not the
bottleneck; raising it hurt both latency and throughput, indicating effective
batch is gated upstream (chunked-prefill / decode kernel sweet spot).

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
- Would `GPU_MEMORY_UTILIZATION = 0.92` (vs current 0.85) gain meaningful KV
  block headroom on AWQ trunk? Untested — low risk, high information value.
- Cross-backend comparison (llama.cpp Q4_K_M with `--kv-unified`) was wired
  up but not benchmarked at parity. Worth a single fair-comparison run.

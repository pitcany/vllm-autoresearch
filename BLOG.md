# What actually moves the needle when serving Llama 3.3 70B on two RTX 5090s

**TL;DR.** I audited vLLM 0.20 tuning knobs for Llama 3.3 70B (AWQ) on 2× RTX 5090. Three findings:

- **Only one config knob mattered**: `KV_CACHE_DTYPE = "fp8"`. Improved all four workload profiles by 7σ to 22σ. Every other plausible scheduler knob I tested was either already at its optimum (`MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, `MAX_NUM_BATCHED_TOKENS`, `BLOCK_SIZE`) or actively a regression when changed.
- **The V1 scheduler rewrite quietly removed several historically-quoted levers.** `--num-scheduler-steps` is gone; `--max-num-partial-prefills > 1` raises `NotImplementedError`; `--swap-space` and `--scheduler-delay-factor` are no-ops. A lot of public vLLM tuning advice predates V1 and won't apply.
- **Cross-stack probes**: FP8 W8A8 weights don't fit on 2× 32 GB (70 GB raw); NVFP4 trades single-stream latency for +11% batch throughput *but* is 55× noisier run-to-run than AWQ; llama.cpp Q4_K_M at parity loses every SLO-weighted profile by 28% to 742%.

Receipts (`results.tsv`, per-iter branches, σ-quantified noise floor) are in [FINDINGS.md](./FINDINGS.md).

---

The Blackwell GPUs in a pair of 5090s give you 64 GB total VRAM and a brand-new FP4/FP8 tensor-core path. With Llama 3.3 70B quantized to AWQ-int4, you can fit the model with KV-cache headroom, and vLLM 0.20 will serve it across the pair via tensor parallelism. The interesting question — given all of that — is which of the dozens of knobs in `vllm serve --help` actually matter.

I spent a couple of evenings turning that question into measurements. Short version: **one change made a real difference, every other config knob I tested was either already at its optimum, removed in the V1 scheduler rewrite, or actively a regression.** Two of the more exotic quantization paths either crashed or only helped one workload. Here's what I learned.

## The first hour was harness, not tuning

My benchmark ran four workload profiles (interactive, coding, batch, long-context), scored each against an SLO, and logged to a TSV. It was broken in three subtle ways. The launcher's port pre-flight accepted any process listening on 8003, so a stale vLLM from the previous run got reported as "ready" with `startup_s=0`. Killing the runner with SIGTERM left workers behind because of process-group confusion. And the batch profile was scoring 0 because the global inter-token SLO penalty applied to all four profiles — and high per-token latency under concurrent batch load is by design, not a regression.

None of these would have surfaced as bugs. They would have surfaced as "the tuning iteration was great" or "the tuning iteration was a wash," interpretable either way. **You cannot tune what you cannot measure reliably**, and you cannot trust a single benchmark run without a variance probe.

## The one change that worked: KV cache in FP8

Switching `KV_CACHE_DTYPE` from `"auto"` (fp16) to `"fp8"` halves the memory the attention cache consumes, freeing room for more concurrent requests and longer effective contexts. Every profile improved by 7σ to 22σ — far past the noise floor, on the order of +1 to +11 % on the SLO-weighted scores. The biggest win was long-context (+11 %), which is unsurprising in retrospect: KV is the bottleneck the moment prompts grow.

That's the entire affirmative finding.

## Every other scheduler knob I touched was already at its optimum (or worse)

I tried five other plausible changes. None landed.

- **`MAX_NUM_BATCHED_TOKENS = 16384`** (up from 8192): helped batch by +6σ, killed interactive latency by −7σ. Bigger prefill chunks make decode wait longer.
- **`MAX_NUM_SEQS = 128`** (up from 64): hurt both latency *and* throughput. The 64-sequence ceiling wasn't the bottleneck; raising it created scheduling contention.
- **`GPU_MEMORY_UTILIZATION = 0.92`** (up from 0.85): lifted the first three profiles by +2 to +7σ, then OOM-crashed on long-context prefill — 195,000 errored requests in 60 seconds. The 0.85 default isn't slack; it's load-bearing headroom for prefill activation buffers, and the only way to find that out was to crash into it.
- **`BLOCK_SIZE = 32`** (up from 16): batch throughput collapsed by 35 % (−1441σ). Larger paged-attention blocks mean more internal fragmentation, and with 64 concurrent short-prompt batch requests, each block-claim wastes more KV than 16-slot blocks did. Block size *is* a high-leverage knob; the vLLM default just happens to sit at the optimum for our prompt-size distribution.
- **`MAX_NUM_PARTIAL_PREFILLS = 4`** (up from 1): didn't even start. `NotImplementedError: Concurrent Partial Prefill is not supported.` Which leads to…

## Many of the historically-quoted vLLM levers don't exist anymore

vLLM 0.20 defaults to the V1 scheduler, which is a substantial rewrite of V0. A surprising amount of tuning advice on the internet references V0 knobs that have been quietly removed or stubbed out:

- `--num-scheduler-steps` (multi-step decode, historically *the* throughput lever): gone.
- `--max-num-partial-prefills > 1`: accepted as an argument, raises `NotImplementedError` at startup.
- `--swap-space`, `--scheduler-delay-factor`: silently dropped by version-aware launchers, no-ops on V1.

The realistic V1 tuning surface for a single-model serving workload is roughly seven knobs: `GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`, `MAX_MODEL_LEN`, `KV_CACHE_DTYPE`, `BLOCK_SIZE`, `MAX_NUM_BATCHED_TOKENS`, plus the chunked-prefill / prefix-caching toggles. I tested five of them. Four were already at their optimum. One mattered.

That's the second affirmative finding, and it's almost more useful than the first: **the V1 vLLM default config is good.** Most tuning effort on V1 is just noise around a well-tuned default.

## The exotic quantization paths were a mixed bag

FP8 W8A8 weights for 70B are 70 GB raw, which is exactly 6 GB too many for two 32-GB cards. No tuning gets you there.

NVFP4 (W4A16) fits comfortably at 43 GB and exercises Blackwell's native FP4 tensor cores — its prefill is genuinely faster than AWQ-marlin's int4 path — but inter-token decode is 7–10 % slower (decode is memory-bandwidth bound, not compute), so NVFP4 only wins on batch workloads, where it delivers +11 % output tok/s.

The really interesting finding wasn't the speed delta, though — it was the noise floor: NVFP4 run-to-run variance was **55× higher than AWQ** on interactive scoring (CV 2.75 % vs 0.05 %). The FP4 stack in vLLM 0.20 is fast but not yet deterministic; you need multiple runs to distinguish signal from "the autotuner picked a different kernel this time."

## llama.cpp on the same model at parity lost every profile

Same GGUF (Q4_K_M), `--kv-unified` (the fairer paged-attention analog), parallel=64, ctx-size=32k. Output throughput gap ranged from +28 % vLLM (interactive) to **+742 % vLLM** (long-context). The sole bright spot for llama.cpp was long-context p99 TTFT — its prefill kernel is genuinely competitive — but its decode collapsed under concurrent load (p99 inter-token: 6.2 s for llama.cpp vs 56 ms for vLLM) and killed the SLO score.

## Does any of this transfer to other models? A quick R1 audit

The whole study was on Llama 3.3 70B. To check whether the findings generalize, I re-ran the audit on `DeepSeek-R1-Distill-Llama-70B-AWQ` — same architecture, same AWQ-int4 quant, same author, but a reasoning-fine-tune that emits long `<think>` traces. I added a fifth `reasoning` workload profile (math/logic prompts with `max_tokens=4096`, concurrency=8) and raised `MAX_MODEL_LEN` to 16384.

- **KV fp8 transferred cleanly** and was, if anything, slightly bigger: coding +0.9%, batch +2.0%, long_context +5.1%, reasoning +15.8%. All five profiles past 2σ on R1's own (very tight) noise floor.
- **A surprise on baseline**: R1's interactive score was 0 because under default fp16 KV the model was KV-starved, p99 TTFT was 25.7 s, and the 1 s SLO zeroed the score. With fp8 KV the TTFT dropped to 0.75 s — back to Llama-level. So R1 *can* serve interactive workloads; it just needs the cache headroom.
- **The two rejections I expected to flip on R1, didn't.** I predicted that `MAX_NUM_BATCHED_TOKENS=16384` would be less harmful on R1 (decode-dominant workload, smaller relative prefill penalty) and that `BLOCK_SIZE=32` would be less catastrophic (R1's reasoning profile uses lower concurrency than Llama's batch). Both predictions were wrong. Iter 2 regressed interactive *more* on R1 than on Llama; iter 5b crashed batch by the same 34% — because the batch profile's concurrency is set in the workload definition, not the model.
- **One R1-specific caveat**: the reasoning profile is intrinsically noisy under these parameters (only ~8 requests per 60 s window, so reasoning σ is ~14 vs ≤1.0 on the other four). Anyone tuning reasoning-specific behavior on this stack would want a longer duration or higher concurrency for a cleaner signal.

The methodology and the headline result transfer. The specific predictions about *which* rejected iters might flip on a different model did not.

## The takeaway

The value of the work wasn't the one knob that mattered. It was the noise floor and the rejection rule. Once you've measured your noise floor properly — a 3-run variance probe on a frozen config — most "improvements" are obviously not improvements, and the few that are jump out as 10σ-plus signal.

Two things I now believe more strongly than I did when I started:

1. **The V1 vLLM default config is good.** Most tuning effort spent on this stack is rediscovering that fact. The single change worth making for this hardware + model class is `KV_CACHE_DTYPE = "fp8"`.
2. **A lot of published vLLM tuning advice predates V1.** Before reaching for a knob you read about in 2023, check that it still exists in the binary you're running. Several historically-large levers (multi-step scheduling, concurrent partial prefill) are gone or stubbed.

The audit was worth doing not because it found a magic config, but because it produced a defensible "no, the default really is the default" for nearly everything else.

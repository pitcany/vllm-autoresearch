# What actually moves the needle when serving Llama 3.3 70B on two RTX 5090s

The Blackwell GPUs in a pair of 5090s give you 64 GB total VRAM and a brand-new FP4/FP8 tensor-core path. With Llama 3.3 70B quantized to AWQ-int4, you can fit the model with KV-cache headroom, and vLLM 0.20 will serve it across the pair via tensor parallelism. The interesting question — given all of that — is which of the dozens of knobs in `vllm serve --help` actually matter.

I spent a couple of evenings turning that question into measurements. Short version: one change made a real difference, three "improvements" were silently regressions, and two of the more exotic quantization paths either crashed or only helped one workload. Here's what I learned.

## The first hour was harness, not tuning

My benchmark ran four workload profiles (interactive, coding, batch, long-context), scored each against an SLO, and logged to a TSV. It was broken in three subtle ways. The launcher's port pre-flight accepted any process listening on 8003, so a stale vLLM from the previous run got reported as "ready" with `startup_s=0`. Killing the runner with SIGTERM left workers behind because of process-group confusion. And the batch profile was scoring 0 because the global inter-token SLO penalty applied to all four profiles — and high per-token latency under concurrent batch load is by design, not a regression.

None of these would have surfaced as bugs. They would have surfaced as "the tuning iteration was great" or "the tuning iteration was a wash," interpretable either way. **You cannot tune what you cannot measure reliably**, and you cannot trust a single benchmark run without a variance probe.

## The one change that worked: KV cache in FP8

Switching `KV_CACHE_DTYPE` from `"auto"` (fp16) to `"fp8"` halves the memory the attention cache consumes, freeing room for more concurrent requests and longer effective contexts. Every profile improved by 7σ to 22σ — far past the noise floor, on the order of +1 to +11 % on the SLO-weighted scores. The biggest win was long-context (+11 %), which is unsurprising in retrospect: KV is the bottleneck the moment prompts grow.

## Three "obvious" tuning ideas were silent regressions

Bigger prefill chunks (`MAX_NUM_BATCHED_TOKENS=16384`) helped batch by +6σ but killed interactive latency by −7σ. Doubling the concurrency ceiling (`MAX_NUM_SEQS=128`) hurt both latency *and* throughput at once — the 64-sequence ceiling wasn't the bottleneck. Pushing `GPU_MEMORY_UTILIZATION` from 0.85 to 0.92 lifted the first three profiles by +2 to +7σ but catastrophically OOM'd on long-context prefill. That last one was the most instructive: the 0.85 default isn't slack, it's load-bearing headroom for prefill activation buffers, and the only way to find that out was to crash into it.

## The exotic quantization paths were a mixed bag

FP8 W8A8 weights for 70B are 70 GB raw, which is exactly 6 GB too many for two 32-GB cards. No tuning gets you there.

NVFP4 (W4A16) fits comfortably at 43 GB and exercises Blackwell's native FP4 tensor cores — its prefill is genuinely faster than AWQ-marlin's int4 path — but inter-token decode is 7–10 % slower (decode is memory-bandwidth bound, not compute), so NVFP4 only wins on batch workloads, where it delivers +11 % output tok/s.

The really interesting finding wasn't the speed delta, though — it was the noise floor: NVFP4 run-to-run variance was **55× higher than AWQ** on interactive scoring (CV 2.75 % vs 0.05 %). The FP4 stack in vLLM 0.20 is fast but not yet deterministic; you need multiple runs to distinguish signal from "the autotuner picked a different kernel this time."

## llama.cpp on the same model at parity lost every profile

Same GGUF (Q4_K_M), `--kv-unified` (the fairer paged-attention analog), parallel=64, ctx-size=32k. Output throughput gap ranged from +28 % vLLM (interactive) to **+742 % vLLM** (long-context). The sole bright spot for llama.cpp was long-context p99 TTFT — its prefill kernel is genuinely competitive — but its decode collapsed under concurrent load (p99 inter-token: 6.2 s for llama.cpp vs 56 ms for vLLM) and killed the SLO score.

## The takeaway

The value of the work wasn't the one knob that mattered. It was the noise floor and the rejection rule. Once you've measured your noise floor properly — a 3-run variance probe on a frozen config — most "improvements" are obviously not improvements, and the few that are jump out as 10σ-plus signal. Without it I would have shipped one of the rejected configs and felt good about it.

Most vLLM tuning posts on the internet are doing exactly that.

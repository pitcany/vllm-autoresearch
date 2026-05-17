"""
The single file the agent edits.

These are the vLLM serving knobs. The agent's job is to find values that
maximize the per-workload scores from ``benchmark.py`` without crashing.

Constants at the bottom (MODEL, QUANTIZATION, TENSOR_PARALLEL_SIZE, …) are
locked — the agent should not change those. Treat them like ``prepare.py`` in
autoresearch.

Note on vLLM compatibility: ``launch_vllm.py`` probes ``vllm serve --help`` and
silently drops any flag that the installed vLLM no longer accepts (e.g.
``--scheduler-delay-factor`` was removed in vLLM V1). Knobs marked
*legacy* below may therefore be no-ops on recent vLLM; they are kept for
backwards compatibility with older builds.
"""

# ---- tunable knobs ---------------------------------------------------------

GPU_MEMORY_UTILIZATION: float = 0.85   # fraction of VRAM vLLM may claim
MAX_NUM_SEQS: int = 64                 # max concurrent sequences in the scheduler
MAX_MODEL_LEN: int = 8192              # max context window served (smaller = more KV headroom)
KV_CACHE_DTYPE: str = "fp8"            # "auto" | "fp8" | "fp8_e5m2" | "fp8_e4m3" | … (auto = safest baseline)
BLOCK_SIZE: int = 16                   # paged-attention block size (8/16/32)

ENABLE_CHUNKED_PREFILL: bool = True    # interleave prefill with decode
ENABLE_PREFIX_CACHING: bool = True     # cache shared prefixes across requests

# Scheduler / batching
MAX_NUM_BATCHED_TOKENS: int = 8192     # max tokens per scheduler step; interacts with chunked prefill

# ---- legacy knobs (likely no-ops on current vLLM, kept for compatibility) --

SWAP_SPACE_GB: int = 4                 # CPU swap for KV blocks (removed in vLLM ≥ 0.20)
SCHEDULER_DELAY_FACTOR: float = 0.0    # removed in vLLM V1 — kept for older builds

# ---- sampling defaults applied during the benchmark (workload may override) -

TEMPERATURE: float = 0.0
TOP_P: float = 1.0

# ---- backend selector ------------------------------------------------------

# "vllm"      → use vLLM with the MODEL/QUANTIZATION below (AWQ, GPTQ, FP8, …)
# "llama_cpp" → use llama-server with LLAMA_CPP_MODEL (a GGUF path)
# The benchmark hits the same OpenAI-compatible endpoint either way.
BACKEND: str = "vllm"

# ---- llama.cpp knobs (only used when BACKEND == "llama_cpp") ---------------
#
# llama.cpp installs and GGUF downloads are NOT managed by this repo.  See
# program.md for setup notes.  When the backend is selected, run.py refuses
# to start if the binary or the GGUF can't be found.
LLAMA_CPP_BIN: str = "/home/yannik/AI/llama.cpp/build/bin/llama-server"
LLAMA_CPP_MODEL: str = "/home/yannik/AI/models/store/gguf/bartowski/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q4_K_M.gguf"

LLAMA_CPP_N_GPU_LAYERS: int = 999        # 999 = offload everything we can
# ctx-size + parallel + kv-unified interact:
#   * Default (kv-unified=False): KV is pre-divided per slot.  Per-slot ctx
#     = LLAMA_CPP_CTX_SIZE // LLAMA_CPP_PARALLEL.  Long prompts that exceed
#     the per-slot cap return HTTP 400.
#   * kv-unified=True: KV is a single shared pool (paged-attention-ish).
#     Slots draw from the pool dynamically; you can run many concurrent
#     sequences as long as the *sum* of their KV fits in CTX_SIZE.
# Use kv-unified for a fairer comparison against vLLM's paged attention.
LLAMA_CPP_CTX_SIZE: int = 32768          # total KV budget (shared if kv-unified)
LLAMA_CPP_PARALLEL: int = 64             # max concurrent sequences
LLAMA_CPP_KV_UNIFIED: bool = True        # shared KV pool (vLLM-ish); False = pre-divided
LLAMA_CPP_BATCH_SIZE: int = 2048         # logical batch
LLAMA_CPP_UBATCH_SIZE: int = 512         # physical/micro batch (matters for prefill speed)
LLAMA_CPP_TENSOR_SPLIT: str = "1,1"      # split across both 5090s
LLAMA_CPP_MAIN_GPU: int | None = 0       # which GPU holds the KV / scratch
LLAMA_CPP_CACHE_TYPE_K: str = "f16"      # KV-K dtype: f16 / q8_0 / q4_0 (analogue of KV_CACHE_DTYPE)
LLAMA_CPP_CACHE_TYPE_V: str = "f16"      # KV-V dtype
LLAMA_CPP_FLASH_ATTN: bool = True        # FlashAttention kernels
LLAMA_CPP_CONT_BATCHING: bool = True     # continuous batching (analogue of vLLM scheduler)
LLAMA_CPP_NO_MMAP: bool = False
LLAMA_CPP_MLOCK: bool = False
LLAMA_CPP_EXTRA_ARGS: tuple[str, ...] = ()   # escape hatch for one-off flags

# ---- locked constants (do not modify) --------------------------------------

MODEL: str = "/home/yannik/AI/models/store/safetensors/RedHatAI/Llama-3.3-70B-Instruct-NVFP4"
QUANTIZATION: str = "compressed-tensors"
TENSOR_PARALLEL_SIZE: int = 2
HOST: str = "127.0.0.1"
PORT: int = 8003
SERVED_MODEL_NAME: str = "llama-3.3-70b"

# Benchmark parameters (also locked — fair-comparison invariants)
BENCH_CONCURRENCY: int = 16            # simultaneous in-flight requests per workload
BENCH_DURATION_SECONDS: int = 60       # how long to fire each workload
BENCH_SLO_INTER_TOKEN_MS: float = 100  # p99 inter-token target; over this, penalty kicks in
BENCH_SLO_TTFT_MS: float = 2000        # p95 time-to-first-token target

# Workload profiles to evaluate.  Each profile has its own score.
#
# SLO override semantics (per profile):
#   - ``slo_inter_token_ms``: if None, no per-token latency penalty applies
#     (use for throughput-bound profiles like batch).
#   - ``slo_ttft_ms``: if None, no TTFT penalty applies (use for prefill-bound
#     profiles like long_context where TTFT is dominated by prompt length, not
#     by serving config).
#   - If a key is absent, the global BENCH_SLO_* defaults are used.
BENCH_PROFILES: tuple[dict, ...] = (
    {
        "name": "interactive",
        "path": "workload/interactive.jsonl",
        "concurrency_override": 16,
        "slo_ttft_ms": 1000,
        "slo_inter_token_ms": 80,
    },
    {
        "name": "coding",
        "path": "workload/coding.jsonl",
        "concurrency_override": 16,
        "slo_ttft_ms": 2000,
        "slo_inter_token_ms": 100,
    },
    {
        # Batch is throughput-only — neither TTFT nor per-token latency matters.
        "name": "batch",
        "path": "workload/batch.jsonl",
        "concurrency_override": 64,
        "slo_ttft_ms": None,
        "slo_inter_token_ms": None,
    },
    {
        # Long-context TTFT is dominated by prefill length, not by config —
        # judge on output throughput + per-token cadence only.
        "name": "long_context",
        "path": "workload/long_context.jsonl",
        "concurrency_override": 4,
        "slo_ttft_ms": None,
        "slo_inter_token_ms": 100,
    },
)

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
KV_CACHE_DTYPE: str = "auto"           # "auto" | "fp8" | "fp8_e5m2" | "fp8_e4m3" | … (auto = safest baseline)
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

# ---- locked constants (do not modify) --------------------------------------

MODEL: str = "casperhansen/llama-3.3-70b-instruct-awq"
QUANTIZATION: str = "awq_marlin"
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
# Set ``concurrency_override`` to None to use BENCH_CONCURRENCY.
BENCH_PROFILES: tuple[dict, ...] = (
    {"name": "interactive",  "path": "workload/interactive.jsonl",  "concurrency_override": 16},
    {"name": "coding",       "path": "workload/coding.jsonl",       "concurrency_override": 16},
    {"name": "batch",        "path": "workload/batch.jsonl",        "concurrency_override": 64},
    {"name": "long_context", "path": "workload/long_context.jsonl", "concurrency_override": 4},
)

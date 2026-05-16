"""
The single file the agent edits.

These are the vLLM serving knobs. The agent's job is to find values that maximize
the score from benchmark.py without crashing.

Constants at the bottom (MODEL, QUANTIZATION, TENSOR_PARALLEL_SIZE) are locked —
the agent should not change those. Treat them like prepare.py in autoresearch.
"""

# ---- tunable knobs ---------------------------------------------------------

GPU_MEMORY_UTILIZATION: float = 0.85   # fraction of VRAM vLLM may claim
MAX_NUM_SEQS: int = 64                 # max concurrent sequences in the scheduler
MAX_MODEL_LEN: int = 8192              # max context window served (smaller = more KV headroom)
KV_CACHE_DTYPE: str = "fp8"            # "auto" | "fp8" | "fp8_e5m2"
BLOCK_SIZE: int = 16                   # paged-attention block size (8/16/32)
SWAP_SPACE_GB: int = 4                 # CPU swap for KV blocks (preempt-and-swap path)

ENABLE_CHUNKED_PREFILL: bool = True    # interleave prefill with decode
ENABLE_PREFIX_CACHING: bool = True     # cache shared prefixes across requests

# Scheduler / batching
MAX_NUM_BATCHED_TOKENS: int = 8192     # max tokens per scheduler step; interacts with chunked prefill
SCHEDULER_DELAY_FACTOR: float = 0.0    # 0.0 = greedy schedule; >0 delays prefill to batch larger

# Sampling defaults applied during the benchmark (the workload may override)
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
BENCH_CONCURRENCY: int = 16            # simultaneous in-flight requests
BENCH_DURATION_SECONDS: int = 90       # how long to fire the workload
BENCH_SLO_INTER_TOKEN_MS: float = 100  # p99 inter-token target; over this, penalty kicks in

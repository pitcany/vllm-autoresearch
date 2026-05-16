# vllm-autoresearch

Autonomous search for the optimal vLLM serving configuration on the current hardware.

## Setup

1. **Confirm the branch**: Use `vllm-autoresearch/<tag>` (e.g. `vllm-autoresearch/may16`). Create from `master` if it doesn't exist.
2. **Read the in-scope files**: This repo is small. Read all of them:
   - `README.md` — overview
   - `config.py` — knobs you can change (top section) and constants you cannot (bottom section)
   - `launch_vllm.py` — how the server is started; do NOT edit
   - `benchmark.py` — how the metric is computed; do NOT edit
   - `run.py` — the experiment loop
3. **Verify workload**: `workload/prompts.jsonl` must exist and have at least 50 prompts that are representative of real traffic.
4. **Verify hardware**: Confirm the right number of GPUs are visible and idle (`nvidia-smi` — no other vLLM/training processes).
5. **Initialize results.tsv**: Create with just the header row.
6. **Confirm and go**: Confirm setup looks good with the human.

Then start experimentation.

## Experimentation

Each experiment runs as: `uv run run.py > run.log 2>&1`. This launches vLLM (~2-3 min for a 70B model), runs the 90-second benchmark, tears down vLLM, and prints metrics. Total per iteration: 6-10 min.

**What you CAN do:**
- Modify any knob in the top section of `config.py`: `GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`, `MAX_MODEL_LEN`, `KV_CACHE_DTYPE`, `BLOCK_SIZE`, `SWAP_SPACE_GB`, `ENABLE_CHUNKED_PREFILL`, `ENABLE_PREFIX_CACHING`, `MAX_NUM_BATCHED_TOKENS`, `SCHEDULER_DELAY_FACTOR`.

**What you CANNOT do:**
- Modify the constants at the bottom of `config.py` (MODEL, QUANTIZATION, TENSOR_PARALLEL_SIZE, HOST, PORT, SERVED_MODEL_NAME, BENCH_*). These are invariants for fair comparison.
- Modify `launch_vllm.py`, `benchmark.py`, `workload/prompts.jsonl`. These are the harness.
- Install new packages or change the vLLM version.

**The goal: maximize `score` from the benchmark.** Higher is better (unlike autoresearch's val_bpb).

**VRAM** is a hard constraint: if `GPU_MEMORY_UTILIZATION` is too high vLLM will OOM. If you push it from 0.85 → 0.95 and the server fails to start, that's a crash — log and back off.

**Simplicity criterion**: All else being equal, simpler is better. Don't push knobs to weird values for a 0.1% gain.

**First run**: Always establish the baseline. Run as-is.

## Output format

`run.py` prints lines like:

```
score:           1234.5
throughput:      1400.2 tok/s
p50_inter_token: 35.0 ms
p99_inter_token: 89.0 ms
completed:       128
errored:         0
```

Extract `score` from `run.log`:

```
grep "^score:" run.log
```

## Logging results

TSV with header: `commit\tscore\tp99_ms\tstatus\tdescription`

- commit: short SHA
- score: 0.00 for crashes
- p99_ms: 0.0 for crashes
- status: keep | discard | crash
- description: one line, what you tried

## The experiment loop

```
LOOP FOREVER:
  1. Look at git state
  2. Tune config.py with one experimental idea
  3. git commit
  4. uv run run.py > run.log 2>&1
  5. grep score / p99 / errored from run.log
  6. If errored > 10% of completed → treat as crash even if score is positive
  7. Log to results.tsv (untracked)
  8. If score went UP and errored is low: keep the commit
  9. If score went down or errors spiked: git reset
```

**Crashes**:
- vLLM fails to start → almost always GPU_MEMORY_UTILIZATION too high or MAX_MODEL_LEN too long for the KV budget. Log "crash" and try a less aggressive value.
- vLLM starts but errors during benchmark → check `vllm.log`. If it's a config mismatch (e.g. block_size doesn't divide max_model_len), back out.

**Timeout**: Each iteration should take ~6-10 min. If `uv run run.py` exceeds 20 min, kill it (`pkill -f vllm.entrypoints`), wait for VRAM to free, treat as crash.

**NEVER STOP**: Once the loop has begun, do not pause to ask the human if you should continue. Run until interrupted.

## Knob priors (seed your search)

- `GPU_MEMORY_UTILIZATION`: try 0.85 → 0.90 → 0.92 → 0.95 (more KV cache → higher throughput; OOM risk)
- `MAX_NUM_SEQS`: try 32, 64, 128, 256 (interacts strongly with chunked prefill)
- `MAX_MODEL_LEN`: smaller is much better — set this to the smallest value that fits your real workload
- `KV_CACHE_DTYPE`: fp8 saves a lot of memory at small quality cost — usually a win on tight VRAM
- `MAX_NUM_BATCHED_TOKENS`: scales with `MAX_NUM_SEQS`; usually want ≥ MAX_NUM_SEQS × typical_prompt_len
- `BLOCK_SIZE`: 16 is the default; 8 or 32 occasionally helps depending on workload shape

Don't sweep blindly — gradient direction matters. If GPU_MEM=0.90 wins, try 0.92 next, not 0.85.

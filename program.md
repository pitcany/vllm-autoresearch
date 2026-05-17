# vllm-autoresearch — agent playbook

Autonomous search for the optimal vLLM serving configuration on this hardware.

The agent's job: edit `config.py`, run one benchmark iteration, decide
keep/discard, repeat.  It does **not** edit anything else.

## Setup (read this first)

1. **Branch**: work on a topical branch (e.g. `vllm-autoresearch/<tag>`) cut from
   `main`.  Never commit straight to `main`.
2. **Environment**: this harness expects the `vllm-serve` conda env, which has
   vLLM 0.20.x + PyTorch 2.11 + CUDA 12.9.  Invoke as:

   ```bash
   conda run -n vllm-serve python run.py --baseline
   ```

   `uv sync` is **not** used — vLLM on Blackwell needs the prebuilt wheel that
   already lives in the conda env.
3. **Read the in-scope files** (this repo is small — read all of them):
   - `README.md` — overview and result format
   - `config.py` — tunable knobs (top), legacy compat knobs, locked invariants
   - `launch_vllm.py` — version-aware vLLM spawn; do NOT edit
   - `benchmark.py` — per-profile scoring; do NOT edit
   - `run.py` — experiment loop and results.tsv writer; do NOT edit
   - `workload/*.jsonl` — four profile workloads; do NOT edit
4. **Workload reality**: every prompt is currently marked `"synthetic": true`.
   Until real (anonymised) production prompts replace them, the absolute
   numbers are not trustworthy — only *relative* comparisons between configs
   on the *same* synthetic workload are valid.
5. **Hardware check**: `nvidia-smi` should show 2× RTX 5090 idle.  Kill any
   stray vLLM/training processes first.
6. **results.tsv**: created automatically on the first run.  Header is
   `commit  config_hash  baseline  interactive_score  coding_score  batch_score  long_context_score  worst_p99_ttft_ms  worst_p99_inter_ms  completed  errored  timed_out  startup_s  synthetic  status  description`.

## Experimentation

Each iteration runs as:

```bash
/home/yannik/miniconda3/envs/vllm-serve/bin/python -u run.py \
    --description "raised gpu_mem to 0.90" > run.log 2>&1
```

This launches the configured backend (~2-3 min for the 70B model the first
time, faster once weights are cached), runs **four** workload profiles
back-to-back (interactive, coding, batch, long_context — ~60 s each), tears
the server down, prints metrics, and appends a row to `results.tsv`.

Total per iteration: **~6-10 min**.

> **Why not `conda run`?**  `conda run --no-capture-output` still buffers
> stdout in non-TTY contexts — `run.log` stays empty for minutes even
> though the run is progressing.  Invoke the env's `python` directly to
> get real-time progress.

## Backends

Two backends are supported, both speaking the same OpenAI `/v1/completions`
shape so the same workloads, scoring, and `results.tsv` schema apply to both.

| `config.BACKEND` | model format | launcher          | log file           |
|------------------|--------------|-------------------|--------------------|
| `"vllm"`         | HF (AWQ/GPTQ/FP8) | `launch_vllm.py`     | `vllm.log`         |
| `"llama_cpp"`    | GGUF          | `launch_llama_cpp.py`| `llama_cpp.log`    |

The `backend` column in `results.tsv` records which one produced each row, so
GGUF rows and AWQ rows can be compared side-by-side.

### llama.cpp setup (one-time)

The repo does **not** install llama.cpp or download GGUFs.

1.  Build `llama-server` with CUDA on this box, e.g.:
    ```bash
    git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
    cd ~/llama.cpp && cmake -B build -DGGML_CUDA=on && cmake --build build -j
    ```
    Either symlink the binary onto `PATH` or set `config.LLAMA_CPP_BIN` to
    its absolute path (e.g. `/home/yannik/llama.cpp/build/bin/llama-server`).

2.  Download a Llama-3.3-70B-Instruct GGUF whose bits-per-weight roughly
    match the AWQ baseline (Q4_K_M ≈ 4.8 bpw vs AWQ 4 bpw), e.g.:
    ```
    bartowski/Llama-3.3-70B-Instruct-GGUF  →  Llama-3.3-70B-Instruct-Q4_K_M.gguf
    ```
    Set `config.LLAMA_CPP_MODEL` to its absolute path.

3.  Set `config.BACKEND = "llama_cpp"` and run as usual.  The launcher
    refuses to start with a clear message if the binary or model path is
    missing — no silent fallback.

### Cross-stack comparison caveats

* **Bits-per-weight differs.**  AWQ is 4-bit; Q4_K_M is ~4.8-bit.  Q4_0
  (~4.5 bpw) is closer to AWQ but generally lower quality than Q4_K_M.
* **Tokenizer is shared** (Llama-3 BPE) — token counts compare cleanly.
* **Tensor parallel differs.**  vLLM does true tensor parallel; llama.cpp
  splits layers across GPUs (pipeline-ish).  Don't expect identical
  scaling on the same hardware.
* **Quality is not measured here** — only throughput and latency.  If two
  backends produce wildly different scores, run a few prompts through both
  and eyeball the outputs before declaring a winner.

### What you CAN edit

Only knobs in the *tunable* section of `config.py`:

| knob | typical range |
|---|---|
| `GPU_MEMORY_UTILIZATION` | 0.80 – 0.95 |
| `MAX_NUM_SEQS` | 16, 32, 64, 128, 256 |
| `MAX_MODEL_LEN` | as small as fits the real workload |
| `KV_CACHE_DTYPE` | `"auto"`, `"fp8"`, `"fp8_e5m2"`, `"fp8_e4m3"` |
| `BLOCK_SIZE` | 16 (default), occasionally 8 or 32 |
| `ENABLE_CHUNKED_PREFILL` | `True` / `False` |
| `ENABLE_PREFIX_CACHING` | `True` / `False` |
| `MAX_NUM_BATCHED_TOKENS` | scales with `MAX_NUM_SEQS × typical prompt` |

### Legacy / no-op knobs

`SWAP_SPACE_GB` and `SCHEDULER_DELAY_FACTOR` are still in `config.py` but vLLM
0.20 no longer accepts them; the launcher drops them automatically.  Don't waste
iterations sweeping these.

### What you CANNOT edit

- Locked constants in `config.py` (`MODEL`, `QUANTIZATION`,
  `TENSOR_PARALLEL_SIZE`, `HOST`, `PORT`, `SERVED_MODEL_NAME`, all `BENCH_*`).
- `launch_vllm.py`, `launch_llama_cpp.py`, `benchmark.py`, `run.py`,
  `workload/*.jsonl`.
- The vLLM or llama.cpp version, or any installed package.

### Unsafe configurations (skip these, they will OOM or hang)

- `GPU_MEMORY_UTILIZATION > 0.95` on 2× 32 GB cards with a 70B AWQ model
  rarely succeeds — back off after one crash.
- `MAX_MODEL_LEN > 16384` with `KV_CACHE_DTYPE="auto"` and high concurrency.
- `MAX_NUM_BATCHED_TOKENS < MAX_NUM_SEQS` (scheduler can't make progress).
- `BLOCK_SIZE != 16` with `ENABLE_PREFIX_CACHING=True` on some vLLM builds.

## Choosing configurations

**Goal**: improve *all four* per-profile scores without crashing.  One score
going up while another collapses is **not** a win — that's a dominated
configuration in disguise.

**Pareto rule**: only `keep` if the new config is ≥ baseline on every profile
*and* strictly better on at least one.  Otherwise `discard`.

**Gradient direction**: if 0.90 wins over 0.85, try 0.92 next, not 0.95.  Only
back off after a crash.

**OOM backoff**: when vLLM fails to start, halve the distance you just moved
in the offending direction (e.g. 0.95 crashes from 0.90 → try 0.92, not 0.85).

**Simplicity tiebreak**: if two configs score within 2 %, prefer the one with
fewer non-default knob values.

**Don't sweep blindly**: one knob change per iteration.  After 3 iterations on
the same knob, move to a different knob.

## The experiment loop

```
LOOP UNTIL INTERRUPTED:
  1. Read results.tsv and find the current best.
  2. Pick ONE knob to change.  Justify the direction in `--description`.
  3. Edit config.py.
  4. git add -A && git commit -m "<short description>"
  5. conda run -n vllm-serve python run.py --description "<same>" > run.log 2>&1
  6. Read the SUMMARY line and the per-profile scores.
  7. If status == "crash": git reset --hard HEAD~1; back off; continue.
  8. If Pareto-dominates the current best: keep the commit.
  9. Otherwise: git reset --hard HEAD~1.
 10. Repeat.
```

### Failure modes

- **vLLM fails to start**: almost always `GPU_MEMORY_UTILIZATION` too high or
  `MAX_MODEL_LEN` too large for the KV budget.  Log "crash", back off.
- **vLLM starts but errors during benchmark**: check `vllm.log`.  Config
  mismatch (e.g. `BLOCK_SIZE` not divisible into `MAX_MODEL_LEN`) → back out.
- **Timeout**: each iteration should take ~6-10 min.  If `run.py` exceeds 20
  min, kill it (`pkill -f vllm.entrypoints`), wait 30 s for VRAM, treat as crash.
- **TWO consecutive OOM crashes**: stop touching the offending knob for at
  least 5 iterations.

### Overnight budget

Each iteration is ~6-10 min.  A 10-hour overnight run is **60-100 iterations
max**.  Don't queue more.  If you've done 30 iterations and the best score
hasn't moved by ≥ 5 %, switch to a different knob or stop.

## Output format

`run.py` prints a `SUMMARY {…}` JSON line and a row in `results.tsv`.  Extract
scores like:

```bash
grep '^SUMMARY ' run.log | tail -1
tail -1 results.tsv
```

## Synthetic-workload caveat

Every prompt in `workload/*.jsonl` is currently flagged `"synthetic": true`.
Conclusions from these runs are useful for **relative ranking of configs**, not
for absolute claims like "this serves X tok/s in production".  Until at least
50 real anonymised prompts replace each profile, do not promote any result as
"the production config".

## Never claim "optimised" before a clean baseline

Always run `--baseline` first, with the stock `config.py`, on a clean
worktree.  A run that hasn't beaten the baseline on at least three profiles
isn't an improvement.

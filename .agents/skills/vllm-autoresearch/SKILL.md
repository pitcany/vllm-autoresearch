```markdown
# vllm-autoresearch Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you how to contribute effectively to the `vllm-autoresearch` repository, a Python-based benchmarking harness for large language model (LLM) inference engines. You'll learn the project's coding conventions, how to tune configurations, add backend support, improve the harness launcher, enhance benchmarking, and update documentation. The guide includes step-by-step workflows, code examples, and suggested commands for common tasks.

## Coding Conventions

- **File Naming:**  
  Use `camelCase` for Python files (e.g., `launchVllm.py`, `runBenchmark.py`).

- **Import Style:**  
  Use **relative imports** within the package.
  ```python
  from .config import KV_CACHE_DTYPE
  from . import benchmark
  ```

- **Export Style:**  
  Use **named exports** (define only what should be public).
  ```python
  __all__ = ["runBenchmark", "KV_CACHE_DTYPE"]
  ```

- **Commit Messages:**  
  - Prefixes: `feat`, `fix`, `docs`, `refactor`, `bench`, `config`, `baseline`, `harness`, `iter1`, `iter2`, `iter3`, `revert`
  - Example:  
    ```
    config: iter2 - increase MAX_NUM_BATCHED_TOKENS for throughput experiment
    ```

## Workflows

### Config Parameter Tuning Iteration
**Trigger:** When you want to run a new tuning experiment or revert a previous config change  
**Command:** `/new-tuning-iteration`

1. Edit `config.py` to adjust relevant parameters (e.g., `KV_CACHE_DTYPE`, `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`).
   ```python
   # Example: Change cache dtype to float16
   KV_CACHE_DTYPE = "float16"
   ```
2. Commit with a message indicating the iteration and rationale.
   ```
   config: iter3 - test float16 cache for memory efficiency
   ```
3. Optionally, revert previous iterations if results are unsatisfactory.
   ```
   revert: iter2 - revert cache dtype to default
   ```

---

### Backend Support Extension
**Trigger:** When you want to benchmark against a new backend or inference engine  
**Command:** `/add-backend`

1. Add or modify a launcher script for the new backend (e.g., `launch_llama_cpp.py`).
   ```python
   # launch_llama_cpp.py
   def launch_llama_cpp(args):
       # Backend-specific launch logic
       pass
   ```
2. Update `config.py` to add backend-specific knobs and selection logic.
   ```python
   BACKEND = "llama_cpp"
   ```
3. Update `run.py` to dispatch based on the selected backend.
   ```python
   if config.BACKEND == "llama_cpp":
       from .launch_llama_cpp import launch_llama_cpp
       launch_llama_cpp(args)
   ```
4. Update `benchmark.py` or other scripts if needed.
5. Update documentation (`program.md`) to describe the new backend setup and usage.

---

### Harness Launcher Improvement
**Trigger:** When you want to improve process management, signal handling, or environment setup for the benchmarking harness  
**Command:** `/improve-launcher`

1. Edit `launch_vllm.py` and/or `launch_llama_cpp.py` to improve startup checks, environment variables, or teardown logic.
   ```python
   import signal

   def handle_sigterm(signum, frame):
       # Cleanup logic
       pass

   signal.signal(signal.SIGTERM, handle_sigterm)
   ```
2. Edit `run.py` to improve signal handling or process lifecycle.
3. Optionally, update related documentation.

---

### Per-Profile Benchmarking Enhancement
**Trigger:** When you want to improve the granularity or accuracy of benchmark results per workload/profile  
**Command:** `/enhance-benchmarking`

1. Edit `benchmark.py` to add or refine per-profile metrics/SLO logic.
   ```python
   def report_metrics(profile, metrics):
       print(f"Profile: {profile}, Latency: {metrics['latency']}")
   ```
2. Edit `config.py` to support per-profile configuration.
3. Edit `run.py` to record/report new metrics.
4. Update or add `workload/*.jsonl` files if new profiles or prompt sets are needed.

---

### Documentation Update After Feature or Change
**Trigger:** When you add a new feature, backend, or complete a tuning iteration with notable results  
**Command:** `/update-docs`

1. Edit relevant documentation files (`README.md`, `program.md`, `FINDINGS.md`) to describe the new feature, workflow, or findings.
2. Commit with a message summarizing the documentation update.
   ```
   docs: update program.md with llama.cpp backend instructions
   ```

## Testing Patterns

- **Framework:** Unknown (no explicit framework detected)
- **Test File Pattern:** Files matching `*.test.*` (e.g., `benchmark.test.py`)
- **Style:**  
  - Place test files alongside implementation or in a dedicated test directory.
  - Use descriptive function names for tests.
  - Example:
    ```python
    # benchmark.test.py
    def test_latency_measurement():
        assert measure_latency() < 100
    ```

## Commands

| Command                | Purpose                                                      |
|------------------------|--------------------------------------------------------------|
| /new-tuning-iteration  | Start a new config parameter tuning experiment               |
| /add-backend           | Add support for a new backend/inference engine               |
| /improve-launcher      | Refactor or improve the harness launcher scripts             |
| /enhance-benchmarking  | Enhance per-profile benchmarking and reporting               |
| /update-docs           | Update documentation after a feature or experiment           |
```
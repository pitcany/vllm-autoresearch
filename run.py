"""
The experiment loop.

  uv run run.py --baseline   # one run, record as baseline
  uv run run.py              # one run with current config

The agent edits config.py between iterations, commits, runs `uv run run.py`,
reads run.log, logs the result to results.tsv, and decides keep/discard.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import benchmark
import launch_vllm


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "no_git"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="label this run as the baseline")
    parser.add_argument("--description", type=str, default="", help="short description for results.tsv")
    args = parser.parse_args()

    print(f"--- launching vLLM ({_git_short_sha()}) ---")
    t_launch = time.time()
    proc, teardown = launch_vllm.launch()
    if proc is None:
        print("vLLM failed to start (OOM or config error). See vllm.log.")
        print(f"score:          0.0")
        print(f"status:         crash")
        return 1
    print(f"vLLM ready in {time.time() - t_launch:.0f}s. Starting benchmark.")

    try:
        result = benchmark.run()
    finally:
        print("Tearing down vLLM…")
        teardown()
        time.sleep(5)  # let VRAM settle

    print(f"score:           {result.score:.2f}")
    print(f"throughput:      {result.throughput_tok_per_s:.1f} tok/s")
    print(f"p50_inter_token: {result.p50_inter_token_ms:.1f} ms")
    print(f"p99_inter_token: {result.p99_inter_token_ms:.1f} ms")
    print(f"completed:       {result.n_requests_completed}")
    print(f"errored:         {result.n_requests_errored}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

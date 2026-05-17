"""
The experiment loop.

  uv run run.py --baseline               # one run, recorded as the baseline
  uv run run.py --description "fp8 + chunked"
  uv run run.py                          # one run with current config

When --baseline or --description is set, the result is appended to results.tsv
so the agent has a durable, comparable log of attempts.

The agent edits ``config.py`` between iterations, commits, runs ``uv run
run.py``, reads ``run.log``, then logs to ``results.tsv``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import benchmark
import launch_vllm


_RESULTS_HEADER = (
    "commit\tconfig_hash\tbaseline\tinteractive_score\tcoding_score\t"
    "batch_score\tlong_context_score\tworst_p99_ttft_ms\tworst_p99_inter_ms\t"
    "completed\terrored\ttimed_out\tstartup_s\tsynthetic\tstatus\tdescription\n"
)


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "no_git"


def _profile_score(report: benchmark.BenchReport, name: str) -> float:
    for p in report.profiles:
        if p.name == name:
            return p.score
    return 0.0


def _append_results_row(
    report: benchmark.BenchReport,
    startup_s: float,
    status: str,
    description: str,
    baseline: bool,
) -> None:
    results = Path("results.tsv")
    if not results.exists() or results.stat().st_size == 0:
        results.write_text(_RESULTS_HEADER)
    elif not results.read_text().startswith("commit\t"):
        # legacy header — leave alone but warn
        pass

    interactive = _profile_score(report, "interactive")
    coding = _profile_score(report, "coding")
    batch_s = _profile_score(report, "batch")
    long_ctx = _profile_score(report, "long_context")

    worst_ttft = max((p.ttft_p99_ms for p in report.profiles if p.ttft_p99_ms != float("inf")), default=0.0)
    worst_inter = max((p.inter_token_p99_ms for p in report.profiles if p.inter_token_p99_ms != float("inf")), default=0.0)
    completed = sum(p.completed for p in report.profiles)
    errored = sum(p.errored for p in report.profiles)
    timed_out = sum(p.timed_out for p in report.profiles)

    row = (
        f"{_git_short_sha()}\t{report.config_hash}\t{int(baseline)}\t"
        f"{interactive:.2f}\t{coding:.2f}\t{batch_s:.2f}\t{long_ctx:.2f}\t"
        f"{worst_ttft:.1f}\t{worst_inter:.1f}\t"
        f"{completed}\t{errored}\t{timed_out}\t"
        f"{startup_s:.1f}\t{int(report.synthetic)}\t{status}\t{description}\n"
    )
    with results.open("a") as f:
        f.write(row)


def _crash_row(startup_s: float, dropped_flags: list[str], description: str, baseline: bool) -> None:
    results = Path("results.tsv")
    if not results.exists() or results.stat().st_size == 0:
        results.write_text(_RESULTS_HEADER)
    note = description or ("dropped_flags=" + ",".join(dropped_flags) if dropped_flags else "")
    row = (
        f"{_git_short_sha()}\t-\t{int(baseline)}\t"
        f"0.00\t0.00\t0.00\t0.00\t0.0\t0.0\t0\t0\t0\t"
        f"{startup_s:.1f}\t1\tcrash\t{note}\n"
    )
    with results.open("a") as f:
        f.write(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="label this run as the baseline")
    parser.add_argument("--description", type=str, default="", help="short description for results.tsv")
    args = parser.parse_args()

    print(f"--- launching vLLM ({_git_short_sha()}) ---")
    proc, teardown, linfo = launch_vllm.launch()
    if linfo.dropped_flags:
        print(f"NOTE: dropped flags not supported by installed vLLM: {linfo.dropped_flags}")
    print(f"vLLM version: {linfo.vllm_version}")
    print(f"gpu_snapshot_before: {linfo.gpu_snapshot_before}")
    print(f"startup_s: {linfo.startup_seconds:.1f}")

    if proc is None:
        print("vLLM failed to start (OOM, config error, or timeout). See vllm.log.")
        print("score_interactive:   0.0")
        print("score_coding:        0.0")
        print("score_batch:         0.0")
        print("score_long_context:  0.0")
        print("status:              crash")
        _crash_row(linfo.startup_seconds, linfo.dropped_flags, args.description, args.baseline)
        return 1

    print("vLLM ready. Running benchmark profiles…")
    try:
        report = benchmark.run()
    finally:
        print("Tearing down vLLM…")
        teardown()
        time.sleep(5)  # let VRAM settle

    benchmark.print_report(report)

    # Determine status: too many errors → flag as crash even if scores look positive.
    completed = sum(p.completed for p in report.profiles)
    errored = sum(p.errored for p in report.profiles)
    status = "ok"
    if completed == 0:
        status = "crash"
    elif errored > 0.10 * (completed + errored):
        status = "crash"

    _append_results_row(report, linfo.startup_seconds, status, args.description, args.baseline)

    # Always emit a one-line summary for greppable log parsing.
    summary = {
        "config_hash": report.config_hash,
        "scores": {p.name: p.score for p in report.profiles},
        "status": status,
        "synthetic": report.synthetic,
    }
    print("SUMMARY " + json.dumps(summary))
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())

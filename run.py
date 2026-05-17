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
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import benchmark
import config
import launch_llama_cpp
import launch_vllm


def _register_early_teardown(td: Callable[[], None]) -> None:
    """Make teardown reachable to the signal handler the moment Popen succeeds,
    so SIGTERM during startup (e.g. a slow model load) still kills the server."""
    global _PENDING_TEARDOWN
    _PENDING_TEARDOWN = td


def _launch_for_backend():
    """Dispatch to the configured backend.  Returns (proc, teardown, info)."""
    name = (config.BACKEND or "vllm").lower()
    if name == "vllm":
        return launch_vllm.launch(on_spawn=_register_early_teardown)
    if name == "llama_cpp":
        return launch_llama_cpp.launch(on_spawn=_register_early_teardown)
    raise ValueError(
        f"Unknown BACKEND={config.BACKEND!r}; expected 'vllm' or 'llama_cpp'"
    )


def _log_path_for_backend() -> str:
    return "llama_cpp.log" if (config.BACKEND or "").lower() == "llama_cpp" else "vllm.log"


_PENDING_TEARDOWN: Optional[Callable[[], None]] = None


def _install_signal_teardown() -> None:
    """Tear down vLLM if we get SIGTERM/SIGINT. Otherwise the child survives
    because the launcher puts it in its own session (preexec_fn=os.setsid)."""

    def _handler(signum, _frame):
        global _PENDING_TEARDOWN
        if _PENDING_TEARDOWN is not None:
            try:
                _PENDING_TEARDOWN()
            except Exception:
                pass
            _PENDING_TEARDOWN = None
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


_RESULTS_HEADER = (
    "commit\tbackend\tconfig_hash\tbaseline\tinteractive_score\tcoding_score\t"
    "batch_score\tlong_context_score\treasoning_score\t"
    "worst_p99_ttft_ms\tworst_p99_inter_ms\t"
    "completed\terrored\ttimed_out\tstartup_s\tsynthetic\tstatus\tdescription\n"
)


def _ensure_results_header(results: Path) -> None:
    if not results.exists() or results.stat().st_size == 0:
        results.write_text(_RESULTS_HEADER)
        return
    lines = results.read_text().splitlines()
    if not lines:
        results.write_text(_RESULTS_HEADER)
        return
    header = lines[0]
    if header == _RESULTS_HEADER.rstrip("\n"):
        return
    if not header.startswith("commit\t"):
        return

    new_cols = _RESULTS_HEADER.rstrip("\n").split("\t")
    old_cols = header.split("\t")
    insert_index = None
    if "reasoning_score" not in old_cols:
        insert_index = new_cols.index("reasoning_score")

    updated_lines = ["\t".join(new_cols)]
    for line in lines[1:]:
        if not line:
            continue
        row = line.split("\t")
        if insert_index is not None and len(row) == len(new_cols) - 1:
            row.insert(insert_index, "0.0")
        if len(row) < len(new_cols):
            row += [""] * (len(new_cols) - len(row))
        updated_lines.append("\t".join(row))

    results.write_text("\n".join(updated_lines) + "\n")


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
    backend: str,
) -> None:
    results = Path("results.tsv")
    _ensure_results_header(results)

    interactive = _profile_score(report, "interactive")
    coding = _profile_score(report, "coding")
    batch_s = _profile_score(report, "batch")
    long_ctx = _profile_score(report, "long_context")
    reasoning = _profile_score(report, "reasoning")

    worst_ttft = max((p.ttft_p99_ms for p in report.profiles if p.ttft_p99_ms != float("inf")), default=0.0)
    worst_inter = max((p.inter_token_p99_ms for p in report.profiles if p.inter_token_p99_ms != float("inf")), default=0.0)
    completed = sum(p.completed for p in report.profiles)
    errored = sum(p.errored for p in report.profiles)
    timed_out = sum(p.timed_out for p in report.profiles)

    row = (
        f"{_git_short_sha()}\t{backend}\t{report.config_hash}\t{int(baseline)}\t"
        f"{interactive:.2f}\t{coding:.2f}\t{batch_s:.2f}\t{long_ctx:.2f}\t{reasoning:.2f}\t"
        f"{worst_ttft:.1f}\t{worst_inter:.1f}\t"
        f"{completed}\t{errored}\t{timed_out}\t"
        f"{startup_s:.1f}\t{int(report.synthetic)}\t{status}\t{description}\n"
    )
    with results.open("a") as f:
        f.write(row)


def _crash_row(startup_s: float, dropped_flags: list[str], description: str, baseline: bool, backend: str) -> None:
    results = Path("results.tsv")
    _ensure_results_header(results)
    note = description or ("dropped_flags=" + ",".join(dropped_flags) if dropped_flags else "")
    row = (
        f"{_git_short_sha()}\t{backend}\t-\t{int(baseline)}\t"
        f"0.00\t0.00\t0.00\t0.00\t0.00\t0.0\t0.0\t0\t0\t0\t"
        f"{startup_s:.1f}\t1\tcrash\t{note}\n"
    )
    with results.open("a") as f:
        f.write(row)


def main() -> int:
    global _PENDING_TEARDOWN
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="label this run as the baseline")
    parser.add_argument("--description", type=str, default="", help="short description for results.tsv")
    args = parser.parse_args()

    _install_signal_teardown()

    print(f"--- launching {config.BACKEND} ({_git_short_sha()}) ---", flush=True)
    proc, teardown, linfo = _launch_for_backend()
    if teardown is not None:
        _PENDING_TEARDOWN = teardown
    if linfo.dropped_flags:
        print(f"NOTE: dropped flags not supported by installed {linfo.backend}: {linfo.dropped_flags}")
    print(f"backend: {linfo.backend} {linfo.backend_version}")
    print(f"gpu_snapshot_before: {linfo.gpu_snapshot_before}")
    print(f"startup_s: {linfo.startup_seconds:.1f}")

    if proc is None:
        print(f"{linfo.backend} failed to start (OOM, config error, or timeout). See {_log_path_for_backend()}.")
        print("score_interactive:   0.0")
        print("score_coding:        0.0")
        print("score_batch:         0.0")
        print("score_long_context:  0.0")
        print("score_reasoning:     0.0")
        print("status:              crash")
        _crash_row(linfo.startup_seconds, linfo.dropped_flags, args.description, args.baseline, linfo.backend)
        return 1

    print(f"{linfo.backend} ready. Running benchmark profiles…", flush=True)
    try:
        report = benchmark.run()
    finally:
        print(f"Tearing down {linfo.backend}…", flush=True)
        teardown()
        _PENDING_TEARDOWN = None
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

    _append_results_row(report, linfo.startup_seconds, status, args.description, args.baseline, linfo.backend)

    # Always emit a one-line summary for greppable log parsing.
    summary = {
        "backend": linfo.backend,
        "config_hash": report.config_hash,
        "scores": {p.name: p.score for p in report.profiles},
        "status": status,
        "synthetic": report.synthetic,
    }
    print("SUMMARY " + json.dumps(summary))
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())

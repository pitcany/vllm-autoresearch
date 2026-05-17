"""
Variance probe — re-run the current config N times and report the spread per
profile score, so we know the noise floor before judging Pareto wins.

  python variance_probe.py 3 --label "baseline noise"

Writes N rows to results.tsv (one per iteration), then prints:
  * min / max / range / stddev for each of the four profile scores
  * worst worst_p99_ttft_ms and worst_p99_inter_ms

The noise floor it reports is what subsequent tuning iterations should beat
by a margin — anything under the floor is signal masquerading as a win.

Refuses to start if the working tree is dirty (a config edit without a
commit) — the probe is for the *committed* config only, otherwise we're
measuring noise of a moving target.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev


def _committed_short_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def _working_tree_clean() -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    ).stdout
    # Allow modifications to results.tsv and gguf_download.log (runtime artefacts)
    for line in out.splitlines():
        path = line[3:].strip()
        if path not in {"results.tsv", "gguf_download.log"}:
            return False
    return True


def _read_results() -> list[dict]:
    p = Path("results.tsv")
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _summarise(rows: list[dict]) -> None:
    if not rows:
        print("no rows recorded — probe failed")
        return
    profiles = ["interactive_score", "coding_score", "batch_score", "long_context_score"]
    extras = ["worst_p99_ttft_ms", "worst_p99_inter_ms", "startup_s"]
    print()
    print("=== variance probe summary ===")
    print(f"runs: {len(rows)}    commit: {rows[0]['commit']}    config_hash: {rows[0]['config_hash']}")
    print()
    print(f"{'metric':<25} {'min':>10} {'max':>10} {'mean':>10} {'range':>10} {'stddev':>10} {'cv%':>8}")
    for key in profiles + extras:
        vals = [float(r[key]) for r in rows]
        m = mean(vals)
        s = pstdev(vals) if len(vals) > 1 else 0.0
        cv = (s / m * 100) if m > 0 else float("nan")
        print(
            f"{key:<25} {min(vals):>10.2f} {max(vals):>10.2f} {m:>10.2f} "
            f"{max(vals)-min(vals):>10.2f} {s:>10.2f} {cv:>7.2f}%"
        )
    print()
    print("Interpretation: a future tuning iteration is signal only if it")
    print("improves a score by more than ~2× the stddev on that profile.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-run the current config N times to measure noise floor.")
    parser.add_argument("n", type=int, nargs="?", default=3, help="number of runs (default 3)")
    parser.add_argument("--label", type=str, default="variance probe", help="prefix for the description column")
    args = parser.parse_args()

    if not _working_tree_clean():
        print("ERROR: working tree has uncommitted changes.  Commit or stash first —", file=sys.stderr)
        print("the probe must measure a single, frozen config.", file=sys.stderr)
        return 2

    sha = _committed_short_sha()
    before = {(r["commit"], r["config_hash"], r["description"]) for r in _read_results()}

    new_rows = []
    for i in range(1, args.n + 1):
        desc = f"{args.label} {i}/{args.n} @ {sha}"
        print(f"--- probe iteration {i}/{args.n}: {desc} ---", flush=True)
        rc = subprocess.call([
            sys.executable, "-u", "run.py", "--description", desc,
        ])
        print(f"--- iteration {i} returned {rc} ---", flush=True)
        # Find the newly-appended row
        for r in _read_results():
            key = (r["commit"], r["config_hash"], r["description"])
            if key not in before and r["description"] == desc:
                new_rows.append(r)
                before.add(key)
                break

    _summarise(new_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

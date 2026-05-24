---
name: harness-launcher-improvement
description: Workflow command scaffold for harness-launcher-improvement in vllm-autoresearch.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /harness-launcher-improvement

Use this workflow when working on **harness-launcher-improvement** in `vllm-autoresearch`.

## Goal

Refactor or improve the launcher and harness scripts to handle signals, environment, or startup/teardown robustness.

## Common Files

- `launch_vllm.py`
- `launch_llama_cpp.py`
- `run.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit launch_vllm.py and/or launch_llama_cpp.py to improve startup checks, environment variables, or teardown logic.
- Edit run.py to improve signal handling or process lifecycle.
- Optionally, update related documentation.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.
---
name: backend-support-extension
description: Workflow command scaffold for backend-support-extension in vllm-autoresearch.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /backend-support-extension

Use this workflow when working on **backend-support-extension** in `vllm-autoresearch`.

## Goal

Add support for a new backend (e.g., llama.cpp) so the harness can benchmark multiple inference engines with similar workflows.

## Common Files

- `launch_llama_cpp.py`
- `config.py`
- `run.py`
- `benchmark.py`
- `program.md`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Add or modify launcher script for the new backend (e.g., launch_llama_cpp.py).
- Update config.py to add backend-specific knobs and selection logic.
- Update run.py to dispatch based on selected backend.
- Update benchmark.py or other relevant scripts if needed.
- Update documentation (program.md) to describe new backend setup and usage.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.
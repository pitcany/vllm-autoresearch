---
name: config-parameter-tuning-iteration
description: Workflow command scaffold for config-parameter-tuning-iteration in vllm-autoresearch.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /config-parameter-tuning-iteration

Use this workflow when working on **config-parameter-tuning-iteration** in `vllm-autoresearch`.

## Goal

Change one or more config.py parameters to run an experimental iteration, typically to optimize performance or test a hypothesis. Each iteration is labeled (e.g., iter1, iter2, iter3, revert).

## Common Files

- `config.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit config.py to adjust relevant parameters (e.g., KV_CACHE_DTYPE, MAX_NUM_BATCHED_TOKENS, MAX_NUM_SEQS).
- Commit with a message indicating the iteration and rationale.
- Optionally, revert previous iterations if results are unsatisfactory.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.
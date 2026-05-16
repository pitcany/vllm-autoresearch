# vllm-autoresearch

Autonomous research harness for finding the optimal vLLM serving configuration for your model on your hardware.

Same pattern as [autoresearch](https://github.com/karpathy/autoresearch): an AI agent edits a config file, launches a short benchmark, measures a single metric, keeps or discards, and repeats overnight.

## What's different from autoresearch

| | autoresearch | vllm-autoresearch |
|---|---|---|
| What runs | A 5-min training of a tiny GPT | An 8-min benchmark of vLLM serving |
| Metric | val_bpb (lower better) | throughput × (1 − latency penalty), higher better |
| Knobs | model architecture, optimizer | vLLM flags, KV cache, scheduling |
| Cost per iter | ~5 min | ~6-10 min (vLLM startup is the tax) |
| Model | trained from scratch | pretrained, you bring your own |

## How it works

```
config.py            <- agent edits this
launch_vllm.py       <- spawns vLLM with the config, waits for /health
workload/prompts.jsonl <- fixed benchmark workload
benchmark.py         <- fires the workload, returns a scalar score
run.py               <- the loop: edit → launch → bench → kill → log → repeat
results.tsv          <- commit | score | status | description
program.md           <- agent playbook (what's fair game, what's locked)
```

The agent only ever modifies `config.py`. Everything else is infrastructure.

## Setup

```bash
uv sync                     # install deps
# Put your real (anonymized) production prompts in workload/prompts.jsonl
uv run run.py --baseline    # establish baseline score
uv run run.py --loop        # start the agent (or invoke Claude/Codex with program.md)
```

## Hardware assumptions

This template assumes:
- Some number of GPUs visible (set `TENSOR_PARALLEL_SIZE` in `config.py` to match)
- vLLM installed in the same environment

Tested with: Llama 3.3 70B AWQ on 2× RTX 5090.

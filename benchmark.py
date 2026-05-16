"""
Fire the workload at a running vLLM server and compute a scalar score.

score = throughput_tok_per_s * max(0, 1 - latency_penalty)
where latency_penalty = max(0, (p99_inter_token_ms - SLO_MS) / SLO_MS)

Pure throughput would reward configs that crush concurrency but feel terrible
interactively. The penalty pulls the optimum toward configs that hold a
reasonable inter-token latency SLO.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

import config


@dataclass
class BenchResult:
    score: float
    throughput_tok_per_s: float
    p50_inter_token_ms: float
    p99_inter_token_ms: float
    n_requests_completed: int
    n_requests_errored: int
    duration_s: float


def _load_workload(path: str = "workload/prompts.jsonl") -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


async def _one_request(client: httpx.AsyncClient, prompt: dict, inter_token_times: list[float]) -> tuple[int, bool]:
    """Stream a completion. Returns (tokens_received, errored)."""
    payload = {
        "model": config.SERVED_MODEL_NAME,
        "prompt": prompt["prompt"],
        "max_tokens": prompt.get("max_tokens", 256),
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "stream": True,
    }
    url = f"http://{config.HOST}:{config.PORT}/v1/completions"
    tokens = 0
    last_t = None
    try:
        async with client.stream("POST", url, json=payload, timeout=120) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                tokens += 1
                now = time.perf_counter()
                if last_t is not None:
                    inter_token_times.append((now - last_t) * 1000)
                last_t = now
        return tokens, False
    except Exception:
        return tokens, True


async def _run_workload(workload: list[dict]) -> BenchResult:
    inter_token_times: list[float] = []
    tokens_total = 0
    errors = 0
    completed = 0

    sem = asyncio.Semaphore(config.BENCH_CONCURRENCY)
    start = time.perf_counter()
    deadline = start + config.BENCH_DURATION_SECONDS

    async def fire_one(prompt: dict) -> None:
        nonlocal tokens_total, errors, completed
        async with sem:
            if time.perf_counter() > deadline:
                return
            async with httpx.AsyncClient() as client:
                tokens, errored = await _one_request(client, prompt, inter_token_times)
                tokens_total += tokens
                completed += 1
                if errored:
                    errors += 1

    # Cycle through the workload until time's up
    tasks = []
    i = 0
    while time.perf_counter() < deadline:
        prompt = workload[i % len(workload)]
        tasks.append(asyncio.create_task(fire_one(prompt)))
        i += 1
        await asyncio.sleep(0)  # yield
    await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.perf_counter() - start
    throughput = tokens_total / duration if duration > 0 else 0.0
    p50 = statistics.median(inter_token_times) if inter_token_times else float("inf")
    p99 = statistics.quantiles(inter_token_times, n=100)[98] if len(inter_token_times) >= 100 else float("inf")

    penalty = max(0.0, (p99 - config.BENCH_SLO_INTER_TOKEN_MS) / config.BENCH_SLO_INTER_TOKEN_MS)
    score = throughput * max(0.0, 1.0 - penalty)

    return BenchResult(
        score=score,
        throughput_tok_per_s=throughput,
        p50_inter_token_ms=p50,
        p99_inter_token_ms=p99,
        n_requests_completed=completed,
        n_requests_errored=errors,
        duration_s=duration,
    )


def run() -> BenchResult:
    workload = _load_workload()
    return asyncio.run(_run_workload(workload))


if __name__ == "__main__":
    result = run()
    print(f"score:           {result.score:.2f}")
    print(f"throughput:      {result.throughput_tok_per_s:.1f} tok/s")
    print(f"p50_inter_token: {result.p50_inter_token_ms:.1f} ms")
    print(f"p99_inter_token: {result.p99_inter_token_ms:.1f} ms")
    print(f"completed:       {result.n_requests_completed}")
    print(f"errored:         {result.n_requests_errored}")

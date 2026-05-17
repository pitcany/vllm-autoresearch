"""
Fire one or more workload profiles at a running vLLM server and compute
per-profile scores plus rich latency statistics.

Per-profile score:

    score = output_tok_per_s
            * max(0, 1 - inter_token_penalty)
            * max(0, 1 - ttft_penalty)

where
    inter_token_penalty = max(0, (p99_inter_token_ms - SLO_INTER) / SLO_INTER)
    ttft_penalty        = max(0, (p95_ttft_ms          - SLO_TTFT)  / SLO_TTFT)

Pure throughput would reward configs that crush concurrency but feel terrible
interactively.  The penalties pull the optimum toward configs that hold both
TTFT and inter-token latency SLOs.  We never collapse the per-profile scores
into a single number on purpose — overnight optimisation should see the
tradeoffs.

The workload profiles are loaded from ``config.BENCH_PROFILES``.  Each profile
gets its own request stream, run sequentially.  Long-context profiles use a
synthetic ``prompt_template`` rendered with a repeated filler corpus so we can
target a specific input-token count without shipping huge JSONL files.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

import config


# ---- workload synthesis -----------------------------------------------------

_FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank where the "
    "willow trees grow tall and the cattails sway in the warm summer breeze. "
)


def _render_long_prompt(template: str, approx_input_tokens: int) -> str:
    """Synthesise a long-context prompt.  We approximate 1 token ≈ 4 chars,
    which is conservative for English text under the Llama 3 tokenizer."""
    target_chars = approx_input_tokens * 4
    body = (_FILLER * ((target_chars // len(_FILLER)) + 1))[:target_chars]
    if template == "long_repeat_summary":
        return ("Summarise the following passage in three bullet points.\n\n"
                + body + "\n\nSummary:")
    if template == "long_repeat_qa":
        return ("Below is a passage.  After reading it, answer the question.\n\n"
                + body + "\n\nQuestion: How many sentences mention the willow?\n"
                "Answer:")
    return body


def _load_profile(path: str) -> list[dict]:
    out: list[dict] = []
    for raw in Path(path).read_text().splitlines():
        if not raw.strip():
            continue
        rec = json.loads(raw)
        if "prompt" not in rec and "prompt_template" in rec:
            rec["prompt"] = _render_long_prompt(
                rec["prompt_template"],
                int(rec.get("approx_input_tokens", 4096)),
            )
        out.append(rec)
    return out


# ---- result types -----------------------------------------------------------


@dataclass
class RequestStat:
    started_at: float
    ttft_ms: float | None
    finished_at: float | None
    output_tokens: int
    input_tokens_est: int
    ok: bool
    error_class: str | None  # "timeout" | "http" | "stream" | None
    inter_token_ms: list[float] = field(default_factory=list)


@dataclass
class ProfileResult:
    name: str
    score: float
    requests_per_s: float
    output_tok_per_s: float
    total_tok_per_s: float
    completed: int
    errored: int
    timed_out: int
    duration_s: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    inter_token_p50_ms: float
    inter_token_p90_ms: float
    inter_token_p95_ms: float
    inter_token_p99_ms: float
    request_p50_ms: float
    request_p95_ms: float
    request_p99_ms: float
    synthetic: bool


@dataclass
class BenchReport:
    config_hash: str
    config_snapshot: dict
    profiles: list[ProfileResult]
    synthetic: bool
    notes: list[str] = field(default_factory=list)


# ---- HTTP firing ------------------------------------------------------------


_REQUEST_TIMEOUT_S = 180.0


async def _fire_one(client: httpx.AsyncClient, prompt: dict) -> RequestStat:
    payload = {
        "model": config.SERVED_MODEL_NAME,
        "prompt": prompt["prompt"],
        "max_tokens": int(prompt.get("max_tokens", 256)),
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"http://{config.HOST}:{config.PORT}/v1/completions"
    started = time.perf_counter()
    stat = RequestStat(
        started_at=started,
        ttft_ms=None,
        finished_at=None,
        output_tokens=0,
        input_tokens_est=len(prompt["prompt"]) // 4,
        ok=False,
        error_class=None,
    )
    last_t = None
    try:
        async with client.stream(
            "POST", url, json=payload, timeout=_REQUEST_TIMEOUT_S
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                now = time.perf_counter()
                # try to parse the chunk for an authoritative token count
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    obj = None
                # First content chunk → TTFT
                if stat.ttft_ms is None and obj is not None:
                    choices = obj.get("choices") or []
                    if choices and (choices[0].get("text") or "").strip() != "":
                        stat.ttft_ms = (now - started) * 1000.0
                        last_t = now
                        stat.output_tokens += 1
                        continue
                # Subsequent chunks → inter-token timing
                if last_t is not None:
                    stat.inter_token_ms.append((now - last_t) * 1000.0)
                last_t = now
                if obj is not None:
                    if obj.get("usage"):
                        usage = obj["usage"]
                        if isinstance(usage.get("completion_tokens"), int):
                            stat.output_tokens = max(
                                stat.output_tokens, usage["completion_tokens"]
                            )
                        if isinstance(usage.get("prompt_tokens"), int):
                            stat.input_tokens_est = usage["prompt_tokens"]
                    else:
                        stat.output_tokens += 1
        stat.finished_at = time.perf_counter()
        stat.ok = True
    except (asyncio.TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout):
        stat.error_class = "timeout"
    except httpx.HTTPStatusError:
        stat.error_class = "http"
    except Exception:
        stat.error_class = "stream"
    return stat


async def _run_profile(
    name: str,
    prompts: list[dict],
    concurrency: int,
    duration_s: float,
    slo_ttft_ms: float | None,
    slo_inter_token_ms: float | None,
) -> ProfileResult:
    sem = asyncio.Semaphore(concurrency)
    stats: list[RequestStat] = []
    start = time.perf_counter()
    deadline = start + duration_s

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        async def fire(idx: int) -> None:
            async with sem:
                if time.perf_counter() > deadline:
                    return
                prompt = prompts[idx % len(prompts)]
                stat = await _fire_one(client, prompt)
                stats.append(stat)

        running: set[asyncio.Task] = set()
        i = 0
        while time.perf_counter() < deadline:
            # back-pressure: only schedule when the semaphore could accept
            if sem.locked() and len(running) >= concurrency * 2:
                done, _ = await asyncio.wait(
                    running, return_when=asyncio.FIRST_COMPLETED
                )
                running.difference_update(done)
                continue
            t = asyncio.create_task(fire(i))
            running.add(t)
            t.add_done_callback(running.discard)
            i += 1
            await asyncio.sleep(0)
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    duration = time.perf_counter() - start

    ttfts = [s.ttft_ms for s in stats if s.ok and s.ttft_ms is not None]
    inter = [x for s in stats if s.ok for x in s.inter_token_ms]
    req_latencies = [
        (s.finished_at - s.started_at) * 1000.0
        for s in stats if s.ok and s.finished_at is not None
    ]
    output_tokens = sum(s.output_tokens for s in stats if s.ok)
    input_tokens = sum(s.input_tokens_est for s in stats if s.ok)
    completed = sum(1 for s in stats if s.ok)
    errored = sum(1 for s in stats if not s.ok)
    timed_out = sum(1 for s in stats if s.error_class == "timeout")

    def _pct(values: list[float], p: float) -> float:
        if not values:
            return float("inf")
        if len(values) == 1:
            return values[0]
        s = sorted(values)
        k = (len(s) - 1) * p
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    ttft_p50 = _pct(ttfts, 0.50)
    ttft_p95 = _pct(ttfts, 0.95)
    ttft_p99 = _pct(ttfts, 0.99)
    it_p50 = _pct(inter, 0.50)
    it_p90 = _pct(inter, 0.90)
    it_p95 = _pct(inter, 0.95)
    it_p99 = _pct(inter, 0.99)
    req_p50 = _pct(req_latencies, 0.50)
    req_p95 = _pct(req_latencies, 0.95)
    req_p99 = _pct(req_latencies, 0.99)

    out_tps = output_tokens / duration if duration > 0 else 0.0
    total_tps = (output_tokens + input_tokens) / duration if duration > 0 else 0.0
    rps = completed / duration if duration > 0 else 0.0

    # Per-profile SLOs: None means "this dimension does not gate the score".
    # E.g. batch is throughput-only; long_context's TTFT is prefill-dominated.
    if slo_inter_token_ms is None or it_p99 == float("inf"):
        inter_factor = 1.0 if slo_inter_token_ms is None else 0.0
    else:
        inter_penalty = max(0.0, (it_p99 - slo_inter_token_ms) / slo_inter_token_ms)
        inter_factor = max(0.0, 1.0 - inter_penalty)
    if slo_ttft_ms is None or ttft_p95 == float("inf"):
        ttft_factor = 1.0 if slo_ttft_ms is None else 0.0
    else:
        ttft_penalty = max(0.0, (ttft_p95 - slo_ttft_ms) / slo_ttft_ms)
        ttft_factor = max(0.0, 1.0 - ttft_penalty)
    score = out_tps * inter_factor * ttft_factor

    return ProfileResult(
        name=name,
        score=score,
        requests_per_s=rps,
        output_tok_per_s=out_tps,
        total_tok_per_s=total_tps,
        completed=completed,
        errored=errored,
        timed_out=timed_out,
        duration_s=duration,
        ttft_p50_ms=ttft_p50,
        ttft_p95_ms=ttft_p95,
        ttft_p99_ms=ttft_p99,
        inter_token_p50_ms=it_p50,
        inter_token_p90_ms=it_p90,
        inter_token_p95_ms=it_p95,
        inter_token_p99_ms=it_p99,
        request_p50_ms=req_p50,
        request_p95_ms=req_p95,
        request_p99_ms=req_p99,
        synthetic=any(p.get("synthetic") for p in prompts),
    )


def _config_snapshot() -> dict:
    keys = [
        "BACKEND",
        "GPU_MEMORY_UTILIZATION", "MAX_NUM_SEQS", "MAX_MODEL_LEN",
        "KV_CACHE_DTYPE", "BLOCK_SIZE", "ENABLE_CHUNKED_PREFILL",
        "ENABLE_PREFIX_CACHING", "MAX_NUM_BATCHED_TOKENS",
        "SWAP_SPACE_GB", "SCHEDULER_DELAY_FACTOR",
        "MODEL", "QUANTIZATION", "TENSOR_PARALLEL_SIZE",
        "BENCH_CONCURRENCY", "BENCH_DURATION_SECONDS",
        "BENCH_SLO_INTER_TOKEN_MS", "BENCH_SLO_TTFT_MS",
    ]
    if (config.BACKEND or "").lower() == "llama_cpp":
        keys += [
            "LLAMA_CPP_MODEL", "LLAMA_CPP_N_GPU_LAYERS", "LLAMA_CPP_CTX_SIZE",
            "LLAMA_CPP_PARALLEL", "LLAMA_CPP_BATCH_SIZE", "LLAMA_CPP_UBATCH_SIZE",
            "LLAMA_CPP_TENSOR_SPLIT", "LLAMA_CPP_MAIN_GPU",
            "LLAMA_CPP_CACHE_TYPE_K", "LLAMA_CPP_CACHE_TYPE_V",
            "LLAMA_CPP_FLASH_ATTN", "LLAMA_CPP_CONT_BATCHING",
            "LLAMA_CPP_KV_UNIFIED",
            "LLAMA_CPP_NO_MMAP", "LLAMA_CPP_MLOCK", "LLAMA_CPP_EXTRA_ARGS",
        ]
    snap = {k: getattr(config, k) for k in keys}
    # Profile-level SLO overrides materially change the score; include them.
    snap["BENCH_PROFILES"] = [
        {
            "name": p["name"],
            "concurrency": p.get("concurrency_override") or config.BENCH_CONCURRENCY,
            "slo_ttft_ms": p["slo_ttft_ms"] if "slo_ttft_ms" in p else config.BENCH_SLO_TTFT_MS,
            "slo_inter_token_ms": p["slo_inter_token_ms"] if "slo_inter_token_ms" in p else config.BENCH_SLO_INTER_TOKEN_MS,
        }
        for p in config.BENCH_PROFILES
    ]
    return snap


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def run() -> BenchReport:
    profiles = list(config.BENCH_PROFILES)
    results: list[ProfileResult] = []
    notes: list[str] = []

    for prof in profiles:
        path = prof["path"]
        if not Path(path).exists():
            notes.append(f"workload missing: {path}")
            continue
        prompts = _load_profile(path)
        if not prompts:
            notes.append(f"workload empty: {path}")
            continue
        concurrency = prof.get("concurrency_override") or config.BENCH_CONCURRENCY
        duration = float(prof.get("duration_s") or config.BENCH_DURATION_SECONDS)
        # If the profile dict has the key but it's None, "no SLO".
        # If the key is absent, fall back to the global default.
        slo_ttft = prof["slo_ttft_ms"] if "slo_ttft_ms" in prof else config.BENCH_SLO_TTFT_MS
        slo_inter = prof["slo_inter_token_ms"] if "slo_inter_token_ms" in prof else config.BENCH_SLO_INTER_TOKEN_MS
        r = asyncio.run(_run_profile(
            prof["name"], prompts, concurrency, duration, slo_ttft, slo_inter,
        ))
        results.append(r)

    snapshot = _config_snapshot()
    return BenchReport(
        config_hash=_config_hash(snapshot),
        config_snapshot=snapshot,
        profiles=results,
        synthetic=any(r.synthetic for r in results),
        notes=notes,
    )


def print_report(rep: BenchReport) -> None:
    print(f"config_hash:     {rep.config_hash}")
    print(f"synthetic_data:  {rep.synthetic}")
    for n in rep.notes:
        print(f"note:            {n}")
    for p in rep.profiles:
        print(f"--- profile: {p.name} ---")
        print(f"score_{p.name:<14} {p.score:.2f}")
        print(f"  output_tok/s:    {p.output_tok_per_s:.1f}")
        print(f"  total_tok/s:     {p.total_tok_per_s:.1f}")
        print(f"  req/s:           {p.requests_per_s:.2f}")
        print(f"  ttft p50/p95/p99 ms: {p.ttft_p50_ms:.0f} / {p.ttft_p95_ms:.0f} / {p.ttft_p99_ms:.0f}")
        print(f"  itok p50/p90/p95/p99 ms: {p.inter_token_p50_ms:.0f} / {p.inter_token_p90_ms:.0f} / {p.inter_token_p95_ms:.0f} / {p.inter_token_p99_ms:.0f}")
        print(f"  request p50/p95/p99 ms: {p.request_p50_ms:.0f} / {p.request_p95_ms:.0f} / {p.request_p99_ms:.0f}")
        print(f"  completed/errored/timed_out: {p.completed} / {p.errored} / {p.timed_out}")


if __name__ == "__main__":
    rep = run()
    print_report(rep)
    print("REPORT_JSON " + json.dumps({
        "config_hash": rep.config_hash,
        "synthetic": rep.synthetic,
        "profiles": [asdict(p) for p in rep.profiles],
        "notes": rep.notes,
    }, default=str))

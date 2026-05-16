"""
Spawn vLLM with the current config.py and wait for /health.

Returns (process, teardown_fn). On startup failure, teardown is already done and
process is None — the loop should treat that as a crash and move on.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Callable

import requests

import config


def _build_command() -> list[str]:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.MODEL,
        "--quantization", config.QUANTIZATION,
        "--tensor-parallel-size", str(config.TENSOR_PARALLEL_SIZE),
        "--host", config.HOST,
        "--port", str(config.PORT),
        "--served-model-name", config.SERVED_MODEL_NAME,
        "--max-model-len", str(config.MAX_MODEL_LEN),
        "--max-num-seqs", str(config.MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(config.MAX_NUM_BATCHED_TOKENS),
        "--gpu-memory-utilization", str(config.GPU_MEMORY_UTILIZATION),
        "--kv-cache-dtype", config.KV_CACHE_DTYPE,
        "--block-size", str(config.BLOCK_SIZE),
        "--swap-space", str(config.SWAP_SPACE_GB),
        "--scheduler-delay-factor", str(config.SCHEDULER_DELAY_FACTOR),
        "--trust-remote-code",
    ]
    if config.ENABLE_CHUNKED_PREFILL:
        cmd.append("--enable-chunked-prefill")
    if config.ENABLE_PREFIX_CACHING:
        cmd.append("--enable-prefix-caching")
    return cmd


def _health_url() -> str:
    return f"http://{config.HOST}:{config.PORT}/health"


def launch(startup_timeout_s: int = 600, log_path: str = "vllm.log"):
    """Spawn vLLM and block until /health returns 200, or fail.

    Returns (process, teardown_fn) on success.
    Returns (None, None) on failure (process is already cleaned up).
    """
    log_file = open(log_path, "w")
    cmd = _build_command()
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # so we can kill the whole group later
    )

    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.close()
            return None, None  # process died during startup
        try:
            r = requests.get(_health_url(), timeout=2)
            if r.status_code == 200:
                def teardown():
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        proc.wait(timeout=30)
                    except Exception:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            pass
                    finally:
                        log_file.close()
                return proc, teardown
        except requests.RequestException:
            pass
        time.sleep(2)

    # Timeout — kill everything
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    log_file.close()
    return None, None

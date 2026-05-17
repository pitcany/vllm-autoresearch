"""
Spawn vLLM with the current config.py and wait for /health.

Returns (process, teardown_fn, info). On startup failure, teardown is already
done and process is None — the loop should treat that as a crash and move on.

The command is built defensively: every CLI flag is checked against
``vllm serve --help`` (cached on first launch) so that a stale flag does not
silently break a 70B startup.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

import config


@dataclass
class LaunchInfo:
    """What we record about each vLLM launch."""

    vllm_version: str = ""
    command: list[str] = field(default_factory=list)
    startup_seconds: float = 0.0
    dropped_flags: list[str] = field(default_factory=list)
    gpu_snapshot_before: str = ""


# ---- vLLM CLI introspection -------------------------------------------------

_CACHED_HELP: str | None = None


def _vllm_help() -> str:
    """Cache `vllm serve --help` so we only spawn it once."""
    global _CACHED_HELP
    if _CACHED_HELP is None:
        for help_arg in ("--help=all", "--help"):
            try:
                out = subprocess.run(
                    [sys.executable, "-m", "vllm.entrypoints.cli.main",
                     "serve", help_arg],
                    capture_output=True, text=True, timeout=60,
                )
                text = (out.stdout or "") + (out.stderr or "")
                # require flags to actually appear, otherwise try the next form
                if "--max-model-len" in text or "--gpu-memory-utilization" in text:
                    _CACHED_HELP = text
                    break
            except Exception:
                continue
        if _CACHED_HELP is None:
            _CACHED_HELP = ""
    return _CACHED_HELP


def _flag_supported(flag: str) -> bool:
    """Return True if `flag` appears in vllm serve --help. Conservative: if we
    couldn't read help (e.g. cli not installed yet), assume yes — let vLLM
    itself reject it with a clear error rather than silently dropping flags."""
    help_text = _vllm_help()
    if not help_text:
        return True
    # match exact flag word boundary, e.g. "--max-model-len " or "--max-model-len="
    return re.search(rf"(^|[\s,]){re.escape(flag)}([\s=,]|$)", help_text, re.MULTILINE) is not None


def _vllm_version() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import vllm,sys;sys.stdout.write(vllm.__version__)"],
            capture_output=True, text=True, timeout=20,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _gpu_snapshot() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _build_command(info: LaunchInfo) -> list[str]:
    """Build the vLLM launch command, dropping any flag the installed vLLM
    no longer recognises. Records dropped flags into ``info``."""

    base = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.MODEL,
        "--quantization", config.QUANTIZATION,
        "--tensor-parallel-size", str(config.TENSOR_PARALLEL_SIZE),
        "--host", config.HOST,
        "--port", str(config.PORT),
        "--served-model-name", config.SERVED_MODEL_NAME,
        "--trust-remote-code",
    ]

    # Optional, version-checked flags.  (flag, value-or-None)
    optional: list[tuple[str, str | None]] = [
        ("--max-model-len", str(config.MAX_MODEL_LEN)),
        ("--max-num-seqs", str(config.MAX_NUM_SEQS)),
        ("--max-num-batched-tokens", str(config.MAX_NUM_BATCHED_TOKENS)),
        ("--gpu-memory-utilization", str(config.GPU_MEMORY_UTILIZATION)),
        ("--kv-cache-dtype", config.KV_CACHE_DTYPE),
        ("--block-size", str(config.BLOCK_SIZE)),
        ("--swap-space", str(config.SWAP_SPACE_GB)),
        ("--scheduler-delay-factor", str(config.SCHEDULER_DELAY_FACTOR)),
    ]
    for flag, value in optional:
        if _flag_supported(flag):
            base.extend([flag, value] if value is not None else [flag])
        else:
            info.dropped_flags.append(flag)

    for flag, enabled in [
        ("--enable-chunked-prefill", config.ENABLE_CHUNKED_PREFILL),
        ("--enable-prefix-caching", config.ENABLE_PREFIX_CACHING),
    ]:
        if not enabled:
            continue
        if _flag_supported(flag):
            base.append(flag)
        else:
            info.dropped_flags.append(flag)

    return base


def _health_url() -> str:
    return f"http://{config.HOST}:{config.PORT}/health"


def launch(startup_timeout_s: int = 900, log_path: str = "vllm.log"):
    """Spawn vLLM and block until /health returns 200, or fail.

    Returns ``(process, teardown_fn, info)`` on success.
    Returns ``(None, None, info)`` on failure (process is already cleaned up).
    ``info`` is always populated so the caller can log what was attempted.
    """
    info = LaunchInfo(
        vllm_version=_vllm_version(),
        gpu_snapshot_before=_gpu_snapshot(),
    )
    info.command = _build_command(info)

    log_file = open(log_path, "w")
    log_file.write(f"# vllm version: {info.vllm_version}\n")
    log_file.write(f"# dropped flags: {info.dropped_flags}\n")
    log_file.write(f"# gpu snapshot: {info.gpu_snapshot_before}\n")
    log_file.write(f"# command: {' '.join(info.command)}\n")
    log_file.flush()

    t0 = time.time()
    proc = subprocess.Popen(
        info.command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # so we can kill the whole group later
    )

    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.close()
            info.startup_seconds = time.time() - t0
            return None, None, info  # process died during startup
        try:
            r = requests.get(_health_url(), timeout=2)
            if r.status_code == 200:
                info.startup_seconds = time.time() - t0

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
                return proc, teardown, info
        except requests.RequestException:
            pass
        time.sleep(2)

    # Timeout — kill everything
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    info.startup_seconds = time.time() - t0
    log_file.close()
    return None, None, info

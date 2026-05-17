"""
Spawn llama.cpp's ``llama-server`` with the current config.py and wait for
``/health``.  Returns ``(process, teardown_fn, info)``, identical contract to
``launch_vllm.launch()``.

llama-server speaks the same OpenAI-compatible ``/v1/completions`` shape as
vLLM, so ``benchmark.py`` does not need to know which backend it is hitting.

This launcher is a near-mirror of ``launch_vllm.py``:

  * port pre-flight (refuse to start if HOST:PORT is already bound)
  * subprocess.Popen with setsid so the whole group can be killed
  * /health poll loop with timeout
  * teardown closure (SIGTERM, then SIGKILL)

llama.cpp install / GGUF download is *not* handled here; see ``program.md``
for setup instructions.  If ``LLAMA_CPP_BIN`` isn't found we fail loudly with
a clear message rather than half-launch.
"""
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

import config
import launch_vllm  # re-use LaunchInfo so run.py stays backend-agnostic


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


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


def _llama_cpp_version(binary: str) -> str:
    """Best-effort version probe; llama-server prints version on --version
    or as part of its banner."""
    for arg in ("--version", "-h"):
        try:
            out = subprocess.run(
                [binary, arg],
                capture_output=True, text=True, timeout=10,
            )
            text = (out.stdout or "") + (out.stderr or "")
            for line in text.splitlines():
                if "version" in line.lower() and any(c.isdigit() for c in line):
                    return line.strip()[:120]
        except Exception:
            continue
    return "unknown"


def _build_command(info: "launch_vllm.LaunchInfo", binary: str) -> list[str]:
    """Build the llama-server command from config.LLAMA_CPP_*.  Any optional
    flag we don't recognise lands in info.dropped_flags so the agent can see
    it didn't take effect."""

    if not config.LLAMA_CPP_MODEL:
        info.dropped_flags.append("--model (LLAMA_CPP_MODEL is empty)")

    base = [
        binary,
        "--model", config.LLAMA_CPP_MODEL,
        "--host", config.HOST,
        "--port", str(config.PORT),
        "--ctx-size", str(config.LLAMA_CPP_CTX_SIZE),
        "--n-gpu-layers", str(config.LLAMA_CPP_N_GPU_LAYERS),
        "--parallel", str(config.LLAMA_CPP_PARALLEL),
        "--batch-size", str(config.LLAMA_CPP_BATCH_SIZE),
        "--ubatch-size", str(config.LLAMA_CPP_UBATCH_SIZE),
        "--cache-type-k", config.LLAMA_CPP_CACHE_TYPE_K,
        "--cache-type-v", config.LLAMA_CPP_CACHE_TYPE_V,
    ]
    if config.LLAMA_CPP_TENSOR_SPLIT:
        base += ["--tensor-split", config.LLAMA_CPP_TENSOR_SPLIT]
    if config.LLAMA_CPP_MAIN_GPU is not None:
        base += ["--main-gpu", str(config.LLAMA_CPP_MAIN_GPU)]
    # Recent llama-server changed --flash-attn to take a value (on|off|auto).
    # Always pass the explicit value so the next flag isn't consumed as its arg.
    base += ["--flash-attn", "on" if config.LLAMA_CPP_FLASH_ATTN else "off"]
    if config.LLAMA_CPP_CONT_BATCHING:
        base.append("--cont-batching")
    if config.LLAMA_CPP_NO_MMAP:
        base.append("--no-mmap")
    if config.LLAMA_CPP_MLOCK:
        base.append("--mlock")
    if config.LLAMA_CPP_EXTRA_ARGS:
        base.extend(config.LLAMA_CPP_EXTRA_ARGS)
    return base


def _health_url() -> str:
    return f"http://{config.HOST}:{config.PORT}/health"


def launch(startup_timeout_s: int = 900, log_path: str = "llama_cpp.log"):
    """Spawn llama-server and block until /health returns 200.

    Same return shape as ``launch_vllm.launch()``:
    ``(proc, teardown_fn, info)`` on success, ``(None, None, info)`` on
    failure.
    """
    binary = shutil.which(config.LLAMA_CPP_BIN) or config.LLAMA_CPP_BIN
    info = launch_vllm.LaunchInfo(
        backend="llama_cpp",
        backend_version=_llama_cpp_version(binary),
        gpu_snapshot_before=_gpu_snapshot(),
    )
    info.command = _build_command(info, binary)

    if not os.path.exists(binary) and shutil.which(config.LLAMA_CPP_BIN) is None:
        with open(log_path, "w") as f:
            f.write(
                f"# LLAMA_CPP_BIN not found: tried {config.LLAMA_CPP_BIN!r}.\n"
                f"# Install llama.cpp (with CUDA) and either put llama-server\n"
                f"# on PATH or set LLAMA_CPP_BIN to an absolute path.\n"
            )
        return None, None, info

    if not config.LLAMA_CPP_MODEL or not os.path.exists(config.LLAMA_CPP_MODEL):
        with open(log_path, "w") as f:
            f.write(
                f"# LLAMA_CPP_MODEL not found: {config.LLAMA_CPP_MODEL!r}.\n"
                f"# Download a GGUF (e.g. bartowski/Llama-3.3-70B-Instruct-GGUF\n"
                f"# Q4_K_M) and set LLAMA_CPP_MODEL to its absolute path.\n"
            )
        return None, None, info

    if _port_in_use(config.HOST, config.PORT):
        with open(log_path, "w") as f:
            f.write(
                f"# PORT_BUSY: {config.HOST}:{config.PORT} is already in use. "
                f"Kill the stale server (or change config.PORT) before launching.\n"
            )
        return None, None, info

    log_file = open(log_path, "w")
    log_file.write(f"# backend: {info.backend} {info.backend_version}\n")
    log_file.write(f"# gpu snapshot: {info.gpu_snapshot_before}\n")
    log_file.write(f"# command: {' '.join(info.command)}\n")
    log_file.flush()

    t0 = time.time()
    proc = subprocess.Popen(
        info.command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.close()
            info.startup_seconds = time.time() - t0
            return None, None, info
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

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    info.startup_seconds = time.time() - t0
    log_file.close()
    return None, None, info

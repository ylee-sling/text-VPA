# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
execution.py -- run generated programs against their tests in separate, resource-limited processes.

Each program runs as ``python -I -B program.py`` in its own temporary directory and process group, with an
empty environment, a wall-clock timeout (the whole group is killed), and POSIX limits on CPU time, address
space, file size and open files. A program passes iff it exits with status 0.

This isolates crashes, hangs and runaway memory, but it is NOT a security sandbox: generated code can still
use the network and read files the user can read. Run evaluations inside a VM or container you can discard
(e.g. the training VM), never on a machine holding secrets.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

__all__ = ["ExecutionResult", "run_program", "run_programs"]

_RUNNER = """\
import runpy
import sys

cpu, memory = int(sys.argv[2]), int(sys.argv[3])
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError:
    pass
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


@dataclass(frozen=True)
class ExecutionResult:
    passed: bool
    status: str                 # "passed" | "failed" | "timeout"
    detail: str = ""            # the last lines of stderr (failures only)


def run_program(program: str, timeout: float = 10.0, memory_mb: int = 4096) -> ExecutionResult:
    """Run ``program`` (tests included) and report whether it exits with status 0 within ``timeout`` s.

    The limits are set by a small runner inside the child (``preexec_fn`` is unsafe in threaded callers).
    """
    with tempfile.TemporaryDirectory(prefix="text-vpa-exec-") as workdir:
        path = os.path.join(workdir, "program.py")
        runner = os.path.join(workdir, "runner.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(program)
        with open(runner, "w", encoding="utf-8") as handle:
            handle.write(_RUNNER)
        env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "HOME": workdir,
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        posix = os.name == "posix"
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", runner, path, str(int(timeout) + 1), str(memory_mb << 20)],
            cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            start_new_session=posix,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc, posix)
            return ExecutionResult(False, "timeout")
        finally:
            if proc.poll() is None:
                _kill(proc, posix)
    if proc.returncode == 0:
        return ExecutionResult(True, "passed")
    tail = stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
    return ExecutionResult(False, "failed", "\n".join(tail) or f"exit status {proc.returncode}")


def _kill(proc: subprocess.Popen, posix: bool) -> None:
    try:
        if posix:
            os.killpg(proc.pid, signal.SIGKILL)
        else:                                          # pragma: no cover
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


def run_programs(programs: Sequence[str], timeout: float = 10.0, workers: int = 4,
                 memory_mb: int = 4096) -> list[ExecutionResult]:
    """``run_program`` for each program, ``workers`` at a time; results in input order."""
    if workers <= 1:
        return [run_program(p, timeout, memory_mb) for p in programs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: run_program(p, timeout, memory_mb), programs))

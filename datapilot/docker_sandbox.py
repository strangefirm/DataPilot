"""Docker variant of the sandbox: same guardrails, but code runs in an isolated container.

Container: no network, memory/CPU/pid limits, non-root user, only the run folder mounted.
Build the image once:  docker build -t datapilot-sandbox docker/
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .sandbox import HOOKS_DIR, SandboxExecutor, _strip_paths, _truncate, compact_traceback

IMAGE = os.getenv("DATAPILOT_DOCKER_IMAGE", "datapilot-sandbox")


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@dataclass
class DockerSandboxExecutor(SandboxExecutor):
    def __post_init__(self) -> None:
        if not docker_available():
            raise RuntimeError("Docker is not running. Start Docker Desktop or set DATAPILOT_EXECUTOR=local.")

    def _run(self, script: Path) -> tuple[int, str]:
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "--memory", "1g", "--cpus", "1", "--pids-limit", "128",
            "--security-opt", "no-new-privileges",
            "-v", f"{self.workspace.root}:/work", "-v", f"{HOOKS_DIR}:/hooks:ro",
            "-w", "/work", "-e", "PYTHONPATH=/hooks", "-e", "MPLBACKEND=Agg", "-e", "PYTHONHASHSEED=0",
            IMAGE, "python", script.name,
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=self.timeout + 15)
            out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
            return p.returncode, _truncate(compact_traceback(_strip_paths(out, self.workspace.root)))
        except subprocess.TimeoutExpired:
            return 124, f"TimeoutError: execution exceeded {self.timeout}s."

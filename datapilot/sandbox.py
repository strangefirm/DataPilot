"""Sandboxed code execution.

Each run gets its own folder runs/<run_id>/ holding a copy of the data, every script the
Coder writes, charts (PNG) and the final analysis.py. Code only runs there.

SandboxExecutor implements AutoGen's CodeExecutor so it plugs into CodeExecutorAgent:
  1. guardrails.check_code() — static AST policy (blocked => returned as an error to the Coder)
  2. run with a timeout, scrubbed environment (no API keys), cwd = sandbox folder
  3. on success save the script as analysis.py and collect chart metadata
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock, CodeExecutor, CodeResult

from .guardrails import GuardrailResult, check_chart_meta, check_code

HOOKS_DIR = Path(__file__).resolve().parent / "sandbox_hooks"
MAX_OUTPUT_CHARS = 6_000
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
# Only these env vars reach sandboxed code; API keys and other secrets are dropped.
_ENV_PASSTHROUGH = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
                    "APPDATA", "LOCALAPPDATA", "LANG", "LC_ALL", "PROGRAMDATA", "NUMBER_OF_PROCESSORS")


@dataclass
class Workspace:
    run_id: str
    root: Path
    data_filename: str

    @classmethod
    def create(cls, data_path: Path, runs_dir: Path) -> "Workspace":
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        root = (runs_dir / run_id).resolve()
        root.mkdir(parents=True, exist_ok=False)
        name = _SAFE_NAME.sub("_", Path(data_path).name)
        shutil.copy2(data_path, root / name)
        return cls(run_id, root, name)

    def charts(self) -> list[Path]:
        return sorted(self.root.glob("*.png"))

    def chart_meta(self) -> list[dict]:
        f = self.root / "_charts_meta.jsonl"
        if not f.exists():
            return []
        latest: dict[str, dict] = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                m = json.loads(line)
                latest[m["file"]] = m  # last save of each file wins
            except (json.JSONDecodeError, KeyError):
                continue
        return list(latest.values())

    @property
    def analysis_file(self) -> Path:
        return self.root / "analysis.py"


@dataclass
class ExecutionRecord:
    step: int
    code: str
    exit_code: int
    output: str
    duration_s: float
    guardrail: dict
    chart_check: dict | None = None


@dataclass
class SandboxExecutor(CodeExecutor):
    workspace: Workspace
    timeout: int = 60
    on_execution: Callable[[ExecutionRecord], None] | None = None
    history: list[ExecutionRecord] = field(default_factory=list)

    # -- CodeExecutor API -------------------------------------------------------------
    async def execute_code_blocks(self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken) -> CodeResult:
        if not code_blocks:
            return CodeResult(exit_code=1, output="No code block found. Send exactly one ```python block.")
        # Several blocks in one message are concatenated into one script, so analysis.py is self-contained.
        bad_lang = [b.language for b in code_blocks if b.language.lower() not in {"python", "py", "python3"}]
        code = "\n\n".join(b.code for b in code_blocks)
        policy = check_code(code, bad_lang[0] if bad_lang else "python")
        step = len(self.history) + 1
        if not policy.ok:
            msg = "STATUS: BLOCKED\n" + policy.message() + "\nRewrite the code without these operations."
            return self._record(step, code, 1, msg, 0.0, policy)

        script = self.workspace.root / f"step_{step:02d}.py"
        script.write_text(code, encoding="utf-8")
        t0 = time.perf_counter()
        exit_code, output = await asyncio.to_thread(self._run, script)
        dt = time.perf_counter() - t0

        chart_check = None
        if exit_code == 0:
            shutil.copy2(script, self.workspace.analysis_file)
            chart_check = check_chart_meta(self.workspace.chart_meta())
            files = [p.name for p in self.workspace.charts()] + ["analysis.py"]
            output = "STATUS: SUCCESS\n" + output.rstrip() + "\n\nFILES: " + " | ".join(files)
            if not chart_check.ok:
                output += "\n\nCHART CHECK FAILED:\n" + "\n".join(f"- {x}" for x in chart_check.violations)
        else:
            output = f"STATUS: ERROR (exit code {exit_code})\n" + output
        return self._record(step, code, exit_code, output, dt, policy, chart_check)

    async def restart(self) -> None:  # stateless: every script runs in a fresh process
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    # -- internals --------------------------------------------------------------------
    def _env(self) -> dict[str, str]:
        env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
        env.update(MPLBACKEND="Agg", PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8",
                   PYTHONPATH=str(HOOKS_DIR), PYTHONDONTWRITEBYTECODE="1")
        return env

    def _run(self, script: Path) -> tuple[int, str]:
        try:
            p = subprocess.run([sys.executable, script.name], cwd=self.workspace.root, env=self._env(),
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=self.timeout)
            out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
            return p.returncode, _truncate(compact_traceback(_strip_paths(out, self.workspace.root)))
        except subprocess.TimeoutExpired:
            return 124, f"TimeoutError: execution exceeded {self.timeout}s. Make the code faster or simpler."

    def _record(self, step, code, exit_code, output, dt, policy: GuardrailResult, chart_check=None) -> CodeResult:
        rec = ExecutionRecord(step, code, exit_code, output, round(dt, 3), policy.to_dict(),
                              chart_check.to_dict() if chart_check else None)
        self.history.append(rec)
        if self.on_execution:
            self.on_execution(rec)
        return CodeResult(exit_code=exit_code, output=output)

    @property
    def successful_outputs(self) -> list[str]:
        return [r.output for r in self.history if r.exit_code == 0]


def compact_traceback(s: str) -> str:
    """Keep only traceback frames from the sandbox script (no library internals / host paths)."""
    if "Traceback (most recent call last):" not in s:
        return s
    out, lines, i = [], s.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'\s*File "([^"]+)", line \d+', line)
        if m:
            block = [line]
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                block.append(lines[i])
                i += 1
            fname = m.group(1)
            if not re.search(r"[\\/]", fname):  # a script in the sandbox folder (e.g. step_01.py)
                out.extend(b for b in block if not re.fullmatch(r"\s*[~^]+\s*", b))
            elif not out or out[-1] != "  [library frames hidden]":
                out.append("  [library frames hidden]")
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _truncate(s: str) -> str:
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    half = MAX_OUTPUT_CHARS // 2
    return s[:half] + f"\n... [{len(s) - MAX_OUTPUT_CHARS} chars truncated] ...\n" + s[-half:]


def _strip_paths(s: str, root: Path) -> str:
    """Hide host paths from agents (and from the UI)."""
    return s.replace(str(root) + os.sep, "").replace(str(root), ".")


def make_executor(workspace: Workspace, kind: str = "local", timeout: int = 60,
                  on_execution: Callable[[ExecutionRecord], None] | None = None) -> CodeExecutor:
    if kind == "local":
        return SandboxExecutor(workspace, timeout, on_execution)
    if kind == "docker":
        from .docker_sandbox import DockerSandboxExecutor
        return DockerSandboxExecutor(workspace, timeout, on_execution)
    raise ValueError(f"Unknown executor '{kind}' (use 'local' or 'docker').")

"""Observability: structured trace, per-question metrics, run index, reasoning capture.

Files per session folder runs/<run_id>/:
  trace.jsonl   - one JSON event per line (session, question, message, reasoning, execution, guardrail, outcome, human)
  metrics.json  - list of per-question metrics
Global:
  runs/index.jsonl - one summary line per question across all sessions (run history / dashboards)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, ThoughtEvent

log = logging.getLogger("datapilot")
MAX_TEXT = 20_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clip(s: str) -> str:
    return s if len(s) <= MAX_TEXT else s[:MAX_TEXT] + f"... [{len(s) - MAX_TEXT} chars clipped]"


# ---------------------------------------------------------------------------- reasoning parsing

def parse_plan(text: str) -> dict:
    """Structured view of the Planner's reasoning: assumptions, steps, expected outputs."""
    def section(name: str) -> str:
        m = re.search(rf"{name}:\s*(.*?)(?=\n[A-Z]{{4,}}:|\Z)", text, re.S)
        return m.group(1).strip() if m else ""
    bullets = lambda s: [re.sub(r"^\s*(?:[-*]|\d+[).])\s*", "", l).strip() for l in s.splitlines() if l.strip()]
    steps = re.findall(r"\d+\)\s*(.+?)(?=\s+\d+\)|\n|$)", section("PLAN"))
    return {"assumptions": bullets(section("ASSUMPTIONS")), "steps": [s.strip() for s in steps],
            "outputs": bullets(section("OUTPUTS"))}


def parse_review(text: str) -> dict:
    t = text.strip()
    if t.upper().startswith("APPROVED"):
        return {"verdict": "APPROVED", "reasons": []}
    reasons = [l.strip("-* ").strip() for l in t.split("REVISE:", 1)[-1].splitlines() if l.strip("-* ").strip()]
    return {"verdict": "REVISE" if "REVISE" in t.upper() else "OTHER", "reasons": reasons}


# ---------------------------------------------------------------------------- metrics

@dataclass
class QuestionMetrics:
    run_id: str
    question_index: int
    question: str
    provider: str
    model: str
    team_mode: str
    started_at: str
    duration_s: float = 0.0
    messages: int = 0
    executions: int = 0
    execution_errors: int = 0
    blocked_by_guardrail: int = 0
    chart_check_failures: int = 0
    self_corrections: int = 0
    review_rounds: int = 0
    revise_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_by_agent: dict[str, int] = field(default_factory=dict)
    time_by_agent_s: dict[str, float] = field(default_factory=dict)
    status: str = "running"
    stop_reason: str = ""
    grounding_ok: bool | None = None
    grounding_warnings: int = 0
    retried_for_grounding: bool = False
    human_decision: str | None = None
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------- tracer

class RunTracer:
    def __init__(self, run_dir: Path, run_id: str, runs_index: Path, meta: dict[str, Any]):
        self.run_dir, self.run_id, self.index_path, self.meta = Path(run_dir), run_id, Path(runs_index), meta
        self.trace_path = self.run_dir / "trace.jsonl"
        self.metrics_path = self.run_dir / "metrics.json"
        self.all_metrics: list[QuestionMetrics] = []
        self._lock = threading.Lock()
        self._current: QuestionMetrics | None = None
        self._t0 = self._last = 0.0
        self.emit("session_start", **meta)

    # -- low level --------------------------------------------------------------------
    def emit(self, event: str, **data: Any) -> None:
        rec = {"ts": _now(), "run_id": self.run_id, "event": event,
               "q": self._current.question_index if self._current else None, **data}
        with self._lock, self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        log.debug("%s %s", event, {k: v for k, v in data.items() if k not in ("text", "code", "output")})

    # -- question lifecycle -----------------------------------------------------------
    def question_start(self, question: str) -> None:
        self._current = QuestionMetrics(
            run_id=self.run_id, question_index=len(self.all_metrics) + 1, question=question,
            provider=self.meta.get("provider", ""), model=self.meta.get("model", ""),
            team_mode=self.meta.get("team_mode", ""), started_at=_now())
        self._t0 = self._last = time.perf_counter()
        self.emit("question_start", question=question)

    def on_item(self, item: BaseAgentEvent | BaseChatMessage) -> None:
        m = self._current
        if m is None:
            return
        now = time.perf_counter()
        elapsed, self._last = now - self._last, now
        usage = getattr(item, "models_usage", None)
        pt, ct = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
        if isinstance(item, ThoughtEvent):
            self.emit("reasoning", agent=item.source, kind="model_thought", text=_clip(item.content))
            return
        if not isinstance(item, BaseChatMessage):
            self.emit("agent_event", agent=item.source, type=type(item).__name__)
            return
        text = item.to_text()
        if item.source != "user":
            m.messages += 1
            m.prompt_tokens += pt
            m.completion_tokens += ct
            if pt + ct:
                m.tokens_by_agent[item.source] = m.tokens_by_agent.get(item.source, 0) + pt + ct
            m.time_by_agent_s[item.source] = round(m.time_by_agent_s.get(item.source, 0.0) + elapsed, 3)
        else:
            m.messages += 1
        self.emit("message", agent=item.source, type=type(item).__name__, text=_clip(text),
                  prompt_tokens=pt, completion_tokens=ct, latency_s=round(elapsed, 3))
        if item.source == "planner" and "PLAN" in text:
            self.emit("reasoning", agent="planner", kind="plan", **parse_plan(text))
        elif item.source == "reviewer":
            r = parse_review(text)
            m.review_rounds += 1
            m.revise_count += r["verdict"] == "REVISE"
            self.emit("reasoning", agent="reviewer", kind="review", **r)

    def on_execution(self, rec) -> None:  # sandbox.ExecutionRecord
        m = self._current
        if m is not None:
            m.executions += 1
            m.execution_errors += rec.exit_code != 0
            m.blocked_by_guardrail += not rec.guardrail["ok"]
            m.chart_check_failures += bool(rec.chart_check and not rec.chart_check["ok"])
        self.emit("execution", step=rec.step, exit_code=rec.exit_code, duration_s=rec.duration_s,
                  code_sha=hashlib.sha256(rec.code.encode()).hexdigest()[:12], code=_clip(rec.code),
                  output=_clip(rec.output), guardrail=rec.guardrail, chart_check=rec.chart_check)
        if not rec.guardrail["ok"]:
            self.emit("guardrail", **rec.guardrail)
        if rec.chart_check and not rec.chart_check["ok"]:
            self.emit("guardrail", **rec.chart_check)

    def guardrail(self, result) -> None:  # guardrails.GuardrailResult
        self.emit("guardrail", **result.to_dict())

    def question_end(self, outcome) -> QuestionMetrics:  # session.RunOutcome
        m = self._current
        m.duration_s = round(time.perf_counter() - self._t0, 3)
        m.status, m.stop_reason = outcome.status, outcome.stop_reason
        m.self_corrections = outcome.self_corrections
        m.retried_for_grounding = outcome.retried_for_grounding
        if outcome.grounding is not None:
            m.grounding_ok = outcome.grounding.ok
            m.grounding_warnings = len(outcome.grounding.warnings)
            self.guardrail(outcome.grounding)
        self.emit("outcome", status=m.status, stop_reason=m.stop_reason, answer=outcome.answer,
                  charts=[p.name for p in outcome.charts], metrics=asdict(m))
        self.all_metrics.append(m)
        self._flush(m)
        self._current = None
        return m

    def question_error(self, question: str, error: BaseException) -> None:
        if self._current is None:
            self.question_start(question)
        m = self._current
        m.status, m.error = "error", f"{type(error).__name__}: {error}"
        m.duration_s = round(time.perf_counter() - self._t0, 3)
        self.emit("error", error=m.error)
        self.all_metrics.append(m)
        self._flush(m)
        self._current = None

    def human_decision(self, question_index: int, decision: str, text: str | None) -> None:
        for m in self.all_metrics:
            if m.question_index == question_index:
                m.human_decision = decision
        self.emit("human", question_index=question_index, decision=decision, text=text)
        self._write_metrics()
        self._rewrite_index_entry(question_index, decision)

    # -- persistence --------------------------------------------------------------------
    def _write_metrics(self) -> None:
        self.metrics_path.write_text(json.dumps([asdict(x) | {"total_tokens": x.total_tokens}
                                                 for x in self.all_metrics], indent=2), encoding="utf-8")

    def _flush(self, m: QuestionMetrics) -> None:
        self._write_metrics()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(m) | {"total_tokens": m.total_tokens}, default=str) + "\n")

    def _rewrite_index_entry(self, question_index: int, decision: str) -> None:
        if not self.index_path.exists():
            return
        with self._lock:
            lines = self.index_path.read_text(encoding="utf-8").splitlines()
            out = []
            for line in lines:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    out.append(line)
                    continue
                if d.get("run_id") == self.run_id and d.get("question_index") == question_index:
                    d["human_decision"] = decision
                out.append(json.dumps(d, default=str))
            self.index_path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- reading back

def load_trace(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "trace.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def load_index(runs_dir: Path) -> list[dict]:
    p = Path(runs_dir) / "index.jsonl"
    if not p.exists():
        return []
    out = []
    for l in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return out


def summarize(rows: list[dict]) -> dict:
    """Aggregate KPIs across questions (used by the report CLI and the Streamlit history page)."""
    n = len(rows)
    if not n:
        return {"questions": 0}
    avg = lambda k: round(sum(r.get(k) or 0 for r in rows) / n, 2)
    by_status: dict[str, int] = defaultdict(int)
    for r in rows:
        by_status[r.get("status", "?")] += 1
    with_errors = [r for r in rows if r.get("execution_errors")]
    return {
        "questions": n,
        "approval_rate": round(by_status["approved"] / n, 3),
        "status_counts": dict(by_status),
        "self_correction_rate": round(sum(1 for r in with_errors if r.get("self_corrections")) / len(with_errors), 3)
        if with_errors else None,
        "avg_messages": avg("messages"),
        "avg_executions": avg("executions"),
        "avg_tokens": avg("total_tokens"),
        "avg_duration_s": avg("duration_s"),
        "hit_message_limit": sum(1 for r in rows if "Maximum number of messages" in (r.get("stop_reason") or "")),
        "guardrail_blocks": sum(r.get("blocked_by_guardrail") or 0 for r in rows),
        "human_approved": sum(1 for r in rows if r.get("human_decision") == "approved"),
    }

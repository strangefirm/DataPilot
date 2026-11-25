"""A DataPilot session: one dataset, one team, many questions (follow-ups keep the same conversation).

Flow per question:
  input guardrails -> team.run_stream (Planner/Coder/Executor/Reviewer until APPROVED | 12 msgs | tokens)
  -> answer grounding guardrail (one automatic retry if numbers are not traceable to code output)
  -> human approval / follow-up via UserProxyAgent
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken

from . import prompts
from .config import Settings, get_settings
from .dataset import DatasetInfo, describe
from .guardrails import GuardrailResult, check_answer_grounding, check_question, check_upload
from .llm import make_model_client
from .observability import QuestionMetrics, RunTracer
from .sandbox import ExecutionRecord, Workspace, make_executor
from .team import APPROVAL_TOKEN, EXECUTOR, REVIEWER, ModelClients, build_team

MAX_GROUNDING_RETRIES = 1
APPROVE_WORDS = {"y", "yes", "approve", "approved", "ok", "okay", "accept", "looks good", "lgtm"}


class GuardrailError(ValueError):
    def __init__(self, result: GuardrailResult):
        super().__init__("; ".join(result.violations))
        self.result = result


@dataclass
class RunOutcome:
    question: str
    approved_by_reviewer: bool
    answer: str  # final answer text (without the APPROVED marker); empty if not approved
    stop_reason: str
    messages: int
    self_corrections: int  # executor errors followed by a later success
    executions: list[ExecutionRecord]
    grounding: GuardrailResult | None
    charts: list[Path]
    analysis_file: Path | None
    human_decision: str | None = None  # "approved" | "follow_up" | None
    retried_for_grounding: bool = False

    @property
    def status(self) -> str:
        if not self.approved_by_reviewer:
            return "not_approved"
        if self.grounding and not self.grounding.ok:
            return "ungrounded"
        return "approved"


@dataclass
class DataPilotSession:
    workspace: Workspace
    dataset: DatasetInfo
    team: object
    clients: ModelClients
    settings: Settings
    history: list[RunOutcome] = field(default_factory=list)
    tracer: RunTracer | None = None
    metrics: list[QuestionMetrics] = field(default_factory=list)
    _executor: object = None
    _execution_listeners: list[Callable[[ExecutionRecord], None]] = field(default_factory=list)

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, data_path: Path, *, settings: Settings | None = None,
               clients: ModelClients | None = None) -> "DataPilotSession":
        s = settings or get_settings()
        data_path = Path(data_path)
        up = check_upload(data_path.name, data_path.stat().st_size, s.max_upload_mb)
        if not up.ok:
            raise GuardrailError(up)
        ws = Workspace.create(data_path, s.runs_dir)
        try:
            info = describe(ws.root / ws.data_filename)
        except Exception as e:
            shutil.rmtree(ws.root, ignore_errors=True)
            raise GuardrailError(GuardrailResult("upload", False, [f"Could not read the file: {e}"])) from e
        model_name = s.model if clients is None else type(clients.planner).__name__
        clients = clients or ModelClients.single(make_model_client(s))
        session = cls(ws, info, None, clients, s)
        session.tracer = RunTracer(ws.root, ws.run_id, s.runs_dir / "index.jsonl", meta={
            "provider": s.provider, "model": model_name, "team_mode": s.team,
            "executor": s.executor, "max_messages": s.max_messages, "max_tokens": s.max_tokens_per_run,
            "data_file": ws.data_filename, "rows": info.rows, "columns": info.columns,
        })
        session.tracer.guardrail(up)
        session.tracer.guardrail(info.injection_check)
        session.add_execution_listener(session.tracer.on_execution)
        session._executor = make_executor(ws, s.executor, s.exec_timeout, on_execution=session._notify_execution)
        session.team = build_team(clients, session._executor, mode=s.team,
                                  max_messages=s.max_messages, max_tokens=s.max_tokens_per_run)
        return session

    def add_execution_listener(self, fn: Callable[[ExecutionRecord], None]) -> None:
        self._execution_listeners.append(fn)

    def _notify_execution(self, rec: ExecutionRecord) -> None:
        for fn in self._execution_listeners:
            fn(rec)

    # ------------------------------------------------------------------ asking
    async def ask(self, question: str) -> AsyncIterator[BaseAgentEvent | BaseChatMessage | RunOutcome]:
        """Stream every agent message, then yield the RunOutcome last."""
        q = check_question(question)
        self.tracer.question_start(question)
        self.tracer.guardrail(q)
        if not q.ok:
            err = GuardrailError(q)
            self.tracer.question_error(question, err)
            raise err
        try:
            async for item in self._ask(question):
                if isinstance(item, RunOutcome):
                    self.metrics.append(self.tracer.question_end(item))
                else:
                    self.tracer.on_item(item)
                yield item
        except Exception as e:
            self.tracer.question_error(question, e)
            raise

    async def _ask(self, question: str) -> AsyncIterator[BaseAgentEvent | BaseChatMessage | RunOutcome]:
        first = not self.history
        task = (prompts.task_message(question, self.dataset.schema_text, self.workspace.data_filename)
                if first else prompts.followup_message(question))
        n_exec_before = len(self._executor.history)

        result: TaskResult | None = None
        retried = False
        for attempt in range(1 + MAX_GROUNDING_RETRIES):
            async for item in self.team.run_stream(task=task):
                if isinstance(item, TaskResult):
                    result = item
                else:
                    yield item
            outcome = self._outcome(question, result, n_exec_before, retried)
            if outcome.status != "ungrounded" or attempt == MAX_GROUNDING_RETRIES:
                break
            retried = True
            task = ("GROUNDING CHECK FAILED - the answer quotes numbers that do not appear in any code output:\n"
                    + "\n".join(outcome.grounding.violations)
                    + "\nCoder: print every figure the answer needs. Reviewer: quote only printed numbers.")
        self._snapshot_analysis(len(self.history) + 1)
        self.history.append(outcome)
        yield outcome

    def _outcome(self, question: str, result: TaskResult, n_exec_before: int, retried: bool) -> RunOutcome:
        execs = self._executor.history[n_exec_before:]
        chat = [m for m in result.messages if isinstance(m, BaseChatMessage)]
        last_review = next((m for m in reversed(chat) if m.source == REVIEWER), None)
        approved = bool(last_review and APPROVAL_TOKEN in last_review.to_text())
        answer = _strip_marker(last_review.to_text()) if approved else ""
        grounding = check_answer_grounding(answer, [e.output for e in execs if e.exit_code == 0]) if approved else None
        corrections = sum(1 for i, e in enumerate(execs) if e.exit_code != 0
                          and any(x.exit_code == 0 for x in execs[i + 1:]))
        return RunOutcome(
            question=question, approved_by_reviewer=approved, answer=answer,
            stop_reason=result.stop_reason or "", messages=len(chat), self_corrections=corrections,
            executions=list(execs), grounding=grounding, charts=self.workspace.charts(),
            analysis_file=self.workspace.analysis_file if self.workspace.analysis_file.exists() else None,
            retried_for_grounding=retried,
        )

    def _snapshot_analysis(self, n: int) -> None:
        if self.workspace.analysis_file.exists():
            shutil.copy2(self.workspace.analysis_file, self.workspace.root / f"analysis_q{n}.py")

    async def run(self, question: str) -> RunOutcome:
        outcome = None
        async for item in self.ask(question):
            if isinstance(item, RunOutcome):
                outcome = item
        return outcome

    # ------------------------------------------------------------------ human in the loop
    async def human_review(self, outcome: RunOutcome,
                           input_func: Callable[[str], str] | Callable[[str, CancellationToken | None], Awaitable[str]]
                           ) -> tuple[str, str | None]:
        """UserProxyAgent gate: returns ("approved", None) or ("follow_up", question)."""
        proxy = UserProxyAgent("user", input_func=input_func)
        shown = outcome.answer or f"(No approved answer: {outcome.stop_reason})"
        resp = await proxy.on_messages([TextMessage(content=shown, source=REVIEWER)], CancellationToken())
        reply = resp.chat_message.to_text().strip()
        index = self.history.index(outcome) + 1 if outcome in self.history else len(self.history)
        if reply.lower().rstrip(".!") in APPROVE_WORDS:
            outcome.human_decision = "approved"
            self.tracer.human_decision(index, "approved", None)
            return "approved", None
        outcome.human_decision = "follow_up"
        self.tracer.human_decision(index, "follow_up", reply)
        return "follow_up", reply

    async def close(self) -> None:
        seen = set()
        for c in (self.clients.planner, self.clients.coder, self.clients.reviewer, self.clients.selector):
            if c is not None and id(c) not in seen:
                seen.add(id(c))
                await c.close()


def _strip_marker(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].strip().upper().startswith(APPROVAL_TOKEN):
        lines = lines[1:]
    return "\n".join(lines).strip()

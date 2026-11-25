"""Build the DataPilot agent team.

Agents (spec 3.2/3.4): Planner, Coder, Reviewer = AssistantAgent; Executor = CodeExecutorAgent.
Termination (spec 3.4): TextMentionTermination("APPROVED", reviewer only) | MaxMessageTermination(12)
                        | TokenUsageTermination (cost guardrail).

Team modes:
- "round_robin" (spec baseline): RoundRobinGroupChat [planner, coder, executor, reviewer].
- "routed": SelectorGroupChat with a deterministic selector_func that sends executor errors straight
   back to the Coder (no wasted Reviewer/Planner turns) - matches the flow in spec 3.6.
- "selector" (stretch): SelectorGroupChat where the LLM picks the next speaker; executor errors still
   route to the Coder deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from autogen_agentchat.agents import ApprovalRequest, ApprovalResponse, AssistantAgent, CodeExecutorAgent
from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination, TokenUsageTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_core.code_executor import CodeExecutor
from autogen_core.models import ChatCompletionClient

from . import prompts
from .guardrails import check_code

PLANNER, CODER, EXECUTOR, REVIEWER = "planner", "coder", "executor", "reviewer"
APPROVAL_TOKEN = "APPROVED"


@dataclass
class ModelClients:
    """One client per role (lets tests script each agent; lets you mix models per role)."""
    planner: ChatCompletionClient
    coder: ChatCompletionClient
    reviewer: ChatCompletionClient
    selector: ChatCompletionClient | None = None

    @classmethod
    def single(cls, client: ChatCompletionClient) -> "ModelClients":
        return cls(client, client, client, client)


def code_approval(request: ApprovalRequest) -> ApprovalResponse:
    """Automatic approval gate in front of the executor: the code safety policy decides."""
    r = check_code(request.code)
    return ApprovalResponse(approved=r.ok, reason="Passed code policy" if r.ok else "; ".join(r.violations))


def build_termination(max_messages: int, max_tokens: int) -> TerminationCondition:
    return (TextMentionTermination(APPROVAL_TOKEN, sources=[REVIEWER])
            | MaxMessageTermination(max_messages)
            | TokenUsageTermination(max_total_token=max_tokens))


def build_agents(clients: ModelClients, executor: CodeExecutor):
    planner = AssistantAgent(PLANNER, model_client=clients.planner, system_message=prompts.PLANNER,
                             description="Writes the step-by-step analysis plan. Speaks first for each question.")
    coder = AssistantAgent(CODER, model_client=clients.coder, system_message=prompts.CODER,
                           description="Writes and fixes Python (pandas + matplotlib) code.")
    executor_agent = CodeExecutorAgent(EXECUTOR, code_executor=executor, sources=[CODER],
                                       supported_languages=["python"], approval_func=code_approval,
                                       description="Runs the Coder's latest code in the sandbox.")
    reviewer = AssistantAgent(REVIEWER, model_client=clients.reviewer, system_message=prompts.REVIEWER,
                              description="Checks results; replies APPROVED with the final answer, or REVISE.")
    return planner, coder, executor_agent, reviewer


def _last_chat(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> BaseChatMessage | None:
    for m in reversed(messages):
        if isinstance(m, BaseChatMessage):
            return m
    return None


def route_next(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    """Deterministic workflow: user->planner->coder->executor->(error? coder : reviewer)->(REVISE? planner/coder)."""
    last = _last_chat(messages)
    if last is None or last.source == "user":
        return PLANNER
    text = last.to_text()
    if last.source == PLANNER:
        return CODER
    if last.source == CODER:
        return EXECUTOR
    if last.source == EXECUTOR:
        return REVIEWER if _succeeded(text) else CODER
    if last.source == REVIEWER:
        return PLANNER if "plan" in text.lower().split("revise:", 1)[-1][:200] else CODER
    return None


def route_errors_only(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    """For LLM-selected mode: only force the obvious moves, let the model pick the rest."""
    last = _last_chat(messages)
    if last is None or last.source == "user":
        return PLANNER
    if last.source == CODER:
        return EXECUTOR
    if last.source == EXECUTOR and not _succeeded(last.to_text()):
        return CODER
    return None


def _succeeded(executor_text: str) -> bool:
    # CodeExecutorAgent may prefix failures with its own sentence, so check for the status line anywhere.
    return "STATUS: SUCCESS" in executor_text and "STATUS: ERROR" not in executor_text


def build_team(clients: ModelClients, executor: CodeExecutor, *, mode: str = "round_robin",
               max_messages: int = 12, max_tokens: int = 200_000):
    agents = list(build_agents(clients, executor))
    termination = build_termination(max_messages, max_tokens)
    if mode == "round_robin":
        return RoundRobinGroupChat(agents, termination_condition=termination)
    if mode in ("routed", "selector"):
        return SelectorGroupChat(
            agents,
            model_client=clients.selector or clients.planner,
            termination_condition=termination,
            selector_func=route_next if mode == "routed" else route_errors_only,
            allow_repeated_speaker=False,
        )
    raise ValueError(f"Unknown team mode '{mode}' (round_robin | routed | selector).")

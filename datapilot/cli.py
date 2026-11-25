"""Command-line DataPilot: python -m datapilot.cli --data data/sales_2026.csv --question "..."."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from autogen_agentchat.messages import BaseChatMessage, CodeExecutionEvent

from .session import DataPilotSession, GuardrailError, RunOutcome

COLORS = {"planner": "\033[36m", "coder": "\033[33m", "executor": "\033[35m", "reviewer": "\033[32m", "user": "\033[37m"}
RESET = "\033[0m"


def _print_message(m: BaseChatMessage) -> None:
    text = m.to_text()
    if m.source == "user":
        text = text.split("\n", 1)[0]  # don't dump the schema
    c = COLORS.get(m.source, "")
    print(f"\n{c}── {m.source.upper()} ──{RESET}\n{text}")


def _print_outcome(o: RunOutcome) -> None:
    print("\n" + "=" * 70)
    print(f"Status: {o.status} | stop: {o.stop_reason} | messages: {o.messages} | "
          f"self-corrections: {o.self_corrections} | executions: {len(o.executions)}")
    if o.grounding and o.grounding.warnings:
        print("Grounding warnings:\n  " + "\n  ".join(o.grounding.warnings))
    if o.grounding and not o.grounding.ok:
        print("GROUNDING FAILED:\n  " + "\n  ".join(o.grounding.violations))
    print("Charts: " + ", ".join(str(p) for p in o.charts))
    print(f"Script: {o.analysis_file}")
    print("=" * 70)


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DataPilot: self-correcting data analyst")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--question", required=True)
    ap.add_argument("--no-human", action="store_true", help="skip the approval/follow-up prompt")
    args = ap.parse_args(argv)

    try:
        session = DataPilotSession.create(args.data)
    except GuardrailError as e:
        print(f"Input rejected: {e}")
        return 2
    print(f"Run folder: {session.workspace.root}")
    question = args.question
    try:
        while question:
            outcome = None
            try:
                async for item in session.ask(question):
                    if isinstance(item, RunOutcome):
                        outcome = item
                    elif isinstance(item, BaseChatMessage):
                        _print_message(item)
                    elif isinstance(item, CodeExecutionEvent):
                        pass  # executor result also arrives as a chat message
            except GuardrailError as e:
                print(f"Question rejected: {e}")
                return 2
            _print_outcome(outcome)
            if args.no_human:
                break
            decision, question = await session.human_review(
                outcome, lambda prompt: input("\nApprove the answer? Type 'yes', or ask a follow-up question: "))
            if decision == "approved":
                print("Answer approved by the user.")
    finally:
        await session.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

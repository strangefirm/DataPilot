"""Inspect runs from the terminal.

  python -m datapilot.report                 # KPIs + recent questions
  python -m datapilot.report --run <run_id>  # timeline of one session (messages, reasoning, executions, guardrails)
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import get_settings
from .observability import load_index, load_trace, summarize


def _short(s: str, n: int = 70) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def print_summary(limit: int) -> None:
    rows = load_index(get_settings().runs_dir)
    print(json.dumps(summarize(rows), indent=2))
    print(f"\n{'run_id':24} {'q':>2} {'status':13} {'msgs':>4} {'exec':>4} {'err':>3} {'fix':>3} {'tokens':>7} {'sec':>6}  question")
    for r in rows[-limit:]:
        print(f"{r['run_id']:24} {r['question_index']:>2} {r['status']:13} {r['messages']:>4} {r['executions']:>4} "
              f"{r['execution_errors']:>3} {r['self_corrections']:>3} {r['total_tokens']:>7} {r['duration_s']:>6.1f}  "
              f"{_short(r['question'], 50)}")


def print_timeline(run_id: str) -> int:
    run_dir = get_settings().runs_dir / run_id
    events = load_trace(run_dir)
    if not events:
        print(f"No trace for {run_id}")
        return 1
    for e in events:
        t, ev = e["ts"][11:23], e["event"]
        if ev == "message":
            tok = f" [{e['prompt_tokens']}+{e['completion_tokens']} tok, {e['latency_s']}s]" if e["agent"] != "user" else ""
            print(f"{t} {e['agent']:>9}: {_short(e['text'].splitlines()[0] if e['text'] else '')}{tok}")
        elif ev == "reasoning":
            if e["kind"] == "plan":
                print(f"{t} {'reasoning':>9}: plan {len(e['steps'])} steps; assumptions: {_short('; '.join(e['assumptions']), 60)}")
            elif e["kind"] == "review":
                print(f"{t} {'reasoning':>9}: review {e['verdict']} {_short('; '.join(e['reasons']), 60)}")
            else:
                print(f"{t} {'reasoning':>9}: {e['agent']} thought: {_short(e['text'])}")
        elif ev == "execution":
            print(f"{t} {'sandbox':>9}: step {e['step']} exit={e['exit_code']} {e['duration_s']}s code={e['code_sha']}")
        elif ev == "guardrail" and (not e["ok"] or e.get("warnings")):
            print(f"{t} {'GUARDRAIL':>9}: {e['guardrail']} ok={e['ok']} {_short('; '.join(e['violations'] + e['warnings']))}")
        elif ev in ("question_start", "outcome", "human", "error", "session_start"):
            detail = {"question_start": e.get("question"), "outcome": f"{e.get('status')} ({e.get('stop_reason')})",
                      "human": f"{e.get('decision')} {e.get('text') or ''}", "error": e.get("error"),
                      "session_start": f"{e.get('provider')}/{e.get('model')} team={e.get('team_mode')}"}[ev]
            print(f"{t} {ev.upper():>9}: {_short(detail, 90)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run_id (folder name under runs/)")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)
    if args.run:
        return print_timeline(args.run)
    print_summary(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

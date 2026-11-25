"""Run the golden eval set against the live LLM (or the offline harness check).

  python -m evals.run                                   # all cases, team modes round_robin + routed
  python -m evals.run --modes routed --cases spec_drop  # subset
  python -m evals.run --repeat 3                        # repeat to measure run-to-run variance
  python -m evals.run --offline                         # scripted agents, spec case only (no API cost)

Writes eval_results/<timestamp>/{results.jsonl, summary.json, report.md}.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from datapilot.config import PROJECT_ROOT, get_settings
from datapilot.llm import MissingAPIKeyError
from datapilot.session import DataPilotSession, GuardrailError

from .golden import build_cases
from .scoring import reproducible, score_case

DATA = PROJECT_ROOT / "data" / "sales_2026.csv"
CHECKS = ["passed", "approved", "within_limit", "self_corrected", "entities_ok", "numbers_ok", "grounded",
          "charts_ok", "reproducible"]


async def run_case(case, mode: str, runs_dir: Path, offline: bool) -> dict:
    s = replace(get_settings(), team=mode, runs_dir=runs_dir)
    clients = None
    if offline:
        from datapilot.demo import demo_clients
        clients = demo_clients()
    session = DataPilotSession.create(DATA, settings=s, clients=clients)
    outcome, rejected, error, repro = None, False, None, None
    t0 = time.perf_counter()
    try:
        outcome = await session.run(case.question)
        last_ok = next((e for e in reversed(outcome.executions) if e.exit_code == 0), None)
        analysis = session.workspace.root / f"analysis_q{len(session.history)}.py"
        repro = await reproducible(analysis if analysis.exists() else None, DATA, last_ok.output if last_ok else "")
    except GuardrailError:
        rejected = True
    except Exception as e:  # API errors, rate limits... recorded, never crash the whole eval
        error = f"{type(e).__name__}: {e}"
    finally:
        await session.close()
    r = score_case(case, outcome, rejected=rejected, error=error, session=session,
                   max_messages=s.max_messages, reproducible_ok=repro)
    r.update(mode=mode, run_id=session.workspace.run_id, wall_s=round(time.perf_counter() - t0, 2))
    return r


def rate(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(bool(v) for v in vals) / len(vals), 3) if vals else None


def summarize_mode(rows: list[dict]) -> dict:
    ran = [r for r in rows if r["kind"] != "rejected" and not r.get("error")]
    avg = lambda k: round(sum(r.get(k) or 0 for r in ran) / len(ran), 2) if ran else None
    return {
        "cases": len(rows), "errors": sum(1 for r in rows if r.get("error")),
        **{f"{k}_rate": rate(rows, k) for k in CHECKS},
        "avg_messages": avg("messages"), "avg_tokens": avg("tokens"), "avg_duration_s": avg("duration_s"),
        "max_messages_seen": max((r.get("messages") or 0 for r in ran), default=0),
    }


def report_md(results: list[dict], summary: dict, meta: dict) -> str:
    fmt = lambda v: "-" if v is None else ("✅" if v is True else "❌" if v is False else f"{v:.0%}" if isinstance(v, float) and v <= 1 else str(v))
    lines = [f"# DataPilot eval report", "", f"- Date: {meta['date']}", f"- Provider/model: {meta['provider']} / {meta['model']}",
             f"- Offline (scripted agents): {meta['offline']}", f"- Repeats: {meta['repeat']}", "",
             "## Summary by team mode", "", "| metric | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    for k in next(iter(summary.values())):
        lines.append(f"| {k} | " + " | ".join(fmt(summary[m][k]) for m in summary) + " |")
    lines += ["", "## Cases", "", "| mode | case | kind | pass | approved | msgs | fixed err | entities | numbers | grounded | charts | reproducible | tokens | sec | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        note = r.get("error") or ("rejected by input guardrail" if r.get("rejected") else r.get("stop_reason", ""))
        lines.append("| " + " | ".join(str(x) for x in [
            r["mode"], r["case"], r["kind"], fmt(r["passed"]), fmt(r.get("approved")), r.get("messages", "-"),
            fmt(r.get("self_corrected")), fmt(r.get("entities_ok")), r.get("numbers_matched", "-"), fmt(r.get("grounded")),
            fmt(r.get("charts_ok")), fmt(r.get("reproducible")), r.get("tokens", "-"), r.get("duration_s", "-"),
            str(note)[:60].replace("|", "/")]) + " |")
    lines += ["", "## Done checklist (spec 3.8) evidence", ""]
    allr = [r for r in results if not r.get("error")]
    lines.append(f"- D1 self-correction seen: {any(r.get('self_corrected') for r in allr)}")
    lines.append(f"- D3 every run within the message limit: {all(r.get('within_limit', True) for r in allr)}")
    lines.append(f"- D4 analysis.py reproducible: {rate(allr, 'reproducible')}")
    lines.append(f"- D5 charts labelled: {rate(allr, 'charts_ok')}")
    def verdict(case_id: str) -> str:
        rs = [r["passed"] for r in allr if r["case"] == case_id]
        return "not run" if not rs else str(all(rs))
    lines.append(f"- Guardrails: injection rejected = {verdict('injection')}, "
                 f"destructive request safe = {verdict('destructive')}")
    return "\n".join(lines) + "\n"


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["round_robin", "routed"])
    ap.add_argument("--cases", nargs="+", help="case ids (default: all)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between cases (rate limits)")
    ap.add_argument("--offline", action="store_true", help="scripted agents on the spec case (harness check)")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval_results")
    args = ap.parse_args(argv)

    s = get_settings()
    if not args.offline and not s.api_key:
        print(f"{s.provider_cfg['key_env']} is not set; add it to .env, or use --offline.")
        return 2
    cases = build_cases(DATA)
    if args.offline:
        cases, args.modes = [c for c in cases if c.id == "spec_drop"], ["routed"]
    elif args.cases:
        cases = [c for c in cases if c.id in args.cases]

    out = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    results: list[dict] = []
    total = len(args.modes) * len(cases) * args.repeat
    i = 0
    for mode in args.modes:
        for rep in range(args.repeat):
            for case in cases:
                i += 1
                try:
                    r = await run_case(case, mode, out / "runs", args.offline)
                except MissingAPIKeyError as e:
                    print(e)
                    return 2
                r["repeat"] = rep
                results.append(r)
                with (out / "results.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, default=str) + "\n")
                print(f"[{i}/{total}] {mode:11} {case.id:24} {'PASS' if r['passed'] else 'FAIL'}  "
                      f"msgs={r.get('messages', '-')} tokens={r.get('tokens', '-')} {r.get('error') or ''}")
                if not args.offline and i < total:
                    await asyncio.sleep(args.sleep)

    summary = {m: summarize_mode([r for r in results if r["mode"] == m]) for m in args.modes}
    meta = {"date": datetime.now().isoformat(timespec="seconds"), "provider": s.provider,
            "model": "scripted (offline)" if args.offline else s.model, "offline": args.offline, "repeat": args.repeat}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
    (out / "report.md").write_text(report_md(results, summary, meta), encoding="utf-8")
    print(f"\nReport: {out / 'report.md'}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

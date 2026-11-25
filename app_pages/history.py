from pathlib import Path

import pandas as pd
import streamlit as st

from datapilot.config import get_settings
from datapilot.observability import load_index, load_trace, summarize

runs_dir = get_settings().runs_dir
rows = load_index(runs_dir)

st.title("Run history", anchor=False)
st.caption("Every question asked, with the metrics from its trace (runs/index.jsonl).")

if not rows:
    st.info("No runs yet. Ask a question on the Analyst page.", icon=":material/info:")
    st.stop()

k = summarize(rows)
pct = lambda x: "-" if x is None else f"{x:.0%}"
with st.container(horizontal=True):
    st.metric("Questions", k["questions"], border=True)
    st.metric("Reviewer approval rate", pct(k["approval_rate"]), border=True)
    st.metric("Self-correction rate", pct(k["self_correction_rate"]), border=True,
              help="Of questions where code failed at least once, the share where the agents fixed it.")
    st.metric("Avg messages", k["avg_messages"], border=True)
    st.metric("Avg tokens", f"{k['avg_tokens']:,.0f}", border=True)
    st.metric("Avg duration", f"{k['avg_duration_s']:.1f}s", border=True)
    st.metric("Hit message limit", k["hit_message_limit"], border=True)
    st.metric("Guardrail blocks", k["guardrail_blocks"], border=True)

df = pd.DataFrame(rows)
cols = ["started_at", "run_id", "question_index", "question", "status", "human_decision", "team_mode", "model",
        "messages", "executions", "execution_errors", "self_corrections", "revise_count", "total_tokens", "duration_s"]
df = df[[c for c in cols if c in df.columns]].sort_values("started_at", ascending=False)
st.dataframe(
    df, hide_index=True, alt="Table of past questions and their run metrics",
    column_config={
        "started_at": st.column_config.DatetimeColumn("Started", format="YYYY-MM-DD HH:mm"),
        "run_id": "Run", "question_index": st.column_config.NumberColumn("Q#", width="small"),
        "question": st.column_config.TextColumn("Question", width="large"),
        "duration_s": st.column_config.NumberColumn("Seconds", format="%.1f"),
        "total_tokens": st.column_config.NumberColumn("Tokens", format="%d"),
    },
)

st.subheader("Trace timeline", anchor=False)
run_id = st.selectbox("Run", df["run_id"].unique().tolist())
events = load_trace(runs_dir / run_id)


def summary(e: dict) -> str:
    ev = e["event"]
    if ev == "message":
        return (e.get("text") or "").splitlines()[0][:160] if e.get("text") else ""
    if ev == "reasoning":
        if e["kind"] == "plan":
            return f"Plan: {' → '.join(e['steps'])} | Assumptions: {'; '.join(e['assumptions'])}"
        if e["kind"] == "review":
            return f"{e['verdict']}: {'; '.join(e['reasons'])}"
        return (e.get("text") or "")[:160]
    if ev == "execution":
        return f"step {e['step']} exit={e['exit_code']} in {e['duration_s']}s (code {e['code_sha']})"
    if ev == "guardrail":
        return f"{e['guardrail']} ok={e['ok']} {'; '.join(e['violations'] + e['warnings'])}"[:200]
    if ev == "outcome":
        return f"{e['status']} ({e['stop_reason']})"
    if ev == "human":
        return f"{e['decision']} {e.get('text') or ''}"
    return {"session_start": lambda: f"{e.get('provider')}/{e.get('model')} team={e.get('team_mode')}",
            "question_start": lambda: e.get("question", ""), "error": lambda: e.get("error", "")}.get(ev, lambda: "")()


tl = pd.DataFrame([{"time": e["ts"][11:23], "q": e.get("q"), "event": e["event"],
                    "agent": e.get("agent", ""), "tokens": (e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0),
                    "detail": summary(e)} for e in events])
st.dataframe(tl, hide_index=True, alt=f"Trace events for run {run_id}",
             column_config={"detail": st.column_config.TextColumn("Detail", width="large")})

run_path = runs_dir / run_id
with st.container(horizontal=True):
    for name in ("trace.jsonl", "metrics.json"):
        p = run_path / name
        if p.exists():
            st.download_button(name, p.read_bytes(), file_name=f"{run_id}_{name}", icon=":material/download:")

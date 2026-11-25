"""Streamlit rendering helpers shared by the app pages (business logic stays in the package)."""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st
from autogen_agentchat.messages import BaseChatMessage, ThoughtEvent

AGENTS = {
    "planner": ("Planner", ":material/route:"),
    "coder": ("Coder", ":material/code:"),
    "executor": ("Executor", ":material/terminal:"),
    "reviewer": ("Reviewer", ":material/fact_check:"),
}
_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
SECTIONS = ("ANSWER", "WHERE IT CAME FROM", "FILES", "SUGGESTED NEXT QUESTION")


def to_event(item) -> dict | None:
    """Convert an AutoGen stream item into a plain, re-renderable dict (None = not shown)."""
    if isinstance(item, ThoughtEvent):
        return {"agent": item.source, "kind": "thought", "text": item.content}
    if not isinstance(item, BaseChatMessage) or item.source == "user":
        return None
    text = item.to_text()
    ev = {"agent": item.source, "kind": "message", "text": text, "status": "ok"}
    usage = item.models_usage
    ev["tokens"] = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    if item.source == "coder":
        m = _CODE_RE.search(text)
        ev["code"] = m.group(1).strip() if m else ""
        ev["text"] = _CODE_RE.sub("", text).strip()
    elif item.source == "executor":
        if "STATUS: BLOCKED" in text or "not approved" in text.lower():
            ev["status"] = "blocked"
        elif "STATUS: SUCCESS" in text and "STATUS: ERROR" not in text:
            ev["status"] = "chart_fail" if "CHART CHECK FAILED" in text else "ok"
        else:
            ev["status"] = "error"
    elif item.source == "reviewer":
        ev["status"] = "approved" if text.lstrip().upper().startswith("APPROVED") else "revise"
    return ev


def _label(ev: dict, prev: dict | None) -> tuple[str, str]:
    name, icon = AGENTS.get(ev["agent"], (ev["agent"].title(), ":material/smart_toy:"))
    if ev["kind"] == "thought":
        return f"{icon} {name}: model reasoning", "complete"
    s = ev.get("status")
    if ev["agent"] == "planner":
        return f"{icon} {name}: " + ("revised plan" if "PLAN:" in ev["text"] and prev else
                                     "plan" if "PLAN:" in ev["text"] else "plan unchanged"), "complete"
    if ev["agent"] == "coder":
        fixing = prev and prev.get("agent") in ("executor", "reviewer") and prev.get("status") != "ok"
        return f"{icon} {name}: " + ("fixed the code" if fixing else "wrote the code"), "complete"
    if ev["agent"] == "executor":
        return {
            "ok": (f"{icon} {name}: ran successfully", "complete"),
            "chart_fail": (f"{icon} {name}: ran, chart check failed", "error"),
            "blocked": (f"{icon} {name}: blocked by guardrail", "error"),
            "error": (f"{icon} {name}: error", "error"),
        }[s]
    if ev["agent"] == "reviewer":
        return (f"{icon} {name}: approved", "complete") if s == "approved" else (f"{icon} {name}: sent back", "error")
    return f"{icon} {name}", "complete"


def render_event(ev: dict, prev: dict | None) -> None:
    label, state = _label(ev, prev)
    with st.status(label, type="step", state=state, expanded=False):
        if ev["agent"] == "executor":
            st.code(_executor_body(ev["text"]), language="text")
        elif ev["agent"] == "coder":
            if ev["text"]:
                st.markdown(ev["text"])
            if ev.get("code"):
                st.code(ev["code"], language="python", line_numbers=True)
        elif ev["agent"] == "reviewer" and ev.get("status") == "approved":
            st.markdown("Checks passed: question answered, numbers sensible, charts labelled.")
        else:
            st.markdown(ev["text"])
        if ev.get("tokens"):
            st.caption(f"{ev['tokens']:,} tokens")


def _executor_body(text: str) -> str:
    # Drop AutoGen's generic prefix so the output starts at our STATUS line.
    i = text.find("STATUS:")
    return text[i:] if i > 0 else text


def parse_answer(answer: str) -> dict[str, str]:
    out: dict[str, str] = {}
    pattern = r"^(%s)\s*$" % "|".join(re.escape(s) for s in SECTIONS)
    parts = re.split(pattern, answer.strip(), flags=re.M)
    if len(parts) < 3:
        return {"ANSWER": answer.strip()}
    for i in range(1, len(parts) - 1, 2):
        out[parts[i]] = parts[i + 1].strip()
    return out


def _pdf_factory(turn: dict):
    """Capture everything now: the download callable runs on another thread without session state."""
    from .pdf_report import build_pdf
    sess = st.session_state.get("dp_session")
    kwargs = dict(question=turn["question"], answer=turn["answer"], charts=[Path(c) for c in turn.get("charts", [])],
                  dataset=st.session_state.get("dp_dataset_label") or "", status=turn["status"],
                  run_id=sess.workspace.run_id if sess else "", metrics=dict(turn.get("metrics") or {}))
    return lambda: build_pdf(**kwargs)


def status_badge(status: str, human: str | None) -> str:
    badge = {"approved": ":green-badge[:material/verified: Reviewer approved]",
             "ungrounded": ":orange-badge[:material/warning: Numbers not verified]",
             "not_approved": ":red-badge[:material/block: Not approved]",
             "error": ":red-badge[:material/error: Error]"}.get(status, status)
    if human == "approved":
        badge += " :blue-badge[:material/person_check: You approved]"
    elif human == "follow_up":
        badge += " :gray-badge[:material/forum: Followed up]"
    return badge


def render_outcome(turn: dict, key: str) -> None:
    """Final answer card: answer sections, charts, downloads, metrics."""
    st.markdown(status_badge(turn["status"], turn.get("human_decision")))
    if turn["status"] in ("approved", "ungrounded") and turn["answer"]:
        sec = parse_answer(turn["answer"])
        with st.container(border=True):
            st.markdown(f"**Answer**\n\n{sec.get('ANSWER', '')}")
            if sec.get("WHERE IT CAME FROM"):
                st.markdown(f"**Where it came from**\n\n{sec['WHERE IT CAME FROM']}")
    else:
        st.warning(f"The team stopped without an approved answer ({turn['stop_reason']}). "
                   "Rephrase the question or ask it again.", icon=":material/report:")
    for w in turn.get("grounding_warnings", []):
        st.caption(f":material/warning: {w}")

    charts = [Path(p) for p in turn.get("charts", []) if Path(p).exists()]
    if charts:
        cols = st.columns(min(len(charts), 2))
        for i, p in enumerate(charts):
            with cols[i % len(cols)]:
                st.image(str(p), caption=p.name, alt=f"Chart {p.stem.replace('_', ' ')}")

    with st.container(horizontal=True):
        for p in charts:
            st.download_button(p.name, p.read_bytes(), file_name=p.name, mime="image/png",
                               icon=":material/image:", key=f"{key}-dl-{p.name}")
        a = turn.get("analysis_file")
        if a and Path(a).exists():
            st.download_button("analysis.py", Path(a).read_bytes(), file_name="analysis.py", mime="text/x-python",
                               icon=":material/download:", key=f"{key}-dl-py")
        if turn["status"] in ("approved", "ungrounded") and turn["answer"]:
            st.download_button("Report (PDF)", _pdf_factory(turn), file_name="datapilot_report.pdf",
                               mime="application/pdf", icon=":material/picture_as_pdf:", key=f"{key}-dl-pdf",
                               type="primary")

    m = turn["metrics"]
    st.caption(f"{m['messages']} messages · {m['executions']} code runs · {m['self_corrections']} self-corrections · "
               f"{m['revise_count']} review rounds sent back · {m['total_tokens']:,} tokens · {m['duration_s']:.1f}s")

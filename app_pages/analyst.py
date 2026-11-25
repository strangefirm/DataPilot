import time
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from datapilot import app_state as A
from datapilot.config import get_settings
from datapilot.demo import SAMPLE_QUESTION
from datapilot.session import GuardrailError, RunOutcome
from datapilot.ui import parse_answer, render_event, render_outcome, to_event

settings = get_settings()
ss = st.session_state
has_key = A.api_key_present()

# ------------------------------------------------------------------ sidebar: data + settings
with st.sidebar:
    st.subheader("Data", divider="gray")
    st.file_uploader("Upload a sales file", type=["csv", "xlsx", "xls"], key="dp_upload", on_change=A.on_upload,
                     help=f"CSV or Excel, up to {settings.max_upload_mb} MB.")
    with st.container(horizontal=True):
        st.button("Use sample data", icon=":material/dataset:", on_click=A.on_sample, disabled=not has_key)
        st.button("Demo (no API key)", icon=":material/play_circle:", on_click=A.on_sample, kwargs={"demo": True},
                  help="Scripted agent replies on the sample data. Code really runs; the LLM is replaced.")

    st.subheader("Settings", divider="gray")
    st.segmented_control("Team", ["round_robin", "routed", "selector"], default=settings.team, key="dp_team_mode",
                         required=True, help="Applies to the next session. round_robin = spec baseline; "
                         "routed = errors go straight back to the Coder; selector = LLM picks the next speaker.")
    if has_key:
        st.caption(f":material/key: {settings.provider} · `{settings.model}`")
    else:
        st.caption(f":material/key_off: No API key. Add `{settings.provider_cfg['key_env']}` to `.env`, or try the demo.")
    st.caption(f"Sandbox: {settings.executor} · limit {settings.max_messages} messages · "
               f"{settings.max_tokens_per_run:,} tokens per question")

    session = ss.dp_session
    if session is not None:
        st.subheader("Session", divider="gray")
        st.markdown(f"**{ss.dp_dataset_label}**  \n{session.dataset.rows:,} rows · {len(session.dataset.columns)} columns")
        if session.dataset.sheets:
            st.caption(f":material/tab: {len(session.dataset.sheets)} sheet(s): {', '.join(session.dataset.sheets)}")
        if ss.dp_demo:
            st.caption(":orange-badge[Demo mode] scripted replies, sample question only")
        with st.expander("Schema the agents see"):
            st.code(session.dataset.schema_text, language="text")
        for w in session.dataset.injection_check.warnings:
            st.warning(w, icon=":material/shield:")
        st.caption(f"Run folder: `runs/{session.workspace.run_id}`")
        st.button("New session", icon=":material/restart_alt:", on_click=A.close_session)

# ------------------------------------------------------------------ main
st.title("DataPilot", anchor=False)
st.caption("Ask a business question about your sales data. A team of agents plans, writes Python, runs it, "
           "fixes its own errors, and returns the answer with charts.")

if ss.dp_error:
    st.error(ss.dp_error, icon=":material/error:")

session = ss.dp_session
if session is None:
    with st.container(border=True):
        st.markdown(":material/upload_file: **Upload a CSV or Excel file** in the sidebar, or start with the sample "
                    "sales data (12,480 orders, 2026).")
        st.markdown("**How it works:** Planner writes a plan → Coder writes pandas code → Executor runs it in a "
                    "sandbox → Reviewer approves or sends it back → you approve or ask a follow-up.")
    st.stop()

# previous turns
for i, turn in enumerate(ss.dp_turns):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar=":material/insights:"):
        m = turn["metrics"]
        with st.status(f"Agents worked {m['duration_s']:.0f}s · {m['messages']} messages · "
                       f"{m['self_corrections']} self-corrections", type="compact", state="complete"):
            prev = None
            for ev in turn["events"]:
                render_event(ev, prev)
                prev = ev
        render_outcome(turn, key=f"t{i}")
        is_last = i == len(ss.dp_turns) - 1
        if is_last and turn["human_decision"] is None and turn["status"] in ("approved", "ungrounded"):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.button("Approve answer", type="primary", icon=":material/check:", on_click=A.approve_turn,
                          args=(i,), key=f"approve{i}")
                nxt = parse_answer(turn["answer"]).get("SUGGESTED NEXT QUESTION", "").strip()
                if nxt and not ss.dp_demo:
                    st.button(nxt, icon=":material/forum:", type="tertiary", on_click=A.ask_suggested, args=(nxt,),
                              key=f"suggest{i}")
                st.caption("…or type a follow-up question below.")

if not ss.dp_turns:
    st.pills("Try asking", [SAMPLE_QUESTION], key="dp_sample_pill", on_change=lambda: A.ask_suggested(
        ss.dp_sample_pill) if ss.dp_sample_pill else None, label_visibility="collapsed")

prompt = st.chat_input("Ask a question about your data", submit_mode="disable", max_chars=1000)
question = prompt or ss.dp_pending_question
ss.dp_pending_question = None

if question and ss.dp_demo and ss.dp_turns:
    st.info("Demo mode has scripted replies for the sample question only. Add an API key to `.env` and start a "
            "new session to ask your own questions.", icon=":material/info:")
    question = None

if question:
    # A question typed while the last answer awaits approval counts as the human's follow-up decision.
    if ss.dp_turns and ss.dp_turns[-1]["human_decision"] is None:
        ss.dp_loop.run(session.human_review(session.history[-1], lambda _: question), timeout=30)
        ss.dp_turns[-1]["human_decision"] = "follow_up"

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/insights:"):
        t0 = time.time()
        events, outcome, prev = [], None, None
        try:
            with st.status(":shimmer[Agents working]", type="compact", expanded=True) as status:
                for item in ss.dp_loop.iterate(session.ask(question)):
                    if isinstance(item, RunOutcome):
                        outcome = item
                        continue
                    ev = to_event(item)
                    if ev:
                        render_event(ev, prev)
                        events.append(ev)
                        prev = ev
                status.update(label=f"Agents worked {time.time() - t0:.0f}s · {outcome.messages} messages · "
                              f"{outcome.self_corrections} self-corrections", state="complete", expanded=False)
        except GuardrailError as e:
            st.warning(f"Question rejected by guardrail: {e}", icon=":material/shield:")
            st.stop()
        except Exception as e:  # LLM/API errors etc. are traced by the session; show them plainly
            st.error(f"{type(e).__name__}: {e}", icon=":material/error:")
            st.stop()

        n = len(session.history)
        metrics = asdict(session.metrics[-1]) | {"total_tokens": session.metrics[-1].total_tokens}
        turn = {
            "question": question, "events": events, "status": outcome.status, "answer": outcome.answer,
            "stop_reason": outcome.stop_reason, "human_decision": None, "metrics": metrics,
            "grounding_warnings": (outcome.grounding.warnings + outcome.grounding.violations) if outcome.grounding else [],
            "charts": [str(p) for p in outcome.charts if p.stat().st_mtime >= t0 - 1],
            "analysis_file": str(session.workspace.root / f"analysis_q{n}.py"),
        }
        ss.dp_turns.append(turn)
    st.rerun()  # re-render through the history loop (adds approve / follow-up controls)

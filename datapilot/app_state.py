"""Per-user app state for Streamlit: one DataPilotSession + its event-loop thread per browser session."""

from __future__ import annotations

import secrets
from dataclasses import replace
from pathlib import Path

import streamlit as st

from .async_bridge import LoopThread
from .config import get_settings
from .guardrails import check_upload
from .llm import MissingAPIKeyError
from .session import DataPilotSession, GuardrailError

SAMPLE = get_settings().data_dir / "sales_2026.csv"
UPLOADS = get_settings().runs_dir / "_uploads"


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("dp_session", None)
    ss.setdefault("dp_loop", None)
    ss.setdefault("dp_turns", [])
    ss.setdefault("dp_demo", False)
    ss.setdefault("dp_dataset_label", None)
    ss.setdefault("dp_error", None)
    ss.setdefault("dp_pending_question", None)


def close_session() -> None:
    ss = st.session_state
    if ss.get("dp_session") is not None and ss.get("dp_loop") is not None:
        try:
            ss.dp_loop.run(ss.dp_session.close(), timeout=10)
        except Exception:
            pass
        ss.dp_loop.close()
    ss.dp_session, ss.dp_loop, ss.dp_turns, ss.dp_dataset_label = None, None, [], None


def start_session(path: Path, label: str, team_mode: str, demo: bool) -> None:
    close_session()
    ss = st.session_state
    ss.dp_error = None
    settings = replace(get_settings(), team="routed" if demo else team_mode)
    clients = None
    if demo:
        from .demo import demo_clients
        clients = demo_clients()
    try:
        ss.dp_session = DataPilotSession.create(path, settings=settings, clients=clients)
    except (GuardrailError, MissingAPIKeyError) as e:
        ss.dp_error = str(e)
        return
    ss.dp_loop = LoopThread()
    ss.dp_demo = demo
    ss.dp_dataset_label = label


def on_upload() -> None:
    f = st.session_state.get("dp_upload")
    if f is None:
        return
    ss = st.session_state
    check = check_upload(f.name, f.size, get_settings().max_upload_mb)
    if not check.ok:
        ss.dp_error = "; ".join(check.violations)
        return
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{secrets.token_hex(4)}_{Path(f.name).name}"
    path.write_bytes(f.getvalue())
    try:
        start_session(path, f.name, ss.get("dp_team_mode", "round_robin"), demo=False)
    finally:
        path.unlink(missing_ok=True)  # the session keeps its own copy in its sandbox folder


def on_sample(demo: bool = False) -> None:
    start_session(SAMPLE, "sales_2026.csv (sample)", st.session_state.get("dp_team_mode", "round_robin"), demo)


def approve_turn(index: int) -> None:
    ss = st.session_state
    turn = ss.dp_turns[index]
    outcome = ss.dp_session.history[index]
    ss.dp_loop.run(ss.dp_session.human_review(outcome, lambda _: "yes"), timeout=30)
    turn["human_decision"] = "approved"


def ask_suggested(question: str) -> None:
    st.session_state.dp_pending_question = question


def api_key_present() -> bool:
    return bool(get_settings().api_key)

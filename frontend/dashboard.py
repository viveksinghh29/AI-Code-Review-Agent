"""
Dashboard — main Streamlit entry point. Called from app.py via run_dashboard().
"""

import os
import streamlit as st

from backend.orchestration import Orchestrator, PipelineStatus, ProgressEvent
from backend.utils.config import get_config, reload_config
from frontend.components import (
    render_header, render_sidebar, render_repo_input, render_progress,
    render_repo_metadata, render_summary_cards, render_all_reviews,
    render_download_buttons, render_error, render_empty_state,
)
from frontend.charts import render_all_charts


def _init_state() -> None:
    for key, val in [("result",None),("events",[]),
                     ("running",False),("last_url","")]:
        if key not in st.session_state:
            st.session_state[key] = val


def _apply_settings(settings: dict) -> None:
    if settings.get("api_key"):
        provider = settings.get("provider", "anthropic")
        env_key  = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        os.environ[env_key]        = settings["api_key"]
        os.environ["LLM_PROVIDER"] = provider
    if settings.get("github_token"):
        os.environ["GITHUB_TOKEN"] = settings["github_token"]
    if settings.get("max_files"):
        os.environ["MAX_FILES"] = str(settings["max_files"])
    reload_config()


def _run_pipeline(url: str, settings: dict) -> None:
    _apply_settings(settings)
    st.session_state["running"] = True
    st.session_state["events"]  = []
    st.session_state["result"]  = None

    events_ref           = st.session_state["events"]
    progress_placeholder = st.empty()

    def on_progress(evt: ProgressEvent) -> None:
        events_ref.append(evt)
        with progress_placeholder.container():
            render_progress(events_ref)

    orch   = Orchestrator(use_cache=settings.get("use_cache", True))
    result = orch.run(url, progress_callback=on_progress)

    st.session_state["result"]   = result
    st.session_state["running"]  = False
    st.session_state["last_url"] = url
    progress_placeholder.empty()


def run_dashboard() -> None:
    _init_state()

    settings = render_sidebar()
    render_header()
    url, submitted = render_repo_input()
    st.divider()

    if submitted:
        if not url:
            st.warning("⚠️ Please enter a GitHub repository URL.")
            return

        orch = Orchestrator(use_cache=False)
        is_valid, err_msg = orch.validate_url(url)
        if not is_valid:
            render_error("Invalid GitHub URL", err_msg)
            return

        cfg    = get_config()
        errors = cfg.validate()
        if errors and not settings.get("api_key"):
            st.warning(
                "⚠️ No API key configured — AST-based findings will still appear. "
                "Add your key in the sidebar for full AI review."
            )

        with st.spinner(""):
            _run_pipeline(url, settings)
        st.rerun()

    result = st.session_state.get("result")

    if result is None:
        render_empty_state()
        return

    if result.status == PipelineStatus.FAILED:
        render_error(f"Pipeline failed for `{result.repo_url}`", result.error)
        if st.session_state["events"]:
            render_progress(st.session_state["events"])
        return

    if st.session_state["events"]:
        render_progress(st.session_state["events"])
        st.divider()

    render_repo_metadata(result)
    st.divider()
    render_summary_cards(result)
    st.divider()
    render_all_charts(result)
    st.divider()
    render_download_buttons(result)
    st.divider()
    render_all_reviews(result, settings)

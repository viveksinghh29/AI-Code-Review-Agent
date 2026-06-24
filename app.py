"""
AI Code Review Agent — Main Entry Point
Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AI Code Review Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Full dashboard is loaded from the frontend package (built in Phase 7)
from frontend.dashboard import run_dashboard  # noqa: E402

if __name__ == "__main__" or True:
    run_dashboard()

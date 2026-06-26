"""UI components for rendering the AI Code Review Agent dashboard."""

import streamlit as st
from typing import Optional

from backend.orchestration import PipelineResult, ProgressEvent, PipelineStatus
from backend.reviewer.ai_reviewer import FileReview, ReviewComment


# ─────────────────────────────────────────────────────────────────────────────
# Theme constants
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_COLOUR = {
    "Critical": "#FF4B4B",
    "High":     "#FF8C00",
    "Medium":   "#FFD700",
    "Low":      "#00CC88",
}
SEVERITY_ICON = {
    "Critical": "🔴",
    "High":     "🟠",
    "Medium":   "🟡",
    "Low":      "🟢",
}
CATEGORY_ICON = {
    "Security":        "🔒",
    "Performance":     "⚡",
    "Readability":     "📖",
    "Maintainability": "🔧",
    "Scalability":     "📈",
    "Best Practices":  "✅",
    "Bug Risk":        "🐛",
}
GRADE_COLOUR = {
    "A": "#00CC88",
    "B": "#66BB6A",
    "C": "#FFD700",
    "D": "#FF8C00",
    "F": "#FF4B4B",
}


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 0.5rem 0;">
        <h1 style="font-size:2.6rem; font-weight:800; margin:0;">
            🔍 AI Code Review Agent
        </h1>
        <p style="color:#888; font-size:1.05rem; margin-top:0.4rem;">
            Autonomous code analysis powered by Claude &amp; static AST parsing
        </p>
    </div>
    <hr style="margin:0.5rem 0 1.5rem 0; border-color:#333;">
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render configuration sidebar. Returns dict of user settings."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.divider()

        st.markdown("### 🤖 LLM Provider")
        provider = st.selectbox(
            "Provider", ["anthropic", "openai"], index=0,
            help="Choose the LLM that powers code review.",
        )
        label      = "Anthropic API Key" if provider == "anthropic" else "OpenAI API Key"
        ph         = "sk-ant-…"          if provider == "anthropic" else "sk-…"
        api_key    = st.text_input(label, type="password", placeholder=ph)

        st.divider()
        st.markdown("### 🐙 GitHub")
        github_token = st.text_input(
            "GitHub Token (optional)", type="password", placeholder="ghp_…",
            help="Required only for private repositories.",
        )

        st.divider()
        st.markdown("### 🔬 Analysis")
        max_files = st.slider("Max files to review", 5, 100, 30, 5)
        use_cache = st.checkbox("Cache results", value=True)

        st.divider()
        st.markdown("### 🔎 Default Filters")
        min_confidence = st.slider("Min confidence %", 0, 100, 50, 5)
        default_severities = st.multiselect(
            "Show severities",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"],
        )

        st.divider()
        st.caption("AI Code Review Agent · v1.0")

    return {
        "provider":           provider,
        "api_key":            api_key,
        "github_token":       github_token,
        "max_files":          max_files,
        "use_cache":          use_cache,
        "min_confidence":     min_confidence,
        "default_severities": default_severities,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Repository Input
# ─────────────────────────────────────────────────────────────────────────────

def render_repo_input() -> tuple[str, bool]:
    """Render URL input + submit. Returns (url, submitted)."""
    st.markdown("### 📦 Repository URL")
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        url = st.text_input(
            "url", placeholder="https://github.com/owner/repository",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.button("🔍 Review", type="primary", use_container_width=True)

    st.markdown(
        "<small>Examples: "
        "<code>https://github.com/realpython/reader</code> &nbsp;|&nbsp; "
        "<code>https://github.com/pallets/flask</code></small>",
        unsafe_allow_html=True,
    )
    return url.strip(), submitted


# ─────────────────────────────────────────────────────────────────────────────
# Progress Display
# ─────────────────────────────────────────────────────────────────────────────

def render_progress(events: list[ProgressEvent]) -> None:
    """Render a progress bar and collapsible event log."""
    if not events:
        return
    latest = events[-1]
    st.progress(latest.pct / 100.0,
                text=f"**{latest.stage}** — {latest.message}")
    with st.expander("📋 Pipeline Log", expanded=False):
        log_lines = []
        for evt in events:
            icon = {
                PipelineStatus.INGESTING: "📥",
                PipelineStatus.PARSING:   "🔬",
                PipelineStatus.ANALYSING: "📊",
                PipelineStatus.REVIEWING: "🤖",
                PipelineStatus.REPORTING: "📄",
                PipelineStatus.COMPLETE:  "✅",
                PipelineStatus.FAILED:    "❌",
            }.get(evt.status, "•")
            log_lines.append(
                f"`{evt.timestamp[11:19]}` {icon} **{evt.stage}** — {evt.message}"
            )
        st.markdown("\n\n".join(log_lines))


# ─────────────────────────────────────────────────────────────────────────────
# Repository Metadata Card
# ─────────────────────────────────────────────────────────────────────────────

def render_repo_metadata(result: PipelineResult) -> None:
    meta = result.metadata
    st.markdown("### 📁 Repository Info")
    cols = st.columns(4)
    cols[0].metric("Owner",      meta.owner)
    cols[1].metric("Repository", meta.name)
    cols[2].metric("Branch",     meta.default_branch)
    cols[3].metric("Elapsed",    f"{result.elapsed_seconds}s")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"**Last Commit:** `{meta.last_commit.sha}` — "
            f"{meta.last_commit.message[:60]}"
        )
        st.markdown(f"**Author:** {meta.last_commit.author}")
    with c2:
        langs = " · ".join(
            f"**{l}** ({n})"
            for l, n in sorted(meta.languages.items(), key=lambda x: -x[1])
        )
        st.markdown(f"**Languages:** {langs or 'N/A'}")
        st.markdown(
            f"**Files:** {meta.supported_files} reviewed "
            f"({meta.skipped_files} skipped) · "
            f"**Lines:** {meta.total_lines:,}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Summary KPI Cards
# ─────────────────────────────────────────────────────────────────────────────

def render_summary_cards(result: PipelineResult) -> None:
    rev = result.review_report
    agg = result.analysis_report.aggregate
    st.markdown("### 📊 Review Summary")

    cols = st.columns(5)
    cols[0].metric("💬 Total",   rev.total_comments)
    _coloured_metric_in(cols[1], "🔴 Critical", rev.critical_count, "#FF4B4B")
    _coloured_metric_in(cols[2], "🟠 High",     rev.high_count,     "#FF8C00")
    _coloured_metric_in(cols[3], "🟡 Medium",   rev.medium_count,   "#FFD700")
    _coloured_metric_in(cols[4], "🟢 Low",      rev.low_count,      "#00CC88")

    st.markdown("")
    cols2 = st.columns(5)
    cols2[0].metric("⭐ Avg Quality",    f"{rev.avg_quality_score:.0f}/100")
    cols2[1].metric("🎯 Avg Confidence", f"{rev.avg_confidence:.0f}%")
    cols2[2].metric("📚 Doc Coverage",   f"{agg.doc_coverage_pct:.0f}%")
    cols2[3].metric("🏷️ Type Hints",    f"{agg.type_hint_pct:.0f}%")
    health_icon = {
        "Excellent":"🟢","Good":"🔵","Fair":"🟡","Poor":"🟠","Critical":"🔴"
    }.get(agg.health_label, "⚪")
    cols2[4].metric("🏥 Health", f"{health_icon} {agg.health_label}")


def _coloured_metric_in(col, label: str, value: int, colour: str) -> None:
    with col:
        st.markdown(
            f'<div style="border:1px solid #333;border-radius:8px;'
            f'padding:0.6rem 0.8rem;text-align:center;">'
            f'<div style="font-size:0.8rem;color:#888;">{label}</div>'
            f'<div style="font-size:1.8rem;font-weight:700;color:{colour};">'
            f'{value}</div></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Review Comment Card
# ─────────────────────────────────────────────────────────────────────────────

def render_review_comment(comment: ReviewComment, index: int = 0) -> None:
    sev_colour = SEVERITY_COLOUR.get(comment.severity, "#888")
    sev_icon   = SEVERITY_ICON.get(comment.severity,   "⚪")
    cat_icon   = CATEGORY_ICON.get(comment.category,   "📌")
    src_badge  = "🤖 AI" if not comment.is_ast_detected else "🔬 AST"

    label = (
        f"{sev_icon} **L{comment.line_number}** · "
        f"`{comment.issue_type}` · "
        f"{comment.severity} · {comment.confidence_score}% confidence"
    )

    with st.expander(label, expanded=(comment.severity == "Critical")):
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown(f"**{cat_icon} Category:** {comment.category}")
            st.markdown("**📝 Explanation:**")
            st.markdown(f"> {comment.explanation}")
        with col_right:
            st.markdown(
                f'<div style="text-align:center;border:1px solid #333;'
                f'border-radius:8px;padding:0.5rem;">'
                f'<div style="font-size:0.75rem;color:#888;">Confidence</div>'
                f'<div style="font-size:2rem;font-weight:700;color:{sev_colour};">'
                f'{comment.confidence_score}%</div>'
                f'<div style="font-size:0.75rem;color:#888;">{src_badge}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if comment.suggested_fix:
            st.markdown("**🔧 Suggested Fix:**")
            if any(kw in comment.suggested_fix
                   for kw in ["def ", "class ", "import ", "return ", "    "]):
                st.code(comment.suggested_fix, language="python")
            else:
                st.info(comment.suggested_fix)

        st.markdown(
            f'<span style="background:{sev_colour}22;color:{sev_colour};'
            f'border:1px solid {sev_colour};border-radius:4px;'
            f'padding:2px 8px;font-size:0.8rem;">{comment.severity}</span>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# File Review Panel
# ─────────────────────────────────────────────────────────────────────────────

def render_file_review(file_review: FileReview, result: PipelineResult) -> None:
    score  = result.analysis_report.get_score(file_review.file_name)
    grade  = score.grade if score else "?"
    colour = GRADE_COLOUR.get(grade, "#888")

    st.markdown(
        f'<div style="border-left:4px solid {colour};padding:0.4rem 0.8rem;'
        f'margin-bottom:0.5rem;background:#1a1a1a;border-radius:0 6px 6px 0;">'
        f'<span style="font-size:1.1rem;font-weight:700;">📄 {file_review.file_name}</span>'
        f'&nbsp;&nbsp;<span style="color:{colour};font-weight:700;">{grade}</span>'
        f'&nbsp;<span style="color:#888;font-size:0.85rem;">'
        f'{file_review.language} · {file_review.line_count} lines · '
        f'score {file_review.overall_score}/100 · '
        f'{len(file_review.comments)} comments</span></div>',
        unsafe_allow_html=True,
    )

    if file_review.summary:
        st.caption(f"💬 {file_review.summary}")
    if file_review.has_errors:
        st.warning(f"⚠️ AI review error: {file_review.review_error}")

    if score:
        bar_cols = st.columns(5)
        dims = [
            ("📚 Docs",    score.documentation),
            ("🔀 Complex", score.complexity),
            ("🔧 Maint",   score.maintainability),
            ("🔒 Security",score.security),
            ("✏️ Style",   score.style),
        ]
        for col, (lbl, val) in zip(bar_cols, dims):
            grade_dim = (
                "A" if val >= 90 else "B" if val >= 80 else
                "C" if val >= 70 else "D" if val >= 60 else "F"
            )
            c = GRADE_COLOUR.get(grade_dim, "#888")
            with col:
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:0.7rem;color:#888;">{lbl}</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:{c};">'
                    f'{val}</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("")
    if file_review.comments:
        for i, comment in enumerate(file_review.comments):
            render_review_comment(comment, i)
    else:
        st.success("✅ No issues found in this file.")


# ─────────────────────────────────────────────────────────────────────────────
# Full Review Panel with Filters
# ─────────────────────────────────────────────────────────────────────────────

def render_all_reviews(result: PipelineResult, settings: dict) -> None:
    rev = result.review_report
    st.markdown("### 🔍 Review Results")

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        sel_sev = st.multiselect(
            "Severity", ["Critical", "High", "Medium", "Low"],
            default=settings.get("default_severities",
                                 ["Critical", "High", "Medium", "Low"]),
            key="filter_severity",
        )
    with fcol2:
        min_conf = st.slider("Min Confidence %", 0, 100,
                             value=settings.get("min_confidence", 50),
                             step=5, key="filter_confidence")
    with fcol3:
        all_files = sorted({r.file_name for r in rev.file_reviews})
        sel_files = st.multiselect("Files", all_files, default=[],
                                   placeholder="All files",
                                   key="filter_files")
    with fcol4:
        all_cats = sorted({
            c.category for r in rev.file_reviews for c in r.comments
        })
        sel_cats = st.multiselect("Category", all_cats, default=[],
                                  placeholder="All categories",
                                  key="filter_category")

    search = st.text_input("🔎 Search comments",
                           placeholder="e.g. eval, injection, docstring…",
                           key="filter_search")

    shown_files = shown_comments = 0
    for fr in rev.file_reviews:
        if sel_files and fr.file_name not in sel_files:
            continue
        filtered = _filter_comments(fr.comments, sel_sev, min_conf,
                                    sel_cats, search)
        if not filtered and sel_sev:
            continue
        shown_files    += 1
        shown_comments += len(filtered)
        clone = FileReview(
            file_name=fr.file_name, language=fr.language,
            line_count=fr.line_count, comments=filtered,
            summary=fr.summary, overall_score=fr.overall_score,
            review_error=fr.review_error,
        )
        render_file_review(clone, result)
        st.divider()

    if shown_files == 0:
        st.info("No files match the current filters.")
    else:
        st.caption(
            f"Showing {shown_comments} comments across {shown_files} files."
        )


def _filter_comments(
    comments: list[ReviewComment],
    severities: list[str],
    min_conf: int,
    categories: list[str],
    search: str,
) -> list[ReviewComment]:
    sl = search.lower().strip()
    out = []
    for c in comments:
        if severities and c.severity not in severities:
            continue
        if c.confidence_score < min_conf:
            continue
        if categories and c.category not in categories:
            continue
        if sl and not (
            sl in c.explanation.lower()
            or sl in c.issue_type.lower()
            or sl in c.suggested_fix.lower()
        ):
            continue
        out.append(c)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Download Buttons
# ─────────────────────────────────────────────────────────────────────────────

def render_download_buttons(result: PipelineResult) -> None:
    st.markdown("### 📥 Download Reports")
    slug = result.metadata.name.replace("/", "_")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download Markdown Report",
            data=result.markdown_report,
            file_name=f"review_{slug}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "⬇️ Download JSON Report",
            data=result.json_report,
            file_name=f"review_{slug}.json",
            mime="application/json",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Error & Empty State
# ─────────────────────────────────────────────────────────────────────────────

def render_error(message: str, detail: Optional[str] = None) -> None:
    st.error(f"❌ **Error:** {message}")
    if detail:
        with st.expander("Error details"):
            st.code(detail)


def render_empty_state() -> None:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem;color:#555;">
            <div style="font-size:4rem;">🔍</div>
            <h3 style="color:#666;margin:1rem 0 0.5rem 0;">Ready to review</h3>
            <p style="max-width:480px;margin:0 auto;">
                Enter a public GitHub repository URL above and click
                <strong>Review</strong> to start the AI-powered analysis.
            </p>
            <br>
            <p style="color:#444;font-size:0.9rem;">
                🔬 AST parsing &nbsp;·&nbsp; 🤖 AI review &nbsp;·&nbsp;
                📊 Quality metrics &nbsp;·&nbsp; 📥 Downloadable reports
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

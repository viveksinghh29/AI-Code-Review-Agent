"""
Charts Module — all Plotly analytics visualisations.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.orchestration import PipelineResult

_BG  = "rgba(0,0,0,0)"
_FONT = dict(family="Inter, sans-serif", color="#CCCCCC")
_GRID = "#2a2a2a"
_SEV_COLOURS = {
    "Critical": "#FF4B4B", "High": "#FF8C00",
    "Medium": "#FFD700",   "Low":  "#00CC88",
}
_CAT_COLOURS = ["#7C83FD","#96FFEA","#FF6B6B","#FFD93D","#6BCB77","#FF922B","#C77DFF"]

def _layout(**kw):
    return dict(paper_bgcolor=_BG, plot_bgcolor=_BG, font=_FONT,
                margin=dict(l=20,r=20,t=40,b=20),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CCC")),
                **kw)

def _empty(title):
    fig = go.Figure()
    fig.add_annotation(text="No data available", showarrow=False,
                       font=dict(color="#555", size=14))
    fig.update_layout(**_layout(title=title))
    return fig

def chart_severity_donut(result: PipelineResult) -> go.Figure:
    rev = result.review_report
    data = {"Severity":["Critical","High","Medium","Low"],
            "Count":[rev.critical_count,rev.high_count,rev.medium_count,rev.low_count]}
    df = pd.DataFrame(data)
    df = df[df["Count"] > 0]
    if df.empty:
        return _empty("Severity Distribution")
    fig = px.pie(df, names="Severity", values="Count", hole=0.55,
                 color="Severity", color_discrete_map=_SEV_COLOURS,
                 title="Severity Distribution")
    fig.update_traces(textposition="outside", textinfo="label+percent")
    fig.update_layout(**_layout())
    return fig

def chart_quality_scores(result: PipelineResult) -> go.Figure:
    scores = result.analysis_report.quality_scores
    if not scores:
        return _empty("Per-File Quality Scores")
    df = pd.DataFrame([
        {"File": s.file_path.split("/")[-1], "Score": s.overall,
         "Grade": s.grade, "Full": s.file_path}
        for s in sorted(scores, key=lambda s: s.overall)
    ])
    colours = [
        _SEV_COLOURS["Low"] if s >= 80 else
        _SEV_COLOURS["Medium"] if s >= 60 else
        _SEV_COLOURS["High"] if s >= 40 else
        _SEV_COLOURS["Critical"]
        for s in df["Score"]
    ]
    fig = go.Figure(go.Bar(
        x=df["Score"], y=df["File"], orientation="h",
        marker_color=colours,
        text=[f"{s}/100 ({g})" for s,g in zip(df["Score"],df["Grade"])],
        textposition="outside",
        customdata=df["Full"],
        hovertemplate="<b>%{customdata}</b><br>Score: %{x}/100<extra></extra>",
    ))
    fig.update_layout(**_layout(title="Per-File Quality Scores"),
                      xaxis=dict(range=[0,110], gridcolor=_GRID, title="Score"),
                      yaxis=dict(gridcolor=_GRID),
                      height=max(250, len(df)*35+80))
    return fig

def chart_complexity_scatter(result: PipelineResult) -> go.Figure:
    rows = []
    for pf in result.analysis_report.parsed_files:
        for fn in pf.functions:
            rows.append({"Function": fn.name, "File": pf.file_path.split("/")[-1],
                         "Complexity": fn.complexity, "Lines": fn.line_count,
                         "HasDoc": "Yes" if fn.has_docstring else "No"})
    if not rows:
        return _empty("Complexity vs Function Length")
    df = pd.DataFrame(rows)
    fig = px.scatter(df, x="Lines", y="Complexity", color="HasDoc",
                     hover_data=["Function","File"],
                     color_discrete_map={"Yes":"#00CC88","No":"#FF4B4B"},
                     title="Complexity vs Function Length",
                     labels={"Lines":"Function Length (lines)",
                             "Complexity":"Cyclomatic Complexity",
                             "HasDoc":"Has Docstring"})
    fig.add_hline(y=10, line_dash="dash", line_color="#FF8C00",
                  annotation_text="Complexity threshold (10)",
                  annotation_font_color="#FF8C00")
    fig.add_vline(x=50, line_dash="dash", line_color="#FFD700",
                  annotation_text="Length threshold (50)",
                  annotation_font_color="#FFD700")
    fig.update_layout(**_layout(), xaxis=dict(gridcolor=_GRID),
                      yaxis=dict(gridcolor=_GRID))
    return fig

def chart_smell_breakdown(result: PipelineResult) -> go.Figure:
    bd = result.analysis_report.aggregate.smell_breakdown
    if not bd:
        return _empty("Code Smell Breakdown")
    df = pd.DataFrame(sorted(bd.items(), key=lambda x: x[1]),
                      columns=["Smell","Count"]).tail(12)
    fig = go.Figure(go.Bar(
        x=df["Count"], y=df["Smell"], orientation="h",
        marker_color="#7C83FD", text=df["Count"], textposition="outside"))
    fig.update_layout(**_layout(title="Code Smell Breakdown"),
                      xaxis=dict(gridcolor=_GRID, title="Occurrences"),
                      yaxis=dict(gridcolor=_GRID),
                      height=max(250, len(df)*32+80))
    return fig

def chart_language_pie(result: PipelineResult) -> go.Figure:
    ld = result.analysis_report.aggregate.language_breakdown
    if not ld:
        return _empty("Language Distribution")
    df = pd.DataFrame(ld.items(), columns=["Language","Files"])
    fig = px.pie(df, names="Language", values="Files",
                 title="Language Distribution",
                 color_discrete_sequence=_CAT_COLOURS)
    fig.update_traces(textposition="inside", textinfo="label+percent")
    fig.update_layout(**_layout())
    return fig

def chart_confidence_histogram(result: PipelineResult) -> go.Figure:
    comments = result.review_report.all_comments()
    if not comments:
        return _empty("Confidence Score Distribution")
    scores = [c.confidence_score for c in comments]
    df = pd.DataFrame({"Confidence": scores})
    fig = px.histogram(df, x="Confidence", nbins=20,
                       title="Confidence Score Distribution",
                       color_discrete_sequence=["#7C83FD"],
                       labels={"Confidence":"Confidence Score (%)"})
    mean = sum(scores)/len(scores)
    fig.add_vline(x=mean, line_dash="dash", line_color="#FFD700",
                  annotation_text=f"Mean: {mean:.0f}%",
                  annotation_font_color="#FFD700")
    fig.update_layout(**_layout(),
                      xaxis=dict(gridcolor=_GRID, range=[0,100]),
                      yaxis=dict(gridcolor=_GRID, title="Comments"))
    return fig

def chart_category_radar(result: PipelineResult) -> go.Figure:
    cats = result.review_report.comments_by_category()
    if not cats:
        return _empty("Issues by Category")
    categories = list(cats.keys())
    counts = [len(v) for v in cats.values()]
    categories += [categories[0]]
    counts += [counts[0]]
    fig = go.Figure(go.Scatterpolar(
        r=counts, theta=categories, fill="toself",
        fillcolor="rgba(124,131,253,0.25)", line_color="#7C83FD", name="Issues"))
    fig.update_layout(**_layout(title="Issues by Category"),
                      polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(visible=True, gridcolor=_GRID, color="#666"),
                                 angularaxis=dict(gridcolor=_GRID, color="#888")))
    return fig

def chart_doc_coverage_gauge(result: PipelineResult) -> go.Figure:
    value = result.analysis_report.aggregate.doc_coverage_pct
    colour = ("#00CC88" if value >= 80 else "#FFD700" if value >= 50
              else "#FF8C00" if value >= 30 else "#FF4B4B")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=value,
        title=dict(text="Documentation Coverage (%)", font=dict(color="#CCC")),
        number=dict(suffix="%", font=dict(color=colour, size=36)),
        delta=dict(reference=80, valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor="#666", tickfont=dict(color="#666")),
            bar=dict(color=colour), bgcolor="rgba(0,0,0,0)", bordercolor="#333",
            steps=[
                dict(range=[0, 30],  color="rgba(255,75,75,0.15)"),
                dict(range=[30,60],  color="rgba(255,140,0,0.15)"),
                dict(range=[60,80],  color="rgba(255,215,0,0.15)"),
                dict(range=[80,100], color="rgba(0,204,136,0.15)"),
            ],
            threshold=dict(line=dict(color="#FFF",width=2), thickness=0.75, value=80),
        ),
    ))
    fig.update_layout(**_layout())
    return fig

def render_all_charts(result: PipelineResult) -> None:
    st.markdown("### 📈 Analytics")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Quality & Complexity", "Issues", "Coverage"]
    )
    with tab1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(chart_severity_donut(result), use_container_width=True)
        c2.plotly_chart(chart_language_pie(result),   use_container_width=True)
    with tab2:
        st.plotly_chart(chart_quality_scores(result),    use_container_width=True)
        st.plotly_chart(chart_complexity_scatter(result), use_container_width=True)
    with tab3:
        c1, c2 = st.columns(2)
        c1.plotly_chart(chart_smell_breakdown(result),  use_container_width=True)
        c2.plotly_chart(chart_category_radar(result),   use_container_width=True)
        st.plotly_chart(chart_confidence_histogram(result), use_container_width=True)
    with tab4:
        c1, c2 = st.columns([1,1])
        c1.plotly_chart(chart_doc_coverage_gauge(result), use_container_width=True)
        with c2:
            agg = result.analysis_report.aggregate
            st.markdown("#### 📊 Key Metrics")
            for label, val in [
                ("Avg Complexity",     f"{agg.avg_complexity:.2f}"),
                ("Max Complexity",     f"{agg.max_complexity}"),
                ("Avg Fn Length",      f"{agg.avg_function_length:.1f} lines"),
                ("Type Hint Coverage", f"{agg.type_hint_pct:.1f}%"),
                ("Files with Errors",  str(agg.files_with_errors)),
                ("Total Imports",      str(agg.total_imports)),
            ]:
                a, b = st.columns(2)
                a.markdown(f"**{label}**")
                b.markdown(val)

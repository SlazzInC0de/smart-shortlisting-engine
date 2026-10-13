"""Nexora - Smart Shortlisting & Resume Intelligence Engine.

Presentation layer: executive analytics, multi-metric candidate comparison,
skills gap matrix, and detailed candidate explanations.
Runs 100% offline with local embeddings and keyword extraction.
"""

import os
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import importlib
import config
import parsing
import keyword_match
import semantic_match
import scoring
import explain
import bias_check

# Guard against stale bytecode/modules if Streamlit was already running during edits
for _m in [config, parsing, keyword_match, semantic_match, scoring, explain, bias_check]:
    try:
        importlib.reload(_m)
    except Exception:
        pass

from scoring import score_candidates
from explain import explain_candidate_details, compare_candidates_structured
from parsing import parse_jd, SUPPORTED_EXTENSIONS
from bias_check import analyze_jd

# Maximum number of candidates rendered in the two comparison charts. The
# full ranking always lives in the table below, so the charts stay readable
# instead of squeezing 100+ bars into a half-width column.
CHART_DISPLAY_LIMIT = 20

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nexora | Smart Shortlisting Engine",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom enterprise CSS. The app is pinned to a light theme via
# .streamlit/config.toml so these colors always render against a light
# background -- no more dark-mode contrast issues.
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
    }

    /* Global clean typography & margins */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 96%;
    }

    /* Header -- plain title treatment, no card/box */
    .nexora-title-wrap {
        padding: 0.2rem 0 1.1rem 0;
        margin-bottom: 1.4rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .nexora-title {
        font-family: Georgia, 'Iowan Old Style', 'Palatino Linotype', Palatino, serif;
        font-size: 2.15rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #0f172a;
        margin: 0;
        line-height: 1.2;
    }
    .nexora-title .accent {
        color: #2563eb;
    }
    .nexora-tagline {
        font-size: 0.92rem;
        color: #64748b;
        margin: 0.5rem 0 0 0;
    }
    .nexora-meta {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 0.6rem 0 0 0;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.08);
    }
    .kpi-title {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1.25;
        overflow-wrap: break-word;
    }
    .kpi-sub {
        color: #94a3b8;
        font-size: 0.72rem;
        margin-top: 0.3rem;
    }

    /* Section headings inside tabs */
    .section-heading {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.2rem 0 0.6rem 0;
    }
    .section-caption {
        font-size: 0.82rem;
        color: #64748b;
        margin-bottom: 0.75rem;
    }

    /* Fit tier badges */
    .fit-badge-strong {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .fit-badge-moderate {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .fit-badge-partial {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .fit-badge-low {
        background-color: #f8fafc;
        color: #475569;
        border: 1px solid #cbd5e1;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Skill tags */
    .skill-pill-good {
        display: inline-block;
        margin: 3px 5px 3px 0;
        padding: 3px 10px;
        border-radius: 6px;
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .skill-pill-critical {
        display: inline-block;
        margin: 3px 5px 3px 0;
        padding: 3px 10px;
        border-radius: 6px;
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .skill-pill-neutral {
        display: inline-block;
        margin: 3px 5px 3px 0;
        padding: 3px 10px;
        border-radius: 6px;
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .rec-box {
        background: #f8fafc;
        border-left: 3px solid #2563eb;
        padding: 0.6rem 0.9rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #1e293b;
        margin-top: 0.5rem;
    }
    .chart-note {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: -0.3rem;
        margin-bottom: 0.6rem;
    }
    .bias-note {
        background: #fffbeb;
        border-left: 3px solid #d97706;
        padding: 0.55rem 0.85rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #78350f;
        margin-bottom: 0.5rem;
    }
    .bias-clean {
        background: #ecfdf5;
        border-left: 3px solid #059669;
        padding: 0.55rem 0.85rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #065f46;
    }
    .compare-result {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.1rem;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #1e293b;
        margin-top: 0.75rem;
    }
    .compare-verdict {
        background: #eff6ff;
        border-left: 3px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        font-size: 0.88rem;
        line-height: 1.55;
        color: #1e293b;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    table.compare-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1rem;
        font-size: 0.86rem;
    }
    .compare-table th {
        background: #f1f5f9;
        color: #475569;
        text-align: left;
        padding: 8px 12px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 2px solid #e2e8f0;
    }
    .compare-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #e2e8f0;
        color: #1e293b;
    }
    .compare-table td.metric-col {
        color: #64748b;
        font-weight: 600;
        width: 34%;
    }
    .compare-table td.compare-winner {
        color: #065f46;
        font-weight: 700;
        background: #ecfdf5;
    }
    .compare-subheading {
        font-size: 0.78rem;
        font-weight: 700;
        color: #334155;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin: 0.7rem 0 0.4rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Caching & Pipeline Runners
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running shortlisting pipeline (parsing documents and computing hybrid scores)...")
def run_pipeline(jd_path: str, resumes_dir: str, jd_mtime: float, resumes_sig: tuple):
    return score_candidates(jd_path, resumes_dir)


def resumes_dir_signature(resumes_dir: str) -> tuple:
    if not os.path.isdir(resumes_dir):
        return ()
    return tuple(sorted(
        (f, os.path.getmtime(os.path.join(resumes_dir, f)))
        for f in os.listdir(resumes_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ))


def get_fit_badge_html(fit_level: str) -> str:
    level = fit_level.lower()
    if "strong" in level:
        return f'<span class="fit-badge-strong">{fit_level}</span>'
    elif "moderate" in level:
        return f'<span class="fit-badge-moderate">{fit_level}</span>'
    elif "partial" in level:
        return f'<span class="fit-badge-partial">{fit_level}</span>'
    return f'<span class="fit-badge-low">{fit_level}</span>'


def skill_tags_html(skills: list, kind: str, empty_text: str | None = None) -> str:
    if not skills:
        if empty_text is not None:
            msg = empty_text
        else:
            msg = "Covers every required skill" if kind == "good" else "None"
        return f"<span style='color:#94a3b8; font-style:italic; font-size:0.85rem;'>{msg}</span>"
    css_class = "skill-pill-good" if kind == "good" else "skill-pill-critical"
    return "".join(f'<span class="{css_class}">{skill}</span>' for skill in skills)


_CANDIDATE_TAG_NAME_RE = re.compile(r"\bcandidate\b", re.IGNORECASE)


def _highlight_ranking_rows(row):
    """Row-level styling for the ranking table:
    - red: placeholder/nameless entries (guessed name is literally
      "Candidate N", not a real person) -- worth a recruiter's attention.
    - orange: zero matched skills -- essentially no keyword overlap with
      the JD at all.
    Red takes priority when both are true on the same row."""
    name = str(row.get("Candidate", ""))
    matched = row.get("Matched Skills", None)
    if _CANDIDATE_TAG_NAME_RE.search(name):
        style = "background-color: #fef2f2; color: #991b1b;"
    elif matched == 0:
        style = "background-color: #fff7ed; color: #9a3412;"
    else:
        style = ""
    return [style] * len(row)


# -----------------------------------------------------------------------------
# App Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="nexora-title-wrap">
        <h1 class="nexora-title">Nexora <span class="accent">Smart Shortlisting Engine</span></h1>
        <p class="nexora-tagline">AI candidate ranking combining exact skill parsing and semantic context matching. Fully local and privacy-first.</p>
        <p class="nexora-meta">Hybrid NLP Engine &nbsp;&middot;&nbsp; Local Embeddings (all-MiniLM-L6-v2) &nbsp;&middot;&nbsp; Zero External API Calls</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Engine Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Scoring Model Tuning")
    st.markdown(
        "<div style='font-size:0.85rem; color:#64748b; margin-bottom:1rem;'>"
        "Adjust how much weight is given to <b>exact keyword matches</b> vs. <b>contextual semantic similarity</b>."
        "</div>",
        unsafe_allow_html=True,
    )

    keyword_weight = st.slider(
        "Keyword Match Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(config.KEYWORD_WEIGHT),
        step=0.05,
        help="Higher values prioritize verbatim technologies and exact tools named in the JD.",
    )
    semantic_weight = round(1.0 - keyword_weight, 2)
    st.metric(label="Semantic Match Weight (Auto-balanced)", value=f"{semantic_weight:.2f}")

    st.caption("Scoring Formula: `final_score = (w_kw × kw_score) + (w_sem × sem_score)`")
    st.divider()

    st.header("Active Data Source")
    st.write(f"**JD File:** `{Path(config.JD_PATH).name}`")
    st.write(f"**Resumes Directory:** `{Path(config.RESUMES_DIR).name}/`")

    st.divider()
    st.subheader("Shortlist Controls")
    top_n = st.slider("Top Candidates to Highlight", min_value=1, max_value=10, value=config.TOP_N_EXPLANATIONS)
    min_score_filter = st.slider("Minimum Final Score Filter", min_value=0.0, max_value=1.0, value=0.0, step=0.05)


# -----------------------------------------------------------------------------
# Pipeline Execution
# -----------------------------------------------------------------------------
if not os.path.isfile(config.JD_PATH):
    st.error(f"Job description not found at `{config.JD_PATH}`. Please place the JD file there.")
    st.stop()

if not os.path.isdir(config.RESUMES_DIR) or not resumes_dir_signature(config.RESUMES_DIR):
    st.error(f"No resumes found in `{config.RESUMES_DIR}`. Please add PDF resumes.")
    st.stop()

raw_results = run_pipeline(
    config.JD_PATH,
    config.RESUMES_DIR,
    os.path.getmtime(config.JD_PATH),
    resumes_dir_signature(config.RESUMES_DIR),
)

# Live recalculation from slider weights (instant without re-embedding)
ranked = []
for c in raw_results["ranked"]:
    c_copy = dict(c)
    recalculated = keyword_weight * c_copy["keyword_score"] + semantic_weight * c_copy["semantic_score"]
    c_copy["final_score"] = round(recalculated, 4)
    # Recalculate fit tier
    if c_copy["final_score"] >= 0.72 and c_copy.get("required_coverage", 0) >= 0.70:
        c_copy["fit_level"] = "Strong Match"
    elif c_copy["final_score"] >= 0.50 and c_copy.get("required_coverage", 0) >= 0.40:
        c_copy["fit_level"] = "Moderate Match"
    elif c_copy["final_score"] >= 0.30:
        c_copy["fit_level"] = "Partial Fit"
    else:
        c_copy["fit_level"] = "Low Fit"
    ranked.append(c_copy)

# Sort with multi-key tie breaking
ranked.sort(
    key=lambda c: (-c["final_score"], -c["keyword_score"], -c["semantic_score"], c["name"])
)

# Apply optional filter
filtered_ranked = [c for c in ranked if c["final_score"] >= min_score_filter]

# -----------------------------------------------------------------------------
# KPI Overview Metrics
# -----------------------------------------------------------------------------
total_candidates = len(ranked)
top_candidate = ranked[0] if ranked else None
avg_fit = (sum(c["final_score"] for c in ranked) / total_candidates) if total_candidates else 0
req_skills = raw_results.get("required_skills", [])
pref_skills = raw_results.get("preferred_skills", [])

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Resumes Processed</div>
            <div class="kpi-value">{total_candidates}</div>
            <div class="kpi-sub">Parsed via PDFPlumber</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi2:
    top_name = top_candidate["name"] if top_candidate else "N/A"
    top_score = f"{top_candidate['final_score'] * 100:.1f}%" if top_candidate else "N/A"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Top Candidate</div>
            <div class="kpi-value" style="font-size:1.15rem; color:#2563eb;">{top_name}</div>
            <div class="kpi-sub">Overall Match: <b>{top_score}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Fit Score</div>
            <div class="kpi-value">{avg_fit * 100:.1f}%</div>
            <div class="kpi-sub">Across all applicants</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Required Skills</div>
            <div class="kpi-value">{len(req_skills)}</div>
            <div class="kpi-sub">High-priority criteria</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Preferred Skills</div>
            <div class="kpi-value">{len(pref_skills)}</div>
            <div class="kpi-sub">Bonus / nice-to-have</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_leaderboard, tab_deepdive, tab_matrix, tab_jd, tab_compare = st.tabs([
    "Shortlist & Visual Analytics",
    "Candidate Deep Dive & Feedback",
    "Skills Gap Matrix",
    "Job Description Criteria",
    "Compare Candidates",
])

# -----------------------------------------------------------------------------
# Tab 1: Leaderboard & Visual Analytics
# -----------------------------------------------------------------------------
with tab_leaderboard:
    st.markdown('<div class="section-heading">Candidate Rankings &amp; Scoring Analysis</div>', unsafe_allow_html=True)

    # Prepare DataFrame for Altair Charts. Only the top CHART_DISPLAY_LIMIT
    # candidates are plotted -- the full list is always in the table below --
    # so the charts stay legible instead of cramming 100+ bars into view.
    chart_source = filtered_ranked[:CHART_DISPLAY_LIMIT]
    chart_data = []
    for i, c in enumerate(chart_source):
        group_label = f"Top {top_n}" if i < top_n else "Other Candidates"
        chart_data.append({
            "Candidate": c["name"],
            "Rank": i + 1,
            "Final Score": c["final_score"],
            "Keyword Score": c["keyword_score"],
            "Semantic Score": c["semantic_score"],
            "Fit Tier": c["fit_level"],
            "Group": group_label,
            "Matched Count": len(c["matched_skills"]),
            "Missing Count": len(c["missing_required_skills"]),
        })
    df_chart = pd.DataFrame(chart_data)

    if not df_chart.empty:
        chart_note = None
        if len(filtered_ranked) > CHART_DISPLAY_LIMIT:
            chart_note = (
                f"Showing top {CHART_DISPLAY_LIMIT} of {len(filtered_ranked)} ranked candidates. "
                "See the full ranking table below for everyone else."
            )

        st.markdown("**Overall Match Leaderboard**")
        if chart_note:
            st.markdown(f'<div class="chart-note">{chart_note}</div>', unsafe_allow_html=True)

        # Horizontal bar chart, full width, with proper band spacing and a
        # padded domain so score labels never clip.
        c_bar = (
            alt.Chart(df_chart)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X(
                    "Final Score:Q",
                    scale=alt.Scale(domain=[0, 1.12]),
                    title="Final Score",
                    axis=alt.Axis(format="0.0%", tickCount=5),
                ),
                y=alt.Y(
                    "Candidate:N",
                    # Explicit order: layered charts drop sort="-x" and fall back to A-Z.
                    sort=df_chart.sort_values("Final Score", ascending=False)["Candidate"].tolist(),
                    title=None,
                    axis=alt.Axis(labelLimit=220, labelFontSize=11, labelFontWeight="bold"),
                ),
                color=alt.Color(
                    "Group:N",
                    scale=alt.Scale(
                        domain=[f"Top {top_n}", "Other Candidates"],
                        range=["#2563eb", "#94a3b8"],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=[
                    "Candidate",
                    alt.Tooltip("Final Score:Q", format=".3f"),
                    alt.Tooltip("Keyword Score:Q", format=".3f"),
                    alt.Tooltip("Semantic Score:Q", format=".3f"),
                    "Fit Tier",
                ],
            )
        )
        c_text = c_bar.mark_text(
            align="left",
            baseline="middle",
            dx=6,
            fontSize=11,
            fontWeight="bold",
        ).encode(
            text=alt.Text("Final Score:Q", format=".1%"),
            color=alt.value("#1e293b"),
        )
        chart1 = (c_bar + c_text).properties(
            height=max(220, len(df_chart) * 32),
        )
        st.altair_chart(chart1, use_container_width=True)

        st.write("")
        st.markdown("**Score Breakdown: Keyword vs. Semantic**")
        if chart_note:
            st.markdown(f'<div class="chart-note">{chart_note}</div>', unsafe_allow_html=True)

        melted_df = df_chart.melt(
            id_vars=["Candidate", "Rank"],
            value_vars=["Keyword Score", "Semantic Score"],
            var_name="Signal",
            value_name="Score",
        )
        grouped_chart = (
            alt.Chart(melted_df)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                y=alt.Y(
                    "Candidate:N",
                    sort=alt.EncodingSortField(field="Rank", order="ascending"),
                    title=None,
                    axis=alt.Axis(labelLimit=220, labelFontSize=11),
                ),
                yOffset="Signal:N",
                x=alt.X(
                    "Score:Q",
                    scale=alt.Scale(domain=[0, 1.05]),
                    title="Score Component",
                    axis=alt.Axis(format="0.0%"),
                ),
                color=alt.Color(
                    "Signal:N",
                    scale=alt.Scale(
                        domain=["Keyword Score", "Semantic Score"],
                        range=["#1d4ed8", "#0d9488"],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["Candidate", "Signal", alt.Tooltip("Score:Q", format=".3f")],
            )
            .properties(height=max(220, len(df_chart) * 32))
        )
        st.altair_chart(grouped_chart, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-heading">Candidates Ranking Table</div>', unsafe_allow_html=True)

    # Clean display dataframe
    table_rows = []
    for i, c in enumerate(filtered_ranked):
        table_rows.append({
            "Rank": f"#{i + 1}",
            "Candidate": c["name"],
            "Role Applied For": c.get("role_applied_for", "Not Specified"),
            "Resume File": c.get("resume_file", ""),
            "Fit Tier": c["fit_level"],
            "Final Score": c["final_score"],
            "Keyword Score": c["keyword_score"],
            "Semantic Score": c["semantic_score"],
            "Matched Skills": len(c["matched_skills"]),
            "Missing Critical": len(c["missing_required_skills"]),
        })
    df_table = pd.DataFrame(table_rows)

    st.markdown(
        "<div class='chart-note'>"
        "<span style='background-color:#fef2f2; color:#991b1b; padding:1px 7px; border-radius:4px; font-weight:600;'>Red</span>"
        "&nbsp;= placeholder/nameless entry (\"Candidate N\") &nbsp;&middot;&nbsp;"
        "<span style='background-color:#fff7ed; color:#9a3412; padding:1px 7px; border-radius:4px; font-weight:600;'>Orange</span>"
        "&nbsp;= zero matched skills"
        "</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        df_table.style.apply(_highlight_ranking_rows, axis=1),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Final Score": st.column_config.ProgressColumn(
                "Final Score",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "Keyword Score": st.column_config.NumberColumn(
                "Keyword Match",
                format="%.3f",
            ),
            "Semantic Score": st.column_config.NumberColumn(
                "Semantic Fit",
                format="%.3f",
            ),
        },
    )

    # -------------------------------------------------------------------
    # Multiple-submission transparency: a candidate who sent in more than
    # one resume file has every submission scored, and only the
    # best-performing one represents them above -- this section shows the
    # full comparison so that auto-selection is never a black box.
    # -------------------------------------------------------------------
    multi_submission_candidates = [c for c in ranked if c.get("submission_count", 1) > 1]
    if multi_submission_candidates:
        st.markdown("---")
        st.markdown('<div class="section-heading">Candidates With Multiple Submissions</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">'
            f"{len(multi_submission_candidates)} candidate(s) submitted more than one resume file. "
            "Every submission was scored independently; the highest-scoring one was auto-selected "
            "to represent that candidate in the ranking above.</div>",
            unsafe_allow_html=True,
        )

        submission_rows = []
        for c in multi_submission_candidates:
            for j, sub in enumerate(c["all_submissions"]):
                submission_rows.append({
                    "Candidate": c["name"],
                    "Submitted File": sub["filename"],
                    "Keyword Score": sub["keyword_score"],
                    "Semantic Score": sub["semantic_score"],
                    "Final Score": sub["final_score"],
                    "Selected": "Yes -- best performing" if j == 0 else "No",
                })
        df_submissions = pd.DataFrame(submission_rows)
        st.dataframe(
            df_submissions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Final Score": st.column_config.ProgressColumn(
                    "Final Score", format="%.2f", min_value=0.0, max_value=1.0,
                ),
                "Keyword Score": st.column_config.NumberColumn("Keyword Match", format="%.3f"),
                "Semantic Score": st.column_config.NumberColumn("Semantic Fit", format="%.3f"),
            },
        )

# -----------------------------------------------------------------------------
# Tab 2: Candidate Deep Dive & Explanations
# -----------------------------------------------------------------------------
with tab_deepdive:
    st.markdown(f'<div class="section-heading">Top {top_n} Candidates &mdash; Recruiter Profiles &amp; Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Comprehensive analysis of candidate competencies, missing critical qualifications, and hiring recommendations.</div>',
        unsafe_allow_html=True,
    )

    for i, candidate in enumerate(filtered_ranked[:top_n]):
        details = explain_candidate_details(candidate, rank=i + 1)
        badge_html = get_fit_badge_html(candidate["fit_level"])

        with st.container(border=True):
            # Card Header
            h_col1, h_col2, h_col3, h_col4 = st.columns([4, 2, 2, 2])
            with h_col1:
                st.markdown(
                    f"<div style='font-size:1.2rem; font-weight:700; color:#0f172a;'>"
                    f"#{i + 1} {candidate['name']} &nbsp;{badge_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:0.85rem; color:#475569; margin-top:0.2rem;'>"
                    f"<b>Driver:</b> {details['headline']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                resume_file = candidate.get("resume_file")
                submission_count = candidate.get("submission_count", 1)
                role_applied_for = candidate.get("role_applied_for")
                if resume_file:
                    file_note = f"Role applied for: <b>{role_applied_for}</b> &nbsp;|&nbsp; Scored from: <code>{resume_file}</code>"
                    if submission_count > 1:
                        file_note += f" &nbsp;(best of {submission_count} submissions -- see Shortlist tab for the full comparison)"
                    st.markdown(
                        f"<div style='font-size:0.78rem; color:#94a3b8; margin-top:0.15rem;'>{file_note}</div>",
                        unsafe_allow_html=True,
                    )
            with h_col2:
                st.metric("Overall Match", f"{candidate['final_score'] * 100:.1f}%")
            with h_col3:
                st.metric("Keyword Match", f"{candidate['keyword_score'] * 100:.1f}%")
            with h_col4:
                st.metric("Semantic Match", f"{candidate['semantic_score'] * 100:.1f}%")

            st.markdown(
                f"""
                <div class="rec-box">
                    <b>Recommended Action:</b> {details['recommendation']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            skills_col1, skills_col2 = st.columns(2)
            with skills_col1:
                st.markdown(f"**Matched Skills ({len(candidate['matched_skills'])}):**")
                st.markdown(skill_tags_html(candidate["matched_skills"], "good"), unsafe_allow_html=True)

            with skills_col2:
                st.markdown(f"**Missing Required Skills ({len(candidate['missing_required_skills'])}):**")
                st.markdown(skill_tags_html(candidate["missing_required_skills"], "critical"), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tab 3: Skills Gap Matrix
# -----------------------------------------------------------------------------
with tab_matrix:
    st.markdown('<div class="section-heading">Skills Coverage &amp; Competency Matrix</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Verify specific skill coverage across all candidates against job requirements.</div>',
        unsafe_allow_html=True,
    )

    all_jd_skills = sorted(raw_results["jd_skill_weights"].keys())
    if all_jd_skills:
        matrix_data = []
        for c in ranked:
            row = {"Candidate": c["name"], "Rank": f"#{ranked.index(c) + 1}", "Fit": c["fit_level"]}
            matched_set = set(c["matched_skills"])
            for skill in all_jd_skills:
                is_required = skill in raw_results["required_skills"]
                if skill in matched_set:
                    status = "Yes"
                elif is_required:
                    status = "Missing (Required)"
                else:
                    status = "–"
                col_name = f"{skill} *" if is_required else skill
                row[col_name] = status
            matrix_data.append(row)

        df_matrix = pd.DataFrame(matrix_data)
        st.caption("Asterisk (*) indicates an explicitly required skill.")
        st.dataframe(df_matrix, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Tab 4: Job Description Criteria
# -----------------------------------------------------------------------------
with tab_jd:
    st.markdown('<div class="section-heading">Job Description Requirements &amp; Extracted Attributes</div>', unsafe_allow_html=True)

    jd_c1, jd_c2 = st.columns(2)
    with jd_c1:
        st.markdown("**Mandatory Requirements**")
        if req_skills:
            st.markdown(skill_tags_html(req_skills, "good"), unsafe_allow_html=True)
        else:
            st.info("No explicitly labeled required skills detected.")

    with jd_c2:
        st.markdown("**Preferred / Bonus Skills**")
        if pref_skills:
            st.markdown(
                "".join(f'<span class="skill-pill-neutral">{s}</span>' for s in pref_skills),
                unsafe_allow_html=True,
            )
        else:
            st.info("No preferred skills detected.")

    st.write("")
    with st.expander("View Parsed Job Description Text", expanded=False):
        parsed_jd_text = parse_jd(config.JD_PATH)
        st.text_area("Cleaned JD Text", parsed_jd_text, height=320, disabled=True)

    st.markdown("---")
    st.markdown('<div class="section-heading">Potential Bias &amp; Narrow-Phrasing Review</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Heuristic, rule-based scan for wording that could unintentionally narrow or '
        "skew the applicant pool. These are prompts to reread a line, not proof of bias -- use judgment.</div>",
        unsafe_allow_html=True,
    )

    jd_text_for_bias = parse_jd(config.JD_PATH)
    bias_report = analyze_jd(jd_text_for_bias, req_skills)

    if not bias_report["has_flags"]:
        st.markdown(
            '<div class="bias-clean">No obvious bias or narrow-phrasing flags detected in this JD.</div>',
            unsafe_allow_html=True,
        )
    else:
        if bias_report["gender_lean_note"]:
            st.markdown(f'<div class="bias-note">{bias_report["gender_lean_note"]}</div>', unsafe_allow_html=True)
        for note in bias_report["exclusionary_notes"]:
            st.markdown(f'<div class="bias-note">{note}</div>', unsafe_allow_html=True)
        if bias_report["high_experience_note"]:
            st.markdown(f'<div class="bias-note">{bias_report["high_experience_note"]}</div>', unsafe_allow_html=True)
        if bias_report["narrow_requirements_note"]:
            st.markdown(f'<div class="bias-note">{bias_report["narrow_requirements_note"]}</div>', unsafe_allow_html=True)

        if bias_report["masculine_coded_terms"] or bias_report["feminine_coded_terms"]:
            term_col1, term_col2 = st.columns(2)
            with term_col1:
                st.markdown("**Masculine-coded terms found:**")
                if bias_report["masculine_coded_terms"]:
                    st.markdown(skill_tags_html(bias_report["masculine_coded_terms"], "critical"), unsafe_allow_html=True)
                else:
                    st.caption("None")
            with term_col2:
                st.markdown("**Feminine-coded terms found:**")
                if bias_report["feminine_coded_terms"]:
                    st.markdown(skill_tags_html(bias_report["feminine_coded_terms"], "critical"), unsafe_allow_html=True)
                else:
                    st.caption("None")

# -----------------------------------------------------------------------------
# Tab 5: Compare Candidates (recruiter Q&A: "why is X ranked above Y?")
# -----------------------------------------------------------------------------
with tab_compare:
    st.markdown('<div class="section-heading">Why Is One Candidate Ranked Above Another?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Pick any two candidates and get a plain-language breakdown of the ranking '
        "gap -- driven entirely by the computed scores below, no external model involved.</div>",
        unsafe_allow_html=True,
    )

    candidate_names = [c["name"] for c in filtered_ranked]

    if len(candidate_names) < 2:
        st.info("Need at least two candidates in the current filtered results to compare.")
    else:
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            candidate_x = st.selectbox("Candidate X", candidate_names, index=0, key="compare_x")
        with cmp_col2:
            default_y_index = 1 if len(candidate_names) > 1 else 0
            candidate_y = st.selectbox("Candidate Y", candidate_names, index=default_y_index, key="compare_y")

        if st.button("Compare", type="primary"):
            by_name = {c["name"]: c for c in filtered_ranked}
            result = compare_candidates_structured(by_name[candidate_x], by_name[candidate_y])

            if result.get("same_candidate"):
                st.warning("That's the same candidate on both sides -- pick two different people to compare.")
            else:
                st.markdown(
                    f'<div class="compare-verdict">{result["verdict"]} {result["driver"]}</div>',
                    unsafe_allow_html=True,
                )

                rows_html = ""
                for row in result["rows"]:
                    a_val, b_val = row["a"], row["b"]
                    a_cls = "compare-winner" if a_val > b_val else ""
                    b_cls = "compare-winner" if b_val > a_val else ""
                    rows_html += (
                        f'<tr><td class="metric-col">{row["metric"]}</td>'
                        f'<td class="{a_cls}">{a_val * 100:.1f}%</td>'
                        f'<td class="{b_cls}">{b_val * 100:.1f}%</td></tr>'
                    )
                rows_html += (
                    f'<tr><td class="metric-col">Fit Level</td>'
                    f'<td>{get_fit_badge_html(result["fit_a"])}</td>'
                    f'<td>{get_fit_badge_html(result["fit_b"])}</td></tr>'
                )

                st.markdown(
                    f"""
                    <table class="compare-table">
                        <thead>
                            <tr><th>Metric</th><th>{result['higher_name']}</th><th>{result['lower_name']}</th></tr>
                        </thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )

                skill_col1, skill_col2 = st.columns(2)
                with skill_col1:
                    st.markdown(
                        f'<div class="compare-subheading">Only {result["higher_name"]} Matches</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        skill_tags_html(result["only_higher_skills"], "good", empty_text="No exclusive matches"),
                        unsafe_allow_html=True,
                    )
                with skill_col2:
                    st.markdown(
                        f'<div class="compare-subheading">Only {result["lower_name"]} Matches</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        skill_tags_html(result["only_lower_skills"], "good", empty_text="No exclusive matches"),
                        unsafe_allow_html=True,
                    )

                gap_col1, gap_col2 = st.columns(2)
                with gap_col1:
                    st.markdown(
                        f'<div class="compare-subheading">{result["higher_name"]} Still Missing (Required)</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        skill_tags_html(result["higher_missing_required"], "critical", empty_text="None -- covers all required skills"),
                        unsafe_allow_html=True,
                    )
                with gap_col2:
                    st.markdown(
                        f'<div class="compare-subheading">{result["lower_name"]} Still Missing (Required)</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        skill_tags_html(result["lower_missing_required"], "critical", empty_text="None -- covers all required skills"),
                        unsafe_allow_html=True,
                    )

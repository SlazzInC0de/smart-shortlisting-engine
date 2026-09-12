"""Template-based explanations for ranked candidates.

No external LLM involved -- builds clear, actionable reasoning
directly from candidate match scores, matched competencies, and
identified skill gaps.
"""

from config import MAX_MATCHED_SKILLS_SHOWN, MAX_MISSING_SKILLS_SHOWN, TOP_N_EXPLANATIONS


def _format_skill_list(skills: list, limit: int) -> str | None:
    if not skills:
        return None
    shown = skills[:limit]
    extra = len(skills) - len(shown)
    text = ", ".join(shown)
    if extra > 0:
        text += f", and {extra} more"
    return text


def _dominant_signal(candidate: dict) -> str:
    """Which signal drove this candidate's ranking."""
    kw, sem = candidate["keyword_score"], candidate["semantic_score"]
    if abs(kw - sem) < 0.08:
        return "a balanced balance of explicit technical skill matches and overall contextual relevance"
    if kw > sem:
        return "strong alignment with explicitly required technologies and tools"
    return "strong contextual and conceptual relevance across project and domain experience"


def generate_recruiter_recommendation(candidate: dict, rank: int) -> str:
    """Generate concise actionable advice for hiring managers/recruiters."""
    final = candidate.get("final_score", 0.0)
    missing = candidate.get("missing_required_skills", [])
    fit = candidate.get("fit_level", "")

    if fit == "Strong Match" or (final >= 0.70 and not missing):
        return "Highly Recommended: Fast-track to technical screening and interview."
    elif missing and len(missing) <= 2 and final >= 0.50:
        return f"Consider: Strong overall background; verify proficiency in {', '.join(missing)} during screening."
    elif final >= 0.40:
        return "Potential Backup: Meets baseline criteria but possesses notable skill gaps compared to top candidates."
    else:
        return "Not Recommended: Substantial misalignment with core job requirements."


def explain_candidate_details(candidate: dict, rank: int) -> dict:
    """Return structured breakdown suitable for rich UI cards."""
    matched_text = _format_skill_list(candidate["matched_skills"], MAX_MATCHED_SKILLS_SHOWN)
    missing_text = _format_skill_list(candidate["missing_required_skills"], MAX_MISSING_SKILLS_SHOWN)
    rec = generate_recruiter_recommendation(candidate, rank)
    driver = _dominant_signal(candidate)

    return {
        "rank": rank,
        "name": candidate["name"],
        "headline": f"Ranked #{rank} primarily due to {driver}.",
        "matched_summary": matched_text or "No verbatim dictionary skills detected.",
        "missing_summary": missing_text or "Covers all explicitly required skills.",
        "recommendation": rec,
        "fit_level": candidate.get("fit_level", "Evaluated"),
    }


def explain_candidate(candidate: dict, rank: int) -> str:
    """Build a short, human-readable explanation string for one candidate."""
    details = explain_candidate_details(candidate, rank)
    lines = [
        f"#{rank}: {candidate['name']}  -  Final Score: {candidate['final_score']:.2f} "
        f"(Keyword: {candidate['keyword_score']:.2f}, Semantic: {candidate['semantic_score']:.2f})",
        details["headline"],
        f"Matched skills: {details['matched_summary']}.",
        f"Missing required skills: {details['missing_summary']}.",
        f"Recommendation: {details['recommendation']}",
    ]
    return "\n".join(lines)


def explain_top_candidates(ranked: list, top_n: int = TOP_N_EXPLANATIONS) -> list:
    """Return explanation strings for the top N ranked candidates."""
    return [
        explain_candidate(candidate, rank=i + 1)
        for i, candidate in enumerate(ranked[:top_n])
    ]


# -----------------------------------------------------------------------------
# Recruiter comparison ("Why is Candidate X ranked above Candidate Y?")
# -----------------------------------------------------------------------------
# Fully template-driven from the already-computed scores/skill sets -- no LLM,
# no external call. Answers the exact recruiter question by walking through
# score gap -> which signal drove it -> skill differences -> remaining gaps.
def compare_candidates(candidate_a: dict, candidate_b: dict) -> str:
    """Build a natural-language explanation of why one candidate outranks another."""
    if candidate_a["name"] == candidate_b["name"]:
        return "That's the same candidate on both sides -- pick two different people to compare."

    higher, lower = (
        (candidate_a, candidate_b)
        if candidate_a["final_score"] >= candidate_b["final_score"]
        else (candidate_b, candidate_a)
    )
    gap = higher["final_score"] - lower["final_score"]

    lines = []

    if gap < 0.005:
        lines.append(
            f"{higher['name']} and {lower['name']} are essentially tied "
            f"({higher['final_score'] * 100:.1f}% vs {lower['final_score'] * 100:.1f}%), "
            f"with {higher['name']} ranked marginally higher on tie-break criteria."
        )
    else:
        lines.append(
            f"{higher['name']} is ranked above {lower['name']}: final score "
            f"{higher['final_score'] * 100:.1f}% vs {lower['final_score'] * 100:.1f}% "
            f"(a gap of {gap * 100:.1f} points)."
        )

    kw_gap = higher["keyword_score"] - lower["keyword_score"]
    sem_gap = higher["semantic_score"] - lower["semantic_score"]
    if abs(kw_gap) >= abs(sem_gap) and abs(kw_gap) > 0.01:
        lines.append(
            f"The gap is driven mainly by explicit skill/keyword match: "
            f"{higher['name']} scored {higher['keyword_score'] * 100:.1f}% on named "
            f"tools and technologies vs {lower['keyword_score'] * 100:.1f}% for {lower['name']}."
        )
    elif abs(sem_gap) > 0.01:
        lines.append(
            f"The gap is driven mainly by contextual/semantic relevance: "
            f"{higher['name']} scored {higher['semantic_score'] * 100:.1f}% on overall "
            f"fit with the JD's context vs {lower['semantic_score'] * 100:.1f}% for {lower['name']}."
        )
    else:
        lines.append("Both the keyword and semantic signals were nearly identical for these two.")

    only_higher = sorted(set(higher["matched_skills"]) - set(lower["matched_skills"]))
    only_lower = sorted(set(lower["matched_skills"]) - set(higher["matched_skills"]))
    if only_higher:
        lines.append(f"{higher['name']} additionally matches: {', '.join(only_higher)}.")
    if only_lower:
        lines.append(
            f"{lower['name']} matches {', '.join(only_lower)} that {higher['name']} doesn't "
            f"-- not enough on its own to close the overall gap."
        )

    if higher["missing_required_skills"]:
        lines.append(
            f"{higher['name']} is still missing required skill(s): "
            f"{', '.join(higher['missing_required_skills'])}."
        )
    if lower["missing_required_skills"]:
        lines.append(
            f"{lower['name']} is missing required skill(s): "
            f"{', '.join(lower['missing_required_skills'])}."
        )

    return " ".join(lines)


def compare_candidates_structured(candidate_a: dict, candidate_b: dict) -> dict:
    """Same underlying logic as compare_candidates(), but returns structured
    data (a metrics table + skill lists) instead of one prose paragraph, so
    the UI can render it as a table/bullet layout rather than a text blob."""
    if candidate_a["name"] == candidate_b["name"]:
        return {"same_candidate": True}

    higher, lower = (
        (candidate_a, candidate_b)
        if candidate_a["final_score"] >= candidate_b["final_score"]
        else (candidate_b, candidate_a)
    )
    gap = higher["final_score"] - lower["final_score"]

    if gap < 0.005:
        verdict = (
            f"{higher['name']} and {lower['name']} are essentially tied "
            f"({higher['final_score'] * 100:.1f}% vs {lower['final_score'] * 100:.1f}%) "
            f"-- {higher['name']} edges ahead only on tie-break criteria."
        )
    else:
        verdict = (
            f"{higher['name']} is ranked above {lower['name']} by a gap of "
            f"{gap * 100:.1f} points ({higher['final_score'] * 100:.1f}% vs "
            f"{lower['final_score'] * 100:.1f}%)."
        )

    kw_gap = higher["keyword_score"] - lower["keyword_score"]
    sem_gap = higher["semantic_score"] - lower["semantic_score"]
    if abs(kw_gap) >= abs(sem_gap) and abs(kw_gap) > 0.01:
        driver = (
            f"Driven mainly by explicit skill/keyword match "
            f"({higher['name']} +{abs(kw_gap) * 100:.1f} pts there)."
        )
    elif abs(sem_gap) > 0.01:
        driver = (
            f"Driven mainly by contextual/semantic relevance "
            f"({higher['name']} +{abs(sem_gap) * 100:.1f} pts there)."
        )
    else:
        driver = "Keyword and semantic signals were nearly identical for both."

    rows = [
        {"metric": "Final Score", "a": higher["final_score"], "b": lower["final_score"]},
        {"metric": "Keyword Score", "a": higher["keyword_score"], "b": lower["keyword_score"]},
        {"metric": "Semantic Score", "a": higher["semantic_score"], "b": lower["semantic_score"]},
        {"metric": "Required Skill Coverage", "a": higher["required_coverage"], "b": lower["required_coverage"]},
        {"metric": "Preferred Skill Coverage", "a": higher["preferred_coverage"], "b": lower["preferred_coverage"]},
    ]

    only_higher = sorted(set(higher["matched_skills"]) - set(lower["matched_skills"]))
    only_lower = sorted(set(lower["matched_skills"]) - set(higher["matched_skills"]))

    return {
        "same_candidate": False,
        "higher_name": higher["name"],
        "lower_name": lower["name"],
        "verdict": verdict,
        "driver": driver,
        "rows": rows,
        "fit_a": higher.get("fit_level", "Evaluated"),
        "fit_b": lower.get("fit_level", "Evaluated"),
        "only_higher_skills": only_higher,
        "only_lower_skills": only_lower,
        "higher_missing_required": higher["missing_required_skills"],
        "lower_missing_required": lower["missing_required_skills"],
    }

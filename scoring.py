"""Combine keyword and semantic signals into a final ranked shortlist.

final_score = KEYWORD_WEIGHT * keyword_score + SEMANTIC_WEIGHT * semantic_score

Both component scores are pre-normalized to [0, 1] (see keyword_match.py
and semantic_match.py), so the weights in config.py directly control
how much each signal contributes.

A candidate who submitted more than one resume file has EVERY submission
scored independently here; only the best-performing one represents them in
the final ranked list. The rest aren't discarded -- they're kept on
"all_submissions" so the UI can show exactly which file was auto-selected
and why, for every candidate who sent in more than one resume.
"""

from config import KEYWORD_WEIGHT, SEMANTIC_WEIGHT
from parsing import parse_jd, parse_resumes
from keyword_match import (
    extract_jd_skills, find_resume_skills, keyword_score,
    collapse_alternatives, REQUIRED_WEIGHT,
)
from semantic_match import embed_document, embed_documents_batch, semantic_score


def categorize_fit(final_score: float, req_coverage: float) -> str:
    """Categorize candidate fit into standardized tiers."""
    if final_score >= 0.72 and req_coverage >= 0.70:
        return "Strong Match"
    if final_score >= 0.50 and req_coverage >= 0.40:
        return "Moderate Match"
    if final_score >= 0.30:
        return "Partial Fit"
    return "Low Fit"


def _score_one_submission(resume_text, resume_embedding, jd_embedding, jd_skill_weights, required_skills, preferred_skills):
    """Score a single resume file against the JD. Returns a plain score dict
    (no "name" -- caller attaches that)."""
    resume_skills = find_resume_skills(resume_text)
    kw_score = keyword_score(jd_skill_weights, resume_skills)
    sem_score = semantic_score(jd_embedding, resume_embedding)
    final = KEYWORD_WEIGHT * kw_score + SEMANTIC_WEIGHT * sem_score

    matched = sorted(resume_skills & set(jd_skill_weights.keys()))

    missing_required = {s for s in required_skills if s not in resume_skills}
    missing_required = sorted(collapse_alternatives(missing_required, resume_skills))

    req_matched_count = len(set(required_skills) - set(missing_required))
    req_coverage = req_matched_count / len(required_skills) if required_skills else 1.0

    pref_matched_count = len(set(preferred_skills) & resume_skills)
    pref_coverage = pref_matched_count / len(preferred_skills) if preferred_skills else 1.0

    return {
        "keyword_score": round(kw_score, 4),
        "semantic_score": round(sem_score, 4),
        "final_score": round(final, 4),
        "matched_skills": matched,
        "missing_required_skills": missing_required,
        "required_coverage": round(req_coverage, 3),
        "preferred_coverage": round(pref_coverage, 3),
    }


def score_candidates(jd_path: str, resumes_dir: str) -> dict:
    """Run the full pipeline and return everything the UI/explainer need."""
    jd_text = parse_jd(jd_path)
    grouped_resumes = parse_resumes(resumes_dir)  # {name: [{"filename","text"}, ...]}

    jd_skill_weights = extract_jd_skills(jd_text)
    required_skills = sorted([s for s, w in jd_skill_weights.items() if w >= REQUIRED_WEIGHT])
    preferred_skills = sorted([s for s, w in jd_skill_weights.items() if w < REQUIRED_WEIGHT])

    jd_embedding = embed_document(jd_text)

    # Flatten every (candidate, submission) pair so all resume files --
    # including every submission from a candidate who sent in more than one
    # -- get embedded together in one efficient batch call.
    flat = []  # (name, filename, text, role)
    for name, submissions in grouped_resumes.items():
        for sub in submissions:
            flat.append((name, sub["filename"], sub["text"], sub.get("role", "Not Specified")))

    all_texts = [text for _, _, text, _ in flat]
    all_embeddings = embed_documents_batch(all_texts)

    variants_by_name = {}
    for (name, filename, resume_text, role), resume_embedding in zip(flat, all_embeddings):
        score = _score_one_submission(
            resume_text, resume_embedding, jd_embedding,
            jd_skill_weights, required_skills, preferred_skills,
        )
        score["filename"] = filename
        score["role"] = role
        variants_by_name.setdefault(name, []).append(score)

    ranked = []
    for name, variants in variants_by_name.items():
        # Best-performing submission for this candidate wins the slot in
        # the ranking; every submission is kept in "all_submissions" so the
        # UI can show what else this candidate sent in and how it scored.
        variants_sorted = sorted(
            variants, key=lambda v: (-v["final_score"], -v["keyword_score"], -v["semantic_score"])
        )
        best = variants_sorted[0]
        fit_level = categorize_fit(best["final_score"], best["required_coverage"])

        ranked.append({
            "name": name,
            "keyword_score": best["keyword_score"],
            "semantic_score": best["semantic_score"],
            "final_score": best["final_score"],
            "matched_skills": best["matched_skills"],
            "missing_required_skills": best["missing_required_skills"],
            "required_coverage": best["required_coverage"],
            "preferred_coverage": best["preferred_coverage"],
            "fit_level": fit_level,
            "resume_file": best["filename"],
            "role_applied_for": best["role"],
            "submission_count": len(variants_sorted),
            "all_submissions": variants_sorted,
        })

    # Deterministic multi-key sort
    ranked.sort(
        key=lambda c: (-c["final_score"], -c["keyword_score"], -c["semantic_score"], c["name"])
    )

    return {
        "jd_skill_weights": jd_skill_weights,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "ranked": ranked,
    }

from pathlib import Path

import pytest

from config import KEYWORD_WEIGHT, SEMANTIC_WEIGHT
from explain import compare_candidates, compare_candidates_structured, explain_top_candidates
from scoring import categorize_fit, score_candidates

DATA = Path(__file__).resolve().parent.parent / "data"

JD = """Backend Developer Intern

Required Skills:
- Python
- Docker
- SQL

Nice to have:
- Kubernetes
"""

STRONG = """Priya Sharma
priya@example.com
Skills: Python, Docker, SQL, Kubernetes, FastAPI
Built containerised Python services backed by PostgreSQL."""

WEAK = """Rahul Mehta
rahul@example.com
Skills: Photoshop, Illustrator
Graphic designer focused on brand identity."""

PRIYA_OLD = """Priya Sharma
priya@example.com
Skills: Python
Older version of my resume."""


@pytest.fixture
def batch(tmp_path):
    jd = tmp_path / "jd.txt"
    jd.write_text(JD)
    resumes = tmp_path / "resumes"
    resumes.mkdir()
    (resumes / "priya_sharma.txt").write_text(STRONG)
    (resumes / "priya_sharma_old.txt").write_text(PRIYA_OLD)
    (resumes / "rahul_mehta.txt").write_text(WEAK)
    return score_candidates(str(jd), str(resumes))


def test_required_and_preferred_skills_come_from_jd_sections(batch):
    required = {s.lower() for s in batch["required_skills"]}
    preferred = {s.lower() for s in batch["preferred_skills"]}
    assert {"python", "docker", "sql"} <= required
    assert "kubernetes" in preferred


def test_strong_candidate_outranks_weak_one(batch):
    names = [c["name"] for c in batch["ranked"]]
    assert names.index("Priya Sharma") < names.index("Rahul Mehta")
    weak = next(c for c in batch["ranked"] if c["name"] == "Rahul Mehta")
    assert weak["keyword_score"] == 0.0
    assert weak["required_coverage"] == 0.0


def test_final_score_is_the_documented_weighted_sum(batch):
    for c in batch["ranked"]:
        expected = KEYWORD_WEIGHT * c["keyword_score"] + SEMANTIC_WEIGHT * c["semantic_score"]
        assert c["final_score"] == pytest.approx(expected, abs=1e-3)
        assert 0.0 <= c["final_score"] <= 1.0


def test_multiple_submissions_keep_the_best_and_record_all(batch):
    priya = next(c for c in batch["ranked"] if c["name"] == "Priya Sharma")
    assert priya["submission_count"] == 2
    assert priya["resume_file"] == "priya_sharma.txt"
    scores = [s["final_score"] for s in priya["all_submissions"]]
    assert scores == sorted(scores, reverse=True)


def test_explanations_and_comparisons_name_the_right_candidates(batch):
    top, second = batch["ranked"][0], batch["ranked"][1]
    assert len(explain_top_candidates(batch["ranked"], top_n=2)) == 2
    text = compare_candidates(top, second)
    assert top["name"] in text and second["name"] in text
    assert isinstance(compare_candidates_structured(top, second), dict)


@pytest.mark.parametrize("final, coverage, tier", [
    (0.80, 0.80, "Strong Match"),
    (0.80, 0.50, "Moderate Match"),
    (0.55, 0.45, "Moderate Match"),
    (0.40, 0.90, "Partial Fit"),
    (0.10, 1.00, "Low Fit"),
])
def test_fit_tiers(final, coverage, tier):
    assert categorize_fit(final, coverage) == tier


def test_bundled_sample_batch_ranks_every_resume():
    jd = next(p for p in (DATA / "jd").iterdir() if p.suffix.lower() in {".pdf", ".txt"})
    result = score_candidates(str(jd), str(DATA / "resumes"))
    ranked = result["ranked"]
    assert len(ranked) == 18
    finals = [c["final_score"] for c in ranked]
    assert finals == sorted(finals, reverse=True)
    assert len({c["name"] for c in ranked}) == 18

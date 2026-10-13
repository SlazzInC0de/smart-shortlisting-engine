from bias_check import analyze_jd
from keyword_match import extract_jd_skills, find_resume_skills, keyword_score


def test_finds_skills_and_tolerates_typos():
    skills = find_resume_skills("Built APIs in Pyhton with Dcoker and PostgreSQL.")
    assert {"python", "docker"} <= {s.lower() for s in skills}


def test_keyword_score_is_weighted_fraction():
    weights = {"python": 1.0, "docker": 0.5}
    assert keyword_score(weights, {"python"}) == 1.0 / 1.5
    assert keyword_score(weights, set()) == 0.0
    assert keyword_score({}, {"python"}) == 0.0


def test_required_section_outweighs_preferred():
    jd = "Required Skills:\n- Python\n\nNice to have:\n- Docker\n"
    weights = {k.lower(): v for k, v in extract_jd_skills(jd).items()}
    assert weights["python"] > weights["docker"]


def test_bias_check_flags_high_experience_and_long_required_list():
    report = analyze_jd("We need a rockstar ninja with 10+ years of experience.", ["s"] * 15)
    assert report["has_flags"]
    assert report["high_experience_note"]
    assert report["narrow_requirements_note"]


def test_neutral_jd_has_no_experience_flag():
    report = analyze_jd("Collaborate with a supportive team building data tools.", ["python"])
    assert report["high_experience_note"] is None
    assert report["narrow_requirements_note"] is None

"""Central configuration for the shortlisting engine.

Judges asked for config-adjustable weights and a clearly documented
scoring formula -- both live here so the whole pipeline reads from
one place.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# final_score = KEYWORD_WEIGHT * keyword_score + SEMANTIC_WEIGHT * semantic_score
# Both component scores are normalized to [0, 1] before combining.
KEYWORD_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5

# Local, offline embedding model (sentence-transformers). No external API calls.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def _find_jd_path() -> str:
    """Pick whichever JD file is sitting in data/jd/, regardless of its name.

    The hackathon doesn't guarantee a fixed JD filename (today it's
    Sample_JD.pdf, the real test JD could be named anything, e.g.
    job_description.pdf) -- so this scans the folder for any supported
    document instead of hardcoding a name. PDF/DOCX are preferred over
    plain text/other formats when more than one file is present; ties
    break alphabetically so the pick stays deterministic.
    """
    from parsing import SUPPORTED_EXTENSIONS, EXTENSION_PRIORITY

    jd_dir = BASE_DIR / "data" / "jd"
    fallback = str(jd_dir / "Sample_JD.pdf")  # surfaces app.py's "not found" error if truly empty
    if not jd_dir.is_dir():
        return fallback

    found = [
        f for f in sorted(jd_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not found:
        return fallback

    found.sort(key=lambda f: (EXTENSION_PRIORITY.get(f.suffix.lower(), 99), f.name))
    return str(found[0])


JD_PATH = _find_jd_path()
RESUMES_DIR = str(BASE_DIR / "data" / "resumes")

# How many required-skill misses are shown per candidate in explanations.
MAX_MISSING_SKILLS_SHOWN = 5
MAX_MATCHED_SKILLS_SHOWN = 8
TOP_N_EXPLANATIONS = 3

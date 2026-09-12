"""Keyword-based matching: curated skill dictionary + regex extraction.

This captures the "explicit tools/technologies named in the JD still
matter" half of the brief. It does NOT try to understand meaning --
that's semantic_match.py's job. This module answers: which named skills
does the JD ask for, and which of those appear in a resume (literally or
via recognized synonyms and variants).
"""

import difflib
import re

# Canonical skill -> list of surface forms that count as a match for that skill (case-insensitive).
SKILL_DICTIONARY = {
    # Core Languages
    "JavaScript": ["javascript", "js", "es6", "es2020", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "Python": ["python", "python3", "py"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp", "c-sharp"],
    "Go": ["golang", "go language"],
    "Rust": ["rust"],
    "PHP": ["php"],
    "Ruby": ["ruby", "ruby on rails", "rails"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],

    # Frontend Frameworks & Web
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "React": ["react", "react.js", "reactjs"],
    "Redux": ["redux", "redux toolkit", "rtk"],
    "Next.js": ["next.js", "nextjs", "next js"],
    "Vue": ["vue", "vue.js", "vuejs"],
    "Angular": ["angular", "angular.js", "angularjs"],
    "Svelte": ["svelte", "sveltekit"],
    "Tailwind": ["tailwind", "tailwindcss", "tailwind css"],
    "Bootstrap": ["bootstrap"],

    # Backend Frameworks & Runtimes
    "Node.js": ["node.js", "nodejs", "node js", "node"],
    "Express": ["express", "express.js", "expressjs"],
    "FastAPI": ["fastapi", "fast-api"],
    "Django": ["django"],
    "Flask": ["flask"],
    "Spring Boot": ["spring boot", "spring framework", "spring"],
    ".NET": [".net", "asp.net", "dotnet"],

    # APIs & Communication
    "REST API": ["rest api", "restful", "rest apis", "restful api", "restful apis", "rest"],
    "GraphQL": ["graphql"],
    "gRPC": ["grpc"],
    "WebSockets": ["websocket", "websockets"],
    "API Development": ["api development", "api integration", "apis"],

    # Databases & Storage
    "MongoDB": ["mongodb", "mongo db", "mongo"],
    "SQL": ["sql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "SQLite": ["sqlite"],
    "Redis": ["redis"],
    "DynamoDB": ["dynamodb", "dynamo db"],
    "Firebase": ["firebase", "firestore"],

    # Cloud & DevOps
    "Git": ["git", "github", "gitlab", "version control"],
    "Docker": ["docker", "containerization", "containers"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Azure": ["azure", "microsoft azure"],
    "CI/CD": ["ci/cd", "ci cd", "continuous integration", "continuous deployment", "github actions"],
    "Linux": ["linux", "unix"],

    # Testing & Practices
    "Jest": ["jest", "unit testing", "unit tests"],
    "PyTest": ["pytest"],
    "Cypress": ["cypress"],
    "Playwright": ["playwright"],
    "Agile": ["agile", "scrum", "kanban", "sprints"],

    # Computer Science & Data
    "Data Structures": ["data structures", "dsa"],
    "Algorithms": ["algorithms"],
    "Full Stack": ["full stack", "fullstack", "full-stack"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-Learn": ["scikit-learn", "sklearn"],
    "Machine Learning": ["machine learning", "ml"],
}

# Section headers that signal requirements vs bonuses
REQUIRED_HEADERS = [
    "requirements", "required skills", "must have", "must-have",
    "qualifications", "what you'll need", "what you will need",
    "you should have", "minimum qualifications", "skills required",
    "who you are", "basic qualifications",
]
PREFERRED_HEADERS = [
    "nice to have", "nice-to-have", "preferred", "preferred qualifications",
    "bonus", "good to have", "is a plus", "a plus", "desired skills",
]

REQUIRED_WEIGHT = 2.0
PREFERRED_WEIGHT = 1.0

# Skills that a JD typically lists as interchangeable alternatives
SKILL_GROUPS = [
    {"MongoDB", "SQL", "PostgreSQL", "MySQL", "SQLite", "Firebase", "DynamoDB"},
    {"AWS", "GCP", "Azure"},
    {"React", "Vue", "Angular", "Svelte"},
    {"Jest", "PyTest", "Cypress", "Playwright"},
]


def collapse_alternatives(missing_skills: set, resume_skills: set) -> set:
    """Drop skills from `missing_skills` whose group is already
    satisfied by something in `resume_skills`."""
    result = set(missing_skills)
    for group in SKILL_GROUPS:
        if resume_skills & group:
            result -= group
    return result


def _build_pattern(surface_forms):
    """Robust regex matching surface forms with fixed-width boundary assertions."""
    escaped_patterns = []
    for form in sorted(surface_forms, key=len, reverse=True):
        pattern = re.escape(form)
        prefix = r"(?<!\w)" if form[0].isalnum() else r"(?<!\S)"
        suffix = r"(?!\w)" if form[-1].isalnum() else r"(?!\S)"
        escaped_patterns.append(f"{prefix}{pattern}{suffix}")
    return re.compile("|".join(escaped_patterns), re.IGNORECASE)


SKILL_PATTERNS = {skill: _build_pattern(forms) for skill, forms in SKILL_DICTIONARY.items()}

# ---------------------------------------------------------------------------
# Typo tolerance
# ---------------------------------------------------------------------------
# Resume text often has small typos ("Pyhton", "Javascrpit", "Dcoker"). Exact
# regex matching above won't catch those, so we add a conservative fuzzy
# fallback: single-word, purely-alphabetic surface forms of at least
# FUZZY_MIN_LEN characters are also checked against resume tokens with a high
# similarity threshold. Multi-word phrases and short/symbol-heavy forms
# (e.g. "c++", "c#", ".net", "go") are excluded -- edit-distance tricks on
# those are unreliable and prone to false positives.
FUZZY_MIN_LEN = 4
FUZZY_MAX_LEN_DIFF = 2
FUZZY_RATIO_THRESHOLD = 0.84

_ALPHA_WORD_RE = re.compile(r"[a-zA-Z]+")


def _fuzzy_eligible_forms(surface_forms):
    return [f for f in surface_forms if re.fullmatch(r"[a-z]{%d,}" % FUZZY_MIN_LEN, f)]


SKILL_FUZZY_FORMS = {skill: _fuzzy_eligible_forms(forms) for skill, forms in SKILL_DICTIONARY.items()}


def _tokenize(text: str) -> set:
    return {t.lower() for t in _ALPHA_WORD_RE.findall(text) if len(t) >= FUZZY_MIN_LEN}


def _is_adjacent_transposition(a: str, b: str) -> bool:
    """True if `a` becomes `b` by swapping exactly one pair of adjacent
    characters -- the single most common real-world typo (e.g. "pyhton" ->
    "python", "dcoker" -> "docker"). Deliberately narrow (same length,
    exactly two swapped positions) so it stays a precise signal rather than
    a fuzzy guess."""
    if len(a) != len(b) or a == b:
        return False
    diffs = [i for i in range(len(a)) if a[i] != b[i]]
    if len(diffs) != 2:
        return False
    i, j = diffs
    return j == i + 1 and a[i] == b[j] and a[j] == b[i]


def _fuzzy_contains(tokens: set, target: str) -> bool:
    for token in tokens:
        if len(token) == len(target) and _is_adjacent_transposition(token, target):
            return True
        if abs(len(token) - len(target)) > FUZZY_MAX_LEN_DIFF:
            continue
        if difflib.SequenceMatcher(None, token, target).ratio() >= FUZZY_RATIO_THRESHOLD:
            return True
    return False


def _skill_present(text: str, tokens: set, skill: str) -> bool:
    """Exact/synonym match first; fuzzy typo-tolerant fallback second."""
    if SKILL_PATTERNS[skill].search(text):
        return True
    for form in SKILL_FUZZY_FORMS.get(skill, ()):
        if _fuzzy_contains(tokens, form):
            return True
    return False


def _split_into_sections(text: str):
    """Split JD text into (header_type, section_text) chunks."""
    lines = text.splitlines()
    sections = []
    current_type = "unlabeled"
    current_lines = []

    def header_type_for(line: str):
        cleaned = line.strip().lower().strip(":#*- ")
        if not cleaned or len(cleaned) > 55 or line.strip().endswith((".", ",")):
            return None
        if any(h == cleaned or cleaned.startswith(f"{h} ") or cleaned.endswith(f" {h}") for h in REQUIRED_HEADERS):
            return "required"
        if any(h == cleaned or cleaned.startswith(f"{h} ") or cleaned.endswith(f" {h}") for h in PREFERRED_HEADERS):
            return "preferred"
        return None

    for line in lines:
        htype = header_type_for(line)
        if htype:
            if current_lines:
                sections.append((current_type, "\n".join(current_lines)))
            current_type = htype
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_type, "\n".join(current_lines)))
    return sections


def extract_jd_skills(jd_text: str) -> dict:
    """Find which dictionary skills the JD mentions, weighted by
    whether they appear under a required-sounding section.

    Returns: {skill_name: weight}
    """
    sections = _split_into_sections(jd_text)
    skill_weights = {}

    for section_type, section_text in sections:
        if section_type == "required":
            weight = REQUIRED_WEIGHT
        elif section_type == "preferred":
            weight = PREFERRED_WEIGHT
        else:
            weight = PREFERRED_WEIGHT

        tokens = _tokenize(section_text)
        for skill in SKILL_DICTIONARY:
            if _skill_present(section_text, tokens, skill):
                skill_weights[skill] = max(skill_weights.get(skill, 0.0), weight)

    return skill_weights


def find_resume_skills(resume_text: str) -> set:
    """Return the set of dictionary skills present anywhere in a resume.

    Combines exact/synonym regex matching with a typo-tolerant fuzzy
    fallback (see FUZZY_* constants above) so a misspelled skill like
    "Pyhton" or "Dcoker" still counts as a match.
    """
    tokens = _tokenize(resume_text)
    return {skill for skill in SKILL_DICTIONARY if _skill_present(resume_text, tokens, skill)}


def keyword_score(jd_skill_weights: dict, resume_skills: set) -> float:
    """Weighted fraction of JD-required skill weight found in the resume.

    Normalized to [0, 1]. If the JD names no dictionary skills at all,
    returns 0.0.
    """
    total_weight = sum(jd_skill_weights.values())
    if total_weight == 0:
        return 0.0
    matched_weight = sum(
        weight for skill, weight in jd_skill_weights.items() if skill in resume_skills
    )
    return matched_weight / total_weight

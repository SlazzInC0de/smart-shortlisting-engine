"""Heuristic scanner for potentially biased or overly narrow language in a
job description.

100% offline and rule-based (curated word lists + regex patterns) -- no
external API, no ML model, no network call of any kind. This is meant as a
*heuristic aid* for a human reviewer, not a verdict: a flagged word or phrase
is a prompt to reread that line, not proof the JD is discriminatory. Always
present results with that framing.
"""

import re

# Masculine/feminine-coded word stems, adapted from the word list published in
# Gaucher, Friesen & Kay (2011), "Evidence That Gendered Wording in Job
# Advertisements Exists and Sustains Gender Inequality", Journal of
# Personality and Social Psychology. Presence of either list on its own does
# not mean a JD is biased -- overall tone and balance matter more than any
# single word.
MASCULINE_CODED_WORDS = [
    "active", "adventurous", "aggressive", "ambitious", "assertive",
    "autonomous", "challenging", "competitive", "confident", "decisive",
    "determined", "dominant", "driven", "fearless", "independent",
    "individualistic", "leader", "lead", "outspoken", "self-reliant",
    "self-sufficient", "superior", "relentless", "hustle",
]
FEMININE_CODED_WORDS = [
    "affectionate", "collaborative", "committed", "communal", "compassionate",
    "considerate", "cooperative", "dependable", "empathetic", "interpersonal",
    "kind", "loyal", "nurturing", "pleasant", "polite", "responsible",
    "sensitive", "supportive", "trustworthy", "understanding", "warm",
]

# (pattern, human-readable note). Each note explains *why* it's flagged and
# suggests a fix, so the report is actionable rather than just a red mark.
EXCLUSIONARY_PATTERNS = [
    (
        r"\bnative english speaker\b",
        "‘Native English speaker’ can exclude fluent non-native speakers and "
        "may read as a national-origin proxy. Consider ‘fluent in English’ instead.",
    ),
    (
        r"\brecent (college |university )?graduate\b",
        "‘Recent graduate’ can act as an age proxy that discourages older "
        "career-changers. Consider stating the actual experience level needed.",
    ),
    (
        r"\byoung(,)? (and )?energetic\b",
        "‘Young and energetic’ is a classic age-discrimination red flag. "
        "Describe the actual pace/energy of the role instead of an age proxy.",
    ),
    (
        r"\bdigital native\b",
        "‘Digital native’ is often used as an unintentional age proxy. "
        "Consider naming the actual tech comfort level required.",
    ),
    (
        r"\bculture fit\b",
        "‘Culture fit’ is vague and prone to unconscious bias (people tend to "
        "pick people like themselves). Consider ‘culture add’ or naming specific "
        "collaboration values instead.",
    ),
    (
        r"\brock ?star\b|\bninja\b|\bguru\b|\bwizard\b",
        "Informal superlatives like this can read as bravado-coded and may "
        "discourage equally qualified candidates who don't self-select into "
        "that language.",
    ),
    (
        r"\b(?:he|him|his)\b(?!\s*/\s*(?:she|her))",
        "Gender-specific pronoun used without a neutral alternative. Consider "
        "‘they’ or rephrasing to avoid pronouns in the JD.",
    ),
]

# Flat years-of-experience requirements above this are worth a second look --
# a high bar screens out strong candidates who gained equivalent skill
# faster or through a non-traditional path.
HIGH_EXPERIENCE_YEARS_THRESHOLD = 8
_EXPERIENCE_YEARS_RE = re.compile(r"\b(\d{1,2})\+?\s*years?\b", re.IGNORECASE)

# A long mandatory-skill list narrows the applicant funnel fast.
NARROW_REQUIRED_SKILLS_THRESHOLD = 10


def _find_coded_terms(jd_text_lower: str, word_list: list) -> list:
    found = []
    for word in word_list:
        pattern = r"(?<!\w)" + re.escape(word) + r"(?!\w)"
        if re.search(pattern, jd_text_lower):
            found.append(word)
    return found


def analyze_jd(jd_text: str, required_skills: list) -> dict:
    """Return a heuristic bias/narrowness report for a job description.

    Returns a dict with:
        has_flags               -- bool, whether anything was flagged at all
        masculine_coded_terms   -- list[str]
        feminine_coded_terms    -- list[str]
        gender_lean_note        -- str | None
        exclusionary_notes      -- list[str]
        high_experience_note    -- str | None
        narrow_requirements_note-- str | None
    """
    text = jd_text or ""
    # Collapse all whitespace (including line wraps from PDF extraction) so
    # multi-word patterns like "recent college graduate" still match even
    # when a real JD happens to wrap mid-phrase across a line break.
    lower = re.sub(r"\s+", " ", text.lower())

    masculine_hits = _find_coded_terms(lower, MASCULINE_CODED_WORDS)
    feminine_hits = _find_coded_terms(lower, FEMININE_CODED_WORDS)

    exclusionary_notes = []
    for pattern, note in EXCLUSIONARY_PATTERNS:
        if re.search(pattern, lower):
            exclusionary_notes.append(note)

    high_experience_note = None
    year_hits = [int(y) for y in _EXPERIENCE_YEARS_RE.findall(lower)]
    max_years = max(year_hits) if year_hits else 0
    if max_years >= HIGH_EXPERIENCE_YEARS_THRESHOLD:
        high_experience_note = (
            f"JD asks for {max_years}+ years of experience. A high flat "
            "years-of-experience bar can screen out strong candidates who "
            "gained equivalent skill through a different path -- confirm "
            "the number is genuinely necessary for the role."
        )

    narrow_requirements_note = None
    if len(required_skills) >= NARROW_REQUIRED_SKILLS_THRESHOLD:
        narrow_requirements_note = (
            f"{len(required_skills)} skills are marked strictly required. "
            "A long mandatory list narrows the funnel fast -- consider "
            "moving the less critical ones to 'preferred' so strong "
            "candidates missing one or two nice-to-haves aren't auto-excluded."
        )

    gender_lean_note = None
    if masculine_hits and not feminine_hits:
        gender_lean_note = (
            "Wording leans masculine-coded. Research links this phrasing to "
            "fewer women applying -- consider balancing with some "
            "collaborative/supportive language."
        )
    elif feminine_hits and not masculine_hits:
        gender_lean_note = (
            "Wording leans feminine-coded. Not necessarily a problem, but "
            "worth a glance if you want maximally broad appeal."
        )

    has_flags = bool(
        masculine_hits or feminine_hits or exclusionary_notes
        or high_experience_note or narrow_requirements_note
    )

    return {
        "has_flags": has_flags,
        "masculine_coded_terms": masculine_hits,
        "feminine_coded_terms": feminine_hits,
        "gender_lean_note": gender_lean_note,
        "exclusionary_notes": exclusionary_notes,
        "high_experience_note": high_experience_note,
        "narrow_requirements_note": narrow_requirements_note,
    }

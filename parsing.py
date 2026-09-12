"""Universal document parsing and text extraction for JDs and resumes.

Supports all major document formats natively without manual conversion:
- PDF (.pdf)
- Microsoft Word (.docx, .docm, .dotx)
- XML (.xml, structured resume exports)
- Plain text (.txt, .md, .markdown)
- Web & markup (.html, .htm)
- Data formats (.json, .csv, .tsv)
- Rich text (.rtf)

Normalizes formatting, whitespace, line breaks, and extracts candidate names.
"""

import os
import re
import json
import glob
import logging
import zipfile
import xml.etree.ElementTree as ET

import pdfplumber

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".docm", ".dotx", ".txt", ".xml",
    ".json", ".html", ".htm", ".rtf", ".md", ".markdown",
    ".csv", ".tsv",
}

EXTENSION_PRIORITY = {
    ".pdf": 1,
    ".docx": 2,
    ".docm": 3,
    ".dotx": 4,
    ".txt": 5,
    ".xml": 6,
    ".json": 7,
    ".html": 8,
    ".htm": 9,
    ".rtf": 10,
    ".md": 11,
    ".markdown": 12,
    ".csv": 13,
    ".tsv": 14,
}


def extract_text_from_pdf(path: str) -> str:
    """Extract raw text from PDF pages."""
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    pages.append(text)
    except Exception as exc:
        logger.warning("Error reading PDF %s: %s", path, exc)
    return "\n".join(pages)


def extract_text_from_docx(path: str) -> str:
    """Extract text from Word .docx file using standard library zipfile and XML parser."""
    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read("word/document.xml")
        root = ET.fromstring(xml_content)
        paragraphs = []
        for p in root.iter():
            if p.tag.endswith("}p"):
                texts = [node.text for node in p.iter() if node.text and node.tag.endswith("}t")]
                p_text = "".join(texts).strip()
                if p_text:
                    paragraphs.append(p_text)
        return "\n".join(paragraphs)
    except Exception as exc:
        logger.warning("Error reading DOCX %s: %s", path, exc)
        return ""


def extract_text_from_xml(path: str) -> str:
    """Extract structured and freeform text from XML documents."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        try:
            root = ET.fromstring(content)
            lines = []
            # Check for candidate name tag at top
            for name_tag in (".//name", ".//applicant/name", ".//candidate/name"):
                name_elem = root.find(name_tag)
                if name_elem is not None and name_elem.text:
                    lines.append(name_elem.text.strip())
                    break

            for elem in root.iter():
                t = (elem.text or "").strip()
                if t and t not in lines:
                    lines.append(t)
            if lines:
                return "\n".join(lines)
        except Exception:
            pass
        # Fallback: strip XML tags
        return re.sub(r"<[^>]+>", " ", content)
    except Exception as exc:
        logger.warning("Error reading XML %s: %s", path, exc)
        return ""


def extract_text_from_json(path: str) -> str:
    """Extract string values from structured JSON resume."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        texts = []

        def _walk(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)
            elif isinstance(obj, str) and obj.strip():
                texts.append(obj.strip())

        _walk(data)
        return "\n".join(texts)
    except Exception as exc:
        logger.warning("Error reading JSON %s: %s", path, exc)
        return ""


def extract_text_from_html(path: str) -> str:
    """Extract text from HTML documents."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        content = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", content)
        return clean
    except Exception as exc:
        logger.warning("Error reading HTML %s: %s", path, exc)
        return ""


def extract_text_from_rtf(path: str) -> str:
    """Extract text from RTF documents."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        clean = re.sub(r"\\[a-z]{1,32}(-?\d+)? ?", "", content)
        clean = re.sub(r"[{}]", "", clean)
        return clean
    except Exception as exc:
        logger.warning("Error reading RTF %s: %s", path, exc)
        return ""


def extract_raw_text_from_file(path: str) -> str:
    """Universal document extractor for any file format without conversion."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".docx", ".docm", ".dotx"):
        return extract_text_from_docx(path)
    elif ext == ".xml":
        return extract_text_from_xml(path)
    elif ext == ".json":
        return extract_text_from_json(path)
    elif ext in (".html", ".htm"):
        return extract_text_from_html(path)
    elif ext == ".rtf":
        return extract_text_from_rtf(path)
    else:
        # Standard text-based: .txt, .md, .markdown, .csv, .tsv, etc.
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as exc:
            logger.warning("Error reading text file %s: %s", path, exc)
            return ""


_MONTH_ABBR = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Numeric month/year date formats collapse to a single "Mon YYYY" spelling so
# resumes that mix "03/2021", "2021-03", "2021/03" etc. read consistently.
# Deliberately narrow (month must be 01-12) to avoid misreading unrelated
# numbers. We do NOT touch words like "Present"/"Current" -- those are
# already valid English and rewriting them risks mangling unrelated
# sentences (e.g. "currently pursuing B.Tech") that get fed straight into
# the semantic embedding model.
_DATE_MM_YYYY_RE = re.compile(r"\b(0[1-9]|1[0-2])[/-](19|20)\d{2}\b")
_DATE_YYYY_MM_RE = re.compile(r"\b((?:19|20)\d{2})[/-](0[1-9]|1[0-2])\b")


def _normalize_date_formats(text: str) -> str:
    """Collapse common numeric MM/YYYY and YYYY/MM date spellings to 'Mon YYYY'."""

    def _mm_yyyy(m: "re.Match") -> str:
        month = int(m.group(1))
        year = m.group(0)[-4:]
        return f"{_MONTH_ABBR[month - 1]} {year}"

    def _yyyy_mm(m: "re.Match") -> str:
        year, month = m.group(1), int(m.group(2))
        return f"{_MONTH_ABBR[month - 1]} {year}"

    text = _DATE_MM_YYYY_RE.sub(_mm_yyyy, text)
    text = _DATE_YYYY_MM_RE.sub(_yyyy_mm, text)
    return text


def clean_text(raw_text: str) -> str:
    """Normalize whitespace/bullets/hyphenation/dates for reliable matching."""
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")

    # Rejoin words hyphenated across line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize bullet glyphs
    text = re.sub(r"[•◦▪●○∙·\u2022\u2023\u25e6\u2043\u2219*]", " ", text)

    # Normalize varied numeric date formats (MM/YYYY, YYYY-MM, ...)
    text = _normalize_date_formats(text)

    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Clean up blank lines while preserving paragraph boundaries
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)

    return text.strip()


def _name_from_filename(filename: str) -> str:
    """Extract candidate name from filenames like 'App_Developer_Resume_1_Siddharth_Rao'."""
    clean = re.sub(r"^(?:strong|medium|weak|edge|sample)[-_]", "", filename, flags=re.IGNORECASE)
    # If pattern like 'Role_Resume_1_First_Last' or 'Role__First_Last'
    m = re.search(r"[-_]{1,2}([A-Za-z]+[-_][A-Za-z]+)$", clean)
    if m:
        return m.group(1).replace("_", " ").replace("-", " ").title()

    # If pattern like 'Aditya Kulkarni CyberSec Resume'
    m2 = re.search(r"^([A-Z][a-z]+\s+[A-Z][a-z]+)", clean)
    if m2:
        return m2.group(1)

    clean = re.sub(r"[-_]+", " ", clean).strip()
    return clean.title() if clean else filename


IGNORE_HEADER_LINES = {
    "resume", "curriculum vitae", "cv", "summary", "profile", "contact",
    "contact info", "personal info", "education", "experience", "skills",
    "objective", "technical skills", "work experience", "about me", "applicant",
}

# Job titles, degrees, and generic resume-header words that should never be
# mistaken for a person's name, even when they appear alone on a line in the
# right shape (Title Case, no digits, short).
NAME_STOPWORDS = {
    "resume", "curriculum", "vitae", "objective", "summary", "profile",
    "contact", "education", "experience", "skills", "portfolio", "career",
    "personal", "details", "information", "about", "applicant", "candidate",
    "engineer", "developer", "manager", "intern", "internship", "analyst",
    "senior", "junior", "lead", "specialist", "associate", "consultant",
    "designer", "scientist", "student", "fresher", "trainee", "executive",
    "software", "web", "backend", "frontend", "full", "stack", "data",
    "machine", "learning", "cyber", "security", "sde", "sales", "marketing",
    "founder", "office", "cloud", "devops", "qa", "product", "business",
    "technical", "writer", "editor", "video", "content", "creation",
    "operations", "finance", "support", "administrator", "coordinator",
}

# A plausible name is 1-4 space-separated tokens, each starting with a
# letter and containing only letters, apostrophes, periods, or hyphens
# (covers "O'Brien", "Jean-Luc", "A. Kumar", etc.).
NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'.\-]*$")


def _looks_like_person_name(candidate_part: str) -> bool:
    if not (2 <= len(candidate_part) <= 40):
        return False
    if "@" in candidate_part or any(ch.isdigit() for ch in candidate_part):
        return False

    words = candidate_part.split()
    if not (1 <= len(words) <= 4):
        return False
    if not all(NAME_TOKEN_RE.match(w) for w in words):
        return False
    if any(w.lower().strip(".") in NAME_STOPWORDS for w in words):
        return False
    return True


CANDIDATE_TAG_RE = re.compile(r"^candidate\s+(\d+)\b", re.IGNORECASE)


def guess_candidate_name(text: str, fallback: str) -> str:
    """Best-effort candidate name from the top non-empty resume lines or filename.

    Scans further into the document than a naive "first line" approach and
    filters out job titles, section headers, and contact-info lines so a
    stray heading doesn't get mistaken for the candidate's name.

    Some placeholder/synthetic resumes carry no real person name at all --
    just an explicit "Candidate 16 - Blockchain Developer" style tag. If we
    let those fall through the normal name-shape check, the scan keeps
    going and can latch onto an unrelated section header further down
    (e.g. "Relevant Tools"), and worse, *every* nameless placeholder
    resume ends up guessing the same wrong header -- silently merging
    what are actually different candidates into one. So that literal
    "Candidate N" tag, when present, is treated as this entry's identifier
    immediately, before the general heuristic below ever gets a chance to
    wander off.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:15]:
        lower_line = line.lower()
        if lower_line in IGNORE_HEADER_LINES:
            continue

        tag_match = CANDIDATE_TAG_RE.match(line)
        if tag_match:
            return f"Candidate {tag_match.group(1)}"

        # Split on common delimiters (pipes, dashes, commas separating name from title/contact)
        candidate_part = re.split(r"[|•\-\–\—,]", line)[0].strip()
        cleaned_part = re.sub(
            r"^(?:name[:\s]*|mr\.|ms\.|mrs\.|dr\.)", "", candidate_part, flags=re.IGNORECASE
        ).strip()

        if _looks_like_person_name(cleaned_part):
            return cleaned_part.title()

    return _name_from_filename(fallback)


# ---------------------------------------------------------------------------
# Role applied for
# ---------------------------------------------------------------------------
# Recruiters looking at the ranking table want to know what role each resume
# actually targets -- especially useful when one batch mixes many different
# applicant profiles (web dev, data science, design, ...) all being scored
# against a single JD. There's no one consistent field for this across
# resumes, so it's resolved in order of reliability:
#   1. An explicit "Candidate N - <role>" placeholder tag, when present --
#      the role is spelled out verbatim right there.
#   2. A curated dictionary of common role phrases, matched against the
#      resume's own header/contact line + profile summary/objective (where a
#      real resume naturally states or implies its target role).
#   3. The same dictionary matched against the filename (many are literally
#      "<Role>_Resume...").
#   4. A last-resort cleanup of the filename itself.
ROLE_DICTIONARY = {
    "Software Development Engineer (SDE)": ["software development engineer", "software engineer", "sde"],
    "Full Stack Developer": ["full stack developer", "full-stack developer", "fullstack developer"],
    "Backend Developer": ["backend developer", "backend engineer"],
    "Frontend Developer": ["frontend developer", "front-end developer", "frontend engineer"],
    "Web Developer": ["web developer", "web dev", "webdev"],
    "App Developer": ["app developer", "mobile developer", "android developer", "ios developer"],
    "Data Scientist": ["data scientist"],
    "Data Analyst": ["data analyst"],
    "Machine Learning Engineer": ["machine learning engineer", "ml engineer"],
    "AI Engineer": ["ai engineer", "artificial intelligence engineer"],
    "Cloud Engineer": ["cloud engineer"],
    "Cloud Support Engineer": ["cloud support engineer", "cloud support"],
    "DevOps Engineer": ["devops engineer", "devops"],
    "Site Reliability Engineer": ["site reliability engineer", "sre"],
    "Developer Relations Engineer": ["developer relations engineer", "devrel engineer", "devrel"],
    "Embedded Systems Engineer": ["embedded systems engineer", "embedded engineer"],
    "Blockchain Developer": ["blockchain developer"],
    "Game Developer": ["game developer"],
    "QA Engineer": ["qa engineer", "quality assurance engineer", "test engineer", "sdet"],
    "UI/UX Designer": ["ui/ux designer", "ui ux designer", "ux designer", "ui designer"],
    "Graphic Designer": ["graphic designer"],
    "Product Manager": ["product manager"],
    "Business Analyst": ["business analyst"],
    "Business Development": ["business development"],
    "Technical Writer": ["technical writer"],
    "Customer Success": ["customer success"],
    "Operations Intern": ["operations intern", "operations associate"],
    "Finance Intern": ["finance intern", "finance associate"],
    "Cybersecurity Analyst": [
        "cybersecurity analyst", "cyber security analyst", "cybersecurity", "cyber security", "cybersec",
    ],
    "Marketing Intern": ["marketing intern", "digital marketing"],
    "Sales Intern": ["sales intern", "sales associate"],
}


def _build_role_patterns(role_dictionary: dict) -> list:
    """Compile (role_name, pattern) pairs, longest surface forms first so
    e.g. 'ui/ux designer' is tried before the shorter 'ux designer'."""
    all_forms = []
    for role, forms in role_dictionary.items():
        for form in forms:
            all_forms.append((role, form))
    all_forms.sort(key=lambda pair: len(pair[1]), reverse=True)

    compiled = []
    for role, form in all_forms:
        # Split first, escape each word separately, then rejoin with a
        # flexible separator -- so "web developer" also matches
        # "web_developer" or "web-developer" in a filename.
        escaped = r"[\s_-]+".join(re.escape(part) for part in form.split(" "))
        pattern = re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)
        compiled.append((role, pattern))
    return compiled


_ROLE_PATTERNS = _build_role_patterns(ROLE_DICTIONARY)

CANDIDATE_TAG_ROLE_RE = re.compile(r"^candidate\s+\d+\s*[-–—:|]\s*(.+)$", re.IGNORECASE)


def _match_role_in(haystack: str):
    """Return the canonical role name for the earliest, longest dictionary
    match in `haystack`, or None if nothing matched."""
    best_key = None
    best_role = None
    for role, pattern in _ROLE_PATTERNS:
        m = pattern.search(haystack)
        if m:
            key = (m.start(), -(m.end() - m.start()))
            if best_key is None or key < best_key:
                best_key = key
                best_role = role
    return best_role


def guess_role_applied_for(text: str, filename: str) -> str:
    """Best-effort guess at the role/position a resume targets. Always
    returns a usable label -- "Not Specified" only when nothing at all
    (text or filename) gives any signal."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # 1. Explicit "Candidate N - Role" placeholder tag. Normalize against the
    #    dictionary first (so "ML Engineer" reads as "Machine Learning
    #    Engineer", "DevRel Engineer" as "Developer Relations Engineer",
    #    etc.); otherwise keep the tag's own casing as-is rather than
    #    blindly .title()-ing it, which would mangle acronyms like "QA" or
    #    "DevOps" into "Qa"/"Devops".
    for line in lines[:5]:
        tag_match = CANDIDATE_TAG_ROLE_RE.match(line)
        if tag_match:
            role_text = tag_match.group(1).strip()
            if role_text:
                dict_hit = _match_role_in(role_text)
                if dict_hit:
                    return dict_hit
                if role_text.isupper() or role_text.islower():
                    return role_text.title()
                return role_text
            break

    # 2. Dictionary match against the filename. Filenames in a resume batch
    #    are very often deliberately named after the role ("web_dev_aditya_
    #    kulkarni.pdf", "SDE_Resume_1.docx") -- that's a stronger, more
    #    direct signal than prose, so it's checked before the resume text.
    filename_normalized = re.sub(r"[_\-]+", " ", os.path.splitext(filename)[0])
    matched = _match_role_in(filename_normalized)
    if matched:
        return matched

    # 3. Dictionary match against the resume's own header/contact line and
    #    profile summary/objective -- used when the filename itself doesn't
    #    give any role away (e.g. it's just the candidate's name).
    matched = _match_role_in(text[:1000])
    if matched:
        return matched

    # 4. Last resort: strip noise words/numbers/lone letters from the
    #    filename and use whatever's left, unless that's just a person's
    #    name (nothing role-ish about it).
    cleaned = re.sub(r"\bresume\b|\bcv\b", "", filename_normalized, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d+\b|\b[A-Za-z]\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and not _looks_like_person_name(cleaned):
        return cleaned.title()

    return "Not Specified"


def parse_jd(jd_path: str) -> str:
    """Parse the Job Description into clean text from any supported format."""
    return clean_text(extract_raw_text_from_file(jd_path))


def parse_resumes(resumes_dir: str) -> dict:
    """Parse every resume in a directory across all supported document formats.

    Two distinct kinds of "duplicate" are handled differently:

    1. Multiple *format exports of the literal same file* (e.g. a
       "Resume.pdf" and "Resume.docx" sharing the same base filename) --
       these collapse to a single parse via EXTENSION_PRIORITY, since
       there's nothing to compare, it's the same document twice.

    2. A candidate who *submitted more than one resume file* under
       different filenames (e.g. a PDF and a plain-text copy, or two
       differently-formatted versions) -- every one of these is kept here.
       They all guess to the same candidate name and are grouped together;
       it's scoring.py's job to score each submission and keep only the
       best-performing one for that candidate, recording the rest for
       transparency instead of silently dropping them.

    A guessed name is always treated as one candidate -- no "(2)", "(3)"
    disambiguation suffixes. Only byte-identical resume text under the same
    name is skipped as a true duplicate.

    Returns: {candidate_name: [{"filename": str, "text": str, "role": str}, ...]}
    """
    if not os.path.isdir(resumes_dir):
        return {}

    all_files = sorted(os.listdir(resumes_dir))

    # Group files by base name to deduplicate multiple extensions of same file
    base_groups = {}
    for f in all_files:
        base, ext = os.path.splitext(f)
        ext_lower = ext.lower()
        if ext_lower in SUPPORTED_EXTENSIONS:
            if base not in base_groups:
                base_groups[base] = []
            base_groups[base].append(f)

    # Pick the highest priority file for each base name
    chosen_files = []
    for base, file_list in sorted(base_groups.items()):
        file_list.sort(key=lambda fn: EXTENSION_PRIORITY.get(os.path.splitext(fn)[1].lower(), 99))
        chosen_files.append(file_list[0])

    candidates: dict = {}
    seen_texts_by_name: dict = {}

    for filename in chosen_files:
        path = os.path.join(resumes_dir, filename)
        raw = extract_raw_text_from_file(path)
        text = clean_text(raw)
        if not text:
            continue

        base_name = os.path.splitext(filename)[0]
        name = guess_candidate_name(text, fallback=base_name)
        role = guess_role_applied_for(text, filename)

        if any(text == prev for prev in seen_texts_by_name.get(name, [])):
            continue  # byte-identical resume content already filed under this name

        seen_texts_by_name.setdefault(name, []).append(text)
        candidates.setdefault(name, []).append({"filename": filename, "text": text, "role": role})

    return candidates

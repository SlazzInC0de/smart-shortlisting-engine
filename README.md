# Nexora — Smart Shortlisting Engine

Hybrid semantic + keyword resume-to-JD ranking engine, built for the InternLoom AI Hackathon (MIT Bengaluru). Given one job description and a batch of resumes, it produces a fully ranked, explainable shortlist.

Runs 100% offline: local sentence-transformer embeddings, a curated skill dictionary, and rule-based heuristics only. **Zero external API calls.**

## Features

- Hybrid scoring — configurable keyword-match + semantic-similarity weighting (default 50/50), live-adjustable from the UI.
- Full ranked shortlist across the entire resume batch, not just a top-N cut.
- Deep-dive explanations for top candidates: matched/missing skills, dominant scoring signal, recruiter recommendation.
- Typo-tolerant keyword matching (adjacent-character-transposition + fuzzy-ratio fallback) and numeric date-format normalization, for messy or inconsistent resume formatting.
- JD bias & narrow-phrasing review — flags gendered wording, exclusionary phrases, high flat experience bars, and overly long required-skill lists, with plain-language reasoning for each.
- Recruiter "Compare Candidates" tool — pick any two candidates and get a natural-language + tabular breakdown of why one outranks the other.
- Multi-submission transparency — if a candidate sends more than one resume file, every submission is scored independently and the best one is auto-selected to represent them, with the full comparison shown. Nothing is silently dropped or duplicated.
- Role-applied-for detection per resume, inferred from placeholder tags, filename, or resume text.
- Row highlighting for quick recruiter triage (placeholder/nameless entries, zero-skill-match entries).

## Stack

Streamlit (UI) · sentence-transformers (`all-MiniLM-L6-v2`, local embeddings) · scikit-learn (cosine similarity) · pandas / Altair (tables & charts) · pdfplumber (PDF parsing).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Drop the job description into `data/jd/` (any supported format — PDF, DOCX, TXT, etc. — any filename) and resumes into `data/resumes/`. The app picks up whatever's there automatically.

## Project layout

- `app.py` — Streamlit UI (dashboard, deep dive, skills matrix, JD review, candidate comparison)
- `config.py` — central scoring-weight and data-path configuration
- `parsing.py` — universal document parsing, candidate name and role extraction
- `keyword_match.py` — curated skill dictionary + typo-tolerant matching
- `semantic_match.py` — local embedding generation and cosine similarity
- `scoring.py` — combines keyword and semantic signals into the final ranked list, including multi-submission handling
- `explain.py` — template-based candidate explanations and candidate-vs-candidate comparisons
- `bias_check.py` — heuristic JD bias / narrow-phrasing scanner

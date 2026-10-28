# Security Policy

## Supported versions

Only the latest commit on the `main` branch receives security fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report it privately by either:

- using **Report a vulnerability** under this repository's **Security** tab, or
- emailing **vsuryansh080405@gmail.com** with the subject `[TalentLens] Security report`.

Include:

- what the issue is and where it lives (module, file type, input)
- steps or a sample file that reproduces it
- the impact you expect

## What to expect

- An acknowledgement within 7 days.
- A fix or a clear decision within 30 days for confirmed issues, depending on severity.
- Credit in the fix commit if you would like it.

## Scope

TalentLens runs fully offline on your own machine and makes no external API calls. In scope:

- crafted PDF, DOCX or TXT resumes or job descriptions that crash the parser, exhaust memory or run code (`parsing.py`)
- path traversal when reading the resumes folder
- unsafe rendering of resume text in the Streamlit dashboard
- any code path that sends resume content off the machine
- vulnerable dependencies

Out of scope: ranking quality or fairness disagreements (open a normal issue for those), and reports from automated scanners without a working proof of concept.

## Handling real resumes

Resumes contain personal data. The sample resumes in `data/resumes/` are fictional. If you run TalentLens on real candidates, keep their files out of this repository and delete them when you no longer need them.

import json
import zipfile

import pytest

from parsing import clean_text, extract_raw_text_from_file, guess_candidate_name

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _docx(path, paragraphs):
    body = "".join(f'<w:p><w:r><w:t>{p}</w:t></w:r></w:p>' for p in paragraphs)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>')


@pytest.mark.parametrize("ext, writer", [
    (".txt", lambda p: p.write_text("Ananya Rao\nSkills: Python, SQL")),
    (".md", lambda p: p.write_text("# Ananya Rao\n- Python\n- SQL")),
    (".docx", lambda p: _docx(p, ["Ananya Rao", "Skills: Python, SQL"])),
    (".xml", lambda p: p.write_text("<resume><name>Ananya Rao</name><skills>Python, SQL</skills></resume>")),
    (".json", lambda p: p.write_text(json.dumps({"name": "Ananya Rao", "skills": ["Python", "SQL"]}))),
    (".html", lambda p: p.write_text("<html><style>p{}</style><body><h1>Ananya Rao</h1><p>Python, SQL</p></body></html>")),
    (".rtf", lambda p: p.write_text(r"{\rtf1\ansi {\b Ananya Rao}\par Skills: Python, SQL}")),
])
def test_every_supported_format_yields_text(tmp_path, ext, writer):
    path = tmp_path / f"resume{ext}"
    writer(path)
    text = extract_raw_text_from_file(str(path))
    assert "Ananya Rao" in text
    assert "Python" in text and "SQL" in text


def test_html_scripts_and_styles_are_stripped(tmp_path):
    path = tmp_path / "r.html"
    path.write_text("<script>steal()</script><p>Real content</p>")
    text = extract_raw_text_from_file(str(path))
    assert "steal" not in text and "Real content" in text


def test_broken_files_return_empty_text_instead_of_crashing(tmp_path):
    for name, payload in (("bad.docx", b"not a zip"), ("bad.json", b"{oops"), ("bad.pdf", b"%PDF-broken")):
        path = tmp_path / name
        path.write_bytes(payload)
        assert extract_raw_text_from_file(str(path)) == ""


def test_malformed_xml_falls_back_to_tag_stripping(tmp_path):
    path = tmp_path / "r.xml"
    path.write_text("<resume><name>Ananya Rao</name><unclosed>")
    assert "Ananya Rao" in extract_raw_text_from_file(str(path))


def test_numeric_dates_are_normalised():
    text = clean_text("Intern 03/2021 - 2022-11 at Acme")
    assert "Mar 2021" in text and "Nov 2022" in text


def test_name_is_taken_from_resume_text_with_filename_fallback():
    assert guess_candidate_name("Ananya Rao\nananya@mail.com\nSkills: Python", "x.txt") == "Ananya Rao"
    assert guess_candidate_name("skills: python, sql", "Resume_07_Kiran_Das") == "Kiran Das"  # caller passes the base name

"""
Tests for the JobFit AI pipeline.
Run with: pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.parser import chunk_document, extract_sections, extract_pdf_text
from src.scraper import clean_text, extract_job_title, detect_platform
from src.models import JobAnalysis


# ── Parser Tests ─────────────────────────────────────────────────────

class TestChunkDocument:
    def test_basic_chunking(self):
        text = "This is a test document. " * 100
        chunks = chunk_document(text, doc_type="resume")
        assert len(chunks) > 1
        assert all("text" in c and "metadata" in c for c in chunks)

    def test_metadata_is_correct(self):
        text = "Short text that fits in one chunk."
        chunks = chunk_document(text, doc_type="job_description")
        assert chunks[0]["metadata"]["doc_type"] == "job_description"
        assert chunks[0]["metadata"]["chunk_index"] == 0

    def test_empty_text(self):
        chunks = chunk_document("", doc_type="resume")
        assert len(chunks) == 0 or chunks[0]["text"] == ""

    def test_overlap_produces_more_chunks(self):
        text = "Word " * 500
        chunks = chunk_document(text, doc_type="resume")
        assert len(chunks) >= 2


class TestExtractSections:
    def test_skills_section(self):
        resume = "John Doe\n\nSKILLS\nPython, SQL, AWS\n\nEXPERIENCE\nData Engineer at ACME"
        sections = extract_sections(resume)
        assert "skills" in sections
        assert "Python" in sections["skills"]

    def test_no_sections(self):
        resume = "Just a plain text resume with no clear headers."
        sections = extract_sections(resume)
        assert "header" in sections


# ── Scraper Tests ────────────────────────────────────────────────────

class TestCleanText:
    def test_collapses_whitespace(self):
        text = "Hello\n\n\n\n\nWorld"
        cleaned = clean_text(text)
        assert "\n\n\n" not in cleaned

    def test_removes_apply_now(self):
        text = "Great job description Apply Now Save Job"
        cleaned = clean_text(text)
        assert "Apply Now" not in cleaned


class TestDetectPlatform:
    def test_linkedin(self):
        result = detect_platform("https://www.linkedin.com/jobs/view/123456")
        assert result is not None

    def test_greenhouse(self):
        result = detect_platform("https://boards.greenhouse.io/company/jobs/123")
        assert result is not None

    def test_unknown(self):
        result = detect_platform("https://randomsite.com/job")
        assert result is None


class TestExtractJobTitle:
    def test_basic_title(self):
        text = "Senior Data Engineer\nSan Francisco, CA\nAbout the role..."
        title = extract_job_title(text)
        assert title == "Senior Data Engineer"


# ── Model Tests ──────────────────────────────────────────────────────

class TestJobAnalysis:
    def test_valid_model(self):
        data = {
            "score": 72,
            "summary": "Good match overall.",
            "matched_skills": ["Python", "SQL"],
            "missing_skills": ["Kubernetes"],
            "matched_experience": ["Data pipeline work"],
            "gaps": ["No cloud cert"],
            "resume_tweaks": ["Add K8s mention"],
            "keywords_to_add": ["Kubernetes"],
            "interview_tips": ["Prepare system design"],
        }
        model = JobAnalysis(**data)
        assert model.score == 72
        assert len(model.matched_skills) == 2

    def test_score_validation(self):
        with pytest.raises(Exception):
            JobAnalysis(
                score=150,  # Invalid — must be 0-100
                summary="Test",
                matched_skills=[],
                missing_skills=[],
                matched_experience=[],
                gaps=[],
                resume_tweaks=[],
                keywords_to_add=[],
                interview_tips=[],
            )

    def test_json_export(self):
        data = {
            "score": 65,
            "summary": "Decent fit.",
            "matched_skills": ["Python"],
            "missing_skills": ["Go"],
            "matched_experience": ["Built ETL"],
            "gaps": ["No Go experience"],
            "resume_tweaks": ["Learn Go"],
            "keywords_to_add": ["Golang"],
            "interview_tips": ["Study concurrency"],
        }
        model = JobAnalysis(**data)
        json_str = model.model_dump_json()
        assert '"score": 65' in json_str

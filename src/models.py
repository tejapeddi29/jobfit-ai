"""
Pydantic models for structured output from the LLM analysis chain.
Uses LangChain's PydanticOutputParser for reliable JSON extraction.
"""

from pydantic import BaseModel, Field


class JobAnalysis(BaseModel):
    """Structured output from the job-resume match analysis."""

    score: int = Field(
        description="Match score from 0-100 representing how well the resume fits the job",
        ge=0,
        le=100,
    )
    summary: str = Field(
        description="2-3 sentence overall assessment of the match"
    )
    matched_skills: list[str] = Field(
        description="Skills from the job description that appear in the resume"
    )
    missing_skills: list[str] = Field(
        description="Skills required by the job that are missing from the resume"
    )
    matched_experience: list[str] = Field(
        description="Resume experiences that align with job requirements"
    )
    gaps: list[str] = Field(
        description="Experience or qualification gaps to address"
    )
    resume_tweaks: list[str] = Field(
        description="Specific, actionable resume edits — rewritten bullet points, added keywords, rephrased sections"
    )
    keywords_to_add: list[str] = Field(
        description="ATS-friendly keywords from the job description to add to the resume"
    )
    interview_tips: list[str] = Field(
        description="3 interview preparation tips specific to this role"
    )


class TailoredBullet(BaseModel):
    """A single rewritten resume bullet point."""
    original: str = Field(description="The original bullet point from the resume")
    rewritten: str = Field(description="The rewritten bullet point tailored to the job")
    reason: str = Field(description="Why this change improves the match (1 sentence)")


class TailoredExperience(BaseModel):
    """A tailored experience section entry."""
    company: str = Field(description="Company name")
    title: str = Field(description="Job title — rewritten to better match the target role if appropriate")
    dates: str = Field(description="Employment dates")
    bullets: list[TailoredBullet] = Field(description="Rewritten bullet points for this role")


class TailoredResume(BaseModel):
    """Complete tailored resume output from the LLM."""
    target_title: str = Field(description="The job title being applied for")
    summary: str = Field(
        description="A 2-3 sentence professional summary tailored to this specific job, "
        "incorporating key requirements and the candidate's relevant strengths"
    )
    skills: list[str] = Field(
        description="Reordered and augmented skills list — job-relevant skills first, "
        "missing skills the candidate can honestly claim added, irrelevant skills removed"
    )
    experience: list[TailoredExperience] = Field(
        description="Each work experience entry with bullet points rewritten to emphasize "
        "job-relevant accomplishments, using keywords from the job description"
    )
    education: list[str] = Field(description="Education entries (usually unchanged)")
    certifications: list[str] = Field(description="Certifications (reordered by relevance)")
    projects: list[str] = Field(
        description="Project lines rewritten to highlight job-relevant technologies. "
        "Format: 'Project Name | Tech Stack | 1-line description'"
    )
    changes_made: list[str] = Field(
        description="Summary of all changes made to the resume, so the candidate "
        "can review what was modified and why"
    )


class SemanticMatch(BaseModel):
    """Result from ChromaDB semantic similarity search."""

    resume_chunk: str
    job_chunk: str
    similarity_score: float

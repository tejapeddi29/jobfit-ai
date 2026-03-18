"""
Document Parsing module — handles PDF resume extraction and text chunking.
Uses PyPDF2 for PDF reading and LangChain's RecursiveCharacterTextSplitter
for intelligent document chunking with overlap.
"""

import io
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, str | None]:
    """
    Extract text content from a PDF file.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        tuple: (extracted_text, error_message)
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())

        full_text = "\n\n".join(pages)

        if not full_text.strip():
            return "", "PDF appears to be image-based (scanned). Paste your resume as text instead."

        return full_text, None

    except Exception as e:
        return "", f"PDF parsing failed: {str(e)}. Try pasting your resume as text."


def chunk_document(text: str, doc_type: str = "general") -> list[dict]:
    """
    Split a document into overlapping chunks for embedding.

    Uses LangChain's RecursiveCharacterTextSplitter which tries to split
    on paragraph boundaries first, then sentences, then words.

    Args:
        text: The full document text.
        doc_type: "resume" or "job_description" — used as metadata.

    Returns:
        List of dicts with 'text', 'metadata' keys.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    chunks = splitter.split_text(text)

    return [
        {
            "text": chunk,
            "metadata": {
                "doc_type": doc_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        }
        for i, chunk in enumerate(chunks)
    ]


def extract_sections(resume_text: str) -> dict[str, str]:
    """
    Attempt to extract common resume sections.

    Returns a dict with keys like 'skills', 'experience', 'education'.
    Falls back to the full text if sections can't be parsed.
    """
    sections = {}
    current_section = "header"
    current_lines = []

    section_keywords = {
        "skills": ["skills", "technical skills", "core competencies", "technologies"],
        "experience": ["experience", "work experience", "professional experience", "employment"],
        "education": ["education", "academic", "degrees"],
        "projects": ["projects", "personal projects", "portfolio"],
        "certifications": ["certifications", "certificates", "licenses"],
        "summary": ["summary", "objective", "profile", "about"],
    }

    for line in resume_text.split("\n"):
        line_lower = line.strip().lower()

        matched_section = None
        for section, keywords in section_keywords.items():
            if any(kw in line_lower for kw in keywords) and len(line.strip()) < 50:
                matched_section = section
                break

        if matched_section:
            # Save previous section
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections

"""
Web Scraping module — extracts job description text from any URL.
Uses requests + BeautifulSoup4 to handle LinkedIn, Indeed, Greenhouse,
Lever, and generic job pages.
"""

import re
import requests
from bs4 import BeautifulSoup
from src.config import REQUEST_TIMEOUT, USER_AGENT


# ── Tag selectors by platform ───────────────────────────────────────
PLATFORM_SELECTORS = {
    "greenhouse.io": {"selector": "div#content", "backup": "div.job-post"},
    "lever.co": {"selector": "div.content", "backup": "div.posting-page"},
    "linkedin.com": {"selector": "div.description", "backup": "div.show-more-less-html"},
    "indeed.com": {"selector": "div#jobDescriptionText", "backup": "div.jobsearch-JobComponent"},
    "myworkdayjobs.com": {"selector": "div[data-automation-id='jobPostingDescription']", "backup": None},
}


def detect_platform(url: str) -> dict | None:
    """Detect the job board platform from the URL."""
    for platform, selectors in PLATFORM_SELECTORS.items():
        if platform in url.lower():
            return selectors
    return None


def clean_text(text: str) -> str:
    """Clean extracted text — remove excess whitespace, artifacts."""
    # Collapse whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Remove common artifacts
    text = re.sub(r"(Apply Now|Save Job|Share|Report)(\s|$)", "", text, flags=re.IGNORECASE)
    return text.strip()


def scrape_job_url(url: str) -> tuple[str, str | None]:
    """
    Scrape a job description from a URL.

    Returns:
        tuple: (extracted_text, error_message)
        If successful, error_message is None.
        If failed, extracted_text is empty and error_message explains why.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return "", "Request timed out. Try pasting the job description text directly."
    except requests.exceptions.ConnectionError:
        return "", "Could not connect to the URL. Check the link and try again."
    except requests.exceptions.HTTPError as e:
        if response.status_code == 403:
            return "", "Access blocked (403). This site requires login. Paste the job description text instead."
        return "", f"HTTP error {response.status_code}. Try pasting the text directly."
    except Exception as e:
        return "", f"Scraping failed: {str(e)}. Try pasting the text directly."

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "iframe"]):
        tag.decompose()

    # Try platform-specific selectors first
    platform = detect_platform(url)
    if platform:
        for key in ["selector", "backup"]:
            sel = platform.get(key)
            if sel:
                element = soup.select_one(sel)
                if element:
                    text = clean_text(element.get_text(separator="\n"))
                    if len(text) > 100:
                        return text, None

    # Generic extraction — try common job description containers
    generic_selectors = [
        "div[class*='job-description']",
        "div[class*='jobDescription']",
        "div[class*='posting-description']",
        "div[class*='job-details']",
        "article[class*='job']",
        "div[id*='job']",
        "main",
        "article",
    ]

    for sel in generic_selectors:
        element = soup.select_one(sel)
        if element:
            text = clean_text(element.get_text(separator="\n"))
            if len(text) > 100:
                return text, None

    # Last resort — get the body text
    body = soup.find("body")
    if body:
        text = clean_text(body.get_text(separator="\n"))
        if len(text) > 100:
            # Truncate to likely job content (first 5000 chars)
            return text[:5000], None

    return "", "Could not extract job description from this page. Paste the text directly."


def extract_job_title(text: str) -> str:
    """Attempt to extract a job title from the first few lines of text."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        # First non-empty line is often the title
        candidate = lines[0]
        if len(candidate) < 100:
            return candidate
    return "Job Position"

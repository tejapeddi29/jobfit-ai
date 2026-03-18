# 🎯 JobFit AI — AI-Powered Job Application Tracker & Resume Tailor

An end-to-end AI application that scrapes job postings, parses your resume, builds a semantic knowledge base with ChromaDB, and uses LangChain + Claude/OpenAI to score your match — then **rewrites your entire resume** tailored to that specific job, ready to download.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🧠 Problem

Every job application requires manually reading a job description, mentally mapping it against your resume, guessing what's missing, and rewriting bullet points. This is slow, subjective, and error-prone — especially when applying to 50+ roles.

## 💡 Solution

**JobFit AI** automates the entire process:

1. **Scrape** any job posting URL (or paste raw text)
2. **Parse** your resume (PDF upload or text)
3. **Embed** both into a ChromaDB vector store for semantic comparison
4. **Analyze** the match using LangChain chains with Claude or OpenAI
5. **Report** a compatibility score, skill gaps, keyword suggestions, and interview tips
6. **Tailor** — generate a fully rewritten resume optimized for that specific role, with before/after diffs for every bullet point

---

## 🏗️ Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="JobFit AI Architecture" width="100%"/>
</p>

<details>
<summary>Text version of architecture (click to expand)</summary>

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                   STREAMLIT UI (app.py)                  │
                    │                                                         │
                    │   ┌──────────┐   ┌──────────┐   ┌───────────────────┐  │
                    │   │ Job URL / │   │ Resume   │   │ Model Selector    │  │
                    │   │ Text Paste│   │ PDF/Text │   │ Claude / GPT-4o   │  │
                    │   └─────┬────┘   └─────┬────┘   └───────────────────┘  │
                    │         │              │                                │
                    └─────────┼──────────────┼────────────────────────────────┘
                              │              │
                              ▼              ▼
                    ┌─────────────┐  ┌──────────────┐
                    │ scraper.py  │  │  parser.py   │
                    │ Beautiful   │  │  PyPDF2 +    │
                    │ Soup +      │  │  LangChain   │
                    │ requests    │  │  TextSplitter│
                    └──────┬──────┘  └──────┬───────┘
                           │                │
                           ▼                ▼
                    ┌──────────────────────────────────┐
                    │     vectorstore.py — ChromaDB     │
                    │                                    │
                    │  Job Chunks ◄──► Resume Chunks     │
                    │  (embedded)      (embedded)        │
                    │                                    │
                    │  → Semantic similarity search      │
                    │  → Cosine similarity scores        │
                    └────────────────┬───────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────┐
                    │      chains.py — LangChain        │
                    │                                    │
                    │  Analysis Chain                    │
                    │  ├─ PromptTemplate                 │
                    │  ├─ ChatAnthropic / ChatOpenAI     │
                    │  └─ PydanticOutputParser           │
                    │                                    │
                    │  Tailoring Chain                   │
                    │  ├─ Resume rewriting prompts       │
                    │  ├─ Before/after bullet diffs      │
                    │  └─ TailoredResume output          │
                    └──────────┬─────────────────────────┘
                               │
                      ┌────────┴────────┐
                      ▼                 ▼
            ┌──────────────┐  ┌──────────────────┐
            │  Tab 1:      │  │  Tab 2:          │
            │  Match       │  │  Tailored Resume │
            │  Analysis    │  │                  │
            │              │  │  • New summary   │
            │  • Score     │  │  • Rewritten     │
            │  • Skills    │  │    bullets        │
            │  • Gaps      │  │  • Before/after  │
            │  • Tweaks    │  │  • Download:     │
            │  • Keywords  │  │    .docx/.md/.json│
            └──────────────┘  └──────────────────┘
```
</details>

---

## 🛠️ Tech Stack

| Technology | Role in Project |
|---|---|
| **LangChain** | Orchestrates analysis + tailoring chains — text splitting, embeddings, prompt templates, `PydanticOutputParser` |
| **Claude API** | Primary LLM for analysis and resume rewriting (via `ChatAnthropic`) |
| **OpenAI API** | Fallback LLM + optional embedding model (`text-embedding-3-small`) |
| **Web Scraping** | `requests` + `BeautifulSoup4` to extract job descriptions from any URL |
| **Streamlit** | Full interactive UI — file upload, URL input, tabbed results dashboard, session history |
| **ChromaDB** | Vector store for semantic embedding of resume and job description chunks |
| **HuggingFace** | Free local embeddings (`all-MiniLM-L6-v2`) — no API key needed |
| **PyPDF2** | Resume PDF text extraction |
| **Pydantic** | Structured output validation for both `JobAnalysis` and `TailoredResume` models |

### Concepts Demonstrated

- **LLM Application Development** — Full production pipeline from input to structured output
- **Prompt Engineering** — System prompts with JSON schema enforcement, ATS-specialist persona
- **RAG Pattern** — Retrieve relevant resume chunks based on job requirements via ChromaDB
- **Document Parsing** — PDF extraction, HTML scraping, text chunking with overlap
- **Semantic Similarity** — Vector embeddings + cosine similarity for skill matching
- **Structured Output** — LangChain `PydanticOutputParser` for reliable JSON responses
- **Multi-Chain Architecture** — Separate analysis and tailoring chains with shared context
- **Error Handling** — Graceful fallbacks at every layer (scraper → text paste, ChromaDB → direct LLM, docx → txt)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (or Docker)
- An API key for **Claude** (Anthropic) — that's it! Embeddings run locally for free.

### Option A: Local Install

```bash
# Clone the repo
git clone https://github.com/tejapeddi29/jobfit-ai.git
cd jobfit-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### Run

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501`

### Option B: Docker (one command)

```bash
docker build -t jobfit-ai .
docker run -p 8501:8501 --env-file .env jobfit-ai
```

Opens at `http://localhost:8501` — embedding model is pre-downloaded in the image.

### Run Tests

```bash
pytest tests/test_pipeline.py -v
```

Tests validate parser, scraper, and models without needing API keys.

---

## 📁 Project Structure

```
jobfit-ai/
├── src/
│   ├── app.py                 # Streamlit UI — input, analysis, tailoring, download
│   ├── scraper.py             # Web scraping (BeautifulSoup + requests)
│   ├── parser.py              # PDF parsing (PyPDF2) + text chunking (LangChain)
│   ├── vectorstore.py         # ChromaDB embedding + semantic similarity search
│   ├── chains.py              # LangChain analysis + tailoring chains
│   ├── models.py              # Pydantic models (JobAnalysis, TailoredResume)
│   ├── resume_generator.py    # DOCX/TXT/MD resume file generation
│   └── config.py              # Environment variables + validation
├── tests/
│   └── test_pipeline.py       # Unit tests for parser, scraper, models
├── data/
│   └── sample_data.json       # Sample job description + resume for testing
├── .streamlit/
│   └── config.toml            # Streamlit dark theme
├── Dockerfile                 # Docker deployment (pre-downloads embedding model)
├── .dockerignore
├── .env.example               # Environment variable template
├── requirements.txt
└── README.md
```

---

## 🔑 Environment Variables

```env
ANTHROPIC_API_KEY=sk-ant-...          # Required for Claude
OPENAI_API_KEY=sk-...                 # Optional (only if using OpenAI model or embeddings)
DEFAULT_MODEL=claude                   # "claude" or "openai"
EMBEDDING_MODEL=free                   # "free" (local HuggingFace) or "openai" (paid)
```

> **Note:** With `EMBEDDING_MODEL=free`, you only need the Anthropic key. The HuggingFace `all-MiniLM-L6-v2` model runs locally at zero cost.

---

## 📝 Usage

1. **Enter a job posting** — Paste a URL (LinkedIn, Indeed, Greenhouse, Lever) or raw job description text
2. **Upload your resume** — PDF upload or paste as text
3. **Select your model** — Claude (recommended) or GPT-4o from the dropdown
4. **🔍 Analyze Match** — Get match score, skill gaps, experience alignment, ATS keywords, interview tips
5. **✨ Tailor My Resume** — Generate a fully rewritten resume optimized for this specific role
6. **Review before/after diffs** — Expand each bullet point to see what changed and why
7. **Download** — Export tailored resume as `.docx`, `.md`, or `.json`
8. **Track history** — All analyses saved in sidebar for multi-job comparison

---

## 🎬 How It Works

### Analyze Match Flow

**Step 1 — Input Collection:** Job description via URL (scraped with BeautifulSoup) or pasted text. Resume via PDF upload (parsed with PyPDF2) or pasted text.

**Step 2 — Document Chunking:** Both documents split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter` (500 chars, 100 overlap).

**Step 3 — Vector Embedding:** All chunks embedded and stored in ChromaDB. Default: free HuggingFace `all-MiniLM-L6-v2`. Optional: OpenAI `text-embedding-3-small`.

**Step 4 — Semantic Similarity:** For each job chunk, ChromaDB retrieves top 3 most similar resume chunks with cosine similarity scores.

**Step 5 — LLM Analysis:** LangChain builds a prompt with job text + resume + semantic evidence → sent to Claude/GPT-4o → parsed into validated `JobAnalysis` Pydantic model.

### Tailor Resume Flow

**Step 6 — Resume Tailoring Chain:** A second LangChain chain receives the original resume + job description + analysis results (matched/missing skills, gaps, keywords). It rewrites every section:
- New professional summary targeted at the role
- Skills reordered with job-relevant ones first
- Every experience bullet rewritten using job keywords
- Before/after diff with reasoning for each change

**Step 7 — Export:** `TailoredResume` Pydantic model rendered in Streamlit with expandable diffs. Download as `.docx` (via Node.js docx-js), `.md`, or `.json`.

---

## 🧪 Technology Deep Dive

### LangChain
- `ChatAnthropic` / `ChatOpenAI` — LLM wrappers with configurable temperature
- `RecursiveCharacterTextSplitter` — Intelligent document chunking with overlap
- `ChatPromptTemplate` — System + user prompt composition with variable injection
- `PydanticOutputParser` — Enforces structured JSON output matching Pydantic schemas
- Two separate chains: analysis (scoring + gaps) and tailoring (full resume rewrite)

### ChromaDB
- In-memory vector store with OpenAI or HuggingFace embeddings
- Stores both job and resume chunks with metadata filtering
- `similarity_search_with_relevance_scores` for ranked semantic matches
- Overall similarity score computed from best-match-per-chunk aggregation

### Web Scraping
- Platform-specific CSS selectors for LinkedIn, Indeed, Greenhouse, Lever, Workday
- Fallback cascade: platform selector → generic selectors → body text
- Strips `<script>`, `<style>`, `<nav>`, `<footer>` before extraction
- Graceful error handling with user-friendly messages

### Streamlit
- Tabbed interface: Match Analysis + Tailored Resume
- `st.file_uploader` for PDF resumes
- `st.status` with real-time pipeline progress
- `st.session_state` for analysis history across runs
- `st.download_button` for .docx/.md/.json export
- Custom CSS with dark theme and styled skill pills

---

## 🐳 Docker

```bash
# Build (pre-downloads the free embedding model)
docker build -t jobfit-ai .

# Run
docker run -p 8501:8501 --env-file .env jobfit-ai

# With inline env vars
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... -e DEFAULT_MODEL=claude -e EMBEDDING_MODEL=free jobfit-ai
```

---

## 📄 License

MIT — use it, fork it, put it on your resume.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

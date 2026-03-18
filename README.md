# 🎯 JobFit AI — AI-Powered Job Application Tracker

An end-to-end AI application that scrapes job postings, parses your resume, builds a semantic knowledge base with ChromaDB, and uses LangChain + Claude/OpenAI to score your match — with actionable resume tweaks.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Screenshots

| Input View | Analysis Results |
|:---:|:---:|
| Paste a URL or job description | Score, gaps, and resume tweaks |

---

## 🧠 Problem

Every job application requires manually reading a job description, mentally mapping it against your resume, guessing what's missing, and rewriting bullet points. This is slow, subjective, and error-prone — especially when applying to 50+ roles.

## 💡 Solution

**JobFit AI** automates the entire process:

1. **Scrape** any job posting URL (or paste raw text)
2. **Parse** your resume (PDF or text)  
3. **Embed** both into a ChromaDB vector store for semantic comparison
4. **Analyze** the match using LangChain chains with Claude or OpenAI
5. **Report** a compatibility score, skill gaps, keyword suggestions, and specific resume rewrites

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ URL Input    │  │ Resume Upload│  │ Results Dashboard      │  │
│  │ + Text Paste │  │ (PDF/Text)   │  │ Score·Gaps·Tweaks      │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────▲────────────┘  │
│         │                │                      │               │
├─────────┼────────────────┼──────────────────────┼───────────────┤
│         ▼                ▼                      │               │
│  ┌─────────────┐  ┌─────────────┐               │               │
│  │ Web Scraper  │  │ Doc Parser  │               │               │
│  │ (BeautifulSoup│  │ (PyPDF2 +  │               │               │
│  │  + requests) │  │  LangChain) │               │               │
│  └──────┬──────┘  └──────┬──────┘               │               │
│         │                │                      │               │
│         ▼                ▼                      │               │
│  ┌──────────────────────────────┐               │               │
│  │        ChromaDB              │               │               │
│  │  ┌────────┐  ┌───────────┐  │               │               │
│  │  │Job Desc │  │  Resume   │  │               │               │
│  │  │Chunks   │  │  Chunks   │  │               │               │
│  │  │(embedded)│  │(embedded) │  │               │               │
│  │  └────────┘  └───────────┘  │               │               │
│  └──────────────┬───────────────┘               │               │
│                 │                               │               │
│                 ▼                               │               │
│  ┌──────────────────────────────┐               │               │
│  │     LangChain Pipeline       │               │               │
│  │                              │               │               │
│  │  1. Semantic Similarity      │───────────────┘               │
│  │     (Vector Comparison)      │                               │
│  │  2. Analysis Chain           │                               │
│  │     (Claude / OpenAI)        │                               │
│  │  3. Structured Output        │                               │
│  │     (JSON Parser)            │                               │
│  └──────────────────────────────┘                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Technology | Role in Project |
|---|---|
| **LangChain** | Orchestrates the full analysis chain — document loading, text splitting, embeddings, prompt templates, output parsing |
| **Claude API** | Primary LLM for job-resume analysis (via `ChatAnthropic`) |
| **OpenAI API** | Fallback LLM + embedding model (`text-embedding-3-small`) |
| **Web Scraping** | `requests` + `BeautifulSoup4` to extract job descriptions from any URL |
| **Streamlit** | Full interactive UI — file upload, URL input, results dashboard, session history |
| **ChromaDB** | Vector store for semantic embedding of resume and job description chunks |
| **PyPDF2** | Resume PDF text extraction |

### Concepts Demonstrated

- **LLM Application Development** — Full production pipeline from input to structured output
- **Prompt Engineering** — System prompts with JSON schema enforcement and few-shot examples
- **Document Parsing** — PDF extraction, HTML scraping, text chunking with overlap
- **Semantic Similarity** — Vector embeddings + cosine similarity for skill matching
- **RAG Pattern** — Retrieve relevant resume chunks based on job requirements
- **Structured Output** — LangChain `PydanticOutputParser` for reliable JSON responses
- **Error Handling** — Graceful fallbacks, retry logic, model switching

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (or Docker)
- An API key for **Claude** (Anthropic) — that's it! Embeddings run locally for free.

### Option A: Local Install

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/jobfit-ai.git
cd jobfit-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501`

### Option B: Docker (one command)

```bash
# Build the image
docker build -t jobfit-ai .

# Run (pass your API keys via .env file)
docker run -p 8501:8501 --env-file .env jobfit-ai
```

Opens at `http://localhost:8501` — embedding model is pre-downloaded in the image.

### Run Tests

```bash
pytest tests/test_pipeline.py -v
```

Tests validate parser, scraper, and models without needing API keys.

### Embedding Options

By default, embeddings use **HuggingFace sentence-transformers** (`all-MiniLM-L6-v2`) which runs locally for free — no OpenAI key needed. To use OpenAI's higher-quality embeddings instead, set `EMBEDDING_MODEL=openai` in your `.env` file.

---

## 📁 Project Structure

```
jobfit-ai/
├── src/
│   ├── app.py              # Streamlit UI + main application logic
│   ├── scraper.py          # Web scraping module (BeautifulSoup)
│   ├── parser.py           # Document parsing (PDF + text chunking)
│   ├── vectorstore.py      # ChromaDB embedding + retrieval
│   ├── chains.py           # LangChain analysis chains
│   ├── models.py           # Pydantic models for structured output
│   └── config.py           # Configuration + environment variables
├── tests/
│   └── test_pipeline.py    # End-to-end pipeline tests
├── data/
│   └── sample_data.json    # Sample job description + resume for testing
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── Dockerfile              # Docker deployment
├── .dockerignore
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🎬 Demo Walkthrough

Here's exactly what happens when you click **Analyze Match**:

**Step 1 — Input Collection**
The app accepts a job description via URL (scraped with BeautifulSoup) or pasted text, plus a resume via PDF upload (parsed with PyPDF2) or pasted text.

**Step 2 — Document Chunking**
Both documents are split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter` (500 chars, 100 overlap). This preserves context across chunk boundaries.

**Step 3 — Vector Embedding**
All chunks are embedded and stored in an in-memory ChromaDB collection. Default uses free HuggingFace `all-MiniLM-L6-v2` (runs locally, no API key). Optional: OpenAI `text-embedding-3-small` for higher quality.

**Step 4 — Semantic Similarity Search**
For each job description chunk, ChromaDB retrieves the top 3 most similar resume chunks (filtered to resume docs only). This produces ranked match evidence with cosine similarity scores.

**Step 5 — LLM Analysis Chain**
LangChain builds a prompt containing: the job description, resume text, semantic match evidence, and overall similarity score. This is sent to Claude (or GPT-4o) with a `PydanticOutputParser` enforcing structured JSON output.

**Step 6 — Results Dashboard**
The validated `JobAnalysis` Pydantic model is rendered in Streamlit: match score ring, skill pills (matched/missing), experience alignment, gap analysis, numbered resume tweaks, ATS keywords, and interview tips. Results are saved to session history for multi-job comparison.

---

## 🔑 Environment Variables

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=claude          # "claude" or "openai"
EMBEDDING_MODEL=openai        # Used for ChromaDB embeddings
```

---

## 📝 Usage

1. **Enter a job posting** — Paste a URL (LinkedIn, Indeed, etc.) or raw job description text
2. **Upload your resume** — PDF upload or paste as text
3. **Select your model** — Claude (recommended) or OpenAI
4. **Click Analyze** — Get your results in ~10 seconds
5. **Review your report:**
   - Match score (0-100)
   - Matched vs. missing skills
   - Experience alignment
   - Gap analysis
   - Specific resume bullet point rewrites
   - ATS keywords to add
   - Interview preparation tips
6. **Track history** — All analyses saved in session for comparison

---

## 🧪 How Each Technology Is Used

### LangChain
- `ChatAnthropic` / `ChatOpenAI` — LLM wrapper with retry
- `RecursiveCharacterTextSplitter` — Chunks documents with 200-char overlap
- `PromptTemplate` — System + user prompt composition
- `PydanticOutputParser` — Forces structured JSON output
- `LLMChain` — Orchestrates the analysis pipeline

### ChromaDB
- Stores embedded chunks of both resume and job description
- Queries job requirement chunks against resume to find semantic matches
- Cosine similarity scores feed into the final analysis prompt
- Persists across session for multi-job comparison

### Web Scraping
- `requests` fetches raw HTML from job URLs
- `BeautifulSoup4` extracts text content, stripping nav/footer/scripts
- Handles LinkedIn, Indeed, Greenhouse, Lever, and generic job pages
- Falls back gracefully if URL is blocked (user can paste text instead)

### Streamlit
- File uploader for PDF resumes
- Text areas for manual input
- `st.session_state` for analysis history
- Custom CSS for dark-themed dashboard
- `st.columns` layout for results grid
- `st.metric` for score display

---

## 📄 License

MIT — use it, fork it, put it on your resume.

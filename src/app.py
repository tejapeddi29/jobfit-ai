"""
JobFit AI — Streamlit Application
Main entry point for the AI-powered job application tracker.
Now with resume tailoring — generates a rewritten resume for any job.
"""

import sys
import tempfile
from pathlib import Path

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.scraper import scrape_job_url, extract_job_title
from src.parser import extract_pdf_text, chunk_document, extract_sections
from src.vectorstore import (
    build_vectorstore,
    find_semantic_matches,
    compute_overall_similarity,
    format_matches_for_prompt,
)
from src.chains import run_analysis, run_quick_analysis, run_tailoring
from src.resume_generator import generate_resume_docx, generate_markdown_resume, generate_resume_pdf
from src.config import validate_keys, DEFAULT_MODEL


# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobFit AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    .stApp { font-family: 'DM Sans', sans-serif; }
    .main-header { text-align: center; padding: 2rem 0 1rem; }
    .main-header h1 {
        font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(135deg, #e4e4e7, #a5b4fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .main-header p { color: rgba(255,255,255,0.4); font-size: 1rem; max-width: 560px; margin: 0 auto; }
    .score-card { text-align: center; padding: 2rem; }
    .score-number { font-size: 4rem; font-weight: 800; line-height: 1; }
    .score-high { color: #4ade80; }
    .score-mid { color: #fbbf24; }
    .score-low { color: #f87171; }
    .skill-pill-green {
        display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem;
        font-weight: 600; background: rgba(34,197,94,0.12); color: #4ade80;
        border: 1px solid rgba(34,197,94,0.25); margin: 3px 4px 3px 0;
    }
    .skill-pill-red {
        display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem;
        font-weight: 600; background: rgba(239,68,68,0.10); color: #f87171;
        border: 1px solid rgba(239,68,68,0.2); margin: 3px 4px 3px 0;
    }
    .skill-pill-blue {
        display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem;
        font-weight: 600; background: rgba(59,130,246,0.10); color: #60a5fa;
        border: 1px solid rgba(59,130,246,0.2); margin: 3px 4px 3px 0;
    }
    .tweak-card {
        background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.12);
        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    }
    .change-card {
        background: rgba(34,197,94,0.06); border: 1px solid rgba(34,197,94,0.12);
        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    }
    .before-after {
        display: flex; gap: 8px; align-items: flex-start; padding: 8px 0;
    }
    .before-text {
        text-decoration: line-through; color: rgba(255,255,255,0.35); font-size: 0.85rem;
    }
    .after-text {
        color: #4ade80; font-size: 0.85rem; font-weight: 500;
    }
    div[data-testid="stStatusWidget"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State ────────────────────────────────────────────────────
for key in ["history", "result", "tailored", "job_text", "resume_text"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "history" else []


# ── Header ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>◆ JobFit AI</h1>
        <p>Analyze your resume against any job posting — then generate a tailored resume optimized for that role, ready to download.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar: History ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Analysis History")
    if st.session_state.history:
        for i, h in enumerate(st.session_state.history):
            score = h["score"]
            color = "score-high" if score >= 75 else "score-mid" if score >= 50 else "score-low"
            st.markdown(f"**{h['title'][:40]}...** — <span class='{color}'>{score}%</span>", unsafe_allow_html=True)
    else:
        st.caption("No analyses yet. Run your first one!")


# ── Input Section ────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Job Description")
    input_method = st.radio("Input method", ["Paste URL", "Paste Text"], horizontal=True, label_visibility="collapsed")
    if input_method == "Paste URL":
        job_url = st.text_input("Job Posting URL", placeholder="https://linkedin.com/jobs/view/...")
        job_text_raw = ""
    else:
        job_url = ""
        job_text_raw = st.text_area("Job Description", height=250, placeholder="Paste the full job description here...")

with col_right:
    st.markdown("#### Your Resume")
    resume_method = st.radio("Resume input", ["Upload PDF", "Paste Text"], horizontal=True, label_visibility="collapsed")
    if resume_method == "Upload PDF":
        uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
        resume_text_raw = ""
    else:
        uploaded_file = None
        resume_text_raw = st.text_area("Resume Text", height=250, placeholder="Paste your resume content here...")

# ── Model + Action Buttons ───────────────────────────────────────────
col_model, col_analyze, col_tailor, _ = st.columns([2, 2, 2, 2])

with col_model:
    model_choice = st.selectbox(
        "LLM Model",
        ["claude", "openai"],
        index=0 if DEFAULT_MODEL == "claude" else 1,
        format_func=lambda x: "Claude (Anthropic)" if x == "claude" else "GPT-4o (OpenAI)",
    )

with col_analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyze Match", type="primary", use_container_width=True)

with col_tailor:
    st.markdown("<br>", unsafe_allow_html=True)
    tailor_clicked = st.button("✨ Tailor My Resume", type="secondary", use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# SHARED: Extract inputs
# ════════════════════════════════════════════════════════════════════
def get_inputs():
    """Extract and validate job description and resume text from the UI inputs."""
    keys_ok, key_error = validate_keys(model_choice)
    if not keys_ok:
        st.error(key_error)
        return None, None

    # Job description
    if input_method == "Paste URL" and job_url:
        jt, err = scrape_job_url(job_url)
        if err:
            st.warning(f"Scraping issue: {err}")
            return None, None
    elif job_text_raw:
        jt = job_text_raw
    else:
        st.error("Please provide a job description (URL or text).")
        return None, None

    if len(jt.strip()) < 50:
        st.error("Job description is too short. Paste a complete posting.")
        return None, None

    # Resume
    if resume_method == "Upload PDF" and uploaded_file:
        rt, pdf_err = extract_pdf_text(uploaded_file.read())
        if pdf_err:
            st.warning(pdf_err)
            return None, None
    elif resume_text_raw:
        rt = resume_text_raw
    else:
        st.error("Please provide your resume (PDF or text).")
        return None, None

    if len(rt.strip()) < 50:
        st.error("Resume is too short. Provide your full resume.")
        return None, None

    return jt, rt


def run_analysis_pipeline(jt, rt):
    """Run the full analysis pipeline and return the result."""
    with st.status("Analyzing job match...", expanded=True) as status:
        st.write("📄 Extracting documents...")
        job_chunks = chunk_document(jt, doc_type="job_description")
        resume_chunks = chunk_document(rt, doc_type="resume")
        st.write(f"  → {len(job_chunks)} job chunks, {len(resume_chunks)} resume chunks")

        try:
            st.write("🧠 Building vector embeddings with ChromaDB...")
            vectorstore = build_vectorstore(resume_chunks, job_chunks)
            st.write("🔍 Computing semantic similarity...")
            matches = find_semantic_matches(vectorstore, job_chunks)
            similarity = compute_overall_similarity(matches)
            match_text = format_matches_for_prompt(matches)
            st.write(f"  → Overall semantic similarity: {similarity:.1%}")

            st.write(f"🤖 Running analysis with {model_choice.title()}...")
            result = run_analysis(
                job_description=jt, resume=rt,
                semantic_matches=match_text, similarity_score=similarity,
                model_choice=model_choice,
            )
        except Exception as e:
            st.write(f"⚠️ ChromaDB failed ({str(e)[:50]}), using direct analysis...")
            result = run_quick_analysis(job_description=jt, resume=rt, model_choice=model_choice)

        st.session_state.result = result
        st.session_state.job_text = jt
        st.session_state.resume_text = rt
        title = extract_job_title(jt)
        st.session_state.history.insert(0, {"title": title, "score": result.score, "data": result})
        status.update(label="Analysis complete!", state="complete")

    return result


# ════════════════════════════════════════════════════════════════════
# ACTION: Analyze Match
# ════════════════════════════════════════════════════════════════════
if analyze_clicked:
    jt, rt = get_inputs()
    if jt and rt:
        run_analysis_pipeline(jt, rt)

# ════════════════════════════════════════════════════════════════════
# ACTION: Tailor My Resume
# ════════════════════════════════════════════════════════════════════
if tailor_clicked:
    jt, rt = get_inputs()
    if jt and rt:
        # Step 1: Run analysis first if we don't have one
        if not st.session_state.result:
            result = run_analysis_pipeline(jt, rt)
        else:
            result = st.session_state.result

        # Step 2: Run tailoring chain
        with st.status("Tailoring your resume...", expanded=True) as status:
            st.write("✨ Generating tailored resume with AI...")
            st.write("  → Rewriting professional summary...")
            st.write("  → Optimizing skills for ATS...")
            st.write("  → Rewriting experience bullet points...")

            try:
                tailored = run_tailoring(
                    job_description=jt, resume=rt,
                    analysis=result, model_choice=model_choice,
                )
                st.session_state.tailored = tailored
                status.update(label="Resume tailored!", state="complete")
            except Exception as e:
                st.error(f"Tailoring failed: {str(e)[:100]}. Try again or switch models.")
                st.session_state.tailored = None


# ════════════════════════════════════════════════════════════════════
# DISPLAY: Analysis Results
# ════════════════════════════════════════════════════════════════════
result = st.session_state.result

if result:
    st.markdown("---")

    # If we also have a tailored resume, show tabs
    if st.session_state.tailored:
        tab_analysis, tab_tailored = st.tabs(["📊 Match Analysis", "📄 Tailored Resume"])
    else:
        tab_analysis = st.container()
        tab_tailored = None

    with tab_analysis:
        # Score Hero
        score_col, summary_col = st.columns([1, 3])
        with score_col:
            color_class = "score-high" if result.score >= 75 else "score-mid" if result.score >= 50 else "score-low"
            label = "Strong Match" if result.score >= 75 else "Moderate Match" if result.score >= 50 else "Needs Work"
            st.markdown(f"""
                <div class="score-card">
                    <div class="score-number {color_class}">{result.score}</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase;">{label}</div>
                </div>
            """, unsafe_allow_html=True)
        with summary_col:
            st.markdown("### Assessment")
            st.write(result.summary)

        # Skills
        sl, sr = st.columns(2)
        with sl:
            st.markdown("#### ✅ Matched Skills")
            st.markdown(" ".join(f'<span class="skill-pill-green">{s}</span>' for s in result.matched_skills), unsafe_allow_html=True)
        with sr:
            st.markdown("#### ❌ Missing Skills")
            st.markdown(" ".join(f'<span class="skill-pill-red">{s}</span>' for s in result.missing_skills), unsafe_allow_html=True)

        # Experience + Gaps
        el, gl = st.columns(2)
        with el:
            st.markdown("#### ⚡ Experience Alignment")
            for item in result.matched_experience:
                st.markdown(f"- {item}")
        with gl:
            st.markdown("#### ⚠️ Gaps to Address")
            for item in result.gaps:
                st.markdown(f"- {item}")

        # Tweaks
        st.markdown("#### ✏️ Resume Tweaks")
        for i, tweak in enumerate(result.resume_tweaks, 1):
            st.markdown(f'<div class="tweak-card"><strong style="color: #818cf8;">#{i}</strong> &nbsp; {tweak}</div>', unsafe_allow_html=True)

        # Keywords + Tips
        kl, tl = st.columns(2)
        with kl:
            st.markdown("#### 🔑 Keywords to Add")
            st.markdown(" ".join(f'<span class="skill-pill-blue">{k}</span>' for k in result.keywords_to_add), unsafe_allow_html=True)
        with tl:
            st.markdown("#### 💡 Interview Tips")
            for tip in result.interview_tips:
                st.markdown(f"- {tip}")

        # If no tailored resume yet, show the button again
        if not st.session_state.tailored:
            st.markdown("---")
            st.info("💡 **Ready to go further?** Click **✨ Tailor My Resume** above to generate a fully rewritten resume optimized for this role.")

        # Export analysis
        st.markdown("---")
        st.download_button("📥 Download Analysis (JSON)", data=result.model_dump_json(indent=2),
                           file_name="jobfit_analysis.json", mime="application/json")


    # ════════════════════════════════════════════════════════════════
    # DISPLAY: Tailored Resume
    # ════════════════════════════════════════════════════════════════
    if tab_tailored and st.session_state.tailored:
        tailored = st.session_state.tailored

        with tab_tailored:
            st.markdown("### ✨ Your Tailored Resume")
            st.caption(f"Optimized for: **{tailored.target_title}**")

            # Changes summary at the top
            st.markdown("#### 🔄 What Was Changed")
            for i, change in enumerate(tailored.changes_made, 1):
                st.markdown(f'<div class="change-card"><strong style="color: #4ade80;">#{i}</strong> &nbsp; {change}</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Professional Summary
            st.markdown("#### Professional Summary")
            st.info(tailored.summary)

            # Skills
            st.markdown("#### Technical Skills")
            st.markdown(" ".join(f'<span class="skill-pill-green">{s}</span>' for s in tailored.skills), unsafe_allow_html=True)

            # Experience with before/after
            st.markdown("#### Experience")
            for exp in tailored.experience:
                st.markdown(f"**{exp.title}** | {exp.company} | {exp.dates}")
                for bullet in exp.bullets:
                    with st.expander(f"✏️ {bullet.rewritten[:70]}...", expanded=False):
                        st.markdown(f'<div class="before-text">Before: {bullet.original}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="after-text">After: {bullet.rewritten}</div>', unsafe_allow_html=True)
                        st.caption(f"💡 {bullet.reason}")
                st.markdown("")

            # Projects
            if tailored.projects:
                st.markdown("#### Projects")
                for proj in tailored.projects:
                    st.markdown(f"- {proj}")

            # Education + Certs
            el2, cl2 = st.columns(2)
            with el2:
                st.markdown("#### Education")
                for edu in tailored.education:
                    st.markdown(f"- {edu}")
            with cl2:
                st.markdown("#### Certifications")
                for cert in tailored.certifications:
                    st.markdown(f"- {cert}")

            # ── Download Options ────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📥 Download Tailored Resume")

            dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

            # Markdown download
            md_content = generate_markdown_resume(tailored)
            with dl_col1:
                st.download_button(
                    "📄 Download (.md)",
                    data=md_content,
                    file_name="tailored_resume.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            # JSON download
            with dl_col2:
                st.download_button(
                    "📋 Download (.json)",
                    data=tailored.model_dump_json(indent=2),
                    file_name="tailored_resume.json",
                    mime="application/json",
                    use_container_width=True,
                )

            # DOCX download
            with dl_col3:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                        docx_path, docx_err = generate_resume_docx(tailored, tmp.name)

                    if docx_err:
                        st.warning(docx_err)

                    with open(docx_path, "rb") as f:
                        file_bytes = f.read()

                    ext = Path(docx_path).suffix
                    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if ext == ".docx" else "text/plain"
                    st.download_button(
                        f"📝 Download ({ext})",
                        data=file_bytes,
                        file_name=f"tailored_resume{ext}",
                        mime=mime,
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"File generation failed: {str(e)[:60]}")
                    # Always offer the text fallback
                    st.download_button(
                        "📄 Download (.txt)",
                        data=md_content,
                        file_name="tailored_resume.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

            # PDF download
            with dl_col4:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        pdf_path, pdf_err = generate_resume_pdf(tailored, tmp.name)

                    if not pdf_err and pdf_path:
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()

                        st.download_button(
                            "📕 Download (.pdf)",
                            data=pdf_bytes,
                            file_name="tailored_resume.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    else:
                        st.warning(pdf_err or "PDF generation failed")
                except Exception as e:
                    st.warning(f"PDF failed: {str(e)[:60]}")

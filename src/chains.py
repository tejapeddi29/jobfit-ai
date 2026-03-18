"""
LangChain Chains module — orchestrates the LLM analysis pipeline.

This module builds the prompt templates, output parsers, and LLM chains
that power the job-resume analysis. Supports both Claude and OpenAI.
"""

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.models import JobAnalysis, TailoredResume
from src.config import (
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    CLAUDE_MODEL_NAME,
    OPENAI_MODEL_NAME,
)


# ── Output Parser ────────────────────────────────────────────────────
parser = PydanticOutputParser(pydantic_object=JobAnalysis)


# ── Prompt Template ──────────────────────────────────────────────────
SYSTEM_TEMPLATE = """You are an expert career coach and ATS (Applicant Tracking System) specialist.
You analyze job descriptions against resumes to provide actionable, specific guidance.

You will receive:
1. A job description
2. A resume
3. Semantic similarity evidence showing which resume sections match which job requirements

Analyze the match thoroughly and provide your assessment.

{format_instructions}

IMPORTANT RULES:
- Score HONESTLY. A 90+ means near-perfect match. Most candidates score 40-75.
- Be SPECIFIC in resume tweaks — don't say "add more detail", say exactly what to write.
- Keywords should be exact terms from the job description, not synonyms.
- Interview tips should be specific to THIS role, not generic advice.
- If the semantic matches show low similarity, reflect that in the score.
"""

HUMAN_TEMPLATE = """
## JOB DESCRIPTION:
{job_description}

## RESUME:
{resume}

## SEMANTIC SIMILARITY EVIDENCE:
{semantic_matches}

Overall semantic similarity score: {similarity_score}

Analyze the match and provide your structured assessment.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", HUMAN_TEMPLATE),
    ]
)


# ── LLM Factory ──────────────────────────────────────────────────────
def get_llm(model_choice: str):
    """
    Create the appropriate LangChain LLM wrapper.

    Args:
        model_choice: "claude" or "openai"

    Returns:
        A ChatAnthropic or ChatOpenAI instance.
    """
    if model_choice == "claude":
        return ChatAnthropic(
            model=CLAUDE_MODEL_NAME,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0.3,
            max_tokens=2000,
        )
    else:
        return ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=2000,
        )


# ── Analysis Chain ───────────────────────────────────────────────────
def run_analysis(
    job_description: str,
    resume: str,
    semantic_matches: str,
    similarity_score: float,
    model_choice: str = "claude",
) -> JobAnalysis:
    """
    Run the full LangChain analysis chain.

    This:
    1. Builds the prompt with all evidence
    2. Sends to Claude or OpenAI
    3. Parses the structured JSON output into a Pydantic model
    4. Validates the response

    Args:
        job_description: Full job description text.
        resume: Full resume text.
        semantic_matches: Formatted string of ChromaDB similarity results.
        similarity_score: Overall cosine similarity (0-1).
        model_choice: "claude" or "openai".

    Returns:
        A validated JobAnalysis Pydantic object.

    Raises:
        ValueError: If the LLM output cannot be parsed.
    """
    llm = get_llm(model_choice)

    # Format the prompt
    formatted_prompt = prompt.format_messages(
        format_instructions=parser.get_format_instructions(),
        job_description=job_description[:4000],  # Truncate to stay within context
        resume=resume[:3000],
        semantic_matches=semantic_matches,
        similarity_score=f"{similarity_score:.2%}",
    )

    # Call the LLM
    response = llm.invoke(formatted_prompt)

    # Parse into structured output
    try:
        result = parser.parse(response.content)
    except Exception as e:
        # Retry with a simpler extraction if Pydantic parsing fails
        import json
        import re

        # Try to extract JSON from the response
        content = response.content
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            raw = json.loads(json_match.group())
            result = JobAnalysis(**raw)
        else:
            raise ValueError(
                f"Could not parse LLM response into structured output: {str(e)}"
            )

    return result


def run_quick_analysis(
    job_description: str,
    resume: str,
    model_choice: str = "claude",
) -> JobAnalysis:
    """
    Simplified analysis without ChromaDB (fallback if embedding fails).
    Sends job + resume directly to the LLM without semantic evidence.
    """
    llm = get_llm(model_choice)

    simple_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            (
                "human",
                "## JOB DESCRIPTION:\n{job_description}\n\n"
                "## RESUME:\n{resume}\n\n"
                "Analyze the match and provide your structured assessment.",
            ),
        ]
    )

    formatted = simple_prompt.format_messages(
        format_instructions=parser.get_format_instructions(),
        job_description=job_description[:4000],
        resume=resume[:3000],
    )

    response = llm.invoke(formatted)

    try:
        return parser.parse(response.content)
    except Exception:
        import json, re

        content = response.content
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return JobAnalysis(**json.loads(json_match.group()))
        raise


# ── Resume Tailoring Chain ───────────────────────────────────────────

tailor_parser = PydanticOutputParser(pydantic_object=TailoredResume)

TAILOR_SYSTEM_TEMPLATE = """You are an expert resume writer and ATS optimization specialist.
Your job is to REWRITE a candidate's resume to be perfectly tailored for a specific job posting.

You will receive:
1. A job description (the TARGET role)
2. The candidate's current resume
3. An analysis showing matched/missing skills and gaps

REWRITING RULES:
- NEVER fabricate experience or skills the candidate doesn't have
- DO reword existing bullets to use the exact terminology from the job description
- DO reorder skills to put job-relevant ones first
- DO add skills the candidate likely has based on their experience but didn't list
- DO write a new professional summary targeted at this specific role
- DO quantify achievements where the original didn't (use reasonable estimates based on context)
- DO incorporate ATS keywords naturally into bullet points
- Every rewritten bullet should start with a strong action verb
- Keep bullet points concise (1 line, max 15 words)
- The resume should read as if the candidate wrote it specifically for THIS job

{format_instructions}
"""

TAILOR_HUMAN_TEMPLATE = """
## TARGET JOB DESCRIPTION:
{job_description}

## CANDIDATE'S CURRENT RESUME:
{resume}

## MATCH ANALYSIS:
Matched skills: {matched_skills}
Missing skills: {missing_skills}
Gaps: {gaps}
Keywords to add: {keywords}

Rewrite the entire resume tailored for this specific job. Preserve the candidate's real experience
but optimize every line for this role.
"""

tailor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", TAILOR_SYSTEM_TEMPLATE),
        ("human", TAILOR_HUMAN_TEMPLATE),
    ]
)


def run_tailoring(
    job_description: str,
    resume: str,
    analysis: JobAnalysis,
    model_choice: str = "claude",
) -> TailoredResume:
    """
    Run the resume tailoring chain.

    Takes the original resume + job description + analysis results,
    and produces a fully rewritten resume optimized for the target role.
    """
    llm = get_llm(model_choice)
    # Use higher max_tokens for the full resume rewrite
    if model_choice == "claude":
        llm = ChatAnthropic(
            model=CLAUDE_MODEL_NAME,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0.4,
            max_tokens=4000,
        )
    else:
        llm = ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.4,
            max_tokens=4000,
        )

    formatted = tailor_prompt.format_messages(
        format_instructions=tailor_parser.get_format_instructions(),
        job_description=job_description[:4000],
        resume=resume[:4000],
        matched_skills=", ".join(analysis.matched_skills),
        missing_skills=", ".join(analysis.missing_skills),
        gaps="; ".join(analysis.gaps),
        keywords=", ".join(analysis.keywords_to_add),
    )

    response = llm.invoke(formatted)

    try:
        return tailor_parser.parse(response.content)
    except Exception:
        import json, re
        content = response.content
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return TailoredResume(**json.loads(json_match.group()))
        raise

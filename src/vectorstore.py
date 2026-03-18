"""
Vector Store module — manages ChromaDB for semantic embedding and retrieval.

This module:
1. Embeds resume and job description chunks into a shared ChromaDB collection
2. Performs semantic similarity search (job reqs → resume matches)
3. Computes an overall semantic similarity score

Supports two embedding backends:
- OpenAI (text-embedding-3-small) — higher quality, requires API key
- HuggingFace (all-MiniLM-L6-v2) — free, runs locally, no API key needed
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_NAME,
    FREE_EMBEDDING_NAME,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
)


def get_embedding_function():
    """
    Initialize the embedding model based on config.

    Returns OpenAI embeddings if EMBEDDING_MODEL=openai,
    otherwise returns free HuggingFace sentence-transformers (no API key).
    """
    if EMBEDDING_MODEL == "openai" and OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_NAME,
            openai_api_key=OPENAI_API_KEY,
        )
    else:
        # Free local embeddings — no API key required
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=FREE_EMBEDDING_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def build_vectorstore(
    resume_chunks: list[dict],
    job_chunks: list[dict],
) -> Chroma:
    """
    Build an in-memory ChromaDB vector store from resume and job chunks.

    Args:
        resume_chunks: List of dicts with 'text' and 'metadata' from parser.
        job_chunks: List of dicts with 'text' and 'metadata' from parser.

    Returns:
        A LangChain Chroma instance with all documents embedded.
    """
    embeddings = get_embedding_function()

    # Convert to LangChain Document objects
    documents = []
    for chunk in resume_chunks:
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"],
            )
        )
    for chunk in job_chunks:
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"],
            )
        )

    # Build in-memory Chroma store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    return vectorstore


def find_semantic_matches(
    vectorstore: Chroma,
    job_chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    For each job description chunk, find the most semantically similar
    resume chunks. This reveals which parts of the resume align with
    which job requirements.

    Args:
        vectorstore: The populated Chroma instance.
        job_chunks: The job description chunks to query with.
        top_k: Number of resume matches per job chunk.

    Returns:
        List of match dicts with job_chunk, resume_chunk, and similarity.
    """
    matches = []

    for jc in job_chunks:
        # Search for similar resume chunks (filter to resume docs only)
        results = vectorstore.similarity_search_with_relevance_scores(
            query=jc["text"],
            k=top_k,
            filter={"doc_type": "resume"},
        )

        for doc, score in results:
            matches.append(
                {
                    "job_chunk": jc["text"],
                    "resume_chunk": doc.page_content,
                    "similarity_score": round(score, 4),
                }
            )

    # Sort by similarity score descending
    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    return matches


def compute_overall_similarity(matches: list[dict]) -> float:
    """
    Compute a single semantic similarity score from all matches.

    Uses the average of the top match per job chunk to avoid
    double-counting resume sections.
    """
    if not matches:
        return 0.0

    # Group by job chunk, take best match per chunk
    seen_job_chunks = {}
    for m in matches:
        key = m["job_chunk"][:100]  # Use first 100 chars as key
        if key not in seen_job_chunks:
            seen_job_chunks[key] = m["similarity_score"]

    scores = list(seen_job_chunks.values())
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def format_matches_for_prompt(matches: list[dict], max_matches: int = 8) -> str:
    """
    Format the top semantic matches into a string for the LLM prompt.
    This gives the LLM evidence of where the resume does/doesn't align.
    """
    top_matches = matches[:max_matches]
    lines = []

    for i, m in enumerate(top_matches, 1):
        lines.append(
            f"Match {i} (similarity: {m['similarity_score']:.2f}):\n"
            f"  Job requires: {m['job_chunk'][:200]}\n"
            f"  Resume has: {m['resume_chunk'][:200]}"
        )

    return "\n\n".join(lines)

"""Question answering orchestration for the RAG application."""

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from src.retrievers import retrieve_from_sources
from utils.config import Settings
from utils.prompts import ANSWER_PROMPT, REWRITE_PROMPT


def create_chat_model(settings: Settings) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )


def format_context(documents: list[Document]) -> str:
    """Include each chunk's source in the context Gemini receives."""
    sections = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")
        citation = f"{source}, page {page}" if page is not None else source
        sections.append(f"[source: {citation}]\n{document.page_content}")
    return "\n\n---\n\n".join(sections)


def answer_question(
    store: PineconeVectorStore,
    model: ChatGoogleGenerativeAI,
    question: str,
    history: list[BaseMessage] | None = None,
    settings: Settings | None = None,
    external_sources: list[str] | None = None,
) -> tuple[str, list[Document]]:
    """Retrieve relevant chunks and answer only from those chunks.

    Args:
        store: Pinecone vector store for indexed documents.
        model: The LLM chat model.
        question: The user's question.
        history: Optional chat history for context-aware retrieval.
        settings: Application settings (needed for external retrievers).
        external_sources: List of external source names to query
            (e.g. ["web", "pubmed", "arxiv", "wikipedia", "youtube"]).
            If None or empty, only Pinecone is used.

    Returns:
        Tuple of (answer_text, list_of_retrieved_documents).
    """
    history = history or []
    retrieval_query = question

    if history:
        retrieval_query = (REWRITE_PROMPT | model | StrOutputParser()).invoke(
            {"chat_history": history, "question": question}
        )

    # 1. Retrieve from Pinecone (indexed documents)
    documents = store.similarity_search(retrieval_query, k=4)

    # 2. Retrieve from external sources if requested
    if external_sources and settings:
        external_docs = retrieve_from_sources(
            query=retrieval_query,
            settings=settings,
            sources=external_sources,
            k=4,
        )
        documents.extend(external_docs)

    answer = (ANSWER_PROMPT | model | StrOutputParser()).invoke(
        {"context": format_context(documents), "chat_history": history, "question": question}
    )
    return answer, documents

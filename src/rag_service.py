"""Question answering orchestration for the RAG application."""

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

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
) -> tuple[str, list[Document]]:
    """Retrieve relevant chunks and answer only from those chunks."""
    history = history or []
    retrieval_query = question

    if history:
        retrieval_query = (REWRITE_PROMPT | model | StrOutputParser()).invoke(
            {"chat_history": history, "question": question}
        )

    documents = store.similarity_search(retrieval_query, k=4)
    answer = (ANSWER_PROMPT | model | StrOutputParser()).invoke(
        {"context": format_context(documents), "chat_history": history, "question": question}
    )
    return answer, documents

"""Command-line entry point for NeuroOceans RAG."""

import argparse
import sys

from src.chat_history import RedisChatHistory
from src.loaders import load_documents, split_documents
from src.rag_service import answer_question, create_chat_model
from src.vector_store import open_vector_store
from utils.config import Settings


AVAILABLE_SOURCES = ["web", "youtube", "pubmed", "arxiv", "wikipedia"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroOceans RAG – Gemini RAG with external retrieval sources"
    )
    parser.add_argument("--ingest", nargs="*", default=[], help="Files or directories to index")
    parser.add_argument("--urls", nargs="*", default=[], help="URLs to index")
    parser.add_argument("--ask", help="Ask one question")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start a persistent Redis-backed conversation",
    )
    parser.add_argument(
        "--session-id",
        default="cli",
        help="Chat session ID (default: cli)",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear this session before chatting",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create the Pinecone index if needed",
    )
    parser.add_argument(
        "--source",
        nargs="*",
        default=[],
        choices=AVAILABLE_SOURCES,
        help=(
            "External retrieval sources to query alongside Pinecone. "
            f"Options: {', '.join(AVAILABLE_SOURCES)}. "
            "Omit to use only indexed documents."
        ),
    )
    return parser.parse_args()


def run_chat(
    settings: Settings,
    store,
    session_id: str,
    clear_history: bool,
    external_sources: list[str] | None,
) -> None:
    history = RedisChatHistory(settings.redis_url, session_id)
    if clear_history:
        history.clear()
        print("Cleared chat history")

    source_label = (
        f"external sources: {', '.join(external_sources)}"
        if external_sources
        else "indexed documents only"
    )
    model = create_chat_model(settings)
    print(f"NeuroOceans RAG chat ({source_label}). Type 'quit' to exit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            return
        if not question:
            continue
        answer, documents = answer_question(
            store,
            model,
            question,
            history.messages,
            settings=settings,
            external_sources=external_sources or None,
        )
        history.add_user_message(question)
        history.add_ai_message(answer)
        print(f"AI: {answer}")
        print("Sources:")
        for doc in documents:
            retriever = doc.metadata.get("retriever", "pinecone")
            source = doc.metadata.get("source", "unknown")
            print(f"  - {source} ({retriever})")


def main() -> None:
    args = parse_args()
    if not (args.ingest or args.urls or args.ask or args.chat):
        print(
            "Choose an action with --ingest, --ask, or --chat. "
            "See --help for details."
        )
        return

    external_sources = args.source if args.source else None

    try:
        settings = Settings.from_env()
        store = open_vector_store(settings, create_index=args.create_index)

        if args.ingest or args.urls:
            documents = load_documents(args.ingest, args.urls)
            if not documents:
                raise RuntimeError("No supported documents were loaded.")
            chunk_ids = store.add_documents(split_documents(documents))
            print(f"Indexed {len(chunk_ids)} chunks in '{settings.pinecone_index}'.")

        if args.ask:
            model = create_chat_model(settings)
            answer, documents = answer_question(
                store,
                model,
                args.ask,
                settings=settings,
                external_sources=external_sources,
            )
            print(f"\nQ: {args.ask}\nA: {answer}\n\nRetrieved sources:")
            for doc in documents:
                retriever = doc.metadata.get("retriever", "pinecone")
                source = doc.metadata.get("source", "unknown")
                print(f"  - {source} ({retriever})")

        if args.chat:
            run_chat(
                settings,
                store,
                args.session_id,
                args.clear_history,
                external_sources,
            )
    except RuntimeError as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()

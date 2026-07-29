"""Command-line entry point for NeuroOceans RAG."""

import argparse
import sys

from src.chat_history import RedisChatHistory
from src.loaders import load_documents, split_documents
from src.rag_service import answer_question, create_chat_model
from src.vector_store import open_vector_store
from utils.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini RAG with Pinecone and Redis")
    parser.add_argument("--ingest", nargs="*", default=[], help="Files or directories to index")
    parser.add_argument("--urls", nargs="*", default=[], help="URLs to index")
    parser.add_argument("--ask", help="Ask one question")
    parser.add_argument("--chat", action="store_true", help="Start a persistent Redis-backed conversation")
    parser.add_argument("--session-id", default="cli", help="Chat session ID (default: cli)")
    parser.add_argument("--clear-history", action="store_true", help="Clear this session before chatting")
    parser.add_argument("--create-index", action="store_true", help="Create the Pinecone index if needed")
    return parser.parse_args()


def run_chat(settings: Settings, store, session_id: str, clear_history: bool) -> None:
    history = RedisChatHistory(settings.redis_url, session_id)
    if clear_history:
        history.clear()
        print("Cleared chat history")

    model = create_chat_model(settings)
    print("NeuroOceans RAG chat. Type 'quit' to exit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            return
        if not question:
            continue
        answer, _ = answer_question(store, model, question, history.messages)
        history.add_user_message(question)
        history.add_ai_message(answer)
        print(f"AI: {answer}")


def main() -> None:
    args = parse_args()
    if not (args.ingest or args.urls or args.ask or args.chat):
        print("Choose an action with --ingest, --ask, or --chat. See --help for details.")
        return

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
            answer, documents = answer_question(store, create_chat_model(settings), args.ask)
            print(f"\nQ: {args.ask}\nA: {answer}\n\nRetrieved sources:")
            for document in documents:
                print(f"- {document.metadata.get('source', 'unknown')}")

        if args.chat:
            run_chat(settings, store, args.session_id, args.clear_history)
    except RuntimeError as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()

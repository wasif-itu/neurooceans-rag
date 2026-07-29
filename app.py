"""A small Gradio interface for indexing documents and asking questions."""

from functools import lru_cache
from pathlib import Path

import gradio as gr

from src.chat_history import RedisChatHistory
from src.loaders import load_documents, split_documents
from src.rag_service import answer_question, create_chat_model
from src.vector_store import open_vector_store
from utils.config import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_store():
    return open_vector_store(get_settings())


@lru_cache(maxsize=1)
def get_model():
    return create_chat_model(get_settings())


def index_content(files: list[str] | None, urls: str) -> str:
    """Index files and newline-separated URLs selected in the interface."""
    try:
        file_paths = [str(Path(file)) for file in files or []]
        url_list = [url.strip() for url in urls.splitlines() if url.strip()]
        if not file_paths and not url_list:
            return "Choose at least one file or enter a URL."

        documents = load_documents(file_paths, url_list)
        if not documents:
            return "No supported content could be loaded."

        chunks = split_documents(documents)
        get_store().add_documents(chunks)
        return f"Indexed {len(chunks)} chunk(s) in Pinecone."
    except Exception as error:
        return f"Indexing failed: {error}"


def ask_question(
    question: str,
    session_id: str,
    conversation: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]], str]:
    """Answer a question and retain the exchange in its Redis session."""
    if not question.strip():
        return "", conversation, "Ask a question to begin."

    try:
        settings = get_settings()
        history = RedisChatHistory(settings.redis_url, session_id or "gradio")
        answer, documents = answer_question(get_store(), get_model(), question, history.messages)
        history.add_user_message(question)
        history.add_ai_message(answer)

        updated_conversation = conversation + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        sources = "\n".join(f"- {document.metadata.get('source', 'unknown')}" for document in documents)
        return "", updated_conversation, sources or "No matching sources returned."
    except Exception as error:
        return "", conversation, f"Request failed: {error}"


def clear_chat(session_id: str) -> tuple[list[dict[str, str]], str]:
    try:
        history = RedisChatHistory(get_settings().redis_url, session_id or "gradio")
        history.clear()
        return [], "Chat history cleared."
    except Exception as error:
        return [], f"Could not clear history: {error}"


with gr.Blocks(title="NeuroOceans RAG") as demo:
    gr.Markdown("# NeuroOceans RAG\nAsk grounded questions over your documents with Gemini, Pinecone, and Redis.")

    with gr.Tab("Ask"):
        session_id = gr.Textbox(label="Session ID", value="gradio", info="Reuse an ID to continue a Redis-backed conversation.")
        chatbot = gr.Chatbot(label="Conversation", height=420)
        question = gr.Textbox(label="Question", placeholder="Ask about indexed documents...", lines=2)
        with gr.Row():
            ask_button = gr.Button("Ask", variant="primary")
            clear_button = gr.Button("Clear chat")
        sources = gr.Markdown("Sources will appear here.")

        ask_button.click(ask_question, [question, session_id, chatbot], [question, chatbot, sources])
        question.submit(ask_question, [question, session_id, chatbot], [question, chatbot, sources])
        clear_button.click(clear_chat, [session_id], [chatbot, sources])

    with gr.Tab("Index documents"):
        gr.Markdown("Upload PDF, TXT, Markdown, or CSV files. You can also paste one URL per line.")
        files = gr.File(label="Files", file_count="multiple", type="filepath")
        urls = gr.Textbox(label="URLs", placeholder="https://example.com/page\nhttps://example.com/another-page", lines=4)
        index_button = gr.Button("Index content", variant="primary")
        index_status = gr.Markdown()
        index_button.click(index_content, [files, urls], index_status)


if __name__ == "__main__":
    demo.launch()

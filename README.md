# NeuroOceans RAG

An evolving retrieval-augmented generation workspace for exploring practical document intelligence workflows. The project brings together Gemini, Pinecone, Redis, and LangChain in a compact Python application that can ingest source material, retrieve relevant context, and produce grounded responses with citations.

## Highlights

- Gemini for response generation and document embeddings
- Pinecone for durable vector search
- Redis-backed chat sessions for follow-up questions
- PDF, TXT, Markdown, CSV, and URL ingestion
- Source-aware responses that point back to the retrieved material

## Architecture

```text
Documents / URLs
       │
       ▼
Load and chunk documents ──► Gemini embeddings ──► Pinecone
                                                   │
User question ──► retrieve relevant chunks ───────┘
       │
       ▼
Gemini response with citations
       │
       ▼
Redis conversation history
```

## Getting started

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and add your Gemini and Pinecone credentials:

```bash
cp .env.example .env
```

Start Redis locally:

```bash
docker compose up -d redis
```

## Usage

Create the Pinecone index on the first ingestion, then index a folder or file:

```bash
python main.py --create-index --ingest ./docs
```

Ask a one-off question:

```bash
python main.py --ask "What does the document say about refunds?"
```

Start a persistent chat session:

```bash
python main.py --chat --session-id wasif
```

Use `--clear-history` to begin a session without prior Redis messages.

## Repository layout

```text
src/       document loading, vector storage, chat memory, and RAG orchestration
utils/     settings and prompt templates
main.py    command-line entry point
```

## Next areas of focus

- richer document preprocessing and metadata filters
- evaluation datasets for retrieval quality
- streaming responses and a lightweight user interface
- support for additional retrieval strategies and sources

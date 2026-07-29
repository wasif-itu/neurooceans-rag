"""Document loading and chunking helpers."""

from pathlib import Path

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOADERS = {".pdf": PyPDFLoader, ".csv": CSVLoader, ".txt": TextLoader, ".md": TextLoader}


def load_documents(paths: list[str], urls: list[str]) -> list[Document]:
    """Load supported local files and web pages into LangChain documents."""
    documents: list[Document] = []

    for raw_path in paths:
        path = Path(raw_path)
        files = list(path.rglob("*")) if path.is_dir() else [path]
        for file_path in files:
            loader_class = LOADERS.get(file_path.suffix.lower())
            if loader_class is None:
                continue
            try:
                loaded = loader_class(str(file_path)).load()
                documents.extend(loaded)
                print(f"Loaded {file_path} ({len(loaded)} document(s))")
            except Exception as error:
                print(f"Could not load {file_path}: {error}")

    if urls:
        loaded = WebBaseLoader(urls).load()
        documents.extend(loaded)
        print(f"Loaded {len(loaded)} document(s) from {len(urls)} URL(s)")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks")
    return chunks

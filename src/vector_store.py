"""Gemini embeddings and Pinecone index management."""

import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from utils.config import Settings


def create_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dimension,
    )


def open_vector_store(settings: Settings, create_index: bool = False) -> PineconeVectorStore:
    """Open the configured Pinecone index, optionally creating it once."""
    client = Pinecone(api_key=settings.pinecone_api_key)
    known_indexes = client.list_indexes().names()

    if settings.pinecone_index not in known_indexes:
        if not create_index:
            raise RuntimeError(
                f"Pinecone index '{settings.pinecone_index}' does not exist. "
                "Run the ingest command once with --create-index."
            )
        print(f"Creating Pinecone index '{settings.pinecone_index}'...")
        client.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        while not client.describe_index(settings.pinecone_index).status["ready"]:
            time.sleep(1)

    return PineconeVectorStore(
        index=client.Index(settings.pinecone_index),
        embedding=create_embeddings(settings),
        namespace=settings.pinecone_namespace,
    )

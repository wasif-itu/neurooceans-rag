"""Environment-based application settings."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    pinecone_api_key: str
    pinecone_index: str
    redis_url: str
    gemini_model: str
    embedding_model: str
    embedding_dimension: int
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        def required(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise RuntimeError(f"Missing {name}. Copy .env.example to .env and set its value.")
            return value

        return cls(
            google_api_key=required("GOOGLE_API_KEY"),
            pinecone_api_key=required("PINECONE_API_KEY"),
            pinecone_index=required("PINECONE_INDEX"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
            embedding_dimension=int(os.getenv("GEMINI_EMBEDDING_DIMENSION", "768")),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "documents"),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        )

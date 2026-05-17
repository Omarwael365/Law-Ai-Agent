"""
Embedding Factory — Provides embeddings from OpenAI or HuggingFace.
Supports text-embedding-3-large (OpenAI) and BAAI/bge-base-en-v1.5 (HuggingFace).
"""

from src.config import (
    DEFAULT_EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    HUGGINGFACE_EMBEDDING_MODEL,
)


def get_embeddings(provider: str = None):
    """
    Factory function to get the appropriate embedding model.

    Args:
        provider: 'openai' or 'huggingface'. Defaults to config setting.

    Returns:
        A LangChain-compatible embeddings instance.
    """
    provider = provider or DEFAULT_EMBEDDING_PROVIDER

    if provider == "openai":
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            print("Warning: OpenAI API key not set. Falling back to HuggingFace embeddings.")
            provider = "huggingface"
        else:
            from langchain_openai import OpenAIEmbeddings
            print(f"  Using OpenAI embeddings: {OPENAI_EMBEDDING_MODEL}")
            return OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY,
            )

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print(f"  Using HuggingFace embeddings: {HUGGINGFACE_EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(f"Unknown embedding provider: {provider}. Use 'openai' or 'huggingface'.")

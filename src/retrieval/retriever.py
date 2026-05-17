"""
Retriever — FAISS-based retriever with configurable top-k and metadata filtering.
Supports searching case law and contracts separately or together.
"""

from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.config import (
    CASE_LAW_INDEX_DIR,
    CONTRACTS_INDEX_DIR,
    RETRIEVAL_TOP_K,
)
from src.embeddings.embeddings import get_embeddings


def load_vectorstore(index_dir: str, embedding_provider: str = None) -> Optional[FAISS]:
    """Load a persisted FAISS vector store from disk."""
    index_path = Path(index_dir)
    if not index_path.exists():
        print(f"Warning: Vector store not found at {index_path}. Run ingestion first.")
        return None

    embeddings = get_embeddings(provider=embedding_provider)
    vectorstore = FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )
    print(f"  Loaded vector store from {index_path} ({vectorstore.index.ntotal} vectors)")
    return vectorstore


def get_case_law_retriever(top_k: int = RETRIEVAL_TOP_K, embedding_provider: str = None):
    """Get a retriever for case law documents."""
    vectorstore = load_vectorstore(str(CASE_LAW_INDEX_DIR), embedding_provider)
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def get_contracts_retriever(top_k: int = RETRIEVAL_TOP_K, embedding_provider: str = None):
    """Get a retriever for contract documents."""
    vectorstore = load_vectorstore(str(CONTRACTS_INDEX_DIR), embedding_provider)
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def search_case_law(query: str, top_k: int = RETRIEVAL_TOP_K, embedding_provider: str = None) -> List[Document]:
    """Search case law documents and return relevant chunks."""
    vectorstore = load_vectorstore(str(CASE_LAW_INDEX_DIR), embedding_provider)
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=top_k)


def search_contracts(query: str, top_k: int = RETRIEVAL_TOP_K, embedding_provider: str = None) -> List[Document]:
    """Search contract documents and return relevant chunks."""
    vectorstore = load_vectorstore(str(CONTRACTS_INDEX_DIR), embedding_provider)
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=top_k)


def search_all(query: str, top_k: int = RETRIEVAL_TOP_K, embedding_provider: str = None) -> List[Document]:
    """Search both case law and contracts, returning combined results ranked by relevance."""
    case_law_results = search_case_law(query, top_k, embedding_provider)
    contract_results = search_contracts(query, top_k, embedding_provider)

    # Combine and return top_k total results
    combined = case_law_results + contract_results
    return combined[:top_k * 2]  # Return results from both stores

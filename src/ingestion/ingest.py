"""
Ingestion Orchestrator — Full pipeline: load → split → embed → store.
Creates and persists FAISS vector stores for case law and contracts.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    CASE_LAW_INDEX_DIR,
    CONTRACTS_INDEX_DIR,
    VECTORSTORE_DIR,
)
from src.ingestion.loader import load_all_documents
from src.ingestion.splitter import split_documents
from src.embeddings.embeddings import get_embeddings
from langchain_community.vectorstores import FAISS


def create_vectorstore(documents, embeddings, persist_dir: str):
    """Create a FAISS vector store from documents and persist to disk."""
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(persist_path))

    print(f"  Vector store saved to: {persist_path}")
    print(f"  Total vectors: {vectorstore.index.ntotal}")

    return vectorstore


def run_ingestion(embedding_provider: str = None):
    """
    Run the full ingestion pipeline:
    1. Load documents from data directories
    2. Split into chunks
    3. Generate embeddings
    4. Store in FAISS vector stores
    """
    print("=" * 60)
    print("LEGAL AI ASSISTANT — Document Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[Step 1/4] Loading documents...")
    doc_collection = load_all_documents()

    if not doc_collection["case_law"] and not doc_collection["contracts"]:
        print("No documents found. Please add documents to data/case_law/ and data/contracts/")
        return None, None

    # Step 2: Split into chunks
    print("\n[Step 2/4] Splitting documents into chunks...")
    case_law_chunks = split_documents(doc_collection["case_law"]) if doc_collection["case_law"] else []
    contract_chunks = split_documents(doc_collection["contracts"]) if doc_collection["contracts"] else []

    # Step 3: Get embeddings
    print("\n[Step 3/4] Initializing embeddings...")
    embeddings = get_embeddings(provider=embedding_provider)
    print(f"  Embedding model ready")

    # Step 4: Create vector stores
    print("\n[Step 4/4] Creating vector stores...")
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    case_law_store = None
    contracts_store = None

    if case_law_chunks:
        print("\n  Building case law vector store...")
        case_law_store = create_vectorstore(
            case_law_chunks, embeddings, str(CASE_LAW_INDEX_DIR)
        )

    if contract_chunks:
        print("\n  Building contracts vector store...")
        contracts_store = create_vectorstore(
            contract_chunks, embeddings, str(CONTRACTS_INDEX_DIR)
        )

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"  Case law chunks: {len(case_law_chunks)}")
    print(f"  Contract chunks: {len(contract_chunks)}")
    print("=" * 60)

    return case_law_store, contracts_store


if __name__ == "__main__":
    run_ingestion()

"""
Text Splitter — Chunks legal documents for embedding and retrieval.
Uses RecursiveCharacterTextSplitter optimized for legal text.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS


def create_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter configured for legal documents.
    Uses 800-token chunks with 200-token overlap as specified in the project guide.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of documents into chunks.
    Preserves metadata from the original documents.
    """
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks")
    print(f"  Chunk size: {chunk_size} | Overlap: {chunk_overlap}")

    return chunks

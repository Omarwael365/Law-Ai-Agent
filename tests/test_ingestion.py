"""
Tests for the document ingestion pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.ingestion.loader import load_text_file, load_directory, load_all_documents
from src.ingestion.splitter import create_text_splitter, split_documents
from src.config import CASE_LAW_DIR, CONTRACTS_DIR


class TestDocumentLoader:
    """Tests for document loading functionality."""

    def test_load_text_file(self):
        """Test loading a single text file."""
        test_file = CASE_LAW_DIR / "case_01_force_majeure.txt"
        if test_file.exists():
            docs = load_text_file(str(test_file))
            assert len(docs) == 1
            assert docs[0].page_content != ""
            assert docs[0].metadata["doc_type"] == "case_law"
            assert docs[0].metadata["doc_id"] == "case_01_force_majeure"

    def test_load_directory_case_law(self):
        """Test loading all documents from case law directory."""
        if CASE_LAW_DIR.exists():
            docs = load_directory(str(CASE_LAW_DIR))
            assert len(docs) >= 1
            for doc in docs:
                assert doc.metadata["doc_type"] == "case_law"

    def test_load_directory_contracts(self):
        """Test loading all documents from contracts directory."""
        if CONTRACTS_DIR.exists():
            docs = load_directory(str(CONTRACTS_DIR))
            assert len(docs) >= 1
            for doc in docs:
                assert doc.metadata["doc_type"] == "contract"

    def test_load_all_documents(self):
        """Test loading all documents from both directories."""
        result = load_all_documents()
        assert "case_law" in result
        assert "contracts" in result
        assert "all" in result
        assert len(result["all"]) == len(result["case_law"]) + len(result["contracts"])

    def test_load_nonexistent_directory(self):
        """Test loading from a directory that doesn't exist."""
        docs = load_directory("/nonexistent/path")
        assert docs == []


class TestTextSplitter:
    """Tests for text splitting functionality."""

    def test_create_text_splitter(self):
        """Test creating a text splitter with default settings."""
        splitter = create_text_splitter()
        assert splitter is not None
        assert splitter._chunk_size == 800
        assert splitter._chunk_overlap == 200

    def test_create_text_splitter_custom(self):
        """Test creating a text splitter with custom settings."""
        splitter = create_text_splitter(chunk_size=500, chunk_overlap=100)
        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 100

    def test_split_documents(self):
        """Test splitting documents into chunks."""
        test_file = CASE_LAW_DIR / "case_01_force_majeure.txt"
        if test_file.exists():
            docs = load_text_file(str(test_file))
            chunks = split_documents(docs)
            assert len(chunks) >= 1
            # Check metadata is preserved
            for chunk in chunks:
                assert "doc_type" in chunk.metadata
                assert "chunk_index" in chunk.metadata

    def test_split_preserves_metadata(self):
        """Test that splitting preserves original document metadata."""
        test_file = CASE_LAW_DIR / "case_01_force_majeure.txt"
        if test_file.exists():
            docs = load_text_file(str(test_file))
            chunks = split_documents(docs)
            for chunk in chunks:
                assert chunk.metadata["doc_type"] == "case_law"
                assert chunk.metadata["doc_id"] == "case_01_force_majeure"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

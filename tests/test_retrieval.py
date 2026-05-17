"""
Tests for the retrieval pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.config import CASE_LAW_INDEX_DIR, CONTRACTS_INDEX_DIR


class TestRetrieval:
    """Tests for retrieval functionality. Requires ingestion to have been run."""

    def _vectorstore_exists(self, index_dir):
        """Check if vector store has been built."""
        return Path(index_dir).exists()

    @pytest.mark.skipif(
        not Path(CASE_LAW_INDEX_DIR).exists(),
        reason="Case law vector store not built. Run ingestion first."
    )
    def test_search_case_law(self):
        """Test searching case law documents."""
        from src.retrieval.retriever import search_case_law
        results = search_case_law("force majeure COVID-19", top_k=3)
        assert len(results) > 0
        assert all(hasattr(doc, "page_content") for doc in results)

    @pytest.mark.skipif(
        not Path(CONTRACTS_INDEX_DIR).exists(),
        reason="Contracts vector store not built. Run ingestion first."
    )
    def test_search_contracts(self):
        """Test searching contract documents."""
        from src.retrieval.retriever import search_contracts
        results = search_contracts("non-compete clause", top_k=3)
        assert len(results) > 0
        assert all(hasattr(doc, "page_content") for doc in results)

    @pytest.mark.skipif(
        not (Path(CASE_LAW_INDEX_DIR).exists() and Path(CONTRACTS_INDEX_DIR).exists()),
        reason="Vector stores not built. Run ingestion first."
    )
    def test_search_all(self):
        """Test searching all documents."""
        from src.retrieval.retriever import search_all
        results = search_all("data breach GDPR", top_k=3)
        assert len(results) > 0

    @pytest.mark.skipif(
        not Path(CASE_LAW_INDEX_DIR).exists(),
        reason="Case law vector store not built. Run ingestion first."
    )
    def test_retriever_returns_metadata(self):
        """Test that retrieved documents include metadata."""
        from src.retrieval.retriever import search_case_law
        results = search_case_law("non-compete California", top_k=1)
        if results:
            doc = results[0]
            assert "doc_type" in doc.metadata
            assert "source" in doc.metadata


class TestEmbeddings:
    """Tests for embedding providers."""

    def test_huggingface_embeddings(self):
        """Test HuggingFace embeddings initialization."""
        from src.embeddings.embeddings import get_embeddings
        embeddings = get_embeddings(provider="huggingface")
        assert embeddings is not None

        # Test embedding a query
        result = embeddings.embed_query("test legal query")
        assert isinstance(result, list)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

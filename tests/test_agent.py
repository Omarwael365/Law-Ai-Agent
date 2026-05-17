"""
Tests for the agent system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.config import get_available_providers, CASE_LAW_INDEX_DIR


class TestConfig:
    """Tests for configuration."""

    def test_config_loads(self):
        """Test that config module loads without errors."""
        from src.config import (
            PROJECT_ROOT, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
            RETRIEVAL_TOP_K, LLM_TEMPERATURE
        )
        assert CHUNK_SIZE == 800
        assert CHUNK_OVERLAP == 200
        assert RETRIEVAL_TOP_K == 5
        assert PROJECT_ROOT.exists()
        assert DATA_DIR.exists()

    def test_available_providers(self):
        """Test provider detection."""
        providers = get_available_providers()
        assert isinstance(providers, list)


class TestTools:
    """Tests for agent tools."""

    def test_tools_defined(self):
        """Test that all tools are properly defined."""
        from src.agents.tools import get_tools
        tools = get_tools()
        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "search_case_law_tool" in tool_names
        assert "search_contracts_tool" in tool_names
        assert "search_all_documents_tool" in tool_names
        assert "summarize_legal_text" in tool_names


class TestPrompts:
    """Tests for prompt templates."""

    def test_rag_prompt(self):
        """Test RAG prompt template."""
        from src.prompts.legal_prompts import RAG_PROMPT
        assert "context" in RAG_PROMPT.input_variables
        assert "question" in RAG_PROMPT.input_variables

    def test_cot_prompt(self):
        """Test chain-of-thought prompt."""
        from src.prompts.legal_prompts import COT_PROMPT
        assert "context" in COT_PROMPT.input_variables
        assert "Step 1" in COT_PROMPT.template

    def test_get_prompt(self):
        """Test prompt factory."""
        from src.prompts.legal_prompts import get_prompt
        for prompt_type in ["rag", "few_shot", "cot"]:
            prompt = get_prompt(prompt_type)
            assert prompt is not None

    def test_get_prompt_invalid(self):
        """Test prompt factory with invalid type."""
        from src.prompts.legal_prompts import get_prompt
        with pytest.raises(ValueError):
            get_prompt("invalid_type")

    def test_system_prompt_contains_disclaimer(self):
        """Test that system prompt includes the required disclaimer."""
        from src.prompts.legal_prompts import LEGAL_SYSTEM_PROMPT
        assert "research assistance only" in LEGAL_SYSTEM_PROMPT.lower()
        assert "licensed attorney" in LEGAL_SYSTEM_PROMPT.lower()


class TestLLMProviders:
    """Tests for LLM provider factory."""

    @pytest.mark.skipif(
        not get_available_providers(),
        reason="No LLM API keys configured"
    )
    def test_get_llm(self):
        """Test LLM initialization with default provider."""
        from src.llm.providers import get_llm
        llm = get_llm()
        assert llm is not None

    def test_get_llm_no_keys(self):
        """Test LLM initialization fails gracefully with no keys."""
        if not get_available_providers():
            from src.llm.providers import get_llm
            with pytest.raises(ValueError):
                get_llm(provider="openai")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

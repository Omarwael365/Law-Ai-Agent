"""
Classic RAG Chain — Retrieval-Augmented Generation pipeline.
Combines retrieval from vector stores with LLM generation for legal Q&A.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from src.llm.providers import get_llm
from src.retrieval.retriever import (
    get_case_law_retriever,
    get_contracts_retriever,
    search_all,
)
from src.prompts.legal_prompts import RAG_PROMPT, LEGAL_SYSTEM_PROMPT


def create_rag_chain(
    doc_type: str = "all",
    llm_provider: str = None,
    embedding_provider: str = None,
    top_k: int = 5,
    prompt_type: str = "rag",
):
    """
    Create a RAG chain for legal question answering.

    Args:
        doc_type: 'case_law', 'contracts', or 'all'
        llm_provider: LLM provider to use
        embedding_provider: Embedding provider to use
        top_k: Number of documents to retrieve
        prompt_type: Prompt template to use

    Returns:
        A RetrievalQA chain instance
    """
    # Get LLM
    llm = get_llm(provider=llm_provider)

    # Get retriever based on doc_type
    if doc_type == "case_law":
        retriever = get_case_law_retriever(top_k, embedding_provider)
    elif doc_type == "contracts":
        retriever = get_contracts_retriever(top_k, embedding_provider)
    else:
        # For "all", use case_law retriever as primary (can be extended)
        retriever = get_case_law_retriever(top_k, embedding_provider)

    if retriever is None:
        raise ValueError(
            f"Could not load vector store for '{doc_type}'. Run ingestion first: "
            "python -m src.ingestion.ingest"
        )

    # Get prompt
    from src.prompts.legal_prompts import get_prompt
    prompt = get_prompt(prompt_type)

    # Create the chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


def query_rag(
    question: str,
    doc_type: str = "all",
    llm_provider: str = None,
    embedding_provider: str = None,
    top_k: int = 5,
) -> dict:
    """
    Query the RAG pipeline with a legal question.

    Args:
        question: The legal question to answer
        doc_type: Document type to search
        llm_provider: LLM provider to use
        embedding_provider: Embedding provider to use
        top_k: Number of documents to retrieve

    Returns:
        Dict with 'answer', 'source_documents', and metadata
    """
    chain = create_rag_chain(
        doc_type=doc_type,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        top_k=top_k,
    )

    result = chain.invoke({"query": question})

    # Format source documents
    sources = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "doc_type": doc.metadata.get("doc_type", "Unknown"),
                "doc_id": doc.metadata.get("doc_id", "Unknown"),
            })

    return {
        "question": question,
        "answer": result.get("result", ""),
        "source_documents": result.get("source_documents", []),
        "sources_summary": sources,
    }


if __name__ == "__main__":
    # Quick test
    question = "Does COVID-19 qualify as force majeure under a supply contract?"
    print(f"Question: {question}\n")
    result = query_rag(question, doc_type="case_law")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources: {len(result['sources_summary'])}")
    for s in result["sources_summary"]:
        print(f"  - {s['doc_id']} ({s['doc_type']})")

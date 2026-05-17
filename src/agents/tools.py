"""
Agent Tools — Tools that the legal agent can use to search and analyze documents.
Each tool wraps the retriever with specific functionality.
"""

from langchain_core.tools import tool
from src.retrieval.retriever import search_case_law, search_contracts, search_all


@tool
def search_case_law_tool(query: str) -> str:
    """Search through case law documents (court judgments, precedents, opinions) for relevant legal information.
    Use this tool when the question involves court decisions, legal precedents, or judicial opinions.
    Input should be a clear legal search query."""
    docs = search_case_law(query, top_k=5)
    if not docs:
        return "No relevant case law found. The vector store may not be initialized — run ingestion first."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("doc_id", "Unknown")
        results.append(f"[Source {i}: {source}]\n{doc.page_content}\n")

    return "\n---\n".join(results)


@tool
def search_contracts_tool(query: str) -> str:
    """Search through contract documents (NDAs, employment agreements, vendor contracts, terms of service) for relevant clauses and provisions.
    Use this tool when the question involves contract terms, clauses, or specific contractual provisions.
    Input should be a clear search query about contract terms."""
    docs = search_contracts(query, top_k=5)
    if not docs:
        return "No relevant contracts found. The vector store may not be initialized — run ingestion first."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("doc_id", "Unknown")
        results.append(f"[Source {i}: {source}]\n{doc.page_content}\n")

    return "\n---\n".join(results)


@tool
def search_all_documents_tool(query: str) -> str:
    """Search through ALL legal documents (both case law and contracts) simultaneously.
    Use this when you need a broad search across all document types, or when unsure which collection is most relevant.
    Input should be a clear legal search query."""
    docs = search_all(query, top_k=5)
    if not docs:
        return "No relevant documents found. The vector stores may not be initialized — run ingestion first."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("doc_id", "Unknown")
        doc_type = doc.metadata.get("doc_type", "Unknown")
        results.append(f"[Source {i}: {source} ({doc_type})]\n{doc.page_content}\n")

    return "\n---\n".join(results)


@tool
def summarize_legal_text(text: str) -> str:
    """Summarize a piece of legal text into key points.
    Use this tool when you have a long legal passage and need to extract the main legal principles, holdings, or clause requirements.
    Input should be the legal text to summarize."""
    from src.llm.providers import get_llm

    llm = get_llm()
    response = llm.invoke(
        f"Summarize the following legal text into concise key points. "
        f"Focus on: legal principles, holdings, obligations, and remedies.\n\n"
        f"TEXT:\n{text}\n\nKEY POINTS SUMMARY:"
    )
    return response.content


def get_tools():
    """Return the list of all available agent tools."""
    return [
        search_case_law_tool,
        search_contracts_tool,
        search_all_documents_tool,
        summarize_legal_text,
    ]

"""
Document Loader — Loads legal documents from PDF and text files.
Supports PyMuPDF for PDFs and LangChain TextLoader for .txt files.
"""

import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document


def load_text_file(file_path: str) -> List[Document]:
    """Load a single text file and return as a LangChain Document."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract metadata from filename
    filename = path.stem
    doc_type = "case_law" if "case_law" in str(path) else "contract"

    metadata = {
        "source": str(path),
        "filename": path.name,
        "doc_type": doc_type,
        "doc_id": filename,
    }

    return [Document(page_content=content, metadata=metadata)]


def load_pdf_file(file_path: str) -> List[Document]:
    """Load a single PDF file using PyMuPDF and return as LangChain Documents."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        # Fallback to LangChain's PyPDFLoader
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        doc_type = "case_law" if "case_law" in file_path else "contract"
        for doc in docs:
            doc.metadata["doc_type"] = doc_type
            doc.metadata["doc_id"] = Path(file_path).stem
        return docs

    path = Path(file_path)
    doc_type = "case_law" if "case_law" in str(path) else "contract"
    documents = []

    pdf_document = fitz.open(str(path))
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text()
        if text.strip():
            metadata = {
                "source": str(path),
                "filename": path.name,
                "doc_type": doc_type,
                "doc_id": path.stem,
                "page": page_num + 1,
            }
            documents.append(Document(page_content=text, metadata=metadata))

    pdf_document.close()
    return documents


def load_directory(directory_path: str) -> List[Document]:
    """Load all supported documents from a directory."""
    documents = []
    dir_path = Path(directory_path)

    if not dir_path.exists():
        print(f"Warning: Directory {directory_path} does not exist.")
        return documents

    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() == ".txt":
            docs = load_text_file(str(file_path))
            documents.extend(docs)
            print(f"  Loaded: {file_path.name} ({len(docs)} document(s))")
        elif file_path.suffix.lower() == ".pdf":
            docs = load_pdf_file(str(file_path))
            documents.extend(docs)
            print(f"  Loaded: {file_path.name} ({len(docs)} page(s))")

    return documents


def load_all_documents() -> dict:
    """
    Load all legal documents from both case_law and contracts directories.
    Returns a dict with 'case_law' and 'contracts' keys.
    """
    from src.config import CASE_LAW_DIR, CONTRACTS_DIR

    print("Loading case law documents...")
    case_law_docs = load_directory(str(CASE_LAW_DIR))
    print(f"Total case law documents: {len(case_law_docs)}\n")

    print("Loading contract documents...")
    contract_docs = load_directory(str(CONTRACTS_DIR))
    print(f"Total contract documents: {len(contract_docs)}\n")

    return {
        "case_law": case_law_docs,
        "contracts": contract_docs,
        "all": case_law_docs + contract_docs,
    }

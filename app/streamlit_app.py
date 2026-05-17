"""
Legal AI Assistant — Streamlit Web Interface
A chat-based interface for querying legal documents with RAG.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ──────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        opacity: 0.7;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: rgba(74, 108, 247, 0.1);
        border-left: 4px solid #4a6cf7;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: inherit;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .disclaimer {
        background-color: rgba(255, 193, 7, 0.15);
        border: 1px solid rgba(255, 193, 7, 0.5);
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.85rem;
        color: inherit;
        margin-top: 1rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    /* Keep chat input always visible at bottom */
    .stChatInput {
        position: sticky;
        bottom: 0;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/law.png", width=60)
    st.markdown("## ⚖️ Legal AI Assistant")
    st.markdown("*Powered by Agentic RAG*")
    st.markdown("---")

    # Provider selection
    st.markdown("### 🤖 Model Configuration")

    from src.config import get_available_providers, get_available_embedding_providers, DEFAULT_LLM_PROVIDER

    available_llm = get_available_providers()
    available_emb = get_available_embedding_providers()

    if not available_llm:
        st.warning("No LLM API keys configured. Please set up your .env file.")
        llm_provider = None
    else:
        provider_labels = {
            "openai": "OpenAI GPT-4o",
            "google": "Google Gemini 2.0 Flash",
            "deepseek": "DeepSeek-V3",
            "groq": "Groq Llama 3.3 70B",
            "mock": "Local Offline Mock (No Keys Needed)",
        }
        default_idx = available_llm.index(DEFAULT_LLM_PROVIDER) if DEFAULT_LLM_PROVIDER in available_llm else 0
        llm_provider = st.selectbox(
            "LLM Provider",
            available_llm,
            format_func=lambda x: provider_labels.get(x, x),
            index=default_idx,
        )

    emb_labels = {
        "huggingface": "HuggingFace BGE-Base",
        "openai": "OpenAI Embedding-3-Large",
    }
    embedding_provider = st.selectbox(
        "Embedding Provider",
        available_emb,
        format_func=lambda x: emb_labels.get(x, x),
    )

    st.markdown("---")

    # Agent mode selection
    st.markdown("### 🧠 Agent Mode")
    agent_mode = st.radio(
        "Query Mode",
        ["RAG Chain (Classic)", "ReAct Agent", "LangGraph Agent"],
        help="Classic RAG for direct Q&A, ReAct for tool-using reasoning, LangGraph for stateful multi-step analysis",
    )

    st.markdown("---")

    # Search options
    st.markdown("### 🔍 Search Options")
    doc_type = st.selectbox(
        "Document Collection",
        ["all", "case_law", "contracts"],
        format_func=lambda x: {"all": "All Documents", "case_law": "Case Law", "contracts": "Contracts"}[x],
    )

    top_k = st.slider("Documents to Retrieve", 1, 10, 5)

    st.markdown("---")

    # Ingestion controls
    st.markdown("### 📥 Document Ingestion")
    if st.button("🔄 Run Ingestion Pipeline", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            try:
                from src.ingestion.ingest import run_ingestion
                run_ingestion(embedding_provider)
                st.success("Ingestion complete!")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    # Sample queries
    st.markdown("---")
    st.markdown("### 💡 Sample Queries")
    sample_queries = [
        "Does COVID-19 qualify as force majeure?",
        "Are non-competes enforceable in California?",
        "What are GDPR Article 32 security obligations?",
        "Can an employer claim IP for personal projects?",
        "What makes a liquidated damages clause a penalty?",
    ]
    for query in sample_queries:
        if st.button(query, key=f"sample_{hash(query)}", use_container_width=True):
            st.session_state.sample_query = query

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ This is research assistance only. '
        "This does not constitute legal advice. Consult a licensed attorney.</div>",
        unsafe_allow_html=True,
    )


# ── Main Content ────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">⚖️ Legal AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask legal questions about case law and contracts — powered by Agentic RAG</div>',
    unsafe_allow_html=True,
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📎 Source Documents"):
                for src in message["sources"]:
                    st.markdown(
                        f'<div class="source-card"><strong>{src["doc_id"]}</strong> '
                        f'({src["doc_type"]})<br>{src["content"]}</div>',
                        unsafe_allow_html=True,
                    )
        if "metrics" in message and message["metrics"]:
            with st.expander("📊 Performance Metrics"):
                cols = st.columns(3)
                m = message["metrics"]
                cols[0].metric("Latency", f"{m.get('latency', 0):.2f}s")
                cols[1].metric("Sources", m.get("num_sources", 0))
                cols[2].metric("Mode", m.get("mode", "RAG"))


# Always show chat input
chat_input = st.chat_input("Ask a legal question...")

# Handle sample query button OR typed input
if "sample_query" in st.session_state:
    prompt = st.session_state.sample_query
    del st.session_state.sample_query
else:
    prompt = chat_input

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal documents..."):
            start_time = time.time()

            try:
                sources_summary = []
                answer = ""

                if agent_mode == "RAG Chain (Classic)":
                    from src.rag.chain import query_rag
                    result = query_rag(
                        prompt,
                        doc_type=doc_type,
                        llm_provider=llm_provider,
                        embedding_provider=embedding_provider,
                        top_k=top_k,
                    )
                    answer = result["answer"]
                    sources_summary = result.get("sources_summary", [])

                elif agent_mode == "ReAct Agent":
                    from src.agents.react_agent import run_react_agent
                    result = run_react_agent(prompt, llm_provider=llm_provider)
                    answer = result["answer"]

                elif agent_mode == "LangGraph Agent":
                    from src.agents.langgraph_agent import run_langgraph_agent
                    result = run_langgraph_agent(prompt, llm_provider=llm_provider)
                    answer = result["answer"]

                latency = time.time() - start_time

                st.markdown(answer)

                metrics = {
                    "latency": latency,
                    "num_sources": len(sources_summary),
                    "mode": agent_mode.split(" ")[0],
                }

                # Show sources
                if sources_summary:
                    with st.expander("📎 Source Documents"):
                        for src in sources_summary:
                            st.markdown(
                                f'<div class="source-card"><strong>{src["doc_id"]}</strong> '
                                f'({src["doc_type"]})<br>{src["content"]}</div>',
                                unsafe_allow_html=True,
                            )

                # Show metrics
                with st.expander("📊 Performance Metrics"):
                    cols = st.columns(3)
                    cols[0].metric("Latency", f"{latency:.2f}s")
                    cols[1].metric("Sources", len(sources_summary))
                    cols[2].metric("Mode", agent_mode.split(" ")[0])

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_summary,
                    "metrics": metrics,
                })

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}\n\nPlease ensure:\n1. Your .env file is configured with valid API keys\n2. The ingestion pipeline has been run\n3. The selected LLM provider is available"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

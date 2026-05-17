# Legal AI Assistant — Powered by Agentic RAG

A production-like Legal AI Assistant that combines Retrieval-Augmented Generation (RAG), agentic reasoning, multi-provider LLMs, and comprehensive evaluation. Built as a Gen AI Capstone Project covering all major topics in modern AI engineering.

## Overview

This system helps lawyers and law students quickly find, analyze, and reason over two core document types:

- **Case Law** — Court judgments, precedents, opinions, and legal summaries
- **Contracts** — NDAs, vendor agreements, employment contracts, and playbooks

Instead of manually searching databases or reading hundreds of pages, the AI assistant retrieves the exact relevant sections from a private document collection and generates clear, cited answers grounded in real text.

## Architecture

```
User Question
    │
    ▼
┌─────────────────┐
│   Agent Layer    │  (ReAct / LangGraph)
│  Decides which   │
│  tools to call   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Case Law│ │Contract│   ← FAISS Vector Stores
│ Search │ │ Search │
└────┬───┘ └───┬────┘
     │         │
     ▼         ▼
┌─────────────────┐
│  LLM Generation  │  (GPT-4o / Gemini / DeepSeek)
│  with citations  │
└─────────────────┘
         │
         ▼
   Cited Answer + Sources
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (Primary) | OpenAI GPT-4o |
| LLM (Long Context) | Google Gemini 1.5 Pro |
| LLM (Budget) | DeepSeek-V3 |
| Embeddings | BAAI/bge-base-en-v1.5 (HuggingFace) |
| Vector Store | FAISS |
| Agent Framework | LangChain + LangGraph |
| Web UI | Streamlit |
| No-Code Automation | N8N |

## Quick Start

### 1. Clone and Set Up Environment

```bash
# Navigate to the project directory
cd "Training Final Project"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# At minimum, you need ONE of: OPENAI_API_KEY, GOOGLE_API_KEY, or DEEPSEEK_API_KEY
```

### 3. Run Document Ingestion

```bash
python -m src.ingestion.ingest
```

This loads the sample legal documents, splits them into chunks, generates embeddings, and stores them in FAISS vector indexes.

### 4. Query the System

**Option A — Streamlit Web UI:**
```bash
streamlit run app/streamlit_app.py
```

**Option B — Python Script:**
```python
from src.rag.chain import query_rag

result = query_rag("Does COVID-19 qualify as force majeure?")
print(result["answer"])
```

**Option C — LangGraph Agent:**
```python
from src.agents.langgraph_agent import run_langgraph_agent

result = run_langgraph_agent("Is a non-compete clause enforceable in California?")
print(result["answer"])
```

### 5. Run Evaluation

```bash
python -m src.evaluation.benchmark
```

## Project Structure

```
├── .env.example              # API key template
├── requirements.txt           # Python dependencies
├── README.md
│
├── data/
│   ├── case_law/             # 10 case law documents
│   └── contracts/            # 10 contract documents
│
├── vectorstore/              # FAISS indexes (auto-generated)
│
├── src/
│   ├── config.py             # Central configuration
│   ├── ingestion/            # Document loading, splitting, embedding
│   ├── embeddings/           # OpenAI & HuggingFace embeddings
│   ├── retrieval/            # FAISS retriever
│   ├── llm/                  # Multi-provider LLM factory
│   ├── prompts/              # Legal prompt templates
│   ├── rag/                  # Classic RAG chain
│   ├── agents/               # ReAct & LangGraph agents
│   └── evaluation/           # Metrics & benchmarking
│
├── app/
│   └── streamlit_app.py      # Web UI
│
├── notebooks/                # Jupyter notebooks for each phase
├── evaluation_report/        # Model selection analysis
├── n8n/                      # N8N workflow template
└── tests/                    # Unit tests
```

## Phases

| Phase | Topic | Implementation |
|-------|-------|---------------|
| 1 | Gen AI Fundamentals + Prompt Engineering | `src/prompts/legal_prompts.py` |
| 2 | Multi-Provider Model Exploration | `src/llm/providers.py` |
| 3 | RAG Pipeline with LangChain | `src/ingestion/`, `src/rag/chain.py` |
| 4 | Building Legal Agents | `src/agents/react_agent.py`, `src/agents/langgraph_agent.py` |
| 5 | Agent Evaluation | `src/evaluation/evaluator.py`, `src/evaluation/benchmark.py` |
| 6 | Model Selection | `evaluation_report/model_selection_report.md` |

## Test Scenarios

The system includes 10 realistic legal test cases:

1. Force Majeure in Supply Chain Contracts
2. Non-Compete Clause Enforceability
3. GDPR Data Breach Liability
4. Intellectual Property Ownership in Employment
5. Liquidated Damages vs. Penalty Clauses
6. SaaS Subscription Auto-Renewal Dispute
7. Wrongful Termination and At-Will Employment
8. Construction Contract Disputes: Scope Creep
9. Confidentiality Breach and Trade Secrets
10. Arbitration Clause Unconscionability

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Faithfulness | Does the answer contain only information from retrieved docs? |
| Task Success Rate | Did the agent address all relevant legal issues? |
| Latency | End-to-end response time in seconds |
| Cost | API token consumption in USD |

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_ingestion.py -v
```

## N8N Workflow

To use the no-code workflow:

```bash
# Start N8N locally with Docker
docker run -it --rm --name n8n -p 5678:5678 n8nio/n8n

# Import the workflow from n8n/legal_agent_workflow.json
```

## Security and Ethics

- This project uses **simulated legal documents only**
- Never upload real client documents or privileged communications to cloud APIs
- All answers include a disclaimer: "This is research assistance only. Consult a licensed attorney."
- API keys are stored in `.env` files (excluded from version control via `.gitignore`)

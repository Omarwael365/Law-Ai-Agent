# Model Selection Analysis Report

## Legal AI Assistant — Choosing the Best LLM

This report analyzes and compares LLM models for use in a Legal AI Assistant, using data from three benchmarking sources: HELM (Stanford), Artificial Analysis, and LLM Arena.

---

## 1. HELM Benchmark Analysis (Stanford CRFM)

Source: https://crfm.stanford.edu/helm/

### LegalBench Scenario Results

LegalBench tests direct legal reasoning capabilities including contract interpretation, statutory analysis, and issue spotting.

| Model | LegalBench (Exact Match) | LegalBench (F1) | BoolQ | NarrativeQA |
|-------|--------------------------|------------------|-------|-------------|
| GPT-4o | 0.82 | 0.86 | 0.93 | 0.78 |
| Gemini 1.5 Pro | 0.79 | 0.83 | 0.91 | 0.82 |
| Mistral-Large | 0.74 | 0.78 | 0.88 | 0.71 |
| DeepSeek-V3 | 0.76 | 0.80 | 0.89 | 0.73 |

### Key Findings
- **GPT-4o** leads on LegalBench exact match and F1 scores, indicating superior legal reasoning for contract interpretation and statutory analysis
- **Gemini 1.5 Pro** excels on NarrativeQA, suggesting better comprehension of long legal documents due to its 1M token context window
- **DeepSeek-V3** performs competitively, particularly notable given its significantly lower cost
- **Mistral-Large** is adequate but trails on legal-specific benchmarks

---

## 2. Artificial Analysis — Cost/Speed Trade-offs

Source: https://artificialanalysis.ai/

### Performance Comparison

| Factor | GPT-4o | Gemini 1.5 Pro | DeepSeek-V3 | Acceptable Range |
|--------|--------|----------------|-------------|------------------|
| Quality Index | 83 | 78 | 75 | > 70 |
| Output Speed (tok/s) | 82 | 135 | 60 | > 50 |
| Context Window | 128K | 1M | 64K | > 32K |
| Cost per 1M input tokens | $2.50 | $3.50 | $0.27 | < $10 |
| Cost per 1M output tokens | $10.00 | $10.50 | $1.10 | < $30 |
| TTFT (seconds) | 0.4 | 0.8 | 1.2 | < 2.0 |

### Cost Analysis for Legal Use Case

Assuming 1,000 legal queries per month, with average 2,000 input tokens and 1,000 output tokens per query:

| Provider | Monthly Input Cost | Monthly Output Cost | Total Monthly |
|----------|-------------------|--------------------| --------------|
| GPT-4o | $5.00 | $10.00 | $15.00 |
| Gemini 1.5 Pro | $7.00 | $10.50 | $17.50 |
| DeepSeek-V3 | $0.54 | $1.10 | $1.64 |

### Key Findings
- **DeepSeek-V3** is 9x cheaper than GPT-4o and 10x cheaper than Gemini, making it ideal for high-volume or budget-constrained deployments
- **Gemini 1.5 Pro** has the fastest output speed (135 tok/s), best for interactive chat experiences
- **GPT-4o** offers the best quality-to-speed ratio with lowest TTFT
- All three models meet the minimum context window requirement (>32K) for legal documents

---

## 3. LLM Arena — Human Preference Rankings

Source: https://lmarena.ai/ (formerly LMSYS Chatbot Arena)

### Elo Ratings (Overall and Category-Specific)

| Model | Overall Elo | Instruction Following | Hard Prompts | Reasoning |
|-------|-------------|----------------------|--------------|-----------|
| GPT-4o | 1287 | 1295 | 1290 | 1285 |
| Gemini 1.5 Pro | 1260 | 1255 | 1270 | 1265 |
| DeepSeek-V3 | 1253 | 1248 | 1260 | 1258 |

### Key Findings
- **GPT-4o** has the highest Elo across all categories, meaning humans consistently prefer its responses
- Legal questions are typically "hard prompts" requiring multi-step reasoning — GPT-4o leads here
- The gap between models is moderate, suggesting all three produce acceptable legal analysis
- **Instruction following** is critical for structured legal output (numbered points, citations) — GPT-4o leads

---

## 4. Final Model Recommendation

### Primary Recommendation: GPT-4o

**Justification:**
1. **Highest legal reasoning scores** on HELM LegalBench (0.82 exact match)
2. **Best instruction following** for structured legal analysis output
3. **Lowest TTFT** (0.4s) for responsive interactive use
4. **Reasonable cost** ($15/month at 1K queries) for professional legal tools
5. **Strong human preference** rankings across all LLM Arena categories

### Secondary Recommendation: Gemini 1.5 Pro (for long documents)

**Use case:** When processing very long contracts (50K+ tokens) that exceed GPT-4o's practical context window. The 1M token context window allows ingesting entire contract suites in a single prompt.

### Budget Option: DeepSeek-V3

**Use case:** High-volume processing, development/testing, or budget-constrained deployments where the 9x cost reduction justifies slightly lower quality scores.

### Recommended Architecture

```
Primary LLM:     GPT-4o (legal Q&A, agent reasoning)
Long Context:    Gemini 1.5 Pro (full document summarization)
Development:     DeepSeek-V3 (testing, iteration, cost savings)
Embeddings:      BAAI/bge-base-en-v1.5 (free, local, no API costs)
Fallback:        OpenAI text-embedding-3-large (if higher quality needed)
```

---

## 5. Evaluation Metrics Summary

Based on our benchmark of 10 legal test cases:

| Metric | Description | Target | Notes |
|--------|-------------|--------|-------|
| Faithfulness | Answer grounded in retrieved docs | > 0.85 | LLM-as-judge evaluation |
| Task Success Rate | All legal issues addressed | > 80% | Per-topic verification |
| Latency | End-to-end response time | < 10s | Includes retrieval + generation |
| Cost | API cost per query | < $0.05 | Token-based estimation |

These metrics should be re-evaluated periodically as model capabilities and pricing evolve.

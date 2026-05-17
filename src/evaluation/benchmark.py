"""
Benchmark Runner — Runs all 10 legal test cases and produces an evaluation report.
Uses the test scenarios from the project guide.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.evaluator import measure_latency, run_full_evaluation


# ── 10 Test Cases from the Project Guide ────────────────────────────────────

TEST_CASES = [
    {
        "id": "case_01",
        "title": "Force Majeure in Supply Chain Contracts",
        "query": "Does COVID-19 qualify as force majeure under a supply contract clause listing natural disasters and governmental action?",
        "expected_topics": [
            "Force majeure clause interpretation",
            "COVID-19 applicability",
            "Ejusdem generis doctrine",
            "Governmental action provision",
        ],
    },
    {
        "id": "case_02",
        "title": "Non-Compete Clause Enforceability",
        "query": "Enforceability of non-compete clauses under California Business and Professions Code Section 16600 for tech employees",
        "expected_topics": [
            "California Section 16600 prohibition",
            "Edwards v. Arthur Andersen precedent",
            "Choice-of-law analysis",
            "Overbreadth of non-compete scope",
        ],
    },
    {
        "id": "case_03",
        "title": "GDPR Data Breach Liability",
        "query": "GDPR Article 32 security obligations for cloud data processors and liability under data processing agreements",
        "expected_topics": [
            "Data processor vs joint controller classification",
            "Article 32 security measures",
            "Encryption requirements",
            "Maximum fine under Article 83",
        ],
    },
    {
        "id": "case_04",
        "title": "IP Ownership in Employment",
        "query": "Employee IP assignment clause enforceability for inventions developed on personal time under California Labor Code 2870",
        "expected_topics": [
            "California Labor Code 2870 protections",
            "Personal time and personal resources test",
            "Relation to employer business",
            "IP clause overbreadth",
        ],
    },
    {
        "id": "case_05",
        "title": "Liquidated Damages vs. Penalty Clauses",
        "query": "Liquidated damages clause enforceability test genuine pre-estimate of loss versus unenforceable penalty clause",
        "expected_topics": [
            "Cavendish Square framework",
            "Genuine pre-estimate of loss test",
            "Legitimate interest analysis",
            "Proportionality assessment",
        ],
    },
    {
        "id": "case_06",
        "title": "SaaS Auto-Renewal Dispute",
        "query": "Auto-renewal clause enforceability conspicuous notice requirement SaaS enterprise contracts electronic signature",
        "expected_topics": [
            "Auto-renewal clause enforceability",
            "Electronic signature validity (E-SIGN Act)",
            "Conspicuous notice requirements",
            "Price increase upon renewal",
        ],
    },
    {
        "id": "case_07",
        "title": "Wrongful Termination and At-Will Employment",
        "query": "Whistleblower retaliation wrongful termination at-will employment exceptions Sarbanes-Oxley private company",
        "expected_topics": [
            "Whistleblower retaliation claim",
            "At-will employment exceptions",
            "SOX applicability to private companies",
            "Pretext analysis",
        ],
    },
    {
        "id": "case_08",
        "title": "Construction Contract Scope Creep",
        "query": "Construction contract change order validity verbal instructions scope of work dispute unjust enrichment",
        "expected_topics": [
            "Change order requirements under AIA terms",
            "Constructive change order doctrine",
            "Verbal instruction binding effect",
            "Unjust enrichment claim",
        ],
    },
    {
        "id": "case_09",
        "title": "Trade Secrets and Confidentiality Breach",
        "query": "Trade secret misappropriation customer list Defend Trade Secrets Act injunctive relief departing employee",
        "expected_topics": [
            "Trade secret qualification under DTSA",
            "Customer list as trade secret",
            "NDA confidentiality coverage",
            "Available remedies (injunction, damages)",
        ],
    },
    {
        "id": "case_10",
        "title": "Arbitration Clause Unconscionability",
        "query": "Mandatory arbitration clause class action waiver unconscionability consumer contracts small value claims",
        "expected_topics": [
            "AT&T Mobility v. Concepcion framework",
            "Effective vindication exception",
            "Procedural and substantive unconscionability",
            "Class action waiver enforceability",
        ],
    },
]


def run_benchmark(
    agent_func=None,
    llm_provider: str = None,
    output_file: str = None,
) -> dict:
    """
    Run all 10 test cases and collect evaluation metrics.

    Args:
        agent_func: Function to call for each query. Defaults to RAG chain.
        llm_provider: LLM provider to use
        output_file: Path to save results JSON

    Returns:
        Dict with all results and aggregate metrics
    """
    if agent_func is None:
        from src.rag.chain import query_rag
        agent_func = lambda q: query_rag(q, llm_provider=llm_provider)

    results = []
    print("=" * 60)
    print("LEGAL AI ASSISTANT — Benchmark Evaluation")
    print(f"Provider: {llm_provider or 'default'}")
    print(f"Test cases: {len(TEST_CASES)}")
    print("=" * 60)

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {test_case['title']}...")

        try:
            # Run the query with latency measurement
            result, latency = measure_latency(agent_func, test_case["query"])

            # Get answer and context
            answer = result.get("answer", "")
            source_docs = result.get("source_documents", [])
            context = "\n\n".join(
                doc.page_content for doc in source_docs
            ) if source_docs else ""

            # Run evaluation
            evaluation = run_full_evaluation(
                question=test_case["query"],
                answer=answer,
                context=context,
                expected_topics=test_case["expected_topics"],
                latency=latency,
                provider=llm_provider or "openai",
                llm_provider=llm_provider,
            )

            test_result = {
                "test_case": test_case,
                "answer": answer,
                "evaluation": evaluation,
                "status": "success",
            }

            print(f"  Faithfulness: {evaluation['summary']['faithfulness_score']:.2f}")
            print(f"  Success Rate: {evaluation['summary']['success_rate']:.2f}")
            print(f"  Latency: {evaluation['summary']['latency_seconds']:.2f}s")

        except Exception as e:
            test_result = {
                "test_case": test_case,
                "answer": "",
                "evaluation": None,
                "status": "error",
                "error": str(e),
            }
            print(f"  ERROR: {e}")

        results.append(test_result)

    # Aggregate metrics
    successful = [r for r in results if r["status"] == "success"]
    aggregate = {}
    if successful:
        aggregate = {
            "avg_faithfulness": sum(
                r["evaluation"]["summary"]["faithfulness_score"] for r in successful
            ) / len(successful),
            "avg_success_rate": sum(
                r["evaluation"]["summary"]["success_rate"] for r in successful
            ) / len(successful),
            "avg_latency": sum(
                r["evaluation"]["summary"]["latency_seconds"] for r in successful
            ) / len(successful),
            "total_cost": sum(
                r["evaluation"]["summary"]["total_cost_usd"] for r in successful
            ),
            "total_tests": len(TEST_CASES),
            "successful_tests": len(successful),
            "failed_tests": len(TEST_CASES) - len(successful),
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "provider": llm_provider or "default",
        "results": results,
        "aggregate": aggregate,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if aggregate:
        print(f"  Avg Faithfulness:  {aggregate['avg_faithfulness']:.3f}")
        print(f"  Avg Success Rate:  {aggregate['avg_success_rate']:.3f}")
        print(f"  Avg Latency:       {aggregate['avg_latency']:.2f}s")
        print(f"  Total Cost:        ${aggregate['total_cost']:.4f}")
        print(f"  Tests Passed:      {aggregate['successful_tests']}/{aggregate['total_tests']}")

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    return report


if __name__ == "__main__":
    report = run_benchmark(
        output_file=str(Path(__file__).parent.parent.parent / "evaluation_report" / "benchmark_results.json")
    )

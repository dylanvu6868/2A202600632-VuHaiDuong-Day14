import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from agent.main_agent import MainAgent, ImprovedAgent


class ExpertEvaluator:
    """Đánh giá RAGAS metrics: Faithfulness, Answer Relevancy, và Retrieval metrics."""

    def __init__(self):
        self._retrieval_eval = RetrievalEvaluator(top_k=3)

    async def score(self, case: dict, response: dict) -> dict:
        retrieval_result = await self._retrieval_eval.evaluate_single(case)

        answer = response.get("answer", "")
        context = case.get("context", "")

        if not context:
            faithfulness = 0.5
        else:
            ctx_words = set(context.lower().split())
            ans_words = set(answer.lower().split())
            overlap = len(ctx_words & ans_words) / max(len(ctx_words), 1)
            faithfulness = min(1.0, overlap * 2.5)

        question = case.get("question", "")
        q_words = set(question.lower().split())
        ans_words = set(answer.lower().split())
        q_overlap = len(q_words & ans_words) / max(len(q_words), 1)
        relevancy = min(1.0, q_overlap * 2.0)

        return {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "retrieval": {
                "hit_rate": retrieval_result["hit_rate"],
                "mrr": retrieval_result["mrr"],
                "precision_at_k": retrieval_result["precision_at_k"],
                "retrieved_ids": retrieval_result["retrieved_ids"][:3],
            },
        }


class MultiModelJudge:
    """Wrapper cho LLMJudge, thu thập scores để tính Cohen's Kappa."""

    def __init__(self):
        self._judge = LLMJudge()
        self._scores_gpt = []
        self._scores_claude = []

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> dict:
        result = await self._judge.evaluate_multi_judge(question, answer, ground_truth)
        self._scores_gpt.append(result["individual_scores"]["gpt-4o"])
        self._scores_claude.append(result["individual_scores"]["claude-sonnet-3-5"])
        return result

    def get_cohens_kappa(self) -> float:
        return LLMJudge.calculate_cohens_kappa(self._scores_gpt, self._scores_claude)


async def run_benchmark(agent, version_label: str, dataset: list) -> tuple:
    print(f"\n🚀 Khởi động Benchmark cho {version_label} ({len(dataset)} cases)...")
    start = time.perf_counter()

    evaluator = ExpertEvaluator()
    judge_wrapper = MultiModelJudge()
    runner = BenchmarkRunner(agent, evaluator, judge_wrapper)
    results = await runner.run_all(dataset)

    elapsed = time.perf_counter() - start
    total = len(results)
    passes = sum(1 for r in results if r["status"] == "pass")

    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    avg_hit_rate = sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total
    avg_mrr = sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total
    avg_faithfulness = sum(r["ragas"]["faithfulness"] for r in results) / total
    avg_relevancy = sum(r["ragas"]["relevancy"] for r in results) / total
    avg_agreement = sum(r["judge"]["agreement_rate"] for r in results) / total
    total_tokens = sum(r.get("tokens_used", 200) for r in results)
    cohens_kappa = judge_wrapper.get_cohens_kappa()

    # Tách conflict cases (delta > 1 giữa 2 judges)
    conflict_cases = [r for r in results if r["judge"].get("delta", 0) > 1]
    conflict_count = len(conflict_cases)

    # Cost breakdown chi tiết theo từng component
    generation_tokens = total_tokens
    judge_tokens_per_case = 300  # mỗi case gọi 2 judges × ~150 tokens
    judge_tokens_total = total * judge_tokens_per_case * 2
    retrieval_cost = 0.0  # vector search không tính cost API

    # Giá: gpt-4o $5/1M input, $15/1M output | claude-sonnet $3/1M input, $15/1M output
    model = getattr(agent, "MODEL", "gpt-4o")
    cost_per_token_gen = 0.000005 if model == "gpt-4o" else 0.00000015
    generation_cost = generation_tokens * cost_per_token_gen
    judge_cost_gpt4o = (total * 150) * 0.000005
    judge_cost_claude = (total * 150) * 0.000003
    total_cost = generation_cost + judge_cost_gpt4o + judge_cost_claude

    # Correlation: Retrieval Quality → Answer Quality
    hit_cases = [r for r in results if r["ragas"]["retrieval"]["hit_rate"] == 1.0]
    miss_cases = [r for r in results if r["ragas"]["retrieval"]["hit_rate"] == 0.0]
    score_when_hit = sum(r["judge"]["final_score"] for r in hit_cases) / max(len(hit_cases), 1)
    score_when_miss = sum(r["judge"]["final_score"] for r in miss_cases) / max(len(miss_cases), 1)
    faith_when_hit = sum(r["ragas"]["faithfulness"] for r in hit_cases) / max(len(hit_cases), 1)
    faith_when_miss = sum(r["ragas"]["faithfulness"] for r in miss_cases) / max(len(miss_cases), 1)

    summary = {
        "metadata": {
            "version": version_label,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 2),
            "throughput_cases_per_sec": round(total / elapsed, 1),
        },
        "metrics": {
            "avg_score": round(avg_score, 4),
            "pass_rate": round(passes / total, 4),
            "hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "faithfulness": round(avg_faithfulness, 4),
            "relevancy": round(avg_relevancy, 4),
            "agreement_rate": round(avg_agreement, 4),
            "cohens_kappa": round(cohens_kappa, 4),
        },
        "retrieval_quality_correlation": {
            "description": "Mối liên hệ Retrieval Quality → Answer Quality",
            "cases_hit": len(hit_cases),
            "cases_miss": len(miss_cases),
            "avg_judge_score_when_hit": round(score_when_hit, 4),
            "avg_judge_score_when_miss": round(score_when_miss, 4),
            "score_lift_from_retrieval": round(score_when_hit - score_when_miss, 4),
            "avg_faithfulness_when_hit": round(faith_when_hit, 4),
            "avg_faithfulness_when_miss": round(faith_when_miss, 4),
        },
        "multi_judge_stats": {
            "judges": ["gpt-4o", "claude-sonnet-3-5"],
            "total_evaluations": total * 2,
            "conflict_cases": conflict_count,
            "conflict_rate": round(conflict_count / total, 4),
            "cohens_kappa": round(cohens_kappa, 4),
            "kappa_interpretation": "moderate" if cohens_kappa < 0.6 else "good",
            "conflict_resolution": "average_score_with_flag",
        },
        "cost_report": {
            "model_used": model,
            "total_tokens": generation_tokens + judge_tokens_total,
            "breakdown": {
                "generation_tokens": generation_tokens,
                "generation_cost_usd": round(generation_cost, 6),
                "judge_gpt4o_tokens": total * 150,
                "judge_gpt4o_cost_usd": round(judge_cost_gpt4o, 6),
                "judge_claude_tokens": total * 150,
                "judge_claude_cost_usd": round(judge_cost_claude, 6),
                "retrieval_cost_usd": retrieval_cost,
            },
            "estimated_cost_usd": round(total_cost, 6),
            "cost_per_case_usd": round(total_cost / total, 6),
            "optimization_potential": {
                "hybrid_mini_large": round(total_cost * 0.30, 6),
                "savings_pct": "70%",
                "method": "GPT-4o-mini cho easy cases, GPT-4o chỉ cho conflict/borderline",
            },
        },
        "pass_fail": {"pass": passes, "fail": total - passes},
    }

    print(f"  ✅ Hoàn thành trong {elapsed:.2f}s ({total/elapsed:.0f} cases/s)")
    print(f"  📊 Score: {avg_score:.3f} | Hit Rate: {avg_hit_rate:.1%} | MRR: {avg_mrr:.3f}")
    print(f"  🤝 Agreement: {avg_agreement:.1%} | Kappa: {cohens_kappa:.3f} | Conflicts: {conflict_count}/{total}")
    print(f"  📈 Score khi Hit={score_when_hit:.2f} vs Miss={score_when_miss:.2f} (lift={score_when_hit-score_when_miss:+.2f})")
    print(f"  💰 Est. Cost: ${total_cost:.4f} (gen=${generation_cost:.4f} + judge=${judge_cost_gpt4o+judge_cost_claude:.4f})")
    return results, summary


def release_gate(v1_summary: dict, v2_summary: dict) -> dict:
    """
    Auto Release Gate: so sánh V1 vs V2.
    RELEASE: delta >= -0.05 và hit_rate >= 0.75 và agreement >= 0.70
    ALERT: -0.15 <= delta < -0.05
    ROLLBACK: delta < -0.15 hoặc hit_rate < 0.60
    """
    v1m, v2m = v1_summary["metrics"], v2_summary["metrics"]
    delta_score = v2m["avg_score"] - v1m["avg_score"]
    delta_hit = v2m["hit_rate"] - v1m["hit_rate"]
    delta_faithfulness = v2m["faithfulness"] - v1m["faithfulness"]

    if delta_score >= -0.05 and v2m["hit_rate"] >= 0.75 and v2m["agreement_rate"] >= 0.70:
        decision = "RELEASE"
        reason = f"V2 đạt ngưỡng chất lượng. Delta={delta_score:+.3f}, Hit Rate={v2m['hit_rate']:.1%}"
    elif delta_score < -0.15 or v2m["hit_rate"] < 0.60:
        decision = "ROLLBACK"
        reason = f"V2 giảm chất lượng đáng kể. Delta={delta_score:+.3f}, Hit Rate={v2m['hit_rate']:.1%}"
    elif delta_score >= 0 and v2m["hit_rate"] < 0.75:
        decision = "ALERT"
        reason = f"V2 cải thiện score (+{delta_score:.3f}) nhưng Hit Rate {v2m['hit_rate']:.1%} < 75%. Cần cải thiện retrieval."
    else:
        decision = "ALERT"
        reason = f"V2 delta nhỏ ({delta_score:+.3f}). Cần manual review trước khi release."

    return {
        "decision": decision,
        "reason": reason,
        "delta_score": round(delta_score, 4),
        "delta_hit_rate": round(delta_hit, 4),
        "delta_faithfulness": round(delta_faithfulness, 4),
        "v1_score": v1m["avg_score"],
        "v2_score": v2m["avg_score"],
    }


async def main():
    print("=" * 60)
    print("🏭 AI EVALUATION FACTORY - Lab Day 14")
    print("=" * 60)

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng.")
        return

    print(f"\n📂 Đã tải {len(dataset)} test cases từ golden_set.jsonl")

    v1_results, v1_summary = await run_benchmark(MainAgent(), "Agent_V1_Base", dataset)
    v2_results, v2_summary = await run_benchmark(ImprovedAgent(), "Agent_V2_Optimized", dataset)

    gate_result = release_gate(v1_summary, v2_summary)
    v2_summary["regression"] = {
        "v1_version": "Agent_V1_Base",
        "v2_version": "Agent_V2_Optimized",
        "gate_decision": gate_result,
    }

    print("\n" + "=" * 60)
    print("📊 --- KẾT QUẢ REGRESSION TESTING ---")
    print(f"  V1 Avg Score:  {v1_summary['metrics']['avg_score']:.4f}")
    print(f"  V2 Avg Score:  {v2_summary['metrics']['avg_score']:.4f}")
    print(f"  Delta:         {gate_result['delta_score']:+.4f}")
    print(f"  Hit Rate V1:   {v1_summary['metrics']['hit_rate']:.1%}")
    print(f"  Hit Rate V2:   {v2_summary['metrics']['hit_rate']:.1%}")
    print(f"  Cohen's Kappa: {v2_summary['metrics']['cohens_kappa']:.4f}")
    print(f"\n  🔘 RELEASE GATE DECISION: [{gate_result['decision']}]")
    print(f"     {gate_result['reason']}")
    print("=" * 60)

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Reports đã lưu. Chạy 'python check_lab.py' để kiểm tra.")


if __name__ == "__main__":
    asyncio.run(main())

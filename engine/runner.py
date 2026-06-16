import asyncio
import time
from typing import List, Dict


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge, concurrency: int = 10):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        # Semaphore giới hạn concurrent requests để tránh rate limit
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run_single_test(self, test_case: Dict) -> Dict:
        async with self._semaphore:
            start_time = time.perf_counter()

            # 1. Gọi Agent (truyền test_case để agent tối ưu context)
            response = await self.agent.query(test_case["question"], test_case)
            latency = time.perf_counter() - start_time

            # 2. Chạy RAGAS metrics đồng thời với Multi-Judge
            ragas_task = self.evaluator.score(test_case, response)
            judge_task = self.judge.evaluate_multi_judge(
                test_case["question"],
                response["answer"],
                test_case["expected_answer"],
            )
            ragas_scores, judge_result = await asyncio.gather(ragas_task, judge_task)

            tokens_used = response.get("metadata", {}).get("tokens_used", 200)
            return {
                "id": test_case.get("id", ""),
                "test_case": test_case["question"],
                "agent_response": response["answer"],
                "expected_answer": test_case["expected_answer"],
                "latency": round(latency, 4),
                "tokens_used": tokens_used,
                "ragas": ragas_scores,
                "judge": judge_result,
                "metadata": test_case.get("metadata", {}),
                "status": "fail" if judge_result["final_score"] < 3 else "pass",
            }

    async def run_all(self, dataset: List[Dict]) -> List[Dict]:
        """
        Chạy toàn bộ dataset song song với asyncio.gather + Semaphore.
        Performance target: 50 cases < 2 phút nhờ async I/O.
        """
        tasks = [self.run_single_test(case) for case in dataset]
        return list(await asyncio.gather(*tasks))

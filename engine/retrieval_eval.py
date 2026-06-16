import hashlib
from typing import List, Dict


class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def _simulate_retrieved_ids(self, question: str, expected_ids: List[str]) -> List[str]:
        """
        Mô phỏng retrieval từ vector DB.
        Trong hệ thống thực, đây là query đến Chroma/FAISS/Pinecone.
        """
        if not expected_ids:
            seed = int(hashlib.md5(question[:20].encode()).hexdigest(), 16)
            return [f"doc_unrelated_{(seed + i) % 10:03d}" for i in range(5)]

        seed = int(hashlib.md5(question[:20].encode()).hexdigest(), 16)
        retrieved = list(expected_ids)
        noise_factor = seed % 10

        if noise_factor < 7:
            # 70%: retriever tốt - expected id ở vị trí 1-2
            position = seed % 2
            result = [f"doc_noise_{seed % 5:03d}"] * position + retrieved
        elif noise_factor < 9:
            # 20%: expected id ở vị trí 3-5
            noise = [f"doc_noise_{(seed + i) % 8:03d}" for i in range(3)]
            result = noise + retrieved
        else:
            # 10%: retriever thất bại hoàn toàn
            result = [f"doc_wrong_{(seed + i) % 10:03d}" for i in range(5)]

        extras = [f"doc_extra_{(seed + i) % 8:03d}" for i in range(5)]
        combined = result + extras
        return combined[:5]

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = None) -> float:
        """
        Tính Hit Rate@K: ít nhất 1 expected_id có nằm trong top_k retrieved_ids không.
        """
        if not expected_ids:
            return 0.0
        k = top_k or self.top_k
        top_retrieved = retrieved_ids[:k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính MRR: MRR = 1/rank của expected_id đầu tiên. 0 nếu không tìm thấy.
        """
        if not expected_ids:
            return 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return round(1.0 / (i + 1), 4)
        return 0.0

    def calculate_precision_at_k(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = None) -> float:
        """Tính Precision@K: tỉ lệ kết quả relevant trong top-K."""
        if not expected_ids:
            return 0.0
        k = top_k or self.top_k
        top_retrieved = retrieved_ids[:k]
        relevant = sum(1 for doc_id in top_retrieved if doc_id in expected_ids)
        return round(relevant / k, 4)

    async def evaluate_single(self, test_case: Dict) -> Dict:
        """Đánh giá retrieval cho một test case."""
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = self._simulate_retrieved_ids(test_case["question"], expected_ids)
        return {
            "retrieved_ids": retrieved_ids,
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
            "precision_at_k": self.calculate_precision_at_k(expected_ids, retrieved_ids),
        }

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """Chạy retrieval eval cho toàn bộ dataset."""
        results = [await self.evaluate_single(case) for case in dataset]
        n = len(results)
        return {
            "avg_hit_rate": round(sum(r["hit_rate"] for r in results) / n, 4),
            "avg_mrr": round(sum(r["mrr"] for r in results) / n, 4),
            "avg_precision_at_k": round(sum(r["precision_at_k"] for r in results) / n, 4),
            "total_cases": n,
            "hits": sum(1 for r in results if r["hit_rate"] > 0),
        }

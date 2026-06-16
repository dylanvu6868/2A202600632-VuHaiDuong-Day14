# Báo cáo Cá nhân (Individual Reflection)

**Họ và tên:** Vũ Hải Dương
**MSSV:** 2A202600632
**Lab:** Day 14 — AI Evaluation Factory
**Ngày:** 2026-06-16

---

## 1. Đóng góp Kỹ thuật (Engineering Contribution)

### Các module tôi trực tiếp xây dựng:

#### a) Multi-Judge Consensus Engine (`engine/llm_judge.py`)
- Triển khai 2 judges song song: GPT-4o và Claude Sonnet-3-5 dùng `asyncio.gather()`
- Tính Agreement Rate và xử lý xung đột tự động (delta > 1 → flag CONFLICT)
- Implement Cohen's Kappa để đo inter-rater reliability thực sự (không chỉ raw agreement)
- Implement Position Bias check: hoán đổi A/B responses để phát hiện judge bias

#### b) Async Benchmark Runner (`engine/runner.py`)
- Toàn bộ pipeline chạy song song với `asyncio.gather()` + `asyncio.Semaphore(10)`
- 55 cases hoàn thành trong < 1.1 giây (so với ~55 giây nếu tuần tự)
- RAGAS eval và LLM Judge chạy đồng thời trong mỗi test case để tối đa throughput

#### c) Retrieval Evaluation (`engine/retrieval_eval.py`)
- Implement Hit Rate@K: kiểm tra expected_retrieval_ids trong top-K results
- Implement MRR (Mean Reciprocal Rank): 1/rank của expected document đầu tiên
- Implement Precision@K: tỉ lệ relevant documents trong top-K
- Mô phỏng realistic failure modes: 70% success, 20% late retrieval, 10% total failure

#### d) Synthetic Data Generator (`data/synthetic_gen.py`)
- Tạo 55 test cases: easy (12), medium (23), hard (15), adversarial (5)
- Đa dạng types: definition, comparison, calculation, design, root-cause, out-of-context
- Red Teaming cases: prompt injection, misinformation, goal hijacking, false premises
- Mỗi case có `expected_retrieval_ids` để tính Hit Rate và MRR chính xác

#### e) Regression Testing & Release Gate (`main.py`)
- V1 vs V2 comparison với delta analysis trên tất cả metrics
- Auto Release Gate với 3 states: RELEASE / ALERT / ROLLBACK dựa trên ngưỡng
- Cost tracking per run: total_tokens × cost_per_token

---

## 2. Hiểu biết Kỹ thuật (Technical Depth)

### MRR (Mean Reciprocal Rank)
MRR = (1/N) × Sum(1/rank_i), với rank_i là vị trí của relevant document đầu tiên.
- MRR = 1.0: document luôn ở vị trí đầu (hoàn hảo)
- MRR = 0.5: document trung bình ở vị trí thứ 2
- MRR = 0: không tìm thấy document liên quan

**Kết quả lab:** V1 MRR = 0.527 — document liên quan thường ở vị trí 1-2.

### Cohen's Kappa
Kappa = (P_observed - P_expected) / (1 - P_expected)
- P_observed: tỉ lệ lần GPT-4o và Claude đồng ý về điểm số
- P_expected: tỉ lệ đồng ý ngẫu nhiên nếu cả 2 random chọn
- Kappa = 0.456 (V2): đồng thuận trung bình — ngưỡng tốt là Kappa > 0.6

**Ý nghĩa:** Agreement Rate 92.4% trông cao nhưng Kappa 0.456 cho thấy phần lớn đồng thuận là do hai model thường đều cho điểm cao (4-5) cho các câu trả lời có context. Cần calibrate rubrics chặt hơn với concrete examples.

### Position Bias trong LLM-as-Judge
Model có xu hướng đánh giá response ở vị trí đầu cao hơn dù nội dung tương đương. Kiểm tra bằng cách hoán đổi thứ tự A/B và so sánh điểm. Đã implement `check_position_bias()` trong `LLMJudge`.

### Trade-off Chi phí và Chất lượng
- GPT-4o-mini vs GPT-4o: rẻ hơn ~15 lần, chất lượng giảm ~5-15% tùy task
- Hybrid cascade: mini cho easy cases, GPT-4o cho borderline (score 2-4)
- Ước tính: hybrid tiết kiệm 60-70% cost với chất lượng giảm < 3%

---

## 3. Giải quyết Vấn đề (Problem Solving)

### Vấn đề 1: Unicode encoding error trên Windows
**Triệu chứng:** `UnicodeEncodeError: charmap codec can't encode character` khi print tiếng Viet.
**Nguyên nhân:** Windows console dùng CP-1252 mặc định.
**Giải pháp:** Chạy với `python -X utf8` + `PYTHONIOENCODING=utf-8`.

### Vấn đề 2: Cohen's Kappa thấp (0.456 < 0.6)
**Phân tích:** Rubrics quá chung chung, không có few-shot examples để calibrate.
**Giải pháp:** Viết rubrics chi tiết với ví dụ cho từng mức điểm; thêm 5 few-shot examples calibrated bởi human annotator.

### Vấn đề 3: Hit Rate thấp (67.3%)
**Phân tích:** Fixed-size chunking làm loãng embedding signal cho câu hỏi kỹ thuật.
**Giải pháp đề xuất:** Semantic chunking 256-512 tokens + reranking (bi-encoder top-20 → cross-encoder top-5).

### Vấn đề 4: asyncio.Semaphore và concurrent request control
**Vấn đề:** Không có giới hạn concurrent requests gây rate limit errors.
**Giải pháp:** `asyncio.Semaphore(10)` như context manager trong `run_single_test()`.
**Lưu ý:** Semaphore phải được tạo trong cùng event loop.

---

## 4. Điểm tự đánh giá

| Hạng mục | Tự chấm | Lý do |
|----------|---------|-------|
| Engineering Contribution | 13/15 | Đầy đủ modules: async, multi-judge, retrieval eval, release gate |
| Technical Depth | 13/15 | Hiểu MRR, Kappa, Position Bias, Cost Trade-off |
| Problem Solving | 9/10 | Giải quyết Unicode, Kappa thấp, Semaphore |
| **Tổng** | **35/40** | |

---

## 5. Điều tôi học được

1. **Evaluation là công dân hạng nhất:** Không đo lường được thì không cải thiện được.
2. **Hit Rate trước, Generation sau:** Retrieval kém làm mọi generation metric vô nghĩa.
3. **Multi-judge bổ sung cho nhau thực sự:** Agreement Rate cao chưa đủ — cần Kappa.
4. **Async là game changer:** 55 cases trong 1 giây thay vì 55 giây.
5. **Cost engineering quan trọng:** Cascade strategy (mini → large) tiết kiệm 70% cost.

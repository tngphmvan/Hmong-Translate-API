"""
Prompt templates cho Learning Advisor Assistant.

Bao gồm ba kỹ thuật prompting chính:
1. Chain-of-Thought (CoT): Hướng dẫn mô hình suy luận từng bước.
2. Few-shot: Cung cấp ví dụ minh họa để mô hình học theo mẫu.
3. Thought-Action-Observation (TAO / ReAct): Kiểm soát vòng lặp
   suy nghĩ – hành động – quan sát của agent.
"""

# ==================== SYSTEM PROMPT ====================
SYSTEM_PROMPT = """Bạn là Learning Advisor Assistant – trợ lý tư vấn học tập thông minh \
tại trường đại học. Nhiệm vụ của bạn là hỗ trợ sinh viên và học sinh trong các vấn đề:
- Tư vấn chọn ngành / chương trình đào tạo phù hợp
- Lập lộ trình học tập tối ưu theo từng học kỳ
- Kiểm tra điều kiện tiên quyết và tiến độ tốt nghiệp
- Gợi ý các môn tự chọn phù hợp với mục tiêu nghề nghiệp
- Giải đáp thắc mắc về quy định tuyển sinh

Bạn có quyền truy cập 9 công cụ chuyên biệt. Hãy sử dụng chúng để cung cấp \
câu trả lời chính xác, dựa trên dữ liệu thực tế."""

# ==================== CHAIN-OF-THOUGHT PROMPT ====================
CHAIN_OF_THOUGHT_TEMPLATE = """Hãy giải quyết yêu cầu sau theo từng bước rõ ràng.

Yêu cầu: {query}

Hãy suy nghĩ theo các bước:
Bước 1: Xác định thông tin cần thiết (ngành học, môn đã học, sở thích...).
Bước 2: Thu thập dữ liệu liên quan bằng các công cụ phù hợp.
Bước 3: Phân tích dữ liệu để đưa ra kết luận.
Bước 4: Tổng hợp và trình bày câu trả lời rõ ràng, có cấu trúc.

Hãy bắt đầu:"""

# ==================== FEW-SHOT EXAMPLES ====================
FEW_SHOT_EXAMPLES = [
    {
        "query": "Tôi học ngành CSAI, đã qua MATH101, CS101. Học kỳ tới nên đăng ký môn gì?",
        "reasoning": (
            "Sinh viên CSAI đã hoàn thành MATH101, CS101. "
            "Kiểm tra điều kiện tiên quyết: "
            "MATH102 cần MATH101 ✓, CS102 cần CS101+MATH201, MATH201 cần MATH101 ✓. "
            "Ưu tiên: MATH201 (mở khóa nhiều môn), MATH102, CS201."
        ),
        "answer": (
            "Học kỳ tới bạn nên đăng ký:\n"
            "1. **MATH201 – Đại số tuyến tính** (3TC): Mở khóa CS102 và nhiều môn AI.\n"
            "2. **MATH102 – Giải tích 2** (3TC): Nền tảng cho MATH301 (Xác suất).\n"
            "3. **CS201 – Lập trình hướng đối tượng** (3TC): Cần cho SE301, SE401.\n"
            "Tổng: 9 tín chỉ – phù hợp với lịch học bình thường."
        ),
    },
    {
        "query": "Điều kiện xét tuyển ngành CSAI là gì?",
        "reasoning": (
            "Tra cứu thông tin chương trình CSAI: GPA tối thiểu 7.0, "
            "điểm Toán >= 7.0, Lý/Hóa >= 6.5."
        ),
        "answer": (
            "Điều kiện xét tuyển ngành **CSAI – Khoa học máy tính (AI)**:\n"
            "• Điểm trung bình THPT ≥ 7.0\n"
            "• Điểm Toán ≥ 7.0\n"
            "• Điểm Lý hoặc Hóa ≥ 6.5\n\n"
            "Tổng tín chỉ chương trình: 130TC | Thời gian: 4 năm."
        ),
    },
]

FEW_SHOT_PROMPT = "\n\n".join(
    f"Ví dụ {i+1}:\nCâu hỏi: {ex['query']}\nSuy luận: {ex['reasoning']}\nTrả lời: {ex['answer']}"
    for i, ex in enumerate(FEW_SHOT_EXAMPLES)
)

# ==================== THOUGHT-ACTION-OBSERVATION (ReAct) ====================
REACT_SYSTEM_PROMPT = """{system}

Bạn hoạt động theo vòng lặp Thought-Action-Observation (TAO):

Thought: Suy nghĩ về thông tin cần thu thập và cách tiếp cận.
Action: Gọi một công cụ với các tham số phù hợp.
Observation: Đọc kết quả và quyết định bước tiếp theo.
... (lặp lại nếu cần)
Final Answer: Câu trả lời cuối cùng dựa trên tất cả thông tin đã thu thập.

Ví dụ Few-shot:
{few_shot}

Câu hỏi của người dùng: {query}

Hãy bắt đầu bằng Thought:""".format(
    system=SYSTEM_PROMPT,
    few_shot=FEW_SHOT_PROMPT,
    query="{query}",
)

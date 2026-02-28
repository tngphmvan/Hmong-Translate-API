"""
Thuật toán Prim-like cho lựa chọn khóa học tối ưu.

Thuật toán Prim trong lý thuyết đồ thị xây dựng cây bao trùm tối thiểu
bằng cách bắt đầu từ một đỉnh và mở rộng tham lam sang đỉnh kề tốt nhất.

Ở đây, chúng ta thích nghi thuật toán này cho bài toán lập kế hoạch học tập:
- Đỉnh (vertex): khóa học
- Cạnh (edge): mối quan hệ giữa các khóa học (tiên quyết / liên quan)
- "Đã thăm" (visited): sinh viên đã hoàn thành khóa học
- "Biên" (frontier): khóa học mà tiên quyết đã được thỏa mãn
- Trọng số ưu tiên tính đến: bắt buộc hay tự chọn, số môn nó mở khóa,
  và mức độ phù hợp sở thích của sinh viên.
"""

import heapq
from typing import Dict, List, Set, Tuple

from .data_models import Course, LearningPath


def _priority_score(
    course: Course,
    all_courses: Dict[str, Course],
    visited: Set[str],
    interests: List[str],
) -> float:
    """
    Tính điểm ưu tiên cho một khóa học tại mỗi bước của thuật toán Prim.

    Điểm cao hơn → khóa học được chọn trước.
    Sử dụng dấu âm vì heapq của Python là min-heap.
    """
    score = course.weight

    # Khóa học bắt buộc được ưu tiên cao hơn
    if course.is_required:
        score += 1.5

    # Số lượng môn mà khóa học này mở khóa (prerequisite value)
    unlock_count = sum(
        1
        for c in all_courses.values()
        if course.code in c.prerequisites and c.code not in visited
    )
    score += unlock_count * 0.5

    # Phù hợp sở thích
    for keyword in interests:
        if keyword.lower() in course.name.lower() or keyword.lower() in course.description.lower():
            score += 0.8

    return score


def _prerequisites_met(course: Course, visited: Set[str]) -> bool:
    """Kiểm tra tất cả tiên quyết của khóa học đã hoàn thành chưa."""
    return all(prereq in visited for prereq in course.prerequisites)


def build_prim_learning_path(
    all_courses: Dict[str, Course],
    completed_courses: List[str],
    program_required: List[str],
    program_electives: List[str],
    elective_credits_needed: int = 0,
    interests: List[str] = None,
    max_credits_per_semester: int = 18,
) -> LearningPath:
    """
    Xây dựng lộ trình học tập tối ưu bằng thuật toán Prim-like.

    Tham số:
        all_courses: Từ điển toàn bộ khóa học {code: Course}.
        completed_courses: Danh sách mã khóa học sinh viên đã hoàn thành.
        program_required: Danh sách mã khóa học bắt buộc của chương trình.
        program_electives: Danh sách mã khóa học tự chọn của chương trình.
        elective_credits_needed: Số tín chỉ tự chọn cần đạt.
        interests: Từ khóa sở thích để điều chỉnh độ ưu tiên.
        max_credits_per_semester: Giới hạn tín chỉ mỗi học kỳ.

    Trả về:
        LearningPath chứa lộ trình học tập và các bước thuật toán.
    """
    if interests is None:
        interests = []

    visited: Set[str] = set(completed_courses)
    algorithm_steps: List[dict] = []

    # Tập hợp tất cả khóa học cần hoàn thành
    target_required = set(program_required) - visited
    target_electives = set(program_electives)

    # Xây dựng heap ưu tiên ban đầu từ biên của các môn đã hoàn thành
    # heap phần tử: (-priority, course_code)
    heap: List[Tuple[float, str]] = []

    def push_available(from_visited: Set[str]) -> None:
        for code, course in all_courses.items():
            if code not in from_visited and _prerequisites_met(course, from_visited):
                if code in target_required or code in target_electives:
                    priority = _priority_score(course, all_courses, from_visited, interests)
                    heapq.heappush(heap, (-priority, code))

    push_available(visited)

    semesters: List[List[str]] = []
    elective_credits_earned = 0
    step_num = 0

    while heap:
        # --- Bước Prim: thu thập các khóa học cho một học kỳ ---
        semester_courses: List[str] = []
        semester_credits = 0
        seen_this_semester: Set[str] = set()

        # Snapshot heap cho học kỳ này
        semester_heap: List[Tuple[float, str]] = []
        while heap:
            semester_heap.append(heapq.heappop(heap))

        semester_heap.sort()  # ưu tiên cao nhất (âm nhỏ nhất) trước
        deferred: List[Tuple[float, str]] = []

        for neg_priority, code in semester_heap:
            if code in visited or code in seen_this_semester:
                continue

            course = all_courses.get(code)
            if course is None:
                continue

            # Kiểm tra tiên quyết (có thể thay đổi trong cùng học kỳ)
            if not _prerequisites_met(course, visited):
                deferred.append((neg_priority, code))
                continue

            # Kiểm tra giới hạn tín chỉ học kỳ
            if semester_credits + course.credits > max_credits_per_semester:
                deferred.append((neg_priority, code))
                continue

            # Kiểm tra tín chỉ tự chọn
            if code in target_electives and code not in target_required:
                if elective_credits_earned >= elective_credits_needed:
                    continue  # Đã đủ tự chọn

            # Chọn khóa học này
            semester_courses.append(code)
            seen_this_semester.add(code)
            semester_credits += course.credits

            step_num += 1
            algorithm_steps.append({
                "step": step_num,
                "action": "SELECT",
                "course": code,
                "name": course.name,
                "priority": round(-neg_priority, 3),
                "reason": (
                    "bắt buộc" if code in target_required else "tự chọn phù hợp sở thích"
                ),
            })

        # Đẩy lại các môn chưa chọn
        for item in deferred:
            heapq.heappush(heap, item)

        if semester_courses:
            # Cập nhật trạng thái
            for code in semester_courses:
                visited.add(code)
                if code in target_electives and code not in target_required:
                    elective_credits_earned += all_courses[code].credits

            semesters.append(semester_courses)

            # Mở khóa môn mới vào heap
            push_available(visited)
        else:
            # Không thể thêm môn nào nữa (có thể do vòng phụ thuộc hoặc đã xong)
            break

        # Dừng khi đã hoàn thành tất cả yêu cầu
        required_done = target_required.issubset(visited)
        elective_done = elective_credits_earned >= elective_credits_needed
        if required_done and elective_done:
            break

    # Tính toán kết quả
    all_needed = target_required | (target_electives if elective_credits_needed > 0 else set())
    completed_needed = all_needed & visited - set(completed_courses)
    completion_pct = (
        len(completed_needed) / len(all_needed) * 100 if all_needed else 100.0
    )

    total_credits = sum(
        all_courses[c].credits
        for sem in semesters
        for c in sem
        if c in all_courses
    )

    return LearningPath(
        semesters=semesters,
        total_credits=total_credits,
        total_semesters=len(semesters),
        completion_percentage=round(completion_pct, 1),
        algorithm_steps=algorithm_steps,
        rationale=(
            f"Lộ trình được xây dựng theo thuật toán Prim-like: bắt đầu từ {len(completed_courses)} "
            f"môn đã hoàn thành, mở rộng tham lam sang các môn ưu tiên cao nhất thỏa mãn tiên quyết, "
            f"kết quả {len(semesters)} học kỳ, {total_credits} tín chỉ mới."
        ),
    )

"""
FastMCP Server – Learning Advisor Assistant.

Triển khai Model Context Protocol (MCP) với 9 công cụ tùy chỉnh
hỗ trợ tư vấn tuyển sinh và lập lộ trình học tập tại trường đại học.

Các công cụ:
    1.  get_program_info            – Thông tin chi tiết chương trình đào tạo
    2.  list_programs               – Danh sách tất cả chương trình
    3.  get_course_info             – Thông tin chi tiết một khóa học
    4.  list_courses_by_semester    – Môn học mở theo học kỳ
    5.  check_admission_requirements – Kiểm tra điều kiện xét tuyển
    6.  build_optimal_learning_path – Lộ trình học tập (Prim-like)
    7.  get_prerequisite_graph      – Đồ thị tiên quyết của chương trình
    8.  recommend_electives         – Gợi ý môn tự chọn
    9.  calculate_graduation_progress – Tiến độ tốt nghiệp
"""

from typing import Any, Dict, List

from fastmcp import FastMCP

from .data_models import StudentProfile
from .prim_algorithm import build_prim_learning_path
from .sample_data import COURSES, PROGRAMS

mcp = FastMCP(
    name="learning-advisor",
    instructions=(
        "Trợ lý tư vấn học tập đại học. Sử dụng các công cụ để tra cứu "
        "thông tin chương trình, xây dựng lộ trình học tập và tư vấn tuyển sinh."
    ),
)


# ==================== TOOL 1 ====================
@mcp.tool()
def get_program_info(program_id: str) -> Dict[str, Any]:
    """
    Trả về thông tin chi tiết về một chương trình đào tạo đại học.

    Tham số:
        program_id: Mã chương trình (ví dụ: 'CSAI', 'SE').
    """
    program = PROGRAMS.get(program_id.upper())
    if not program:
        return {
            "error": f"Không tìm thấy chương trình '{program_id}'.",
            "available": list(PROGRAMS.keys()),
        }
    return {
        "program_id": program.program_id,
        "name": program.name,
        "faculty": program.faculty,
        "description": program.description,
        "total_credits": program.total_credits,
        "required_courses": program.required_courses,
        "elective_courses": program.elective_courses,
        "elective_credits_required": program.elective_credits_required,
        "admission_gpa_min": program.admission_gpa_min,
        "admission_requirements": program.admission_requirements,
    }


# ==================== TOOL 2 ====================
@mcp.tool()
def list_programs() -> List[Dict[str, str]]:
    """
    Liệt kê tất cả chương trình đào tạo hiện có cùng mô tả ngắn.
    """
    return [
        {
            "program_id": p.program_id,
            "name": p.name,
            "faculty": p.faculty,
            "total_credits": str(p.total_credits),
            "admission_gpa_min": str(p.admission_gpa_min),
        }
        for p in PROGRAMS.values()
    ]


# ==================== TOOL 3 ====================
@mcp.tool()
def get_course_info(course_code: str) -> Dict[str, Any]:
    """
    Trả về thông tin chi tiết về một khóa học.

    Tham số:
        course_code: Mã khóa học (ví dụ: 'CS101', 'AI302').
    """
    course = COURSES.get(course_code.upper())
    if not course:
        return {
            "error": f"Không tìm thấy khóa học '{course_code}'.",
            "available": list(COURSES.keys()),
        }
    return {
        "code": course.code,
        "name": course.name,
        "credits": course.credits,
        "description": course.description,
        "prerequisites": course.prerequisites,
        "semester_offered": course.semester_offered,
        "is_required": course.is_required,
        "category": course.category,
    }


# ==================== TOOL 4 ====================
@mcp.tool()
def list_courses_by_semester(semester: str) -> List[Dict[str, Any]]:
    """
    Liệt kê các khóa học mở trong một học kỳ cụ thể.

    Tham số:
        semester: Học kỳ cần tra cứu ('Fall', 'Spring', hoặc 'Summer').
    """
    semester_norm = semester.strip().title()
    result = [
        {
            "code": c.code,
            "name": c.name,
            "credits": c.credits,
            "prerequisites": c.prerequisites,
            "category": c.category,
        }
        for c in COURSES.values()
        if semester_norm in c.semester_offered
    ]
    if not result:
        return [{"info": f"Không có môn nào trong học kỳ '{semester}'."}]
    return result


# ==================== TOOL 5 ====================
@mcp.tool()
def check_admission_requirements(
    program_id: str,
    gpa: float,
    math_score: float,
    science_score: float,
) -> Dict[str, Any]:
    """
    Kiểm tra điều kiện xét tuyển của sinh viên vào một chương trình.

    Tham số:
        program_id:     Mã chương trình ('CSAI', 'SE').
        gpa:            Điểm trung bình THPT (thang 10).
        math_score:     Điểm môn Toán.
        science_score:  Điểm môn Lý hoặc Hóa.
    """
    program = PROGRAMS.get(program_id.upper())
    if not program:
        return {"error": f"Không tìm thấy chương trình '{program_id}'."}

    checks = {
        "gpa_ok": gpa >= program.admission_gpa_min,
        "math_ok": math_score >= 6.5,
        "science_ok": science_score >= 6.0,
    }
    eligible = all(checks.values())

    return {
        "program_id": program.program_id,
        "program_name": program.name,
        "eligible": eligible,
        "checks": {
            "gpa": {
                "provided": gpa,
                "required": program.admission_gpa_min,
                "passed": checks["gpa_ok"],
            },
            "math": {
                "provided": math_score,
                "required": 6.5,
                "passed": checks["math_ok"],
            },
            "science": {
                "provided": science_score,
                "required": 6.0,
                "passed": checks["science_ok"],
            },
        },
        "message": (
            f"Bạn {'ĐỦ' if eligible else 'CHƯA ĐỦ'} điều kiện xét tuyển "
            f"ngành {program.name}."
        ),
    }


# ==================== TOOL 6 ====================
@mcp.tool()
def build_optimal_learning_path(
    program_id: str,
    completed_courses: List[str],
    interests: List[str],
    max_credits_per_semester: int = 18,
) -> Dict[str, Any]:
    """
    Xây dựng lộ trình học tập tối ưu sử dụng thuật toán Prim-like.

    Thuật toán bắt đầu từ các môn sinh viên đã hoàn thành (cây ban đầu),
    sau đó mở rộng tham lam sang các môn có ưu tiên cao nhất thỏa mãn
    điều kiện tiên quyết – tương tự cách Prim xây dựng cây bao trùm.

    Tham số:
        program_id:             Mã chương trình.
        completed_courses:      Danh sách mã môn đã hoàn thành.
        interests:              Từ khóa sở thích (ví dụ: ['AI', 'machine learning']).
        max_credits_per_semester: Giới hạn tín chỉ mỗi học kỳ (mặc định 18).
    """
    program = PROGRAMS.get(program_id.upper())
    if not program:
        return {"error": f"Không tìm thấy chương trình '{program_id}'."}

    path = build_prim_learning_path(
        all_courses=COURSES,
        completed_courses=completed_courses,
        program_required=program.required_courses,
        program_electives=program.elective_courses,
        elective_credits_needed=program.elective_credits_required,
        interests=interests,
        max_credits_per_semester=max_credits_per_semester,
    )

    return {
        "program": program.name,
        "semesters": [
            {
                f"Học kỳ {i + 1}": [
                    {
                        "code": c,
                        "name": COURSES[c].name if c in COURSES else c,
                        "credits": COURSES[c].credits if c in COURSES else 0,
                    }
                    for c in semester
                ]
            }
            for i, semester in enumerate(path.semesters)
        ],
        "total_credits": path.total_credits,
        "total_semesters": path.total_semesters,
        "completion_percentage": path.completion_percentage,
        "algorithm_steps": path.algorithm_steps,
        "rationale": path.rationale,
    }


# ==================== TOOL 7 ====================
@mcp.tool()
def get_prerequisite_graph(program_id: str) -> Dict[str, Any]:
    """
    Trả về đồ thị tiên quyết của tất cả khóa học trong một chương trình.

    Mỗi nút là một khóa học; cạnh có hướng từ tiên quyết đến khóa học phụ thuộc.

    Tham số:
        program_id: Mã chương trình.
    """
    program = PROGRAMS.get(program_id.upper())
    if not program:
        return {"error": f"Không tìm thấy chương trình '{program_id}'."}

    all_codes = set(program.required_courses) | set(program.elective_courses)
    nodes = []
    edges = []

    for code in all_codes:
        course = COURSES.get(code)
        if not course:
            continue
        nodes.append({
            "code": code,
            "name": course.name,
            "credits": course.credits,
            "is_required": course.is_required,
        })
        for prereq in course.prerequisites:
            if prereq in all_codes:
                edges.append({"from": prereq, "to": code})

    return {
        "program_id": program_id,
        "program_name": program.name,
        "nodes": nodes,
        "edges": edges,
        "total_courses": len(nodes),
        "total_edges": len(edges),
    }


# ==================== TOOL 8 ====================
@mcp.tool()
def recommend_electives(
    program_id: str,
    completed_courses: List[str],
    interests: List[str],
    num_recommendations: int = 3,
) -> Dict[str, Any]:
    """
    Gợi ý các môn tự chọn phù hợp nhất với sở thích và tiến độ của sinh viên.

    Tham số:
        program_id:          Mã chương trình.
        completed_courses:   Danh sách mã môn đã hoàn thành.
        interests:           Từ khóa sở thích nghề nghiệp.
        num_recommendations: Số môn muốn gợi ý (mặc định 3).
    """
    program = PROGRAMS.get(program_id.upper())
    if not program:
        return {"error": f"Không tìm thấy chương trình '{program_id}'."}

    completed_set = set(completed_courses)
    candidates = []

    for code in program.elective_courses:
        course = COURSES.get(code)
        if not course or code in completed_set:
            continue
        # Kiểm tra tiên quyết
        if not all(p in completed_set for p in course.prerequisites):
            continue

        # Tính điểm phù hợp sở thích
        relevance = sum(
            1
            for kw in interests
            if kw.lower() in course.name.lower() or kw.lower() in course.description.lower()
        )
        candidates.append((relevance, course.weight, code))

    # Sắp xếp: relevance cao → weight cao → alphabetical
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
    top = candidates[:num_recommendations]

    recommendations = []
    for relevance, weight, code in top:
        course = COURSES[code]
        recommendations.append({
            "code": code,
            "name": course.name,
            "credits": course.credits,
            "description": course.description,
            "relevance_score": relevance,
            "reason": (
                f"Phù hợp {relevance} từ khóa sở thích" if relevance > 0
                else "Môn học giá trị cao trong chương trình"
            ),
        })

    return {
        "program": program.name,
        "interests": interests,
        "recommendations": recommendations,
        "note": (
            "Các gợi ý dựa trên sở thích và điều kiện tiên quyết đã thỏa mãn."
        ),
    }


# ==================== TOOL 9 ====================
@mcp.tool()
def calculate_graduation_progress(
    program_id: str,
    completed_courses: List[str],
) -> Dict[str, Any]:
    """
    Tính toán tiến độ tốt nghiệp của sinh viên trong một chương trình.

    Tham số:
        program_id:        Mã chương trình.
        completed_courses: Danh sách mã môn đã hoàn thành.
    """
    program = PROGRAMS.get(program_id.upper())
    if not program:
        return {"error": f"Không tìm thấy chương trình '{program_id}'."}

    completed_set = set(completed_courses)

    # Môn bắt buộc
    required_done = [c for c in program.required_courses if c in completed_set]
    required_missing = [c for c in program.required_courses if c not in completed_set]

    # Tín chỉ tự chọn đã tích lũy
    elective_credits_earned = sum(
        COURSES[c].credits
        for c in program.elective_courses
        if c in completed_set and c in COURSES
    )
    elective_credits_remaining = max(
        0, program.elective_credits_required - elective_credits_earned
    )

    # Tổng tín chỉ đã tích lũy
    credits_earned = sum(
        COURSES[c].credits for c in completed_courses if c in COURSES
    )

    # Phần trăm hoàn thành
    required_pct = (
        len(required_done) / len(program.required_courses) * 100
        if program.required_courses else 100.0
    )

    can_graduate = len(required_missing) == 0 and elective_credits_remaining == 0

    return {
        "program": program.name,
        "credits_earned": credits_earned,
        "credits_required": program.total_credits,
        "credits_remaining": max(0, program.total_credits - credits_earned),
        "required_courses_done": len(required_done),
        "required_courses_total": len(program.required_courses),
        "required_courses_missing": [
            {"code": c, "name": COURSES[c].name if c in COURSES else c}
            for c in required_missing
        ],
        "elective_credits_earned": elective_credits_earned,
        "elective_credits_required": program.elective_credits_required,
        "elective_credits_remaining": elective_credits_remaining,
        "completion_percentage": round(required_pct, 1),
        "can_graduate": can_graduate,
        "message": (
            "🎓 Đủ điều kiện tốt nghiệp!" if can_graduate
            else f"Còn {len(required_missing)} môn bắt buộc và "
                 f"{elective_credits_remaining}TC tự chọn cần hoàn thành."
        ),
    }


if __name__ == "__main__":
    mcp.run()

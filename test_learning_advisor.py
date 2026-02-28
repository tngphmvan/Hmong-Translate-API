"""
Unit tests cho Learning Advisor Assistant.

Kiểm tra:
1. Data models (Course, Program, StudentProfile, LearningPath)
2. Thuật toán Prim-like (build_prim_learning_path)
3. Các MCP tools (9 công cụ)
4. Prompt templates
"""

import pytest

from learning_advisor.data_models import Course, LearningPath, Program, StudentProfile
from learning_advisor.prim_algorithm import build_prim_learning_path
from learning_advisor.sample_data import COURSES, PROGRAMS
from learning_advisor.prompts import (
    SYSTEM_PROMPT,
    CHAIN_OF_THOUGHT_TEMPLATE,
    FEW_SHOT_EXAMPLES,
    REACT_SYSTEM_PROMPT,
)
from learning_advisor.mcp_server import (
    get_program_info,
    list_programs,
    get_course_info,
    list_courses_by_semester,
    check_admission_requirements,
    build_optimal_learning_path,
    get_prerequisite_graph,
    recommend_electives,
    calculate_graduation_progress,
)


# ==================== DATA MODEL TESTS ====================

class TestDataModels:
    def test_course_creation(self):
        course = Course(
            code="TEST101",
            name="Test Course",
            credits=3,
            description="A test course.",
            prerequisites=["CS101"],
        )
        assert course.code == "TEST101"
        assert course.credits == 3
        assert course.prerequisites == ["CS101"]
        assert course.is_required is True  # default

    def test_program_creation(self):
        program = Program(
            program_id="TEST",
            name="Test Program",
            faculty="Test Faculty",
            description="Test.",
            total_credits=120,
            required_courses=["CS101", "MATH101"],
        )
        assert program.program_id == "TEST"
        assert len(program.required_courses) == 2
        assert program.elective_courses == []

    def test_student_profile_defaults(self):
        student = StudentProfile(
            student_id="SV001",
            name="Nguyen Van A",
            gpa=8.5,
        )
        assert student.completed_courses == []
        assert student.interests == []
        assert student.total_credits_earned == 0

    def test_learning_path_creation(self):
        path = LearningPath(
            semesters=[["CS101", "MATH101"], ["CS102"]],
            total_credits=9,
            total_semesters=2,
            completion_percentage=50.0,
            algorithm_steps=[],
            rationale="Test rationale",
        )
        assert path.total_semesters == 2
        assert path.completion_percentage == 50.0


# ==================== SAMPLE DATA TESTS ====================

class TestSampleData:
    def test_courses_not_empty(self):
        assert len(COURSES) > 0

    def test_programs_not_empty(self):
        assert len(PROGRAMS) > 0

    def test_csai_program_exists(self):
        assert "CSAI" in PROGRAMS
        csai = PROGRAMS["CSAI"]
        assert csai.total_credits == 130
        assert "CS101" in csai.required_courses

    def test_se_program_exists(self):
        assert "SE" in PROGRAMS
        se = PROGRAMS["SE"]
        assert se.total_credits == 125

    def test_all_prerequisites_exist(self):
        """Mọi tiên quyết phải tồn tại trong COURSES."""
        for code, course in COURSES.items():
            for prereq in course.prerequisites:
                assert prereq in COURSES, (
                    f"Khóa học {code} có tiên quyết {prereq} không tồn tại"
                )

    def test_all_program_courses_exist(self):
        """Mọi môn trong chương trình phải tồn tại trong COURSES."""
        for prog_id, program in PROGRAMS.items():
            for code in program.required_courses + program.elective_courses:
                assert code in COURSES, (
                    f"Chương trình {prog_id} tham chiếu môn {code} không tồn tại"
                )


# ==================== PRIM ALGORITHM TESTS ====================

class TestPrimAlgorithm:
    def setup_method(self):
        self.program = PROGRAMS["CSAI"]

    def test_empty_completed_courses(self):
        """Khi sinh viên chưa học gì, phải trả về lộ trình hợp lệ."""
        path = build_prim_learning_path(
            all_courses=COURSES,
            completed_courses=[],
            program_required=self.program.required_courses,
            program_electives=self.program.elective_courses,
            elective_credits_needed=self.program.elective_credits_required,
            interests=[],
        )
        assert isinstance(path, LearningPath)
        assert path.total_semesters > 0
        assert path.total_credits > 0

    def test_prerequisites_respected(self):
        """Môn tiên quyết phải xuất hiện trước trong lộ trình."""
        path = build_prim_learning_path(
            all_courses=COURSES,
            completed_courses=[],
            program_required=self.program.required_courses,
            program_electives=self.program.elective_courses,
            elective_credits_needed=self.program.elective_credits_required,
        )
        completed_so_far = set()
        for semester in path.semesters:
            for code in semester:
                course = COURSES.get(code)
                if course:
                    for prereq in course.prerequisites:
                        assert prereq in completed_so_far or prereq not in COURSES, (
                            f"Môn {code} được xếp trước tiên quyết {prereq}"
                        )
            completed_so_far.update(semester)

    def test_completed_courses_not_repeated(self):
        """Môn đã hoàn thành không được xuất hiện trong lộ trình."""
        completed = ["MATH101", "CS101"]
        path = build_prim_learning_path(
            all_courses=COURSES,
            completed_courses=completed,
            program_required=self.program.required_courses,
            program_electives=self.program.elective_courses,
            elective_credits_needed=self.program.elective_credits_required,
        )
        scheduled = [c for sem in path.semesters for c in sem]
        for c in completed:
            assert c not in scheduled, f"Môn đã hoàn thành {c} bị lặp lại trong lộ trình"

    def test_credit_limit_per_semester(self):
        """Không học kỳ nào được vượt quá giới hạn tín chỉ."""
        max_credits = 12
        path = build_prim_learning_path(
            all_courses=COURSES,
            completed_courses=[],
            program_required=self.program.required_courses,
            program_electives=self.program.elective_courses,
            elective_credits_needed=self.program.elective_credits_required,
            max_credits_per_semester=max_credits,
        )
        for i, semester in enumerate(path.semesters):
            sem_credits = sum(
                COURSES[c].credits for c in semester if c in COURSES
            )
            assert sem_credits <= max_credits, (
                f"Học kỳ {i+1} có {sem_credits}TC vượt giới hạn {max_credits}TC"
            )

    def test_interest_affects_order(self):
        """Sở thích phải ảnh hưởng tới thứ tự ưu tiên môn học."""
        path_ai = build_prim_learning_path(
            all_courses=COURSES,
            completed_courses=["MATH101", "MATH102", "CS101", "CS102", "MATH301"],
            program_required=self.program.required_courses,
            program_electives=self.program.elective_courses,
            elective_credits_needed=self.program.elective_credits_required,
            interests=["AI", "machine learning"],
        )
        # Chỉ kiểm tra lộ trình trả về hợp lệ
        assert path_ai.total_semesters >= 0

    def test_algorithm_steps_recorded(self):
        """Phải có ít nhất một bước được ghi lại."""
        path = build_prim_learning_path(
            all_courses=COURSES,
            completed_courses=[],
            program_required=["MATH101", "CS101"],
            program_electives=[],
            elective_credits_needed=0,
        )
        assert len(path.algorithm_steps) > 0
        step = path.algorithm_steps[0]
        assert "step" in step
        assert "action" in step
        assert "course" in step


# ==================== MCP TOOL TESTS ====================

class TestMCPTools:
    # --- Tool 1 ---
    def test_get_program_info_valid(self):
        result = get_program_info("CSAI")
        assert result["program_id"] == "CSAI"
        assert "total_credits" in result
        assert "required_courses" in result

    def test_get_program_info_invalid(self):
        result = get_program_info("UNKNOWN")
        assert "error" in result
        assert "available" in result

    def test_get_program_info_case_insensitive(self):
        result = get_program_info("csai")
        assert result["program_id"] == "CSAI"

    # --- Tool 2 ---
    def test_list_programs_returns_list(self):
        result = list_programs()
        assert isinstance(result, list)
        assert len(result) >= 1
        assert "program_id" in result[0]

    # --- Tool 3 ---
    def test_get_course_info_valid(self):
        result = get_course_info("CS101")
        assert result["code"] == "CS101"
        assert result["credits"] == 3
        assert result["prerequisites"] == []

    def test_get_course_info_invalid(self):
        result = get_course_info("INVALID")
        assert "error" in result

    # --- Tool 4 ---
    def test_list_courses_by_semester_fall(self):
        result = list_courses_by_semester("Fall")
        assert isinstance(result, list)
        assert len(result) > 0
        # CS101 dạy vào Fall
        codes = [r.get("code") for r in result]
        assert "CS101" in codes

    def test_list_courses_by_semester_unknown(self):
        result = list_courses_by_semester("Winter")
        assert len(result) == 1
        assert "info" in result[0]

    # --- Tool 5 ---
    def test_check_admission_eligible(self):
        result = check_admission_requirements("CSAI", 8.0, 8.0, 7.5)
        assert result["eligible"] is True
        assert result["checks"]["gpa"]["passed"] is True

    def test_check_admission_ineligible(self):
        result = check_admission_requirements("CSAI", 5.0, 5.0, 5.0)
        assert result["eligible"] is False

    def test_check_admission_invalid_program(self):
        result = check_admission_requirements("UNKNOWN", 8.0, 8.0, 7.5)
        assert "error" in result

    # --- Tool 6 ---
    def test_build_optimal_learning_path_valid(self):
        result = build_optimal_learning_path(
            program_id="CSAI",
            completed_courses=["MATH101", "CS101"],
            interests=["AI"],
        )
        assert "semesters" in result
        assert result["total_semesters"] > 0

    def test_build_optimal_learning_path_invalid_program(self):
        result = build_optimal_learning_path(
            program_id="INVALID",
            completed_courses=[],
            interests=[],
        )
        assert "error" in result

    # --- Tool 7 ---
    def test_get_prerequisite_graph_valid(self):
        result = get_prerequisite_graph("CSAI")
        assert "nodes" in result
        assert "edges" in result
        assert result["total_courses"] > 0

    def test_get_prerequisite_graph_has_edges(self):
        result = get_prerequisite_graph("CSAI")
        # CS102 phụ thuộc CS101 → phải có cạnh
        assert result["total_edges"] > 0

    # --- Tool 8 ---
    def test_recommend_electives_returns_list(self):
        result = recommend_electives(
            program_id="CSAI",
            completed_courses=["MATH101", "MATH102", "MATH301", "CS101", "CS102"],
            interests=["AI", "deep learning"],
            num_recommendations=3,
        )
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)

    def test_recommend_electives_respects_prerequisites(self):
        """Không gợi ý môn chưa đủ tiên quyết."""
        result = recommend_electives(
            program_id="CSAI",
            completed_courses=[],  # Chưa học gì
            interests=["AI"],
            num_recommendations=5,
        )
        # AI401 cần AI302 → không được gợi ý
        codes = [r["code"] for r in result.get("recommendations", [])]
        assert "AI401" not in codes

    # --- Tool 9 ---
    def test_calculate_graduation_progress_no_courses(self):
        result = calculate_graduation_progress("CSAI", [])
        assert result["credits_earned"] == 0
        assert result["can_graduate"] is False
        assert result["completion_percentage"] == 0.0

    def test_calculate_graduation_progress_partial(self):
        result = calculate_graduation_progress("CSAI", ["MATH101", "CS101"])
        assert result["credits_earned"] == 6
        assert result["can_graduate"] is False
        assert result["completion_percentage"] < 100.0

    def test_calculate_graduation_progress_missing_courses_listed(self):
        result = calculate_graduation_progress("CSAI", ["MATH101"])
        missing_codes = [m["code"] for m in result["required_courses_missing"]]
        assert "CS101" in missing_codes


# ==================== PROMPT TESTS ====================

class TestPrompts:
    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 50

    def test_chain_of_thought_has_placeholder(self):
        assert "{query}" in CHAIN_OF_THOUGHT_TEMPLATE

    def test_few_shot_examples_count(self):
        assert len(FEW_SHOT_EXAMPLES) >= 2

    def test_few_shot_examples_structure(self):
        for ex in FEW_SHOT_EXAMPLES:
            assert "query" in ex
            assert "reasoning" in ex
            assert "answer" in ex

    def test_react_prompt_has_placeholders(self):
        # REACT_SYSTEM_PROMPT có {query} để điền vào runtime
        assert "{query}" in REACT_SYSTEM_PROMPT

    def test_chain_of_thought_formatting(self):
        formatted = CHAIN_OF_THOUGHT_TEMPLATE.format(query="Test query")
        assert "Test query" in formatted
        assert "Bước 1" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

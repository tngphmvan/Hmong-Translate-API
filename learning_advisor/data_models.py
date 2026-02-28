"""
Data models cho Learning Advisor Assistant.

Định nghĩa các cấu trúc dữ liệu cốt lõi:
- Course: Thông tin khóa học
- Program: Chương trình đào tạo
- StudentProfile: Hồ sơ sinh viên
- LearningPath: Lộ trình học tập được tạo bởi thuật toán Prim
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Course:
    """Thông tin chi tiết về một khóa học."""

    code: str
    name: str
    credits: int
    description: str
    prerequisites: List[str] = field(default_factory=list)
    semester_offered: List[str] = field(default_factory=lambda: ["Fall", "Spring"])
    is_required: bool = True
    category: str = "core"
    weight: float = 1.0  # Trọng số độ ưu tiên trong thuật toán Prim


@dataclass
class Program:
    """Thông tin chương trình đào tạo đại học."""

    program_id: str
    name: str
    faculty: str
    description: str
    total_credits: int
    required_courses: List[str] = field(default_factory=list)
    elective_courses: List[str] = field(default_factory=list)
    elective_credits_required: int = 0
    admission_gpa_min: float = 0.0
    admission_requirements: List[str] = field(default_factory=list)


@dataclass
class StudentProfile:
    """Hồ sơ sinh viên."""

    student_id: str
    name: str
    gpa: float
    completed_courses: List[str] = field(default_factory=list)
    current_program: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    total_credits_earned: int = 0


@dataclass
class LearningPath:
    """Lộ trình học tập được xây dựng bởi thuật toán Prim."""

    semesters: List[List[str]]  # Danh sách khóa học theo từng học kỳ
    total_credits: int
    total_semesters: int
    completion_percentage: float
    algorithm_steps: List[Dict]  # Các bước của thuật toán Prim
    rationale: str

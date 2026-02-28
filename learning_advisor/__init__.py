"""
Learning Advisor Assistant - Agent System
-----------------------------------------
Hệ thống tư vấn học tập sử dụng:
- FastMCP: Model Context Protocol với 9 công cụ tùy chỉnh
- Google ADK: Agent Development Kit
- Prim-like Algorithm: Lựa chọn khóa học tối ưu
- Chain-of-Thought, Few-shot, Thought-Action-Observation prompting
"""

from .data_models import Course, Program, StudentProfile, LearningPath
from .prim_algorithm import build_prim_learning_path
from .mcp_server import mcp

__all__ = [
    "Course",
    "Program",
    "StudentProfile",
    "LearningPath",
    "build_prim_learning_path",
    "mcp",
]

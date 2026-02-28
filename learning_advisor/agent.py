"""
Google ADK Agent – Learning Advisor Assistant.

Cấu hình và khởi chạy agent sử dụng Google Agent Development Kit (ADK)
tích hợp với MCP server qua FastMCP.

Luồng hoạt động:
    1. Agent nhận câu hỏi từ người dùng.
    2. Sử dụng REACT_SYSTEM_PROMPT (Chain-of-Thought + Few-shot + TAO).
    3. Gọi các công cụ MCP để thu thập thông tin.
    4. Tổng hợp kết quả và trả lời người dùng.

Yêu cầu biến môi trường:
    GOOGLE_API_KEY hoặc GOOGLE_GENAI_API_KEY: API key của Google AI Studio.
"""

import os
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

from .prompts import SYSTEM_PROMPT


def create_learning_advisor_agent(model: str = "gemini-2.0-flash") -> LlmAgent:
    """
    Tạo và trả về Learning Advisor Agent sử dụng Google ADK.

    Tham số:
        model: Tên mô hình Gemini (mặc định 'gemini-2.0-flash').

    Trả về:
        LlmAgent đã được cấu hình với 9 MCP tools.
    """
    mcp_server_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")

    toolset = MCPToolset(
        connection_params=StdioServerParameters(
            command="python",
            args=[mcp_server_path],
        )
    )

    agent = LlmAgent(
        name="learning_advisor",
        model=model,
        description=(
            "Trợ lý tư vấn học tập đại học: lộ trình môn học, "
            "điều kiện xét tuyển và gợi ý tự chọn."
        ),
        instruction=SYSTEM_PROMPT,
        tools=[toolset],
    )
    return agent


def get_api_key() -> Optional[str]:
    """Lấy Google API key từ biến môi trường."""
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")

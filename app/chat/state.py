from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """LangGraph 状态定义

    - messages: 完整对话历史（add_messages reducer 自动合并）
    - intent: 意图分析结果 — general_chat / legal_assist / human_handoff
    - legal_context: 法律 RAG 检索结果（文本）
    - references: 引用元数据列表 [{"source":..., "chunk_index":..., "minio_url":..., "relevance":...}]
    - need_human: 是否需要转人工
    - human_feedback: 人工客服回复
    - current_response: 本轮最终回复内容
    - session_id: 会话 ID
    """
    messages: Annotated[List[BaseMessage], add_messages]
    intent: str
    legal_context: Optional[str]
    references: List[dict]
    need_human: bool
    human_feedback: Optional[str]
    current_response: Optional[str]
    session_id: str

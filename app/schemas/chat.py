from typing import Optional, List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    session_id: Optional[str] = Field(None, description="会话 ID，不传则自动生成")


class ReferenceItem(BaseModel):
    source: str = Field("", description="文档/法条名称")
    chunk_index: int = Field(0, description="片段序号")
    minio_url: str = Field("", description="MinIO 链接")
    relevance: str = Field("", description="引用理由")


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    response: Optional[str] = Field(None, description="AI 回复内容")
    intent: str = Field("", description="意图分类")
    need_human: bool = Field(False, description="是否需要转人工")
    interrupted: bool = Field(False, description="是否中断等待人工")
    interrupt_data: Optional[dict] = Field(None, description="中断信息（人工侧参考）")
    references: List[ReferenceItem] = Field(default_factory=list, description="引用来源")
    minio_url: str = Field("", description="关联 MinIO 链接")


class ResumeRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    message: str = Field(..., min_length=1, max_length=2000, description="人工客服回复")


class HistoryItem(BaseModel):
    role: str = Field(..., description="user / ai / human")
    content: str = Field(..., description="消息内容")
    intent: str = Field("", description="意图")
    references: List[ReferenceItem] = Field(default_factory=list, description="引用来源")
    minio_url: str = Field("", description="关联 MinIO 链接")
    created_at: str = Field("", description="时间")


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[HistoryItem]


class PendingSessionItem(BaseModel):
    session_id: str
    last_message: str
    message_count: int

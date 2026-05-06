from typing import List, Optional
from pydantic import BaseModel, Field


class ReferenceItem(BaseModel):
    """单个引用来源"""
    source: str = Field(..., description="文档名称/法条名称")
    chunk_index: int = Field(0, description="片段序号/页码")
    minio_url: str = Field("", description="MinIO 文件链接")
    relevance: str = Field("", description="引用理由")


class StructuredResponse(BaseModel):
    """LLM 结构化输出约束 — 强制带引用来源"""
    answer: str = Field(..., description="回答内容")
    references: List[ReferenceItem] = Field(
        default_factory=list,
        description="引用来源列表，无引用时为空数组",
    )
    need_human: bool = Field(False, description="是否需要人工介入")

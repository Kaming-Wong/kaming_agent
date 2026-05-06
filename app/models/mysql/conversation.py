import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, Integer, String, Text, DateTime, Index

from app.utils.db import Base


class Conversation(Base):
    """对话记录表"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False, comment="user / ai / human")
    content = Column(Text, nullable=False)
    intent = Column(String(32), default="", comment="当前轮次的意图分类")
    references = Column(Text, default="", comment="引用来源 JSON 列表 [source, chunk_index, minio_url, ...]")
    minio_url = Column(String(512), default="", comment="关联的 MinIO 链接")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_session_created", "session_id", "created_at"),
    )

    def get_references(self) -> List[dict]:
        if self.references:
            try:
                return json.loads(self.references)
            except (json.JSONDecodeError, TypeError):
                pass
        return []

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "intent": self.intent,
            "references": self.get_references(),
            "minio_url": self.minio_url or "",
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

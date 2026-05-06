from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime

from app.utils.db import Base


class Session(Base):
    """会话表 — 记录摘要和元信息"""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, unique=True, index=True)
    summary = Column(Text, default="", comment="对话摘要")
    message_count = Column(Integer, default=0, comment="总消息数")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

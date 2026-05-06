from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Index

from app.utils.db import Base


class DocumentSource(Base):
    """文档来源表 — 记录 MinIO URL 与源文件的映射"""
    __tablename__ = "document_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(256), nullable=False, comment="源文件名")
    session_id = Column(String(64), default="", comment="上传者会话 ID")
    object_name = Column(String(512), nullable=False, comment="MinIO 对象路径")
    minio_url = Column(String(512), nullable=False, comment="MinIO 访问链接")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_doc_source", "source"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source,
            "session_id": self.session_id,
            "object_name": self.object_name,
            "minio_url": self.minio_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

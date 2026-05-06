import json
import logging
from typing import List, Optional

from sqlalchemy.orm import Session
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.utils.db import SessionLocal
from app.models.mysql.conversation import Conversation
from app.models.mysql.document_source import DocumentSource
from app.models.mysql.session import Session as SessionModel

logger = logging.getLogger(__name__)

# 对话摘要压缩阈值：超过此消息数触发压缩
SUMMARY_THRESHOLD = 20
# 压缩后保留的最近完整消息数
KEEP_RECENT = 10

SUMMARY_PROMPT = """请用中文简要总结以下对话的核心内容，包括：
1. 用户问了哪些关键问题
2. AI给出了什么关键回答
3. 还有哪些未解决的问题或待办事项

控制在200字以内，只输出摘要文本，不要多余的解释。

对话记录："""


def _get_llm():
    from app.adapters.llm_adapter import LLMAdapter
    return LLMAdapter().get_llm()


class MemoryManager:
    """对话记忆管理

    三层存储架构：
    1. MySQL conversations 表 — 完整消息记录（最近 KEEP_RECENT 条）
    2. MySQL sessions 表 — 对话摘要（超过阈值后自动压缩）
    3. document_sources 表 — 文件 MinIO 链接映射

    自动压缩机制：
    - 每 20 条消息触发一次
    - LLM 将旧消息生成摘要，合并到已有摘要中
    - 只保留最近 10 条完整消息
    """

    # ── 消息持久化 ──

    def save_message(self, session_id: str, role: str, content: str,
                     intent: str = "", references: Optional[List[dict]] = None,
                     minio_url: str = ""):
        """保存一条对话记录到 MySQL

        保存后自动检查是否需要触发摘要压缩。
        """
        db: Session = SessionLocal()
        try:
            record = Conversation(
                session_id=session_id,
                role=role,
                content=content,
                intent=intent,
                references=json.dumps(references, ensure_ascii=False) if references else "",
                minio_url=minio_url,
            )
            db.add(record)
            db.commit()

            self._increment_count(db, session_id)
            self._maybe_summarize(db, session_id)

        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            db.rollback()
        finally:
            db.close()

    def _increment_count(self, db: Session, session_id: str):
        """递增会话消息计数，不存在则新建会话记录"""
        sess = db.query(SessionModel).filter(
            SessionModel.session_id == session_id
        ).first()
        if sess:
            sess.message_count = sess.message_count + 1
        else:
            db.add(SessionModel(
                session_id=session_id,
                message_count=1,
            ))
        db.commit()

    # ── 摘要压缩 ──

    def _maybe_summarize(self, db: Session, session_id: str, force: bool = False):
        """检查并触发摘要压缩

        当会话消息数达到 SUMMARY_THRESHOLD（默认 20）时：
        1. 取最早的 (count - KEEP_RECENT) 条消息
        2. 调用 LLM 生成摘要
        3. 合并到已有摘要
        4. 删除旧消息，只保留最近 KEEP_RECENT 条
        """
        sess = db.query(SessionModel).filter(
            SessionModel.session_id == session_id
        ).first()
        if not sess:
            return
        if not force and sess.message_count < SUMMARY_THRESHOLD:
            return

        delete_count = sess.message_count - KEEP_RECENT
        if delete_count < 1:
            return

        old_records = (
            db.query(Conversation)
            .filter(Conversation.session_id == session_id)
            .order_by(Conversation.created_at.asc())
            .limit(delete_count)
            .all()
        )

        if len(old_records) < 2:
            return

        # 拼接旧消息文本喂给 LLM
        dialog = "\n".join(
            f"{'用户' if r.role == 'user' else 'AI'}: {r.content[:200]}"
            for r in old_records
        )

        new_summary = self._generate_summary(dialog)

        # 合并：已有摘要 + 新摘要
        old_summary = sess.summary or ""
        if old_summary:
            merged = f"{old_summary}\n【后续对话】{new_summary}"
        else:
            merged = new_summary

        # 删除旧消息，更新摘要
        old_ids = [r.id for r in old_records]
        db.query(Conversation).filter(Conversation.id.in_(old_ids)).delete(synchronize_session=False)

        sess.summary = merged[:1000]
        sess.message_count = KEEP_RECENT
        db.commit()

        logger.info(
            f"Summarized session {session_id}: "
            f"removed {len(old_records)} old messages, "
            f"summary now {len(merged)} chars"
        )

    def _generate_summary(self, dialog: str) -> str:
        """LLM 生成摘要，失败时用前几条消息的首句兜底"""
        try:
            llm = _get_llm()
            result = llm.invoke(SUMMARY_PROMPT + "\n" + dialog)
            return result.content.strip()[:500]
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            lines = dialog.split("\n")[:4]
            return " | ".join(l.split(": ", 1)[-1][:50] for l in lines if ": " in l)

    # ── 历史读取 ──

    def get_history(self, session_id: str, limit: int = 50) -> List[Conversation]:
        """获取原始 Conversation 记录列表"""
        db: Session = SessionLocal()
        try:
            records = (
                db.query(Conversation)
                .filter(Conversation.session_id == session_id)
                .order_by(Conversation.created_at.asc())
                .limit(limit)
                .all()
            )
            return records
        finally:
            db.close()

    def get_history_messages(self, session_id: str, limit: int = 50) -> List[BaseMessage]:
        """获取 LangChain 格式的历史消息

        结构：SystemMessage(摘要) + HumanMessage/AIMessage(最近记录)

        这样 LLM 既能通过摘要理解前文，又能看到最近对话的完整细节。
        """
        db: Session = SessionLocal()
        try:
            sess = db.query(SessionModel).filter(
                SessionModel.session_id == session_id
            ).first()
            summary = sess.summary if sess and sess.summary else ""

            records = (
                db.query(Conversation)
                .filter(Conversation.session_id == session_id)
                .order_by(Conversation.created_at.asc())
                .limit(limit)
                .all()
            )

            messages: List[BaseMessage] = []

            if summary:
                messages.append(SystemMessage(
                    content=f"以下是之前的对话摘要，请基于此理解对话上下文：\n{summary}"
                ))

            for r in records:
                if r.role == "user":
                    messages.append(HumanMessage(content=r.content))
                elif r.role in ("ai", "human"):
                    messages.append(AIMessage(content=r.content))

            return messages
        finally:
            db.close()

    # ── 文档来源 ──

    def save_document_source(self, source: str, session_id: str,
                              object_name: str, minio_url: str):
        """保存源文件 → MinIO URL 映射，用于 RAG 引用追溯"""
        db: Session = SessionLocal()
        try:
            record = DocumentSource(
                source=source,
                session_id=session_id,
                object_name=object_name,
                minio_url=minio_url,
            )
            db.add(record)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save document source: {e}")
            db.rollback()
        finally:
            db.close()

    def get_minio_url(self, source: str) -> str:
        """根据源文件名查询最新的 MinIO URL"""
        db: Session = SessionLocal()
        try:
            record = (
                db.query(DocumentSource)
                .filter(DocumentSource.source == source)
                .order_by(DocumentSource.created_at.desc())
                .first()
            )
            return record.minio_url if record else ""
        finally:
            db.close()

    # ── 会话管理 ──

    def clear_history(self, session_id: str):
        """清除会话的所有记录（conversations + sessions）"""
        db: Session = SessionLocal()
        try:
            db.query(Conversation).filter(Conversation.session_id == session_id).delete()
            db.query(SessionModel).filter(SessionModel.session_id == session_id).delete()
            db.commit()
            logger.info(f"Cleared history for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            db.rollback()
        finally:
            db.close()

    def get_pending_sessions(self) -> List[str]:
        """获取有转人工标记的会话 ID 列表"""
        db: Session = SessionLocal()
        try:
            subquery = (
                db.query(Conversation.session_id, Conversation.created_at)
                .filter(Conversation.intent == "human_handoff")
                .order_by(Conversation.created_at.desc())
                .limit(100)
                .all()
            )
            seen = set()
            result = []
            for row in subquery:
                if row.session_id not in seen:
                    seen.add(row.session_id)
                    result.append(row.session_id)
            return result
        finally:
            db.close()

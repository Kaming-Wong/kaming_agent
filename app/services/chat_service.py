import logging
import uuid
from typing import Optional, List

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph.state import CompiledStateGraph

from app.chat.graph import get_graph
from app.chat.state import AgentState
from app.manager.memory_manager import MemoryManager
from app.schemas.chat import HistoryItem
from app.utils.redis_utils import get_redis

logger = logging.getLogger(__name__)


class ChatService:
    """对话服务 — 编排 LangGraph 状态机

    职责：
    1. 构造初始状态（加载 MySQL 历史 + Redis 缓存）
    2. 调用 graph.stream() 执行图
    3. 检测 interrupt（转人工）
    4. 持久化对话记录到 MySQL
    5. 管理 Redis 缓存失效和速率限制
    """

    def __init__(self):
        self.graph: CompiledStateGraph = get_graph()
        self.memory = MemoryManager()

    def _make_config(self, session_id: str) -> dict:
        """构造 LangGraph 配置，thread_id 用于 checkpoint 持久化"""
        return {"configurable": {"thread_id": session_id}}

    def _build_initial_state(self, session_id: str, message: str) -> AgentState:
        """构造初始状态

        1. 优先从 Redis 读取历史（5 分钟缓存）
        2. Redis 未命中则读 MySQL
        3. 读到的历史写回 Redis
        4. 追加当前用户消息
        """
        redis = get_redis()
        cached = redis.get_cached_messages(session_id)

        if cached:
            history = [self._dict_to_message(m) for m in cached]
        else:
            history = self.memory.get_history_messages(session_id)
            cache_data = [self._message_to_dict(m) for m in history]
            redis.cache_messages(session_id, cache_data, ttl=300)

        history.append(HumanMessage(content=message))

        return {
            "messages": history,
            "intent": "",
            "legal_context": None,
            "references": [],
            "need_human": False,
            "human_feedback": None,
            "current_response": None,
            "session_id": session_id,
        }

    def process_message(self, session_id: str, message: str) -> dict:
        """处理单条用户消息

        Returns:
            正常完成: {"session_id", "response", "intent", "references", "minio_url", ...}
            转人工:   {"session_id", "interrupted": True, "interrupt_data": {...}}
        """
        config = self._make_config(session_id)
        # state 构造在 try 外面：如果 MySQL 连不上提前报错，不污染业务逻辑
        state = self._build_initial_state(session_id, message)

        try:
            # 执行图，stream 模式逐节点产出事件
            events = []
            for event in self.graph.stream(state, config, stream_mode="values"):
                events.append(event)

            final_state = events[-1] if events else state

            # ── 检测 interrupt（转人工） ──
            # human_handoff 节点调了 interrupt() 后图会暂停。
            # 在不同 LangGraph 版本中检测方式不同，这里用三种方法兜底：
            snapshot = self.graph.get_state(config)
            has_interrupt = False
            interrupt_val = None

            if snapshot:
                # 方法1: tasks 中的 interrupt 对象（LangGraph >= 0.2）
                if snapshot.tasks:
                    for task in snapshot.tasks:
                        if getattr(task, "interrupt", None):
                            has_interrupt = True
                            interrupt_val = task.interrupt.value
                            break

                # 方法2: next 中有待执行节点（说明图没跑到 END）
                if not has_interrupt and snapshot.next:
                    node_list = list(snapshot.next)
                    logger.info(f"Graph paused at nodes: {node_list}")
                    if "human_handoff" in node_list:
                        has_interrupt = True
                        interrupt_val = {"type": "human_handoff", "session_id": session_id}

            if has_interrupt:
                # 只存用户消息，AI 回复等人工提供
                self.memory.save_message(session_id, "user", message)
                logger.info(f"Session {session_id}: interrupted for human handoff")
                return {
                    "session_id": session_id,
                    "interrupted": True,
                    "interrupt_data": interrupt_val,
                }

            # ── 正常完成 ──
            response = self._extract_response(final_state)
            intent = final_state.get("intent", "")
            need_human = final_state.get("need_human", False)
            references = final_state.get("references", [])
            minio_url = final_state.get("minio_url", "")
            if not minio_url and references:
                minio_url = references[0].get("minio_url", "")

            # 兜底：图执行完了但没产生有效回复（比如 interrupt 没被识别的情况）
            if not response and need_human:
                self.memory.save_message(session_id, "user", message)
                logger.info(f"Session {session_id}: no response but need_human, treating as handoff")
                return {
                    "session_id": session_id,
                    "interrupted": True,
                    "interrupt_data": {"type": "human_handoff", "session_id": session_id},
                }

            # 持久化到 MySQL
            self.memory.save_message(session_id, "user", message, intent)
            self.memory.save_message(
                session_id, "ai", response,
                references=references, minio_url=minio_url,
            )

            # Redis 缓存失效，下次请求重新从 MySQL 加载
            get_redis().delete_cached_messages(session_id)

            logger.info(f"Session {session_id}: completed (intent={intent}, refs={len(references)})")
            return {
                "session_id": session_id,
                "response": response,
                "intent": intent,
                "need_human": need_human,
                "interrupted": False,
                "references": references,
                "minio_url": minio_url,
            }

        except Exception as e:
            logger.error(f"Session {session_id} error: {e}", exc_info=True)
            self.memory.save_message(session_id, "user", message)
            fallback = "抱歉，系统出现异常，请稍后再试。"
            self.memory.save_message(session_id, "ai", fallback)
            return {
                "session_id": session_id,
                "response": fallback,
                "intent": "error",
                "need_human": True,
                "interrupted": False,
                "references": [],
                "minio_url": "",
            }

    def resume_after_human(self, session_id: str, human_message: str) -> dict:
        """人工客服回复后，用 Command(resume) 恢复图执行"""
        config = self._make_config(session_id)

        try:
            events = []
            for event in self.graph.stream(
                Command(resume={"message": human_message}),
                config,
                stream_mode="values",
            ):
                events.append(event)

            final_state = events[-1] if events else {}

            # 检查 resume 后是否再次中断（极少出现，但需兜底）
            snapshot = self.graph.get_state(config)
            if snapshot and snapshot.tasks:
                for task in snapshot.tasks:
                    if getattr(task, "interrupt", None):
                        logger.warning(f"Session {session_id}: double interrupt after resume")
                        return {
                            "session_id": session_id,
                            "interrupted": True,
                            "interrupt_data": task.interrupt.value,
                        }

            response = self._extract_response(final_state)
            references = final_state.get("references", [])
            minio_url = final_state.get("minio_url", "")
            if not minio_url and references:
                minio_url = references[0].get("minio_url", "")

            self.memory.save_message(session_id, "human", human_message)
            self.memory.save_message(
                session_id, "ai", response,
                references=references, minio_url=minio_url,
            )

            logger.info(f"Session {session_id}: human handoff completed")
            return {
                "session_id": session_id,
                "response": response,
                "intent": "human_handoff",
                "need_human": False,
                "interrupted": False,
                "references": references,
                "minio_url": minio_url,
            }

        except Exception as e:
            logger.error(f"Session {session_id} resume error: {e}", exc_info=True)
            fallback = "人工回复已记录，请继续提问。"
            self.memory.save_message(session_id, "human", human_message)
            self.memory.save_message(session_id, "ai", fallback)
            return {
                "session_id": session_id,
                "response": fallback,
                "intent": "human_handoff",
                "need_human": False,
                "interrupted": False,
                "references": [],
                "minio_url": "",
            }

    def get_history(self, session_id: str, limit: int = 50) -> list:
        """获取会话历史（含引用信息和 MinIO 链接）"""
        records = self.memory.get_history(session_id, limit)
        result = []
        for r in records:
            item = HistoryItem(
                role=r.role,
                content=r.content,
                intent=r.intent or "",
                references=r.get_references(),
                minio_url=r.minio_url or "",
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
            result.append(item)
        return result

    def get_pending_sessions(self) -> list:
        """获取待人工处理的会话列表"""
        from app.schemas.chat import PendingSessionItem
        session_ids = self.memory.get_pending_sessions()
        result = []
        for sid in session_ids:
            records = self.memory.get_history(sid, limit=1)
            last_msg = records[-1].content if records else ""
            count = len(self.memory.get_history(sid, limit=1000))
            result.append(PendingSessionItem(
                session_id=sid,
                last_message=last_msg,
                message_count=count,
            ))
        return result

    # ── 工具方法 ──

    @staticmethod
    def _message_to_dict(msg) -> dict:
        """BaseMessage → dict，用于 Redis 序列化"""
        return {"role": msg.type, "content": msg.content}

    @staticmethod
    def _dict_to_message(d: dict):
        """dict → BaseMessage，从 Redis 反序列化恢复"""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        role = d.get("role", "human")
        content = d.get("content", "")
        if role == "human":
            return HumanMessage(content=content)
        elif role == "ai":
            return AIMessage(content=content)
        elif role == "system":
            return SystemMessage(content=content)
        return HumanMessage(content=content)

    @staticmethod
    def _extract_response(state: dict) -> str:
        """从图最终状态提取回复文本

        优先用 current_response（结构化输出时设置），
        兜底从 messages 列表取最后一条。
        """
        if state.get("current_response"):
            return state["current_response"]
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            return last.content if hasattr(last, "content") else str(last)
        return ""

    @staticmethod
    def generate_session_id() -> str:
        return uuid.uuid4().hex[:16]

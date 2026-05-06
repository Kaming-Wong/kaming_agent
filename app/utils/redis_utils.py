import json
import logging
import hashlib
import time
from typing import Optional, Any

from redis import Redis, RedisError

from app.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 缓存客户端

    三层缓存用途：
    1. 对话历史缓存 — 减少 MySQL 读取，提升历史加载速度
    2. LLM 响应缓存 — 相同问题直接返回，避免重复调用 Ollama
    3. 速率限制 — 防刷，每 session 每 10 秒最多 5 次

    所有方法在 Redis 不可用时自动降级（返回 None / True），不影响主流程。
    """

    def __init__(self):
        self._client: Optional[Redis] = None

    def _get_client(self) -> Optional[Redis]:
        """惰性初始化 + 健康检查，失败后标记不可用"""
        if self._client is None:
            try:
                self._client = Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=3,
                )
                self._client.ping()
                logger.info(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            except (RedisError, ConnectionError) as e:
                logger.warning(f"Redis unavailable (proceeding without cache): {e}")
                self._client = None
        return self._client

    @property
    def available(self) -> bool:
        return self._get_client() is not None

    # ── 对话历史缓存 ──
    # key: session:{session_id}:messages
    # value: JSON 序列化的 BaseMessage 列表
    # TTL: 5 分钟，有新消息时主动删除

    def cache_messages(self, session_id: str, messages: list, ttl: int = 300):
        client = self._get_client()
        if not client:
            return
        try:
            key = f"session:{session_id}:messages"
            client.setex(key, ttl, json.dumps(messages, ensure_ascii=False))
        except RedisError as e:
            logger.warning(f"Redis cache_messages failed: {e}")

    def get_cached_messages(self, session_id: str) -> Optional[list]:
        client = self._get_client()
        if not client:
            return None
        try:
            key = f"session:{session_id}:messages"
            data = client.get(key)
            return json.loads(data) if data else None
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Redis get_cached_messages failed: {e}")
            return None

    def delete_cached_messages(self, session_id: str):
        client = self._get_client()
        if not client:
            return
        try:
            client.delete(f"session:{session_id}:messages")
        except RedisError as e:
            logger.warning(f"Redis delete failed: {e}")

    # ── 会话摘要缓存（2 小时）──

    def cache_summary(self, session_id: str, summary: str, ttl: int = 7200):
        client = self._get_client()
        if not client:
            return
        try:
            client.setex(f"session:{session_id}:summary", ttl, summary)
        except RedisError as e:
            logger.warning(f"Redis cache_summary failed: {e}")

    def get_cached_summary(self, session_id: str) -> Optional[str]:
        client = self._get_client()
        if not client:
            return None
        try:
            return client.get(f"session:{session_id}:summary")
        except RedisError:
            return None

    # ── LLM 响应缓存 ──
    # 用户重复发送相同问题时直接返回，避免 LLM 重复计算
    # key: llm:{prefix}:{input_md5}

    def cache_llm_response(self, question: str, response: str, ttl: int = 300):
        client = self._get_client()
        if not client:
            return
        try:
            key = f"llm:q:{self._hash(question)}"
            client.setex(key, ttl, response)
        except RedisError:
            pass

    def get_cached_llm_response(self, question: str) -> Optional[str]:
        client = self._get_client()
        if not client:
            return None
        try:
            key = f"llm:q:{self._hash(question)}"
            return client.get(key)
        except RedisError:
            return None

    # ── 速率限制 ──
    # 滑动窗口计数，超过 max_requests 返回 False

    def check_rate_limit(self, key: str, max_requests: int = 10, window: int = 60) -> bool:
        client = self._get_client()
        if not client:
            return True  # Redis 不可用时放行，不影响业务

        try:
            redis_key = f"ratelimit:{key}"
            current = client.get(redis_key)

            if current is None:
                client.setex(redis_key, window, 1)
                return True

            count = int(current)
            if count >= max_requests:
                logger.warning(f"Rate limit exceeded for {key}: {count}/{max_requests}")
                return False

            client.incr(redis_key)
            return True
        except (RedisError, ValueError):
            return True

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]


# 全局单例
_redis: Optional[RedisClient] = None


def get_redis() -> RedisClient:
    global _redis
    if _redis is None:
        _redis = RedisClient()
    return _redis

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.routers.base_router import BaseRouter
from app.schemas.common import ApiResponse
from app.schemas.chat import ChatRequest, ResumeRequest, ChatResponse
from app.services.chat_service import ChatService
from app.utils.redis_utils import get_redis

logger = logging.getLogger(__name__)


class ChatRouter(BaseRouter):
    def __init__(self):
        super().__init__()
        self.chat_service = ChatService()
        self.router = self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "/chat",
            self.chat_endpoint,
            methods=["POST"],
            response_model=ApiResponse,
            summary="发送消息",
        )
        self.router.add_api_route(
            "/chat/resume",
            self.resume_endpoint,
            methods=["POST"],
            response_model=ApiResponse,
            summary="人工回复后继续对话",
        )
        self.router.add_api_route(
            "/chat/{session_id}/history",
            self.history_endpoint,
            methods=["GET"],
            response_model=ApiResponse,
            summary="获取会话历史",
        )
        self.router.add_api_route(
            "/admin/pending",
            self.pending_endpoint,
            methods=["GET"],
            response_model=ApiResponse,
            summary="获取待人工处理的会话",
        )
        return self.router

    async def chat_endpoint(self, req: ChatRequest, request: Request) -> ApiResponse:
        """发送消息到智能客服（含限流 + Redis 缓存）"""
        session_id = req.session_id or self.chat_service.generate_session_id()

        # 速率限制：每 session 每 10 秒最多 5 次
        redis = get_redis()
        if not redis.check_rate_limit(f"chat:{session_id}", max_requests=5, window=10):
            raise HTTPException(status_code=429, detail="请求太频繁，请稍后再试")
        result = self.chat_service.process_message(session_id, req.message)

        if result.get("interrupted"):
            return ApiResponse(
                status=200,
                message="human_handoff",
                data={
                    "session_id": session_id,
                    "interrupt_data": result["interrupt_data"],
                },
            )

        return ApiResponse(
            status=200,
            message="success",
            data=ChatResponse(
                session_id=session_id,
                response=result["response"],
                intent=result["intent"],
                need_human=result["need_human"],
                interrupted=False,
                references=result.get("references", []),
                minio_url=result.get("minio_url", ""),
            ).model_dump(),
        )

    async def resume_endpoint(self, req: ResumeRequest) -> ApiResponse:
        """人工客服回复后继续图执行"""
        if not req.session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        result = self.chat_service.resume_after_human(req.session_id, req.message)

        if result.get("interrupted"):
            return ApiResponse(
                status=200,
                message="human_handoff",
                data={"session_id": req.session_id, "interrupt_data": result["interrupt_data"]},
            )

        return ApiResponse(
            status=200,
            message="success",
            data=ChatResponse(
                session_id=req.session_id,
                response=result["response"],
                intent=result["intent"],
                need_human=False,
                interrupted=False,
                references=result.get("references", []),
                minio_url=result.get("minio_url", ""),
            ).model_dump(),
        )

    async def history_endpoint(self, session_id: str, limit: int = Query(50, ge=1, le=200)) -> ApiResponse:
        """获取会话历史"""
        messages = self.chat_service.get_history(session_id, limit)
        return ApiResponse(
            status=200,
            message="success",
            data={
                "session_id": session_id,
                "messages": [m.model_dump() for m in messages],
            },
        )

    async def pending_endpoint(self) -> ApiResponse:
        """获取等待人工处理的会话列表"""
        sessions = self.chat_service.get_pending_sessions()
        return ApiResponse(
            status=200,
            message="success",
            data={"sessions": [s.model_dump() for s in sessions]},
        )

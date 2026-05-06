import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn

from app.routers.chat_router import ChatRouter
from app.routers.document_router import DocumentRouter
from app.utils.db import Base, engine, migrate_db
# 以下 import 确保 SQLAlchemy 在 create_all 时能发现这些模型
from app.models.mysql.document_source import DocumentSource  # noqa: F401
from app.models.mysql.session import Session  # noqa: F401
from app.utils.neo4j_utils import get_neo4j

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="智能客服 Agent", version="1.0.0")

# CORS：允许所有来源（开发环境），生产环境应收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """启动时初始化数据库

    1. create_all：新建不存在的表（不会改已有表结构）
    2. migrate_db：对已有表执行 ALTER TABLE 加列（幂等）
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created / verified")
    except Exception as e:
        logger.warning(f"Table creation skipped (may not be available): {e}")

    try:
        migrate_db()
    except Exception as e:
        logger.warning(f"Migration skipped: {e}")

    try:
        neo4j = get_neo4j()
        if neo4j.available:
            neo4j.init_schema()
            logger.info("Neo4j schema initialized")
    except Exception as e:
        logger.warning(f"Neo4j initialization skipped: {e}")


# ── 挂载路由 ──
chat_router = ChatRouter()
app.include_router(chat_router.router, prefix="/api/v1")

doc_router = DocumentRouter()
app.include_router(doc_router.router, prefix="/api/v1")


# ── 静态文件和前端入口 ──
import os
static_dir = os.path.join(os.path.dirname(__file__), "app", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """根路径重定向到聊天页面"""
    return RedirectResponse(url="/static/chat.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "kaming-agent"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8018,
        reload=True,
    )

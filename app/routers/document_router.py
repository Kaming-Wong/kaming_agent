import asyncio
import logging
import os
import tempfile
import time
import uuid
from typing import List, Optional

from fastapi import UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse

from app.routers.base_router import BaseRouter
from app.schemas.common import ApiResponse
from app.utils.minio_utils import MinioClient
from app.utils.document_parser import DocumentParser
from app.utils.milvus_utils import MilvusClient
from app.utils.knowledge_graph import get_knowledge_graph
from app.manager.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


class FileTask:
    """单个文件处理任务的状态和结果"""

    def __init__(self, filename: str, size: int):
        self.filename = filename
        self.size = size
        self.status = "pending"  # pending / processing / success / error
        self.error = ""
        self.chunk_count = 0
        self.inserted = 0
        self.object_name = ""
        self.minio_url = ""
        self.total_chars = 0


class DocumentRouter(BaseRouter):
    def __init__(self):
        super().__init__()
        self.minio = MinioClient()
        self.parser = DocumentParser()
        self.milvus = MilvusClient()
        self.router = self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "/documents/upload",
            self.upload_endpoint,
            methods=["POST"],
            summary="批量异步上传文档并导入知识库",
        )
        self.router.add_api_route(
            "/documents",
            self.list_endpoint,
            methods=["GET"],
            summary="列出已上传文档",
        )
        self.router.add_api_route(
            "/documents/{source:path}",
            self.delete_endpoint,
            methods=["DELETE"],
            summary="删除文档及其 chunks",
        )
        self.router.add_api_route(
            "/knowledge/stats",
            self.stats_endpoint,
            methods=["GET"],
            summary="知识库统计",
        )
        return self.router

    # ── 批量异步上传入口 ──

    async def upload_endpoint(
        self,
        files: List[UploadFile] = File(..., description="多个文件（支持 txt/md/pdf/docx）"),
        session_id: str = Form(""),
    ):
        """批量上传文档 → 异步并发处理 → 全部入库

        - 每个文件独立解析、切分、向量化、入库
        - 全部完成后一次性返回结果列表
        - 单文件失败不影响其他文件
        """
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="请至少选择一个文件")

        # 校验并构建任务列表
        tasks: list[FileTask] = []
        for f in files:
            if not f.filename:
                continue
            if not DocumentParser.is_supported(f.filename):
                tasks.append(self._make_error_task(f.filename, f"不支持的文件类型"))
                continue
            tasks.append(FileTask(f.filename, f.size or 0))

        # 用 dict 做 filename → UploadFile 的快速查找
        file_map = {f.filename: f for f in files if f.filename}

        # 并发处理所有文件
        async def process(task: FileTask):
            if task.status == "error":
                return task
            task.status = "processing"
            try:
                await self._process_single_file(task, file_map.get(task.filename), session_id)
            except Exception as e:
                task.status = "error"
                task.error = str(e)[:200]
                logger.error(f"File failed [{task.filename}]: {e}")
            return task

        start = time.time()
        results = await asyncio.gather(*[process(t) for t in tasks])
        elapsed = time.time() - start

        success_count = sum(1 for r in results if r.status == "success")
        error_count = sum(1 for r in results if r.status == "error")

        logger.info(f"Batch upload: {len(results)} files, {success_count} success, "
                    f"{error_count} failed, {elapsed:.1f}s")

        return ApiResponse(
            status=200,
            message="success",
            data={
                "total": len(results),
                "success": success_count,
                "error": error_count,
                "elapsed_seconds": round(elapsed, 1),
                "files": [
                    {
                        "filename": r.filename,
                        "size": r.size,
                        "status": r.status,
                        "error": r.error,
                        "chunk_count": r.chunk_count,
                        "inserted": r.inserted,
                        "object_name": r.object_name,
                        "minio_url": r.minio_url,
                        "total_chars": r.total_chars,
                    }
                    for r in results
                ],
            },
        )

    # ── 单文件处理流水线 ──

    async def _process_single_file(self, task: FileTask, file: UploadFile | None, session_id: str):
        """单个文件的完整处理链路（在 executor 中运行）"""
        if file is None:
            task.status = "error"
            task.error = "文件未找到"
            return

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            task.status = "error"
            task.error = f"文件过大 ({len(content)//1024//1024}MB)，最大 50MB"
            return

        suffix = os.path.splitext(task.filename)[1]

        def _pipeline() -> FileTask:
            """同步流水线：写临时文件 → MinIO → 解析 → 切分 → Milvus"""
            tmp_path = None
            try:
                # 写临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                # 1. MinIO 上传
                object_name = f"{session_id or 'anonymous'}/{uuid.uuid4().hex}_{task.filename}"
                with open(tmp_path, "rb") as f:
                    self.minio.upload_stream(
                        object_name, f, len(content),
                        content_type=file.content_type,
                    )
                task.object_name = object_name
                task.minio_url = self.minio.get_url(object_name)

                # 2. 解析 + 切分
                source_info = {"session_id": session_id} if session_id else {}
                chunks = self.parser.parse_and_chunk(tmp_path, task.filename, source_info)

                if not chunks:
                    raise ValueError("文件内容为空，无法提取有效文本")

                task.total_chars = len(chunks[0].page_content) if chunks else 0
                task.chunk_count = len(chunks)

                # 3. 向量化 + Milvus 入库
                inserted = self.milvus.insert_chunks(chunks, session_id)
                task.inserted = inserted

                # 4. 持久化 MinIO URL 映射到 MySQL
                MemoryManager().save_document_source(
                    source=task.filename,
                    session_id=session_id,
                    object_name=object_name,
                    minio_url=task.minio_url,
                )

                # 5. 知识图谱关联（增强版：自动创建新概念节点）
                kg = get_knowledge_graph()
                if kg.available and chunks:
                    try:
                        full_text = " ".join(c.page_content for c in chunks[:5])

                        # 调用增强后的分析方法（一次性返回已有法条+新概念+关系）
                        analysis = kg.analyze_and_suggest_nodes(full_text)

                        existing_articles = analysis.get("existing_articles", [])
                        new_concepts = analysis.get("new_concepts", [])
                        new_relations = analysis.get("new_relations", [])

                        # 5a. 关联已有法条
                        if existing_articles:
                            kg.neo4j.link_document_to_articles(
                                doc_name=task.filename,
                                session_id=session_id,
                                matched_articles=existing_articles,
                            )
                            logger.info(f"Linked to existing articles: {existing_articles}")

                        # 5b. 创建新概念节点
                        if new_concepts:
                            created = kg.neo4j.create_concept_nodes(new_concepts)
                            logger.info(f"Created {created} new concept nodes")

                        # 5c. 创建概念间关系
                        if new_relations:
                            created = kg.neo4j.create_relations(new_relations)
                            logger.info(f"Created {created} new relations")

                        # 5d. 将新概念关联到文档
                        if new_concepts:
                            concept_names = [c.get("name") for c in new_concepts if c.get("name")]
                            if concept_names:
                                kg.neo4j.link_document_to_articles(
                                    doc_name=task.filename,
                                    session_id=session_id,
                                    matched_articles=concept_names,
                                )

                    except Exception as e:
                        logger.warning(f"Graph linking failed (non-fatal): {e}")

                task.status = "success"
                return task

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        # CPU 密集任务丢到线程池执行
        await asyncio.to_thread(_pipeline)

    @staticmethod
    def _make_error_task(filename: str, error: str) -> FileTask:
        t = FileTask(filename, 0)
        t.status = "error"
        t.error = error
        return t

    # ── 列表 / 删除 / 统计 ──

    async def list_endpoint(self, session_id: Optional[str] = Query(None)):
        docs = self.milvus.list_user_docs(session_id)
        for doc in docs:
            try:
                files = self.minio.list_files(prefix=doc.get("source", ""))
                doc["minio_files"] = files
            except Exception:
                doc["minio_files"] = []
        return ApiResponse(status=200, message="success", data={"documents": docs})

    async def delete_endpoint(self, source: str):
        if not source:
            raise HTTPException(status_code=400, detail="source 不能为空")
        milvus_deleted = self.milvus.delete_user_doc(source)
        self.minio.delete_prefix(prefix=source)
        return ApiResponse(
            status=200, message="success",
            data={"source": source, "milvus_chunks_deleted": milvus_deleted},
        )

    async def stats_endpoint(self):
        legal_count = self.milvus.count("legal_docs")
        user_count = self.milvus.count("user_docs")
        user_docs = self.milvus.list_user_docs()
        return ApiResponse(
            status=200, message="success",
            data={
                "legal_docs_count": legal_count,
                "user_docs_count": user_count,
                "user_docs_files": len(user_docs),
                "user_docs_list": user_docs,
            },
        )

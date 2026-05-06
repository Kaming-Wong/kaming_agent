import logging
from typing import List, Optional, Dict, Any

from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility,
)
from langchain_core.embeddings import Embeddings
from langchain.schema import Document

from app.config import settings
from app.adapters.embedding_adapter import EmbeddingAdapter

logger = logging.getLogger(__name__)

# Milvus 集合名称
LEGAL_COLLECTION = "legal_docs"    # 预置法律条文
USER_DOCS_COLLECTION = "user_docs" # 用户上传的文档

# 统一列名约定（两个集合共用）
FIELD_PK = "pk"
FIELD_TEXT = "text"
FIELD_TITLE = "title"
FIELD_VECTOR = "vector"
FIELD_SOURCE = "source"       # user_docs: 来源文件名
FIELD_CHUNK_INDEX = "chunk_index"  # user_docs: 片段序号
FIELD_SESSION_ID = "session_id"


class MilvusClient:
    """Milvus 向量数据库客户端

    管理两个集合：
    - legal_docs：预置的法律条文（通过 seed 脚本导入）
    - user_docs：用户上传文档切分后的 chunks

    搜索时同时查两个集合，合并排序后返回。
    """

    def __init__(self, embeddings: Optional[Embeddings] = None):
        self._embeddings = embeddings or EmbeddingAdapter().get_embeddings()
        self._connected = False

    def _connect(self):
        if self._connected:
            return
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    # ── legal_docs：预置法律知识库 ──

    def _ensure_legal_collection(self):
        """确保 legal_docs 集合存在，不存在则创建"""
        self._connect()
        if utility.has_collection(LEGAL_COLLECTION):
            collection = Collection(LEGAL_COLLECTION)
            collection.load()
            return collection

        fields = [
            FieldSchema(name=FIELD_PK, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=FIELD_TITLE, dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name=FIELD_TEXT, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=FIELD_VECTOR, dtype=DataType.FLOAT_VECTOR, dim=settings.MILVUS_VECTOR_DIM),
        ]
        schema = CollectionSchema(fields, description="法律知识库")
        collection = Collection(LEGAL_COLLECTION, schema)

        index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        collection.create_index(FIELD_VECTOR, index_params)
        collection.load()
        logger.info(f"Created Milvus collection: {LEGAL_COLLECTION}")
        return collection

    # ── user_docs：用户上传文档 ──

    def _ensure_user_docs_collection(self):
        """确保 user_docs 集合存在

        比 legal_docs 多了 source / chunk_index / session_id 三个元数据字段，
        用于追踪每个 chunk 的来源文件和归属会话。
        """
        self._connect()
        if utility.has_collection(USER_DOCS_COLLECTION):
            collection = Collection(USER_DOCS_COLLECTION)
            collection.load()
            return collection

        fields = [
            FieldSchema(name=FIELD_PK, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=FIELD_TEXT, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=FIELD_VECTOR, dtype=DataType.FLOAT_VECTOR, dim=settings.MILVUS_VECTOR_DIM),
            FieldSchema(name=FIELD_SOURCE, dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name=FIELD_CHUNK_INDEX, dtype=DataType.INT64),
            FieldSchema(name=FIELD_SESSION_ID, dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields, description="用户上传文档")
        collection = Collection(USER_DOCS_COLLECTION, schema)

        index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        collection.create_index(FIELD_VECTOR, index_params)
        collection.load()
        logger.info(f"Created Milvus collection: {USER_DOCS_COLLECTION}")
        return collection

    # ── 搜索 ──

    def search(self, query: str, top_k: int = 3, expr: Optional[str] = None,
               include_user_docs: bool = True) -> List[Dict[str, Any]]:
        """跨集合搜索

        同时检索 legal_docs 和 user_docs，按 score 降序合并后取 top_k。

        Args:
            query: 搜索文本
            top_k: 返回结果数
            expr: Milvus 标量过滤表达式（仅 legal_docs）
            include_user_docs: 是否包含用户上传文档
        """
        try:
            legal_results = self._search_collection(
                self._ensure_legal_collection(),
                query, top_k, expr,
                output_fields=[FIELD_TITLE, FIELD_TEXT],
                transform_func=lambda hit: {
                    "title": hit.entity.get(FIELD_TITLE),
                    "text": hit.entity.get(FIELD_TEXT),
                    "score": hit.score,
                    "source_type": "legal_docs",
                }
            )

            user_results = []
            if include_user_docs:
                user_results = self._search_collection(
                    self._ensure_user_docs_collection(),
                    query, top_k, None,
                    output_fields=[FIELD_SOURCE, FIELD_TEXT, FIELD_CHUNK_INDEX, FIELD_SESSION_ID],
                    transform_func=lambda hit: {
                        "title": f"【用户文档】{hit.entity.get(FIELD_SOURCE, '未知')} (片段 {hit.entity.get(FIELD_CHUNK_INDEX, 0)})",
                        "text": hit.entity.get(FIELD_TEXT, ""),
                        "score": hit.score,
                        "source_type": "user_docs",
                        "source_file": hit.entity.get(FIELD_SOURCE, ""),
                        "chunk_index": hit.entity.get(FIELD_CHUNK_INDEX, 0),
                    }
                )

            # 合并排序：两个集合的结果按相似度分数混合排序
            all_results = legal_results + user_results
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            logger.info(f"Milvus search: '{query[:50]}...' → legal={len(legal_results)}, user={len(user_results)}")
            return all_results[:top_k]

        except Exception as e:
            logger.warning(f"Milvus search failed: {e}")
            return []

    def _search_collection(self, collection: Collection, query: str, top_k: int,
                           expr: Optional[str], output_fields: List[str],
                           transform_func) -> List[Dict]:
        """在单个集合中执行向量搜索"""
        query_vector = self._embeddings.embed_query(query)
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

        results = collection.search(
            data=[query_vector],
            anns_field=FIELD_VECTOR,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
        )

        docs = []
        for hits in results:
            for hit in hits:
                docs.append(transform_func(hit))
        return docs

    # ── 插入 ──

    def insert_legal(self, title: str, text: str) -> bool:
        """插入一条法律条文"""
        try:
            collection = self._ensure_legal_collection()
            vector = self._embeddings.embed_query(text)
            collection.insert([[title], [text], [vector]])
            collection.flush()
            logger.info(f"Inserted legal doc: {title}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert legal doc: {e}")
            return False

    def batch_insert_legal(self, docs: List[Dict[str, str]]) -> int:
        success = 0
        for doc in docs:
            if self.insert_legal(doc["title"], doc["text"]):
                success += 1
        return success

    def insert_chunks(self, chunks: List[Document], session_id: str = "") -> int:
        """批量插入文档 chunks 到 user_docs

        批量向量化后一次插入，减少 Milvus 写入次数。
        """
        try:
            collection = self._ensure_user_docs_collection()
            texts = [c.page_content for c in chunks]
            source = chunks[0].metadata.get("source", "unknown") if chunks else "unknown"
            chunk_indices = [c.metadata.get("chunk_index", i) for i, c in enumerate(chunks)]

            vectors = self._embeddings.embed_documents(texts)

            collection.insert([
                texts,
                vectors,
                [source] * len(chunks),
                chunk_indices,
                [session_id] * len(chunks),
            ])
            collection.flush()
            logger.info(f"Inserted {len(chunks)} chunks into {USER_DOCS_COLLECTION}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to insert chunks: {e}")
            return 0

    # ── 文档管理 ──

    def list_user_docs(self, session_id: Optional[str] = None) -> List[Dict]:
        """列出用户上传文档的摘要信息（按 source 去重汇总）"""
        try:
            collection = self._ensure_user_docs_collection()
            expr = f'{FIELD_SESSION_ID} == "{session_id}"' if session_id else None

            results = collection.query(
                expr=expr or f"{FIELD_PK} > 0",
                output_fields=[FIELD_SOURCE, FIELD_CHUNK_INDEX, FIELD_SESSION_ID],
                limit=10000,
            )

            doc_summary = {}
            for r in results:
                src = r.get(FIELD_SOURCE, "unknown")
                if src not in doc_summary:
                    doc_summary[src] = {"source": src, "chunk_count": 0, "session_id": r.get(FIELD_SESSION_ID, "")}
                doc_summary[src]["chunk_count"] += 1

            return list(doc_summary.values())
        except Exception as e:
            logger.warning(f"List user docs failed: {e}")
            return []

    def delete_user_doc(self, source: str) -> int:
        """按源文件名删除所有对应的 chunks"""
        try:
            collection = self._ensure_user_docs_collection()
            expr = f'{FIELD_SOURCE} == "{source}"'
            result = collection.delete(expr)
            logger.info(f"Deleted {source} from {USER_DOCS_COLLECTION}: {result}")
            return len(result) if result else 0
        except Exception as e:
            logger.error(f"Failed to delete user doc: {e}")
            return 0

    def count(self, collection_name: str = LEGAL_COLLECTION) -> int:
        """获取集合的文档数量"""
        try:
            self._connect()
            if not utility.has_collection(collection_name):
                return 0
            collection = Collection(collection_name)
            return collection.num_entities
        except Exception:
            return 0

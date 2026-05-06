import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.messages import HumanMessage, AIMessage

from app.utils.neo4j_utils import get_neo4j, Neo4jClient
from app.adapters.llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = LLMAdapter().get_llm()
    return _llm


class KnowledgeGraph:
    """知识图谱服务

    职责：
    1. 从用户问题中抽取法律实体关键词
    2. 在 Neo4j 中做多跳关联查询
    3. 将图检索结果与向量检索结果合并
    """

    def __init__(self):
        self.neo4j = get_neo4j()

    @property
    def available(self) -> bool:
        return self.neo4j.available

    # ── 实体抽取 ──

    def extract_entities(self, question: str) -> List[str]:
        """用 LLM 从用户问题中抽取法律实体关键词

        例如："我试用期被辞退了有赔偿吗"
        → ["试用期", "辞退", "赔偿"]

        这些关键词用来在 Neo4j 中匹配法条/概念节点。
        """
        prompt = f"""从用户的法律咨询问题中提取关键法律实体词，每个词是一个法律概念或法条关键词。
只返回逗号分隔的词列表，不要解释。

用户问题：{question}

提取的关键词："""

        try:
            result = _get_llm().invoke(prompt)
            keywords = [k.strip() for k in result.content.strip().split(",") if k.strip()]
            return keywords[:5]  # 最多取 5 个
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            # 兜底：直接从问题中取最长名词短语
            import re
            words = re.findall(r'[一-鿿]{2,}', question)
            return words[:3] if words else []

    # ── 图检索 ──

    def search(self, question: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """知识图谱检索入口

        Args:
            question: 用户原始问题
            top_k: 返回结果数
        Returns:
            (context_text, references)
        """
        if not self.neo4j.available:
            return "", []

        keywords = self.extract_entities(question)
        if not keywords:
            return "", []

        logger.info(f"Graph search keywords: {keywords}")

        all_results = []
        seen = set()

        for kw in keywords:
            # 多跳关联查询
            results = self.neo4j.search_related(kw, limit=top_k)
            for r in results:
                key = r.get("source", "")
                if key and key not in seen:
                    seen.add(key)
                    all_results.append(r)

            # 关键字直查补充
            direct = self.neo4j.search_by_keyword(kw, limit=2)
            for r in direct:
                key = r.get("source", "")
                if key and key not in seen:
                    seen.add(key)
                    all_results.append(r)

        # 按 score 降序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        top = all_results[:top_k]

        context_parts = []
        references = []
        for r in top:
            source = r.get("source", "")
            text = r.get("text", "")
            context_parts.append(f"【图谱·{source}】\n{text}")
            references.append({
                "source": f"图谱·{source}",
                "chunk_index": 0,
                "minio_url": "",
                "relevance": f"知识图谱关联 {r.get('category', '')}",
            })

        context = "\n\n".join(context_parts)
        logger.info(f"Graph search: {len(top)} results from {len(keywords)} keywords")
        return context, references

    # ── 双通道合并 ──

    @staticmethod
    def merge_results(
        vector_context: str,
        vector_refs: List[Dict],
        graph_context: str,
        graph_refs: List[Dict],
        max_total: int = 5,
    ) -> Tuple[str, List[Dict]]:
        """合并向量检索和图检索结果

        策略：向量优先（语义匹配更准），图谱补充（关系推理）。
        先取向量结果，再用图谱结果填满 max_total。
        """
        combined_refs = list(vector_refs)

        # 去重：source 不重复
        seen_sources = {r.get("source", "") for r in combined_refs}

        for r in graph_refs:
            if len(combined_refs) >= max_total:
                break
            source = r.get("source", "")
            if source and source not in seen_sources:
                seen_sources.add(source)
                combined_refs.append(r)

        # 合并上下文文本
        parts = []
        if vector_context:
            parts.append(vector_context)
        if graph_context:
            parts.append(graph_context)

        context = "\n\n".join(parts)
        return context, combined_refs

    # ── 文档实体抽取(上传时用) ──

    def extract_articles_from_doc(self, doc_text: str) -> List[str]:
        """从文档文本中抽取可能引用的法条名称

        用于上传文档时自动关联到知识图谱中的法条节点。
        """
        # 先用图谱中的法条名称做精确匹配
        articles = self.neo4j.query(
            "MATCH (a:LawArticle) RETURN a.name AS name"
        )
        matched = []
        for art in articles:
            name = art.get("name", "")
            if name and name in doc_text:
                matched.append(name)

        # 再用 LLM 补充抽取
        if len(matched) < 3:
            try:
                prompt = f"""从以下文档文本中识别涉及的法律法规名称（如：劳动合同法、劳动法等）。
只返回逗号分隔的名称列表，不要解释。

文本开头：{doc_text[:500]}

涉及的法律法规："""
                result = _get_llm().invoke(prompt)
                llm_matched = [m.strip() for m in result.content.strip().split(",") if m.strip()]
                for m in llm_matched:
                    if m and m not in matched:
                        matched.append(m)
            except Exception as e:
                logger.warning(f"LLM article extraction failed: {e}")

        return matched[:5]


# 全局单例
_kg: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg

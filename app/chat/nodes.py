import logging
import hashlib

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import interrupt

from app.chat.state import AgentState
from app.adapters.llm_adapter import LLMAdapter
from app.manager.memory_manager import MemoryManager
from app.utils.redis_utils import get_redis

logger = logging.getLogger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = LLMAdapter().get_llm()
    return _llm


# ── LLM 调用封装（带 Redis 缓存） ──

# 同一次会话中可能出现完全相同的用户问题（用户重复发送、刷新重试等）。
# 用 Redis 缓存 LLM 响应，避免重复调用浪费算力。
# 缓存粒度：按输入文本的 md5 前缀分 key，不同类型的调用用不同前缀（intent/chat/legal/conf）避免冲突。


def _llm_invoke(messages, cache_key: str = "", cache_ttl: int = 300):
    """调用 LLM，相同输入走 Redis 缓存（默认 5 分钟过期）"""
    if cache_key:
        redis = get_redis()
        cached = redis.get_cached_llm_response(cache_key)
        if cached:
            logger.info(f"LLM cache hit: {cache_key[:40]}...")
            # 用 SimpleNamespace 模拟 LLMResult，保持返回类型一致
            return AIMessage(content=cached)

    result = _get_llm().invoke(messages)

    if cache_key:
        get_redis().cache_llm_response(cache_key, result.content, ttl=cache_ttl)

    return result


def _cache_key(text: str, prefix: str = "llm") -> str:
    """生成缓存 key：前缀 + 输入文本的 md5 指纹"""
    return f"{prefix}:{hashlib.md5(text.encode()).hexdigest()[:16]}"


# ════════════════════════════════════════════════
# 1. 意图分析
#    → 分类用户消息，决定走日常对话 / 法律咨询 / 转人工
#    → LLM 只输出一个词，string 匹配兜底
# ════════════════════════════════════════════════

def analyze_intent(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    prompt = f"""你是一个智能客服系统的意图分析器。分析用户最新一条消息的意图，只返回以下三类之一：

- general_chat: 日常闲聊、一般性问题、非法律咨询
- legal_assist: 法律相关问题（合同、法规、权益、诉讼、知识产权等）
- human_handoff: 明确要求转人工、投诉、复杂问题、或无法归类的请求

只返回一个词，不要解释。

用户消息：{last_message}"""

    try:
        result = _llm_invoke(prompt, cache_key=_cache_key(last_message, "intent"))
        intent = result.content.strip().lower()
        # 防止小模型乱输出，不在三分类内的统统归为 general_chat
        if intent not in ("general_chat", "legal_assist", "human_handoff"):
            intent = "general_chat"
    except Exception as e:
        logger.error(f"Intent analysis failed: {e}")
        intent = "general_chat"

    logger.info(f"Intent: {intent} | {last_message[:50]}...")
    return {"intent": intent}


# ════════════════════════════════════════════════
# 2. 主 Agent — 日常对话
#    → 通用问答，无 RAG
#    → 不要求 JSON 输出（小模型能力有限），引用信息单独走 state.references
# ════════════════════════════════════════════════

def general_chat(state: AgentState) -> dict:
    last = state["messages"][-1].content if state["messages"] else ""
    try:
        response = _llm_invoke(state["messages"], cache_key=_cache_key(last, "chat"))
        return {
            "messages": [response],
            "current_response": response.content,
        }
    except Exception as e:
        logger.error(f"General chat failed: {e}")
        return {
            "messages": [AIMessage(content="抱歉，我暂时无法回答您的问题。")],
            "need_human": True,
        }


# ════════════════════════════════════════════════
# 3. 法律 RAG 检索
#    → Milvus 中同时搜 legal_docs（预置法条）+ user_docs（用户上传文档）
#    → 返回两份数据：legal_context（LLM 用的文本） + references（前端用的元数据）
#    → 元数据从 MySQL document_sources 表反查 minio_url，实现可追溯
# ════════════════════════════════════════════════

def legal_search(state: AgentState) -> dict:
    """双通道检索：Milvus 向量 + Neo4j 知识图谱，合并后返回"""
    last_message = state["messages"][-1].content
    memory = MemoryManager()

    try:
        from app.utils.milvus_utils import MilvusClient
        from app.utils.knowledge_graph import get_knowledge_graph, KnowledgeGraph

        mc = MilvusClient()
        kg = get_knowledge_graph()

        # ── 通道1: Milvus 向量检索 ──
        vector_results = mc.search(last_message, top_k=3, include_user_docs=True)

        vector_context_parts = []
        vector_refs = []

        for r in vector_results:
            title = r.get("title", "知识条目")
            text = r.get("text", "")
            source_type = r.get("source_type", "legal_docs")

            vector_context_parts.append(f"【{title}】\n{text}")

            minio_url = ""
            source_file = r.get("source_file", "")
            if source_file:
                minio_url = memory.get_minio_url(source_file)

            chunk_index = r.get("chunk_index", 0) if source_type == "user_docs" else 0

            vector_refs.append({
                "source": title.strip("【】"),
                "chunk_index": chunk_index,
                "minio_url": minio_url,
                "relevance": f"语义相似度 {r.get('score', 0):.2f}",
            })

        vector_context = "\n\n".join(vector_context_parts)

        # ── 通道2: Neo4j 知识图谱检索 ──
        graph_context = ""
        graph_refs = []
        if kg.available:
            try:
                graph_context, graph_refs = kg.search(last_message, top_k=2)
            except Exception as e:
                logger.warning(f"Graph search failed (proceeding without): {e}")

        # 合并双通道
        context, references = KnowledgeGraph.merge_results(
            vector_context, vector_refs,
            graph_context, graph_refs,
            max_total=5,
        )

        logger.info(f"RAG: vector={len(vector_results)}, graph={len(graph_refs)}, merged={len(references)}")

    except Exception as e:
        logger.warning(f"RAG search failed: {e}")
        context = "（知识库暂不可用）"
        references = []

    return {"legal_context": context, "references": references}


# ════════════════════════════════════════════════
# 4. 法律子 Agent
#    → 基于 RAG 结果回答法律问题
#    → 引用来源单独走 state.references 传给前端，LLM 只需要自然说话
#    → 强制声明"仅供参考，不构成法律意见"
# ════════════════════════════════════════════════

def legal_assist(state: AgentState) -> dict:
    context = state.get("legal_context", "")

    system_prompt = f"""你是一名专业的法律助手。请基于以下法律条文回答用户的法律问题。
回答中请用《书名号》引用你参考的具体法条或文档名称。
如果提供的法律条文不足以回答，请说明并给出一般性法律指引。
注意：你必须声明"以上回答仅供参考，不构成法律意见，具体问题请咨询专业律师"。

相关法律条文：
{context}"""

    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m.type, "content": m.content} for m in state["messages"]
    ]

    try:
        last = state["messages"][-1].content if state["messages"] else ""
        response = _llm_invoke(messages, cache_key=_cache_key(f"legal:{last}", "legal"))
        refs = state.get("references", [])
        minio_url = refs[0].get("minio_url", "") if refs else ""
        return {
            "messages": [response],
            "current_response": response.content,
            "references": refs,
            "minio_url": minio_url,
        }
    except Exception as e:
        logger.error(f"Legal assist failed: {e}")
        return {
            "messages": [AIMessage(content="抱歉，法律助手暂时无法提供服务。")],
            "need_human": True,
        }


# ════════════════════════════════════════════════
# 5. 可信度检查
#    → LLM 自评上一轮回答的质量
#    → not_confident → 转人工
#    → 目的是兜底：避免 LLM 胡编乱造直接给用户
# ════════════════════════════════════════════════

def check_confidence(state: AgentState) -> dict:
    # 日常聊天不检查可信度，直接放行
    if state.get("intent") == "general_chat":
        return {"need_human": False}

    last_ai_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None
    )
    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    if not last_ai_msg:
        return {"need_human": True}

    prompt = f"""用户问题：{last_user_msg}
AI回答：{last_ai_msg[:300]}

只判断两种情况：
- not_confident: AI 承认无法回答、回答有事实性错误、或回答了"我不清楚"/"我无法回答"
- confident: 其他所有情况（包括回答不够完整但无硬伤）

只返回 confident 或 not_confident，不要解释。"""

    try:
        result = _llm_invoke(prompt, cache_key=_cache_key(f"conf:{last_user_msg[:100]}", "conf"))
        need_human = "not_confident" in result.content.strip().lower()
    except Exception as e:
        logger.error(f"Confidence check failed: {e}")
        need_human = True

    logger.info(f"Confidence: {'→ human' if need_human else '→ respond'}")
    return {"need_human": need_human}


# ════════════════════════════════════════════════
# 6. 转人工
#    → interrupt() 暂停图执行，等人工客服回复
#    → 人工回复后 Command(resume=...) 恢复，interrupt() 的返回值就是人工输入
#    → 整个会话上下文被打包传给人工后台
# ════════════════════════════════════════════════

def human_handoff(state: AgentState) -> dict:
    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    payload = {
        "type": "human_handoff",
        "session_id": state["session_id"],
        "user_message": last_user_msg,
        "intent": state.get("intent", "unknown"),
        "conversation_summary": _summarize(state["messages"]),
    }

    # interrupt() 在此暂停，人工后台看到 "待处理会话" 后回复
    human_input = interrupt(payload)

    # interrupt() 恢复后拿到人工客服的输入
    msg = ""
    if isinstance(human_input, dict):
        msg = human_input.get("message", "")
    elif isinstance(human_input, str):
        msg = human_input

    if not msg:
        msg = "已为您转接人工客服，请稍候，客服人员正在查看您的问题。"

    logger.info(f"Human handoff done for {state['session_id']}")
    return {
        "messages": [AIMessage(content=msg)],
        "human_feedback": msg,
        "need_human": True,
    }


def _summarize(messages) -> str:
    """给人工客服看的对话摘要，取最近 10 条"""
    parts = []
    for m in messages[-10:]:
        role = "用户" if isinstance(m, HumanMessage) else "AI"
        parts.append(f"{role}: {m.content[:200]}")
    return "\n".join(parts)


# ════════════════════════════════════════════════
# 7. 路由条件函数
#    → 供 graph.py 的 add_conditional_edges 使用
#    → 返回字符串匹配节点名称
# ════════════════════════════════════════════════

def route_decision(state: AgentState) -> str:
    """intent 路由：general_chat / legal_assist / human_handoff"""
    return state.get("intent", "general_chat")


def after_check(state: AgentState) -> str:
    """可信度路由：confident → END，not_confident → human_handoff"""
    return "not_confident" if state.get("need_human") else "confident"

import logging
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from app.chat.state import AgentState
from app.chat import nodes as n

logger = logging.getLogger(__name__)


def build_graph() -> CompiledStateGraph:
    """构建并编译 LangGraph 客服状态机

    图结构（有向图）：
        START
          │
          ▼
     analyze_intent  ←─── LLM 分类用户意图
        │   │   │
        │   │   └──────────┐
        │   │              │
        │   └──────┐       │
        │          │       │
        ▼          ▼       ▼
  general_chat  legal_search  human_handoff
        │          │              │
        │          ▼              │
        │    legal_assist         │
        │          │              │
        └─────┬────┘              │
              │                   │
              ▼                   │
     check_confidence  ←── LLM 自评回答质量
        │        │               │
   confident  not_confident      │
        │          │              │
        ▼          ▼              │
       END    human_handoff ──────┘
                   │
                   ▼
                  END

    MemorySaver 用于支持 interrupt()：当 human_handoff 被触发时，
    图在此暂停，等待人工客服回复后通过 Command(resume) 恢复。
    """
    builder = StateGraph(AgentState)

    # ── 注册所有节点 ──
    builder.add_node("analyze_intent", n.analyze_intent)
    builder.add_node("general_chat", n.general_chat)
    builder.add_node("legal_search", n.legal_search)
    builder.add_node("legal_assist", n.legal_assist)
    builder.add_node("check_confidence", n.check_confidence)
    builder.add_node("human_handoff", n.human_handoff)

    # ── 起点 ──
    builder.add_edge(START, "analyze_intent")

    # ── 意图路由 ──
    # route_decision 返回 "general_chat" / "legal_assist" / "human_handoff"
    # 映射到对应的下游节点
    builder.add_conditional_edges(
        "analyze_intent",
        n.route_decision,
        {
            "general_chat": "general_chat",
            "legal_assist": "legal_search",
            "human_handoff": "human_handoff",
        },
    )

    # ── 法律检索 → 法律回答 ──
    builder.add_edge("legal_search", "legal_assist")

    # ── 两个 Agent 节点之后都接可信度检查 ──
    builder.add_edge("general_chat", "check_confidence")
    builder.add_edge("legal_assist", "check_confidence")

    # ── 可信度路由 ──
    # after_check 返回 "confident" → END
    # after_check 返回 "not_confident" → human_handoff
    builder.add_conditional_edges(
        "check_confidence",
        n.after_check,
        {
            "confident": END,
            "not_confident": "human_handoff",
        },
    )

    # ── 转人工后结束本轮（下次用户消息重新走 START）──
    builder.add_edge("human_handoff", END)

    # ── 编译 ──
    # MemorySaver 是必须的：interrupt() 依赖 checkpoint 机制
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("LangGraph state machine built successfully")
    return graph


# 全局单例：服务启动时构建一次，后续复用
_graph: Optional[CompiledStateGraph] = None


def get_graph() -> CompiledStateGraph:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

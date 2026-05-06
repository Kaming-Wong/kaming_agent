import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel

from app.config import settings

logger = logging.getLogger(__name__)


class LLMAdapter:
    """LLM 策略适配器：支持 ollama / 硅基流动 / OpenAI 兼容"""

    def __init__(self, strategy: Optional[str] = None, model: Optional[str] = None):
        self.strategy = strategy or settings.LLM_STRATEGY
        self.model = model or settings.LLM_MODEL
        self._llm: Optional[BaseChatModel] = None

    def _init_llm(self) -> BaseChatModel:
        if self.strategy == "ollama":
            from langchain_ollama import ChatOllama
            logger.info(f"Initializing Ollama LLM: model={self.model}, base_url={settings.OLLAMA_BASE_URL}")
            return ChatOllama(
                model=self.model,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.1,
            )
        elif self.strategy in ("silicon_flow", "openai"):
            from langchain_openai import ChatOpenAI
            api_key = settings.SILICON_FLOW_API_KEY or "sk-placeholder"
            base_url = settings.SILICON_FLOW_BASE_URL or "https://api.siliconflow.cn/v1"
            logger.info(f"Initializing OpenAI-compatible LLM: model={self.model}, base_url={base_url}")
            return ChatOpenAI(
                model=self.model,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0.1,
            )
        else:
            raise ValueError(f"Unsupported LLM strategy: {self.strategy}")

    def get_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = self._init_llm()
        return self._llm

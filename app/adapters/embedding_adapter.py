import logging
from typing import Optional

from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingAdapter:
    """Embedding 策略适配器：支持 ollama / 硅基流动"""

    def __init__(self, strategy: Optional[str] = None, model: Optional[str] = None):
        self.strategy = strategy or settings.EMBEDDING_STRATEGY
        self.model = model or settings.EMBEDDING_MODEL
        self._embeddings: Optional[Embeddings] = None

    def _init_embeddings(self) -> Embeddings:
        if self.strategy == "ollama":
            from langchain_ollama import OllamaEmbeddings
            logger.info(f"Initializing Ollama Embeddings: model={self.model}, base_url={settings.OLLAMA_BASE_URL}")
            return OllamaEmbeddings(
                model=self.model,
                base_url=settings.OLLAMA_BASE_URL,
            )
        elif self.strategy in ("silicon_flow", "openai"):
            from langchain_openai import OpenAIEmbeddings
            api_key = settings.SILICON_FLOW_API_KEY or "sk-placeholder"
            base_url = settings.SILICON_FLOW_BASE_URL or "https://api.siliconflow.cn/v1"
            logger.info(f"Initializing OpenAI-compatible Embeddings: model={self.model}, base_url={base_url}")
            return OpenAIEmbeddings(
                model=self.model,
                openai_api_key=api_key,
                openai_api_base=base_url,
            )
        else:
            raise ValueError(f"Unsupported Embedding strategy: {self.strategy}")

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = self._init_embeddings()
        return self._embeddings

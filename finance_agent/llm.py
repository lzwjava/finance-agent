"""DeepSeek LLM adapter for LlamaIndex.

DeepSeek serves an OpenAI-compatible API, so we reuse llama-index's OpenAI
integration and only override the model metadata (DeepSeek model names are not
in OpenAI's internal whitelist, which would otherwise raise a ValueError).
"""

import os

from llama_index.core.base.llms.base import LLMMetadata
from llama_index.core.base.llms.types import MessageRole
from llama_index.llms.openai import OpenAI as LLOpenAI


class DeepSeekLLM(LLOpenAI):
    """OpenAI-compatible LLM pointed at https://api.deepseek.com."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=65536,
            num_output=self.max_tokens or 8192,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )


def get_llm() -> DeepSeekLLM:
    """Build the DeepSeek LLM from environment variables (or a .env file)."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export DEEPSEEK_API_KEY in your shell."
        )
    return DeepSeekLLM(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.3")),
        timeout=float(os.environ.get("DEEPSEEK_TIMEOUT", "90")),
    )
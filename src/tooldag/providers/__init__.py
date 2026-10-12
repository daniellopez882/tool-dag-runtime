"""Chat-model providers: an OpenAI-compatible HTTP adapter and a scripted fake."""

from tooldag.providers.base import ChatMessage, ChatProvider, ProviderError, complete
from tooldag.providers.fake import FakeProvider
from tooldag.providers.openai_compat import OpenAICompatibleProvider

__all__ = [
    "ChatMessage",
    "ChatProvider",
    "FakeProvider",
    "OpenAICompatibleProvider",
    "ProviderError",
    "complete",
]

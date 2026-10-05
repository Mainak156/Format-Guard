from __future__ import annotations

from .base import LLMProvider
from .mock import MockProvider

__all__ = [
    "LLMProvider",
    "MockProvider",
    "GroqProvider",
    "LangChainGroqProvider",
]


def __getattr__(name: str):
    if name == "GroqProvider":
        from .groq import GroqProvider

        return GroqProvider

    if name == "LangChainGroqProvider":
        from .langchain_groq import LangChainGroqProvider

        return LangChainGroqProvider

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
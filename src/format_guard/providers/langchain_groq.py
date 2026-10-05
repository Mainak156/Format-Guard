from __future__ import annotations

import os
from typing import TypeVar

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel


load_dotenv()


T = TypeVar("T", bound=BaseModel)


class LangChainGroqProvider:
    """
    LangChain-backed Groq provider for Format Guard.

    Supports both:
    1. Plain-text generation through generate()
    2. Native structured output through generate_structured()

    The plain-text interface keeps compatibility with the existing
    Format Guard validation and repair pipeline.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Groq API key is required. "
                "Pass api_key=... or set GROQ_API_KEY."
            )

        self.model = model
        self.temperature = temperature

        self.llm = ChatGroq(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a plain-text response through LangChain.
        """

        response = self.llm.invoke(prompt)

        content = response.content

        if not isinstance(content, str):
            raise RuntimeError(
                "LangChain Groq returned a non-text response."
            )

        return content

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
    ) -> T:
        """
        Generate a response using LangChain's native structured
        output support and return a validated Pydantic object.
        """

        structured_llm = self.llm.with_structured_output(
            schema
        )

        result = structured_llm.invoke(prompt)

        if not isinstance(result, schema):
            raise RuntimeError(
                "LangChain Groq returned an unexpected "
                "structured-output type."
            )

        return result
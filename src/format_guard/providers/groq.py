from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class GroqProvider:
    """
    Groq-backed LLM provider for Format Guard.

    The provider converts a prompt into a plain text response,
    keeping the rest of Format Guard independent of Groq.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-oss-120b",
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Groq API key is required. "
                "Pass api_key=... or set GROQ_API_KEY."
            )

        self.model = model
        self.client = Groq(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to Groq and return the raw text response.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        content = response.choices[0].message.content

        if content is None:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        return content
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


@dataclass(frozen=True)
class ModelPricing:
    """
    Token pricing for a benchmark model.

    Prices are expressed in USD per 1 million tokens.
    """

    input_per_million: float
    output_per_million: float


@dataclass
class BenchmarkUsage:
    """
    Token usage accumulated by a benchmark provider.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens used."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class GroqModelSpec:
    """
    Configuration for a Groq benchmark model.
    """

    name: str
    model_id: str
    pricing: ModelPricing


GROQ_MODEL_CATALOG: dict[str, GroqModelSpec] = {
    "gpt-oss-120b": GroqModelSpec(
        name="GPT-OSS 120B",
        model_id="openai/gpt-oss-120b",
        pricing=ModelPricing(
            input_per_million=0.15,
            output_per_million=0.60,
        ),
    ),
    "gpt-oss-20b": GroqModelSpec(
        name="GPT-OSS 20B",
        model_id="openai/gpt-oss-20b",
        pricing=ModelPricing(
            input_per_million=0.075,
            output_per_million=0.30,
        ),
    ),
    "qwen3.8-27b": GroqModelSpec(
        name="Qwen 3.8 27B",
        model_id="qwen/qwen3.8-27b",
        pricing=ModelPricing(
            input_per_million=0.80,
            output_per_million=4.00,
        ),
    ),
}


class GroqBenchmarkProvider:
    """
    Cost-aware Groq provider for benchmark execution.

    It behaves like the normal LLM callable expected by
    BenchmarkRunner while also exposing token usage and
    accumulated estimated cost.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = (
            os.getenv("GROQ_API_KEY")
            if api_key is None
            else api_key
        )

        if not self.api_key:
            raise ValueError(
                "Groq API key is required. "
                "Pass api_key=... or set GROQ_API_KEY."
            )

        self.model = model
        self.temperature = temperature
        self.client = Groq(api_key=self.api_key)
        self.usage = BenchmarkUsage()
        self._pricing = self._find_pricing(model)

    @staticmethod
    def _find_pricing(
        model: str,
    ) -> ModelPricing:
        """
        Find pricing configuration for a Groq model.
        """

        for spec in GROQ_MODEL_CATALOG.values():
            if spec.model_id == model:
                return spec.pricing

        raise ValueError(
            f"No benchmark pricing configured "
            f"for Groq model '{model}'."
        )

    @property
    def total_cost(self) -> float:
        """
        Return accumulated estimated cost in USD.
        """

        input_cost = (
            self.usage.prompt_tokens
            / 1_000_000
            * self._pricing.input_per_million
        )

        output_cost = (
            self.usage.completion_tokens
            / 1_000_000
            * self._pricing.output_per_million
        )

        return input_cost + output_cost

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate one benchmark response.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=self.temperature,
        )

        usage = response.usage

        if usage is not None:
            self.usage.prompt_tokens += (
                getattr(
                    usage,
                    "prompt_tokens",
                    0,
                )
                or 0
            )

            self.usage.completion_tokens += (
                getattr(
                    usage,
                    "completion_tokens",
                    0,
                )
                or 0
            )

        content = (
            response.choices[0]
            .message
            .content
        )

        if content is None:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        return content

    def __call__(
        self,
        prompt: str,
    ) -> str:
        """
        Allow the provider to be used directly as an LLM callable.
        """

        return self.generate(prompt)


def build_groq_benchmark_providers(
    *,
    api_key: str | None = None,
) -> dict[str, GroqBenchmarkProvider]:
    """
    Build all currently configured public Groq
    benchmark providers.

    Returns providers keyed by human-readable
    benchmark names.
    """

    providers: dict[str, GroqBenchmarkProvider] = {}

    for spec in GROQ_MODEL_CATALOG.values():
        providers[spec.name] = GroqBenchmarkProvider(
            model=spec.model_id,
            api_key=api_key,
        )

    return providers
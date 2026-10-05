from __future__ import annotations

import json
from pathlib import Path

from format_guard.benchmark import (
    BenchmarkRunner,
    build_groq_benchmark_providers,
)
from format_guard.cleaner import parse_json
from pydantic import BaseModel, ConfigDict


class Customer(BaseModel):
    """
    Expected structured output for the benchmark.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    age: int
    email: str
    company: str
    role: str


def load_smoke_prompts(limit: int = 5) -> list[str]:
    """
    Load the first few prompts from the 200-prompt dataset.
    """

    dataset_path = (
        Path(__file__).parent
        / "data"
        / "customer_prompts.json"
    )

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    prompts: list[str] = []

    for item in dataset[:limit]:
        prompts.append(item["prompt"])

    return prompts


def main() -> None:
    prompts = load_smoke_prompts(limit=5)

    print("=" * 70)
    print("FORMAT GUARD — GROQ BENCHMARK SMOKE TEST")
    print("=" * 70)
    print()
    print(f"Prompts: {len(prompts)}")
    print("Models: 3")
    print("Maximum retries: 2")
    print("Maximum initial calls: 15")
    print()

    print("Loading Groq benchmark providers...")

    providers = build_groq_benchmark_providers()

    print()
    print("Configured models:")

    for name, provider in providers.items():
        print(
            f"  - {name} "
            f"({provider.model})"
        )

    print()

    runner = BenchmarkRunner(
        schema=Customer,
        max_retries=2,
    )

    for model_name, provider in providers.items():
        print("-" * 70)
        print(f"MODEL: {model_name}")
        print("-" * 70)

        result = runner.run_model(
            model_name=model_name,
            llm_fn=provider,
            prompts=prompts,
        )

        print(
            f"Valid before : "
            f"{result.valid_before}/{result.total_prompts}"
        )

        print(
            f"Valid after  : "
            f"{result.valid_after}/{result.total_prompts}"
        )

        print(
            f"Before rate  : "
            f"{result.valid_rate_before:.2%}"
        )

        print(
            f"After rate   : "
            f"{result.valid_rate_after:.2%}"
        )

        print(
            f"Improvement  : "
            f"{result.improvement:+.2%}"
        )

        print(
            f"Avg attempts : "
            f"{result.average_attempts:.2f}"
        )

        print(
            f"Avg retries  : "
            f"{result.average_retries:.2f}"
        )

        print(
            f"Repair rate  : "
            f"{result.repair_rate:.2%}"
        )

        print(
            f"Flagged      : "
            f"{result.flagged}"
        )

        print(
            f"Cost before  : "
            f"${result.estimated_cost_before:.6f}"
        )

        print(
            f"Cost after   : "
            f"${result.estimated_cost_after:.6f}"
        )

        print(
            f"Extra cost   : "
            f"${result.extra_cost:.6f}"
        )

        print(
            f"Provider tok.: "
            f"{provider.usage.total_tokens}"
        )

        print()

    print("=" * 70)
    print("SMOKE TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
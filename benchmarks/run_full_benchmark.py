from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from format_guard.benchmark import (
    build_groq_benchmark_providers,
)
from format_guard.core import guard
from format_guard.cleaner import parse_json
from pydantic import ValidationError


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "customer_prompts.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "groq_full_benchmark.json"

TOTAL_PROMPTS = 200
MAX_RETRIES = 2

# Groq developer limits can be restrictive.
# A small delay between API calls provides basic rate-limit protection.
CALL_DELAY_SECONDS = 2.1

# Retry API/network failures independently from Format Guard retries.
API_RETRY_ATTEMPTS = 4
API_RETRY_BASE_DELAY = 5.0


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


def load_dataset() -> list[dict[str, Any]]:
    """
    Load the benchmark dataset.
    """

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "Benchmark dataset must contain a JSON list."
        )

    if len(dataset) < TOTAL_PROMPTS:
        raise ValueError(
            f"Expected at least {TOTAL_PROMPTS} prompts, "
            f"found {len(dataset)}."
        )

    return dataset[:TOTAL_PROMPTS]


def load_checkpoint() -> dict[str, Any]:
    """
    Load previously completed benchmark results.

    If no checkpoint exists, start with an empty structure.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not RESULTS_PATH.exists():
        return {
            "metadata": {
                "total_prompts": TOTAL_PROMPTS,
                "max_retries": MAX_RETRIES,
                "call_delay_seconds": CALL_DELAY_SECONDS,
            },
            "models": {},
        }

    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        checkpoint = json.load(file)

    checkpoint.setdefault(
        "metadata",
        {},
    )
    checkpoint.setdefault(
        "models",
        {},
    )

    return checkpoint


def save_checkpoint(
    checkpoint: dict[str, Any],
) -> None:
    """
    Atomically save the current benchmark checkpoint.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = RESULTS_PATH.with_suffix(
        ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            checkpoint,
            file,
            indent=2,
        )

    temporary_path.replace(RESULTS_PATH)


def is_valid_output(
    output: str,
) -> bool:
    """
    Check whether raw model output satisfies Customer.
    """

    try:
        parsed = parse_json(output)
        Customer.model_validate(parsed)
        return True
    except (
        json.JSONDecodeError,
        ValidationError,
    ):
        return False


def call_with_backoff(
    provider: Any,
    prompt: str,
) -> str:
    """
    Call a provider with exponential backoff for transient
    API/network failures.

    Format Guard validation retries are handled separately.
    """

    last_error: Exception | None = None

    for attempt in range(
        API_RETRY_ATTEMPTS
    ):
        try:
            return provider(prompt)

        except Exception as exc:
            last_error = exc

            if attempt == API_RETRY_ATTEMPTS - 1:
                raise

            delay = (
                API_RETRY_BASE_DELAY
                * (2 ** attempt)
            )

            print(
                f"      API error: {exc}"
            )
            print(
                f"      Retrying API call in "
                f"{delay:.1f}s "
                f"({attempt + 1}/"
                f"{API_RETRY_ATTEMPTS - 1})..."
            )

            time.sleep(delay)

    raise RuntimeError(
        "Provider call failed."
    ) from last_error


def run_prompt(
    provider: Any,
    prompt: str,
) -> dict[str, Any]:
    """
    Run one benchmark prompt using the paired
    baseline/repaired methodology.
    """

    cost_before_call = provider.total_cost

    time.sleep(CALL_DELAY_SECONDS)

    raw_output = call_with_backoff(
        provider,
        prompt,
    )

    cost_after_baseline = provider.total_cost

    baseline_cost = (
        cost_after_baseline
        - cost_before_call
    )

    valid_before = is_valid_output(
        raw_output
    )

    first_call = True

    def guarded_llm(
        repair_prompt: str,
    ) -> str:
        nonlocal first_call

        if first_call:
            first_call = False
            return raw_output

        time.sleep(CALL_DELAY_SECONDS)

        return call_with_backoff(
            provider,
            repair_prompt,
        )

    cost_before_guard = provider.total_cost

    guarded = guard(
        schema=Customer,
        llm_fn=guarded_llm,
        prompt=prompt,
        max_retries=MAX_RETRIES,
    )

    cost_after_guard = provider.total_cost

    repair_cost = (
        cost_after_guard
        - cost_before_guard
    )

    return {
        "valid_before": valid_before,
        "valid_after": guarded.success,
        "attempts": guarded.attempts,
        "retries": (
            guarded.metrics.repairs
            if guarded.metrics is not None
            else 0
        ),
        "repaired": guarded.repaired,
        "flagged": guarded.flagged,
        "fallback_used": guarded.fallback_used,
        "baseline_cost": baseline_cost,
        "repair_cost": repair_cost,
        "total_cost": (
            baseline_cost
            + repair_cost
        ),
        "error": guarded.error,
    }


def model_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate aggregate metrics for completed records.
    """

    total = len(records)

    if total == 0:
        return {
            "total_prompts": 0,
            "valid_before": 0,
            "valid_after": 0,
            "valid_rate_before": 0.0,
            "valid_rate_after": 0.0,
            "improvement": 0.0,
            "average_attempts": 0.0,
            "average_retries": 0.0,
            "repair_rate": 0.0,
            "flagged": 0,
            "estimated_cost_before": 0.0,
            "estimated_cost_after": 0.0,
            "extra_cost": 0.0,
        }

    valid_before = sum(
        bool(record["valid_before"])
        for record in records
    )

    valid_after = sum(
        bool(record["valid_after"])
        for record in records
    )

    total_attempts = sum(
        int(record["attempts"])
        for record in records
    )

    total_retries = sum(
        int(record["retries"])
        for record in records
    )

    repaired = sum(
        bool(record["repaired"])
        for record in records
    )

    flagged = sum(
        bool(record["flagged"])
        for record in records
    )

    cost_before = sum(
        float(record["baseline_cost"])
        for record in records
    )

    cost_after = sum(
        float(record["total_cost"])
        for record in records
    )

    valid_rate_before = (
        valid_before / total
    )

    valid_rate_after = (
        valid_after / total
    )

    return {
        "total_prompts": total,
        "valid_before": valid_before,
        "valid_after": valid_after,
        "valid_rate_before": valid_rate_before,
        "valid_rate_after": valid_rate_after,
        "improvement": (
            valid_rate_after
            - valid_rate_before
        ),
        "average_attempts": (
            total_attempts / total
        ),
        "average_retries": (
            total_retries / total
        ),
        "repair_rate": (
            repaired / total
        ),
        "flagged": flagged,
        "estimated_cost_before": cost_before,
        "estimated_cost_after": cost_after,
        "extra_cost": (
            cost_after - cost_before
        ),
    }


def print_summary(
    model_name: str,
    summary: dict[str, Any],
) -> None:
    """
    Print current benchmark metrics.
    """

    print()
    print(
        f"  {model_name}"
    )
    print(
        f"  Valid before : "
        f"{summary['valid_before']}/"
        f"{summary['total_prompts']}"
    )
    print(
        f"  Valid after  : "
        f"{summary['valid_after']}/"
        f"{summary['total_prompts']}"
    )
    print(
        f"  Before rate  : "
        f"{summary['valid_rate_before']:.2%}"
    )
    print(
        f"  After rate   : "
        f"{summary['valid_rate_after']:.2%}"
    )
    print(
        f"  Improvement  : "
        f"{summary['improvement']:+.2%}"
    )
    print(
        f"  Avg attempts : "
        f"{summary['average_attempts']:.2f}"
    )
    print(
        f"  Avg retries  : "
        f"{summary['average_retries']:.2f}"
    )
    print(
        f"  Repair rate  : "
        f"{summary['repair_rate']:.2%}"
    )
    print(
        f"  Flagged      : "
        f"{summary['flagged']}"
    )
    print(
        f"  Cost before  : "
        f"${summary['estimated_cost_before']:.6f}"
    )
    print(
        f"  Cost after   : "
        f"${summary['estimated_cost_after']:.6f}"
    )
    print(
        f"  Extra cost   : "
        f"${summary['extra_cost']:.6f}"
    )


def main() -> None:
    print("=" * 70)
    print(
        "FORMAT GUARD — FULL GROQ BENCHMARK"
    )
    print("=" * 70)
    print()

    dataset = load_dataset()

    print(
        f"Dataset prompts : {len(dataset)}"
    )
    print(
        f"Models          : 3"
    )
    print(
        f"Max retries     : {MAX_RETRIES}"
    )
    print(
        f"Call delay      : "
        f"{CALL_DELAY_SECONDS:.1f}s"
    )
    print(
        f"Checkpoint      : {RESULTS_PATH}"
    )
    print()

    checkpoint = load_checkpoint()

    providers = build_groq_benchmark_providers()

    print("Configured models:")

    for model_name, provider in providers.items():
        print(
            f"  - {model_name} "
            f"({provider.model})"
        )

    print()

    for model_name, provider in providers.items():
        print("=" * 70)
        print(
            f"MODEL: {model_name}"
        )
        print("=" * 70)

        model_state = checkpoint["models"].setdefault(
            model_name,
            {
                "model_id": provider.model,
                "records": {},
            },
        )

        records = model_state.setdefault(
            "records",
            {},
        )

        completed = len(records)

        print(
            f"Completed: "
            f"{completed}/{TOTAL_PROMPTS}"
        )

        for index, item in enumerate(
            dataset,
            start=1,
        ):
            prompt_id = str(
                item["id"]
            )

            if prompt_id in records:
                continue

            prompt = item["prompt"]

            print(
                f"[{index:03d}/{TOTAL_PROMPTS}] "
                f"{prompt_id}"
            )

            try:
                result = run_prompt(
                    provider=provider,
                    prompt=prompt,
                )

                records[prompt_id] = {
                    "id": prompt_id,
                    "category": item.get(
                        "category"
                    ),
                    **result,
                }

                save_checkpoint(
                    checkpoint
                )

                completed += 1

                print(
                    f"      before="
                    f"{'VALID' if result['valid_before'] else 'INVALID'} "
                    f"| after="
                    f"{'VALID' if result['valid_after'] else 'INVALID'} "
                    f"| attempts="
                    f"{result['attempts']} "
                    f"| retries="
                    f"{result['retries']} "
                    f"| extra="
                    f"${result['repair_cost']:.6f}"
                )

            except KeyboardInterrupt:
                save_checkpoint(
                    checkpoint
                )

                print()
                print(
                    "Benchmark interrupted."
                )
                print(
                    "Checkpoint saved."
                )
                print(
                    "Run the same command again "
                    "to resume."
                )
                return

            except Exception as exc:
                save_checkpoint(
                    checkpoint
                )

                print(
                    f"      ERROR: {exc}"
                )
                print(
                    "      Checkpoint saved."
                )
                print(
                    "      Stopping safely so the "
                    "benchmark can be resumed."
                )
                return

        summary = model_summary(
            list(records.values())
        )

        model_state["summary"] = summary

        save_checkpoint(
            checkpoint
        )

        print_summary(
            model_name,
            summary,
        )

    checkpoint["completed"] = True

    checkpoint["overall"] = {
        model_name: model_state.get(
            "summary",
            {},
        )
        for model_name, model_state
        in checkpoint["models"].items()
    }

    save_checkpoint(
        checkpoint
    )

    print()
    print("=" * 70)
    print(
        "FULL GROQ BENCHMARK COMPLETED"
    )
    print("=" * 70)
    print()
    print(
        f"Results saved to:"
    )
    print(
        f"  {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
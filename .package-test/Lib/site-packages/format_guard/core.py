from __future__ import annotations

import json
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .cleaner import parse_json
from .models import GuardMetrics, ValidationResult
from .repair import RepairEngine


T = TypeVar("T", bound=BaseModel)


class FormatGuard:
    """
    Core validation, repair, retry, safe-failure,
    and observability engine for LLM outputs.
    """

    def __init__(
        self,
        max_retries: int = 3,
        fallback: T | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.max_retries = max_retries
        self.fallback = fallback
        self.repair_engine = RepairEngine()

    def validate(
        self,
        schema: type[T],
        llm_fn: Callable[[str], str],
        prompt: str,
    ) -> ValidationResult:
        """
        Generate, validate, repair, retry, and safely fail
        an LLM response while recording execution metrics.
        """

        current_prompt = prompt

        first_output: str | None = None
        last_output: str | None = None
        last_error: str | None = None

        metrics = GuardMetrics()

        for attempt in range(self.max_retries + 1):
            output = llm_fn(current_prompt)

            metrics.attempts = attempt + 1

            if first_output is None:
                first_output = output

            last_output = output

            try:
                parsed = parse_json(output)

                validated = schema.model_validate(parsed)

                metrics.successful = True

                return ValidationResult(
                    success=True,
                    value=validated,
                    attempts=metrics.attempts,
                    repaired=metrics.repairs > 0,
                    raw_output=first_output,
                    final_output=output,
                    flagged=False,
                    fallback_used=False,
                    metrics=metrics,
                )

            except (json.JSONDecodeError, ValidationError) as exc:
                metrics.validation_failures += 1
                last_error = str(exc)

                if attempt >= self.max_retries:
                    break

                metrics.repairs += 1

                current_prompt = self.repair_engine.build_prompt(
                    original_prompt=prompt,
                    invalid_output=output,
                    error=last_error,
                    schema=schema,
                )

        return ValidationResult(
            success=False,
            value=self.fallback,
            attempts=metrics.attempts,
            repaired=metrics.repairs > 0,
            error=last_error,
            raw_output=first_output,
            final_output=last_output,
            flagged=True,
            fallback_used=self.fallback is not None,
            metrics=metrics,
        )


def guard(
    schema: type[T],
    llm_fn: Callable[[str], str],
    prompt: str,
    max_retries: int = 3,
    fallback: T | None = None,
) -> ValidationResult:
    """
    Convenience function for Format Guard.
    """

    formatter = FormatGuard(
        max_retries=max_retries,
        fallback=fallback,
    )

    return formatter.validate(
        schema=schema,
        llm_fn=llm_fn,
        prompt=prompt,
    )
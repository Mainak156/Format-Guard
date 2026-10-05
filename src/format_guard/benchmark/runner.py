from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, ValidationError

from ..cleaner import parse_json
from ..core import guard
from .models import BenchmarkResult, ModelBenchmark


LLMFunction = Callable[[str], str]


def _get_provider_cost(
    provider: Any,
) -> float:
    """
    Return accumulated provider cost when available.

    Ordinary callables remain fully supported and simply
    report zero cost.
    """

    value = getattr(
        provider,
        "total_cost",
        0.0,
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class BenchmarkRunner:
    """
    Compare raw LLM output against the same LLM protected
    by Format Guard.

    The initial LLM response is captured once and then replayed
    into Format Guard so that the "before" and "after" results
    are based on the same initial model output.

    Cost-aware providers can expose a cumulative `total_cost`
    property. Ordinary callables remain supported.
    """

    def __init__(
        self,
        schema: type[BaseModel],
        max_retries: int = 3,
    ) -> None:
        if max_retries < 0:
            raise ValueError(
                "max_retries must be >= 0"
            )

        self.schema = schema
        self.max_retries = max_retries

    def _is_valid(
        self,
        output: str,
    ) -> bool:
        """
        Check whether raw LLM output already satisfies
        the benchmark schema.
        """

        try:
            parsed = parse_json(output)
            self.schema.model_validate(parsed)
            return True

        except (
            json.JSONDecodeError,
            ValidationError,
        ):
            return False

    def run_model(
        self,
        model_name: str,
        llm_fn: LLMFunction,
        prompts: Iterable[str],
    ) -> ModelBenchmark:
        """
        Run a benchmark for one model/provider.
        """

        result = ModelBenchmark(
            model_name=model_name,
        )

        prompt_list = list(prompts)
        result.total_prompts = len(prompt_list)

        for prompt in prompt_list:
            # -------------------------------------------------
            # BEFORE
            # -------------------------------------------------

            cost_before_call = (
                _get_provider_cost(llm_fn)
            )

            raw_output = llm_fn(prompt)

            cost_after_baseline = (
                _get_provider_cost(llm_fn)
            )

            result.estimated_cost_before += (
                cost_after_baseline
                - cost_before_call
            )

            is_valid_before = self._is_valid(
                raw_output
            )

            if is_valid_before:
                result.valid_before += 1
            else:
                result.invalid_before += 1

            # -------------------------------------------------
            # AFTER
            # -------------------------------------------------

            first_call = True

            def guarded_llm(
                repair_prompt: str,
            ) -> str:
                nonlocal first_call

                if first_call:
                    first_call = False
                    return raw_output

                return llm_fn(repair_prompt)

            cost_before_guard = (
                _get_provider_cost(llm_fn)
            )

            guarded = guard(
                schema=self.schema,
                llm_fn=guarded_llm,
                prompt=prompt,
                max_retries=self.max_retries,
            )

            cost_after_guard = (
                _get_provider_cost(llm_fn)
            )

            result.estimated_cost_after += (
                cost_after_guard
                - cost_before_guard
            )

            # The baseline response is already included
            # in estimated_cost_before. The guarded total
            # cost for this prompt therefore consists of
            # baseline + any repair calls.
            result.estimated_cost_after += (
                cost_after_baseline
                - cost_before_call
            )

            result.total_attempts += (
                guarded.attempts
            )

            if guarded.metrics is not None:
                result.total_repairs += (
                    guarded.metrics.repairs
                )

            if guarded.success:
                result.valid_after += 1

            if guarded.repaired:
                result.repaired += 1

            if guarded.flagged:
                result.flagged += 1

        return result

    def run(
        self,
        models: dict[str, LLMFunction],
        prompts: Iterable[str],
    ) -> BenchmarkResult:
        """
        Run the benchmark across multiple models/providers.
        """

        prompt_list = list(prompts)

        benchmark = BenchmarkResult(
            prompts=len(prompt_list),
        )

        for model_name, llm_fn in models.items():
            result = self.run_model(
                model_name=model_name,
                llm_fn=llm_fn,
                prompts=prompt_list,
            )

            benchmark.models.append(result)

        return benchmark
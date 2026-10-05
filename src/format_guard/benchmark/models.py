from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelBenchmark:
    """
    Benchmark results for one LLM/provider.
    """

    model_name: str

    total_prompts: int = 0

    valid_before: int = 0
    valid_after: int = 0

    invalid_before: int = 0
    repaired: int = 0
    flagged: int = 0

    total_attempts: int = 0
    total_repairs: int = 0

    estimated_cost_before: float = 0.0
    estimated_cost_after: float = 0.0

    @property
    def valid_rate_before(self) -> float:
        if self.total_prompts == 0:
            return 0.0

        return self.valid_before / self.total_prompts

    @property
    def valid_rate_after(self) -> float:
        if self.total_prompts == 0:
            return 0.0

        return self.valid_after / self.total_prompts

    @property
    def improvement(self) -> float:
        return (
            self.valid_rate_after
            - self.valid_rate_before
        )

    @property
    def average_attempts(self) -> float:
        if self.total_prompts == 0:
            return 0.0

        return self.total_attempts / self.total_prompts

    @property
    def average_retries(self) -> float:
        if self.total_prompts == 0:
            return 0.0

        return self.total_repairs / self.total_prompts

    @property
    def repair_rate(self) -> float:
        if self.total_prompts == 0:
            return 0.0

        return self.repaired / self.total_prompts

    @property
    def extra_cost(self) -> float:
        return (
            self.estimated_cost_after
            - self.estimated_cost_before
        )


@dataclass
class BenchmarkResult:
    """
    Complete benchmark result across multiple models.
    """

    prompts: int
    models: list[ModelBenchmark] = field(
        default_factory=list
    )

    @property
    def model_count(self) -> int:
        return len(self.models)

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "model": result.model_name,
                "valid_rate_before": result.valid_rate_before,
                "valid_rate_after": result.valid_rate_after,
                "improvement": result.improvement,
                "average_attempts": result.average_attempts,
                "average_retries": result.average_retries,
                "repair_rate": result.repair_rate,
                "extra_cost": result.extra_cost,
            }
            for result in self.models
        ]
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GuardMetrics:
    """
    Observability metrics for a single Format Guard execution.
    """

    attempts: int = 0
    repairs: int = 0
    validation_failures: int = 0
    successful: bool = False

    @property
    def repair_rate(self) -> float:
        """
        Calculate the proportion of attempts that required repair.
        """

        if self.attempts == 0:
            return 0.0

        return self.repairs / self.attempts


@dataclass
class ValidationResult:
    """
    Result returned by Format Guard after validation
    and repair attempts.
    """

    success: bool
    value: Any | None = None

    attempts: int = 0
    repaired: bool = False

    error: str | None = None

    raw_output: str | None = None
    final_output: str | None = None

    flagged: bool = False
    fallback_used: bool = False

    metrics: GuardMetrics | None = None
from pydantic import BaseModel

from format_guard.core import guard
from format_guard.providers import MockProvider


class Customer(BaseModel):
    name: str
    age: int
    email: str


def test_metrics_for_first_attempt_success() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
    )

    assert result.metrics is not None
    assert result.metrics.attempts == 1
    assert result.metrics.repairs == 0
    assert result.metrics.validation_failures == 0
    assert result.metrics.successful is True
    assert result.metrics.repair_rate == 0.0


def test_metrics_track_success_after_repair() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak"}',
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
        max_retries=1,
    )

    assert result.metrics is not None
    assert result.metrics.attempts == 2
    assert result.metrics.repairs == 1
    assert result.metrics.validation_failures == 1
    assert result.metrics.successful is True
    assert result.metrics.repair_rate == 0.5


def test_metrics_track_multiple_repairs() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak"}',
            '{"name": "Mainak"}',
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
        max_retries=2,
    )

    assert result.metrics is not None
    assert result.metrics.attempts == 3
    assert result.metrics.repairs == 2
    assert result.metrics.validation_failures == 2
    assert result.metrics.successful is True
    assert result.metrics.repair_rate == 2 / 3


def test_metrics_track_exhausted_failure() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak"}',
            '{"name": "Mainak"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
        max_retries=1,
    )

    assert result.success is False
    assert result.metrics is not None

    assert result.metrics.attempts == 2
    assert result.metrics.repairs == 1
    assert result.metrics.validation_failures == 2
    assert result.metrics.successful is False
    assert result.metrics.repair_rate == 0.5


def test_metrics_are_attached_to_result() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
    )

    assert result.metrics is not None
    assert result.metrics is not None
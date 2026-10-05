import pytest
from pydantic import BaseModel

from format_guard.core import FormatGuard, guard
from format_guard.providers import MockProvider


class Customer(BaseModel):
    name: str
    age: int
    email: str


def test_valid_response_stops_immediately() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
            '{"name": "Should", "age": 99, "email": "not-used@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
        max_retries=3,
    )

    assert result.success is True
    assert result.attempts == 1
    assert len(provider.calls) == 1


def test_one_retry_allows_second_attempt() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak", "age": "invalid", "email": "mainak@example.com"}',
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
        max_retries=1,
    )

    assert result.success is True
    assert result.attempts == 2
    assert len(provider.calls) == 2


def test_retry_count_is_bounded() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak"}',
            '{"name": "Mainak"}',
            '{"name": "Mainak"}',
            '{"name": "Mainak"}',
            '{"name": "Mainak"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
        max_retries=2,
    )

    assert result.success is False
    assert result.attempts == 3
    assert len(provider.calls) == 3


def test_three_retries_allow_four_total_attempts() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak"}',
            '{"name": "Mainak"}',
            '{"name": "Mainak"}',
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
        max_retries=3,
    )

    assert result.success is True
    assert result.attempts == 4
    assert len(provider.calls) == 4


def test_success_after_retry_stops_further_calls() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak"}',
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
            '{"name": "Unused", "age": 99, "email": "unused@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
        max_retries=3,
    )

    assert result.success is True
    assert result.attempts == 2
    assert len(provider.calls) == 2


def test_negative_retry_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        FormatGuard(max_retries=-1)


def test_zero_retries_means_one_total_attempt() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak"}',
            '{"name": "Should not be called"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
        max_retries=0,
    )

    assert result.success is False
    assert result.attempts == 1
    assert len(provider.calls) == 1
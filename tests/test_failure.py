from pydantic import BaseModel

from format_guard.core import FormatGuard, guard
from format_guard.providers import MockProvider


class Customer(BaseModel):
    name: str
    age: int
    email: str


def test_exhausted_retries_are_flagged() -> None:
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
    assert result.flagged is True
    assert result.fallback_used is False
    assert result.value is None
    assert result.error is not None


def test_safe_fallback_is_returned_after_failure() -> None:
    fallback = Customer(
        name="Unknown",
        age=0,
        email="unknown@example.com",
    )

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
        fallback=fallback,
    )

    assert result.success is False
    assert result.flagged is True
    assert result.fallback_used is True
    assert result.value == fallback


def test_fallback_is_not_used_when_validation_succeeds() -> None:
    fallback = Customer(
        name="Unknown",
        age=0,
        email="unknown@example.com",
    )

    provider = MockProvider(
        [
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
        max_retries=3,
        fallback=fallback,
    )

    assert result.success is True
    assert result.flagged is False
    assert result.fallback_used is False

    assert result.value.name == "Mainak"
    assert result.value.age == 21


def test_no_retry_failure_still_flags_output() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
        max_retries=0,
    )

    assert result.success is False
    assert result.flagged is True
    assert result.attempts == 1


def test_format_guard_accepts_fallback() -> None:
    fallback = Customer(
        name="Fallback",
        age=0,
        email="fallback@example.com",
    )

    formatter = FormatGuard(
        max_retries=0,
        fallback=fallback,
    )

    provider = MockProvider(
        [
            '{"name": "Invalid"}',
        ]
    )

    result = formatter.validate(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract customer information.",
    )

    assert result.success is False
    assert result.flagged is True
    assert result.fallback_used is True
    assert result.value == fallback
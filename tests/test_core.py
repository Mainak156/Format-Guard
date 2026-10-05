import pytest
from pydantic import BaseModel, ConfigDict

from format_guard.core import FormatGuard, guard
from format_guard.providers import MockProvider


class Customer(BaseModel):
    name: str
    age: int
    email: str


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    price: float


def test_valid_output_returns_pydantic_object() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}'
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract the customer information.",
    )

    assert result.success is True
    assert isinstance(result.value, Customer)

    assert result.value.name == "Mainak"
    assert result.value.age == 21
    assert result.value.email == "mainak@example.com"

    assert result.attempts == 1
    assert result.repaired is False


def test_markdown_json_is_accepted() -> None:
    provider = MockProvider(
        [
            """```json
            {
                "name": "Mainak",
                "age": 21,
                "email": "mainak@example.com"
            }
            ```"""
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract the customer information.",
    )

    assert result.success is True
    assert result.value.name == "Mainak"
    assert result.attempts == 1


def test_trailing_comma_is_cleaned() -> None:
    provider = MockProvider(
        [
            """{
                "name": "Mainak",
                "age": 21,
                "email": "mainak@example.com",
            }"""
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract the customer information.",
    )

    assert result.success is True
    assert result.value.name == "Mainak"


def test_missing_required_field_triggers_repair() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak", "age": 21}',
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract the customer information.",
        max_retries=1,
    )

    assert result.success is True
    assert result.value.email == "mainak@example.com"
    assert result.attempts == 2
    assert result.repaired is True


def test_wrong_type_triggers_repair() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak", "age": "twenty-one", "email": "mainak@example.com"}',
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract the customer information.",
        max_retries=1,
    )

    assert result.success is True
    assert result.value.age == 21
    assert result.attempts == 2
    assert result.repaired is True


def test_invalid_json_triggers_repair() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak", "age": 21,',
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract the customer information.",
        max_retries=1,
    )

    assert result.success is True
    assert result.attempts == 2
    assert result.repaired is True


def test_extra_field_is_rejected_when_schema_forbids_it() -> None:
    provider = MockProvider(
        [
            '{"name": "Laptop", "price": 1000, "brand": "Example"}',
            '{"name": "Laptop", "price": 1000}',
        ]
    )

    result = guard(
        schema=Product,
        llm_fn=provider.generate,
        prompt="Extract the product information.",
        max_retries=1,
    )

    assert result.success is True
    assert result.value.name == "Laptop"
    assert result.value.price == 1000
    assert result.attempts == 2


def test_zero_retries_does_not_retry() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak"}',
        ]
    )

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract the customer information.",
        max_retries=0,
    )

    assert result.success is False
    assert result.attempts == 1
    assert len(provider.calls) == 1


def test_format_guard_rejects_negative_retries() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        FormatGuard(max_retries=-1)


def test_core_uses_dedicated_repair_engine() -> None:
    provider = MockProvider(
        [
            '{"name": "Mainak", "age": "twenty-one", "email": "mainak@example.com"}',
            '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}',
        ]
    )

    formatter = FormatGuard(max_retries=1)

    result = formatter.validate(
        schema=Customer,
        llm_fn=provider.generate,
        prompt="Extract the customer information.",
    )

    assert result.success is True
    assert result.attempts == 2
    assert result.repaired is True

    repair_prompt = provider.calls[1]

    assert "Validation error:" in repair_prompt
    assert "Previous response:" in repair_prompt
    assert "Required JSON schema:" in repair_prompt
    assert "Return ONLY valid JSON." in repair_prompt
from pydantic import BaseModel

from format_guard import (
    FormatGuard,
    GuardMetrics,
    ValidationResult,
    __version__,
    guard,
)
from format_guard.providers import MockProvider
from fastapi.testclient import TestClient
from format_guard.api import app, build_dynamic_schema


class Customer(BaseModel):
    name: str
    age: int
    email: str


def test_public_guard_is_importable() -> None:
    assert callable(guard)


def test_public_format_guard_is_importable() -> None:
    formatter = FormatGuard()

    assert isinstance(formatter, FormatGuard)


def test_public_result_type_is_returned() -> None:
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

    assert isinstance(result, ValidationResult)
    assert result.success is True


def test_public_api_exposes_metrics() -> None:
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

    assert isinstance(result.metrics, GuardMetrics)
    assert result.metrics.attempts == 2
    assert result.metrics.repairs == 1
    assert result.metrics.successful is True


def test_public_api_supports_fallback() -> None:
    fallback = Customer(
        name="Unknown",
        age=0,
        email="unknown@example.com",
    )

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
        fallback=fallback,
    )

    assert result.success is False
    assert result.flagged is True
    assert result.fallback_used is True
    assert result.value == fallback


def test_public_api_preserves_raw_and_final_output() -> None:
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

    assert result.raw_output == '{"name": "Mainak"}'

    assert (
        result.final_output
        == '{"name": "Mainak", "age": 21, "email": "mainak@example.com"}'
    )


def test_package_version_is_exposed() -> None:
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "Format Guard API"


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_dynamic_schema_builder() -> None:
    CustomerSchema = build_dynamic_schema(
        {
            "name": "string",
            "age": "integer",
            "score": "number",
            "active": "boolean",
        }
    )

    customer = CustomerSchema(
        name="Mainak",
        age=21,
        score=9.5,
        active=True,
    )

    assert customer.name == "Mainak"
    assert customer.age == 21
    assert customer.score == 9.5
    assert customer.active is True


def test_dynamic_schema_rejects_unknown_type() -> None:
    try:
        build_dynamic_schema(
            {
                "name": "unknown",
            }
        )
    except ValueError as exc:
        assert "Unsupported field type" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for unsupported field type."
        )


def test_dynamic_schema_rejects_empty_schema() -> None:
    try:
        build_dynamic_schema({})
    except ValueError as exc:
        assert "at least one field" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for empty schema."
        )


def test_validate_endpoint_success(monkeypatch) -> None:
    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

        def generate(self, prompt: str) -> str:
            return (
                '{"name": "Mainak", '
                '"age": 21, '
                '"email": "mainak@example.com"}'
            )

    monkeypatch.setattr(
        "format_guard.api.GroqProvider",
        FakeProvider,
    )

    response = client.post(
        "/validate",
        json={
            "prompt": "Return customer information.",
            "schema": {
                "name": "string",
                "age": "integer",
                "email": "string",
            },
            "max_retries": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["value"] == {
        "name": "Mainak",
        "age": 21,
        "email": "mainak@example.com",
    }
    assert data["attempts"] == 1
    assert data["repaired"] is False
    assert data["flagged"] is False
    assert data["metrics"]["successful"] is True


def test_validate_endpoint_rejects_invalid_schema() -> None:
    response = client.post(
        "/validate",
        json={
            "prompt": "Return information.",
            "schema": {
                "age": "invalid-type",
            },
        },
    )

    assert response.status_code == 400
    assert "Unsupported field type" in response.json()["detail"]
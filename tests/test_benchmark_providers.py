from __future__ import annotations

from types import SimpleNamespace

import pytest

from format_guard.benchmark import (
    GROQ_MODEL_CATALOG,
    BenchmarkUsage,
    GroqBenchmarkProvider,
    ModelPricing,
)


def test_model_catalog_contains_public_groq_models() -> None:
    assert "gpt-oss-120b" in GROQ_MODEL_CATALOG
    assert "gpt-oss-20b" in GROQ_MODEL_CATALOG
    assert "qwen3.8-27b" in GROQ_MODEL_CATALOG


def test_model_pricing_values() -> None:
    pricing = ModelPricing(
        input_per_million=0.15,
        output_per_million=0.60,
    )

    assert pricing.input_per_million == 0.15
    assert pricing.output_per_million == 0.60


def test_benchmark_usage_total_tokens() -> None:
    usage = BenchmarkUsage(
        prompt_tokens=100,
        completion_tokens=50,
    )

    assert usage.total_tokens == 150


def test_provider_requires_api_key() -> None:
    with pytest.raises(ValueError):
        GroqBenchmarkProvider(
            model="openai/gpt-oss-120b",
            api_key="",
        )


def test_provider_rejects_unknown_pricing_model() -> None:
    with pytest.raises(ValueError):
        GroqBenchmarkProvider(
            model="unknown/model",
            api_key="test-key",
        )


def test_provider_tracks_usage_and_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GroqBenchmarkProvider(
        model="openai/gpt-oss-120b",
        api_key="test-key",
    )

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"ok": true}'
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=1_000,
            completion_tokens=500,
        ),
    )

    def fake_create(**kwargs):
        assert kwargs["model"] == (
            "openai/gpt-oss-120b"
        )

        return fake_response

    monkeypatch.setattr(
        provider.client.chat.completions,
        "create",
        fake_create,
    )

    output = provider.generate(
        "Return a JSON object."
    )

    assert output == '{"ok": true}'

    assert provider.usage.prompt_tokens == 1_000
    assert provider.usage.completion_tokens == 500
    assert provider.usage.total_tokens == 1_500

    expected_cost = (
        (1_000 / 1_000_000) * 0.15
        + (500 / 1_000_000) * 0.60
    )

    assert provider.total_cost == pytest.approx(
        expected_cost
    )


def test_provider_is_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GroqBenchmarkProvider(
        model="openai/gpt-oss-20b",
        api_key="test-key",
    )

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"ok": true}'
                )
            )
        ],
        usage=None,
    )

    monkeypatch.setattr(
        provider.client.chat.completions,
        "create",
        lambda **kwargs: fake_response,
    )

    assert provider(
        "test"
    ) == '{"ok": true}'
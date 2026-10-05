import pytest

from format_guard.providers.groq import GroqProvider


def test_groq_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Groq API key"):
        GroqProvider()


def test_groq_provider_accepts_explicit_api_key() -> None:
    provider = GroqProvider(
        api_key="test-key",
        model="test-model",
    )

    assert provider.api_key == "test-key"
    assert provider.model == "test-model"
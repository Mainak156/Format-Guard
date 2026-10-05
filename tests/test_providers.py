import pytest

from format_guard.providers import LLMProvider, MockProvider


def test_mock_provider_returns_response() -> None:
    provider = MockProvider(
        ['{"name": "Mainak"}']
    )

    result = provider.generate("Give me the name.")

    assert result == '{"name": "Mainak"}'


def test_mock_provider_records_prompts() -> None:
    provider = MockProvider(
        ['{"name": "Mainak"}']
    )

    prompt = "Give me the name."

    provider.generate(prompt)

    assert provider.calls == [prompt]


def test_mock_provider_returns_responses_in_order() -> None:
    provider = MockProvider(
        [
            '{"name": "First"}',
            '{"name": "Second"}',
        ]
    )

    first = provider.generate("First request")
    second = provider.generate("Second request")

    assert first == '{"name": "First"}'
    assert second == '{"name": "Second"}'


def test_mock_provider_raises_when_exhausted() -> None:
    provider = MockProvider(
        ['{"name": "Mainak"}']
    )

    provider.generate("First request")

    with pytest.raises(RuntimeError, match="no responses remaining"):
        provider.generate("Second request")


def test_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        LLMProvider()
from pydantic import BaseModel
import pytest

from format_guard.providers.langchain_groq import (
    LangChainGroqProvider,
)


class Customer(BaseModel):
    name: str
    age: int
    email: str


def test_langchain_groq_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Groq API key"):
        LangChainGroqProvider()


def test_langchain_groq_provider_configuration() -> None:
    provider = LangChainGroqProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
    )

    assert provider.api_key == "test-key"
    assert provider.model == "test-model"
    assert provider.temperature == 0.2
    assert provider.llm is not None


def test_generate_structured_uses_pydantic_schema() -> None:
    provider = LangChainGroqProvider(
        api_key="test-key",
        model="test-model",
    )

    expected = Customer(
        name="Mainak",
        age=21,
        email="mainak@example.com",
    )

    class FakeStructuredLLM:
        def invoke(self, prompt: str) -> Customer:
            assert "customer" in prompt.lower()
            return expected

    class FakeLLM:
        def with_structured_output(
            self,
            schema: type[BaseModel],
        ) -> FakeStructuredLLM:
            assert schema is Customer
            return FakeStructuredLLM()

    provider.llm = FakeLLM()

    result = provider.generate_structured(
        prompt="Return customer information.",
        schema=Customer,
    )

    assert result == expected
    assert isinstance(result, Customer)


def test_generate_structured_rejects_wrong_result_type() -> None:
    provider = LangChainGroqProvider(
        api_key="test-key",
        model="test-model",
    )

    class FakeStructuredLLM:
        def invoke(self, prompt: str) -> dict:
            return {
                "name": "Mainak",
                "age": 21,
                "email": "mainak@example.com",
            }

    class FakeLLM:
        def with_structured_output(
            self,
            schema: type[BaseModel],
        ) -> FakeStructuredLLM:
            return FakeStructuredLLM()

    provider.llm = FakeLLM()

    with pytest.raises(
        RuntimeError,
        match="unexpected structured-output type",
    ):
        provider.generate_structured(
            prompt="Return customer information.",
            schema=Customer,
        )
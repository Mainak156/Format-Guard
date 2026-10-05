from pydantic import BaseModel
import pytest

from format_guard.benchmark import BenchmarkRunner


class Customer(BaseModel):
    name: str
    age: int
    email: str


def test_benchmark_detects_invalid_output() -> None:
    responses = iter(
        [
            '{"name": "Mainak", "age": "bad", '
            '"email": "mainak@example.com"}',
            '{"name": "Mainak", "age": 21, '
            '"email": "mainak@example.com"}',
        ]
    )

    def provider(prompt: str) -> str:
        return next(responses)

    runner = BenchmarkRunner(
        schema=Customer,
        max_retries=0,
    )

    result = runner.run_model(
        model_name="test-model",
        llm_fn=provider,
        prompts=["Extract customer information."],
    )

    assert result.total_prompts == 1
    assert result.valid_before == 0
    assert result.invalid_before == 1
    assert result.valid_after == 0


def test_benchmark_measures_repair() -> None:
    responses = iter(
        [
            '{"name": "Mainak", "age": "bad", '
            '"email": "mainak@example.com"}',
            '{"name": "Mainak", "age": 21, '
            '"email": "mainak@example.com"}',
        ]
    )

    def provider(prompt: str) -> str:
        return next(responses)

    runner = BenchmarkRunner(
        schema=Customer,
        max_retries=1,
    )

    result = runner.run_model(
        model_name="test-model",
        llm_fn=provider,
        prompts=["Extract customer information."],
    )

    assert result.valid_before == 0
    assert result.invalid_before == 1
    assert result.valid_after == 1
    assert result.repaired == 1
    assert result.total_attempts == 2
    assert result.total_repairs == 1
    assert result.valid_rate_before == 0.0
    assert result.valid_rate_after == 1.0
    assert result.improvement == 1.0


def test_benchmark_supports_multiple_models() -> None:
    valid = (
        '{"name": "Mainak", "age": 21, '
        '"email": "mainak@example.com"}'
    )

    def model_a(prompt: str) -> str:
        return valid

    def model_b(prompt: str) -> str:
        return valid

    runner = BenchmarkRunner(
        schema=Customer,
    )

    result = runner.run(
        models={
            "model-a": model_a,
            "model-b": model_b,
        },
        prompts=[
            "Prompt 1",
            "Prompt 2",
        ],
    )

    assert result.prompts == 2
    assert result.model_count == 2
    assert len(result.models) == 2


def test_benchmark_summary() -> None:
    valid = (
        '{"name": "Mainak", "age": 21, '
        '"email": "mainak@example.com"}'
    )

    def provider(prompt: str) -> str:
        return valid

    runner = BenchmarkRunner(
        schema=Customer,
    )

    result = runner.run(
        models={"test-model": provider},
        prompts=["Prompt 1"],
    )

    summary = result.summary()

    assert len(summary) == 1
    assert summary[0]["model"] == "test-model"
    assert summary[0]["valid_rate_before"] == 1.0
    assert summary[0]["valid_rate_after"] == 1.0


def test_benchmark_tracks_provider_cost() -> None:
    class CostAwareProvider:
        def __init__(self) -> None:
            self.total_cost = 0.0
            self.calls = 0

        def __call__(self, prompt: str) -> str:
            self.calls += 1

            if self.calls == 1:
                self.total_cost += 0.01

                return (
                    '{"name": "Mainak", '
                    '"age": "bad", '
                    '"email": "mainak@example.com"}'
                )

            self.total_cost += 0.02

            return (
                '{"name": "Mainak", '
                '"age": 21, '
                '"email": "mainak@example.com"}'
            )

    provider = CostAwareProvider()

    runner = BenchmarkRunner(
        schema=Customer,
        max_retries=1,
    )

    result = runner.run_model(
        model_name="cost-model",
        llm_fn=provider,
        prompts=[
            "Extract customer information."
        ],
    )

    assert result.valid_before == 0
    assert result.valid_after == 1

    assert result.estimated_cost_before == 0.01
    assert result.estimated_cost_after == 0.03
    assert result.extra_cost == pytest.approx(0.02)
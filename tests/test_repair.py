from pydantic import BaseModel, ConfigDict

from format_guard.repair import RepairEngine, build_repair_prompt


class Customer(BaseModel):
    name: str
    age: int
    email: str


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    price: float


def test_build_repair_prompt_contains_original_request() -> None:
    prompt = build_repair_prompt(
        original_prompt="Extract customer information.",
        invalid_output='{"name": "Mainak"}',
        error="Field required: email",
        schema=Customer,
    )

    assert "Extract customer information." in prompt


def test_build_repair_prompt_contains_invalid_output() -> None:
    invalid_output = '{"name": "Mainak"}'

    prompt = build_repair_prompt(
        original_prompt="Extract customer information.",
        invalid_output=invalid_output,
        error="Field required: email",
        schema=Customer,
    )

    assert invalid_output in prompt


def test_build_repair_prompt_contains_validation_error() -> None:
    error = "Field required: email"

    prompt = build_repair_prompt(
        original_prompt="Extract customer information.",
        invalid_output='{"name": "Mainak"}',
        error=error,
        schema=Customer,
    )

    assert error in prompt


def test_build_repair_prompt_contains_json_schema() -> None:
    prompt = build_repair_prompt(
        original_prompt="Extract customer information.",
        invalid_output='{"name": "Mainak"}',
        error="Field required: email",
        schema=Customer,
    )

    assert '"name"' in prompt
    assert '"age"' in prompt
    assert '"email"' in prompt


def test_build_repair_prompt_requires_json_only() -> None:
    prompt = build_repair_prompt(
        original_prompt="Extract customer information.",
        invalid_output='{"name": "Mainak"}',
        error="Field required: email",
        schema=Customer,
    )

    assert "Return ONLY valid JSON." in prompt
    assert "Do not use Markdown code fences." in prompt


def test_build_repair_prompt_forbids_extra_fields() -> None:
    prompt = build_repair_prompt(
        original_prompt="Extract product information.",
        invalid_output='{"name": "Laptop", "price": 1000, "brand": "Example"}',
        error="Extra inputs are not permitted",
        schema=Product,
    )

    assert "Do not add extra fields." in prompt


def test_repair_engine_matches_function() -> None:
    engine = RepairEngine()

    kwargs = {
        "original_prompt": "Extract customer information.",
        "invalid_output": '{"name": "Mainak"}',
        "error": "Field required: email",
        "schema": Customer,
    }

    assert engine.build_prompt(**kwargs) == build_repair_prompt(**kwargs)
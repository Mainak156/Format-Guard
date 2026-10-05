from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from .core import guard
from .providers import GroqProvider


app = FastAPI(
    title="Format Guard API",
    description=(
        "Validate, repair, retry, and safely handle "
        "LLM-generated structured output."
    ),
    version="0.1.0",
)


class ValidateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)

    schema_definition: dict[str, str] = Field(
        ...,
        alias="schema",
        description=(
            "Schema fields mapped to supported primitive types: "
            "string, integer, number, boolean."
        ),
    )

    max_retries: int = Field(default=3, ge=0, le=10)

    model: str = "openai/gpt-oss-120b"

    model_config = {
        "populate_by_name": True,
    }


class ValidateResponse(BaseModel):
    success: bool
    value: dict[str, Any] | None = None

    attempts: int
    repaired: bool
    flagged: bool
    fallback_used: bool

    error: str | None = None

    raw_output: str | None = None
    final_output: str | None = None

    metrics: dict[str, Any] | None = None


SUPPORTED_TYPES: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def build_dynamic_schema(
    schema_definition: dict[str, str],
) -> type[BaseModel]:
    """
    Build a Pydantic model from a simple JSON-compatible
    field/type definition.
    """

    fields: dict[str, tuple[type[Any], Any]] = {}

    for field_name, field_type in schema_definition.items():
        if field_type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported field type '{field_type}' "
                f"for field '{field_name}'. "
                f"Supported types: "
                f"{', '.join(SUPPORTED_TYPES)}"
            )

        fields[field_name] = (
            SUPPORTED_TYPES[field_type],
            ...,
        )

    if not fields:
        raise ValueError("Schema must contain at least one field.")

    return create_model(
        "DynamicOutputSchema",
        **fields,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Format Guard API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post(
    "/validate",
    response_model=ValidateResponse,
)
def validate_output(
    request: ValidateRequest,
) -> ValidateResponse:
    """
    Generate and validate structured LLM output.
    """

    try:
        schema = build_dynamic_schema(
            request.schema_definition
        )

        provider = GroqProvider(
            model=request.model,
        )

        result = guard(
            schema=schema,
            llm_fn=provider.generate,
            prompt=request.prompt,
            max_retries=request.max_retries,
        )

        metrics = None

        if result.metrics is not None:
            metrics = {
                "attempts": result.metrics.attempts,
                "repairs": result.metrics.repairs,
                "validation_failures": (
                    result.metrics.validation_failures
                ),
                "repair_rate": result.metrics.repair_rate,
                "successful": result.metrics.successful,
            }

        value = None

        if result.value is not None:
            value = result.value.model_dump()

        return ValidateResponse(
            success=result.success,
            value=value,
            attempts=result.attempts,
            repaired=result.repaired,
            flagged=result.flagged,
            fallback_used=result.fallback_used,
            error=result.error,
            raw_output=result.raw_output,
            final_output=result.final_output,
            metrics=metrics,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Format Guard API error: {exc}",
        ) from exc
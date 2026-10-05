from __future__ import annotations

import json

from pydantic import BaseModel


def build_repair_prompt(
    *,
    original_prompt: str,
    invalid_output: str,
    error: str,
    schema: type[BaseModel],
) -> str:
    """
    Build a schema-aware prompt asking the LLM to repair
    a previously invalid response.
    """

    schema_json = json.dumps(
        schema.model_json_schema(),
        indent=2,
    )

    return f"""
Your previous response did not satisfy the required output schema.

Original request:
{original_prompt}

Previous response:
{invalid_output}

Validation error:
{error}

Required JSON schema:
{schema_json}

Repair the previous response.

Rules:
- Return ONLY valid JSON.
- Do not use Markdown code fences.
- Do not add explanations.
- Do not add extra fields.
- Preserve the original meaning wherever possible.
- Fix every validation error.
""".strip()


class RepairEngine:
    """
    Generates repair prompts for invalid LLM outputs.
    """

    def build_prompt(
        self,
        *,
        original_prompt: str,
        invalid_output: str,
        error: str,
        schema: type[BaseModel],
    ) -> str:
        """
        Build a repair prompt for an invalid LLM response.
        """

        return build_repair_prompt(
            original_prompt=original_prompt,
            invalid_output=invalid_output,
            error=error,
            schema=schema,
        )
from __future__ import annotations

import json
import re
from typing import Any


def strip_markdown_fences(text: str) -> str:
    """
    Remove Markdown code fences surrounding JSON.

    Example:

        ```json
        {"name": "Mainak"}
        ```

    becomes:

        {"name": "Mainak"}
    """

    if not isinstance(text, str):
        raise TypeError("JSON output must be a string.")

    text = text.strip()

    pattern = r"^```(?:json)?\s*(.*?)\s*```$"

    match = re.match(
        pattern,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return text


def remove_trailing_commas(text: str) -> str:
    """
    Remove trailing commas before closing JSON objects or arrays.
    """

    return re.sub(
        r",(\s*[}\]])",
        r"\1",
        text,
    )


def clean_json_text(text: str) -> str:
    """
    Apply deterministic JSON cleanup operations.
    """

    cleaned = strip_markdown_fences(text)
    cleaned = remove_trailing_commas(cleaned)

    return cleaned.strip()


def parse_json(text: str) -> Any:
    """
    Clean and parse JSON text.
    """

    cleaned = clean_json_text(text)

    return json.loads(cleaned)
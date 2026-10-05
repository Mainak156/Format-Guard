import pytest

from format_guard.cleaner import (
    clean_json_text,
    parse_json,
    remove_trailing_commas,
    strip_markdown_fences,
)


def test_strip_markdown_json_fence() -> None:
    text = """```json
{"name": "Mainak"}
```"""

    result = strip_markdown_fences(text)

    assert result == '{"name": "Mainak"}'


def test_strip_generic_markdown_fence() -> None:
    text = """```
{"name": "Mainak"}
```"""

    result = strip_markdown_fences(text)

    assert result == '{"name": "Mainak"}'


def test_plain_json_is_unchanged() -> None:
    text = '{"name": "Mainak"}'

    assert strip_markdown_fences(text) == text


def test_remove_trailing_comma_from_object() -> None:
    text = '{"name": "Mainak",}'

    result = remove_trailing_commas(text)

    assert result == '{"name": "Mainak"}'


def test_remove_trailing_comma_from_array() -> None:
    text = '{"items": [1, 2, 3,]}'

    result = remove_trailing_commas(text)

    assert result == '{"items": [1, 2, 3]}'


def test_clean_json_text() -> None:
    text = """```json
{
    "name": "Mainak",
}
```"""

    result = clean_json_text(text)

    assert result == '{\n    "name": "Mainak"\n}'


def test_parse_valid_json() -> None:
    text = '{"name": "Mainak", "age": 21}'

    result = parse_json(text)

    assert result == {
        "name": "Mainak",
        "age": 21,
    }


def test_parse_markdown_json() -> None:
    text = """```json
{
    "name": "Mainak",
    "age": 21
}
```"""

    result = parse_json(text)

    assert result == {
        "name": "Mainak",
        "age": 21,
    }


def test_parse_json_with_trailing_comma() -> None:
    text = """{
        "name": "Mainak",
        "age": 21,
    }"""

    result = parse_json(text)

    assert result == {
        "name": "Mainak",
        "age": 21,
    }


def test_invalid_json_raises_error() -> None:
    text = '{"name": "Mainak",'

    with pytest.raises(Exception):
        parse_json(text)


def test_non_string_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        parse_json({"name": "Mainak"})
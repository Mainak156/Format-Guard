from __future__ import annotations

import json
from pathlib import Path


DATASET_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "data"
    / "customer_prompts.json"
)


def load_dataset() -> list[dict[str, object]]:
    return json.loads(
        DATASET_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_dataset_contains_200_prompts() -> None:
    dataset = load_dataset()

    assert len(dataset) == 200


def test_dataset_ids_are_unique() -> None:
    dataset = load_dataset()

    ids = [
        item["id"]
        for item in dataset
    ]

    assert len(ids) == len(set(ids))


def test_dataset_contains_all_categories() -> None:
    dataset = load_dataset()

    categories = {
        item["category"]
        for item in dataset
    }

    assert categories == {
        "crm",
        "support",
        "sales",
        "meeting",
        "onboarding",
    }


def test_each_category_has_40_prompts() -> None:
    dataset = load_dataset()

    counts: dict[str, int] = {}

    for item in dataset:
        category = str(item["category"])

        counts[category] = (
            counts.get(category, 0) + 1
        )

    assert counts == {
        "crm": 40,
        "support": 40,
        "sales": 40,
        "meeting": 40,
        "onboarding": 40,
    }


def test_dataset_contains_expected_fields() -> None:
    dataset = load_dataset()

    required_fields = {
        "name",
        "age",
        "email",
        "company",
        "role",
    }

    for item in dataset:
        assert set(item) == {
            "id",
            "category",
            "prompt",
            "expected",
        }

        assert isinstance(
            item["prompt"],
            str,
        )

        assert set(
            item["expected"]
        ) == required_fields


def test_first_and_last_prompt_have_expected_structure() -> None:
    dataset = load_dataset()

    assert dataset[0]["id"] == "FG-001"
    assert dataset[-1]["id"] == "FG-200"

    assert isinstance(
        dataset[0]["prompt"],
        str,
    )

    assert isinstance(
        dataset[-1]["prompt"],
        str,
    )
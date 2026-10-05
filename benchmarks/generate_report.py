from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = (
    BASE_DIR
    / "results"
    / "groq_full_benchmark.json"
)

REPORTS_DIR = BASE_DIR / "results"

CHART_PATH = (
    REPORTS_DIR
    / "format_guard_before_after.png"
)


def load_results() -> dict:
    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def generate_chart(
    results: dict,
) -> None:
    models = []
    before_rates = []
    after_rates = []

    for model_name, model_data in results[
        "models"
    ].items():
        summary = model_data["summary"]

        models.append(model_name)
        before_rates.append(
            summary["valid_rate_before"] * 100
        )
        after_rates.append(
            summary["valid_rate_after"] * 100
        )

    x = range(len(models))
    width = 0.35

    before_positions = [
        position - width / 2
        for position in x
    ]

    after_positions = [
        position + width / 2
        for position in x
    ]

    plt.figure(
        figsize=(11, 6.5)
    )

    plt.bar(
        before_positions,
        before_rates,
        width=width,
        label="Before Format Guard",
    )

    plt.bar(
        after_positions,
        after_rates,
        width=width,
        label="After Format Guard",
    )

    plt.title(
        "Format Guard — Valid Output Rate"
    )

    plt.ylabel(
        "Valid Output Rate (%)"
    )

    plt.xlabel(
        "LLM"
    )

    plt.xticks(
        list(x),
        models,
    )

    plt.ylim(
        0,
        110,
    )

    plt.legend()

    for positions, values in [
        (
            before_positions,
            before_rates,
        ),
        (
            after_positions,
            after_rates,
        ),
    ]:
        for position, value in zip(
            positions,
            values,
        ):
            plt.text(
                position,
                value + 2,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        CHART_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Chart generated successfully:"
    )
    print(
        CHART_PATH
    )


def print_summary(
    results: dict,
) -> None:
    print()
    print("=" * 70)
    print(
        "FORMAT GUARD — BENCHMARK SUMMARY"
    )
    print("=" * 70)

    for model_name, model_data in results[
        "models"
    ].items():
        summary = model_data["summary"]

        print()
        print(
            f"Model: {model_name}"
        )
        print(
            f"  Prompts       : "
            f"{summary['total_prompts']}"
        )
        print(
            f"  Before        : "
            f"{summary['valid_rate_before']:.2%}"
        )
        print(
            f"  After         : "
            f"{summary['valid_rate_after']:.2%}"
        )
        print(
            f"  Improvement   : "
            f"{summary['improvement']:+.2%}"
        )
        print(
            f"  Avg retries   : "
            f"{summary['average_retries']:.2f}"
        )
        print(
            f"  Repair rate   : "
            f"{summary['repair_rate']:.2%}"
        )
        print(
            f"  Flagged       : "
            f"{summary['flagged']}"
        )
        print(
            f"  Extra cost    : "
            f"${summary['extra_cost']:.6f}"
        )


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: "
            f"{RESULTS_PATH}"
        )

    results = load_results()

    generate_chart(results)
    print_summary(results)


if __name__ == "__main__":
    main()
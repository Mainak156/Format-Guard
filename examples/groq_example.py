from pydantic import BaseModel

from format_guard import guard
from format_guard.providers import GroqProvider


class Customer(BaseModel):
    name: str
    age: int
    email: str


def main() -> None:
    provider = GroqProvider()

    result = guard(
        schema=Customer,
        llm_fn=provider.generate,
        prompt=(
            "Return customer information as JSON. "
            "Use this customer: Mainak, age 21, "
            "email mainak@example.com."
        ),
        max_retries=2,
    )

    print("\n=== FORMAT GUARD RESULT ===")
    print(f"Success: {result.success}")
    print(f"Value: {result.value}")
    print(f"Attempts: {result.attempts}")
    print(f"Repaired: {result.repaired}")
    print(f"Flagged: {result.flagged}")
    print(f"Error: {result.error}")

    if result.metrics:
        print("\n=== METRICS ===")
        print(f"Attempts: {result.metrics.attempts}")
        print(f"Repairs: {result.metrics.repairs}")
        print(
            f"Validation failures: "
            f"{result.metrics.validation_failures}"
        )
        print(f"Repair rate: {result.metrics.repair_rate:.2%}")
        print(f"Successful: {result.metrics.successful}")


if __name__ == "__main__":
    main()
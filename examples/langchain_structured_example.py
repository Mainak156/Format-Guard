from pydantic import BaseModel

from format_guard.providers import LangChainGroqProvider


class Customer(BaseModel):
    name: str
    age: int
    email: str


def main() -> None:
    provider = LangChainGroqProvider()

    result = provider.generate_structured(
        prompt=(
            "Extract the customer information from this text: "
            "Mainak is 21 years old and his email is "
            "mainak@example.com."
        ),
        schema=Customer,
    )

    print("\n=== LANGCHAIN STRUCTURED OUTPUT ===")
    print(f"Result: {result}")
    print(f"Type: {type(result).__name__}")
    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Email: {result.email}")


if __name__ == "__main__":
    main()
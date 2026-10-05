from __future__ import annotations

from collections.abc import Iterable, Iterator


class MockProvider:
    """
    Deterministic LLM provider used for testing.

    Responses can be supplied as a sequence and are returned
    one by one for each call.
    """

    def __init__(self, responses: Iterable[str]) -> None:
        self._responses: Iterator[str] = iter(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        """
        Return the next configured response.
        """

        self.calls.append(prompt)

        try:
            return next(self._responses)
        except StopIteration as exc:
            raise RuntimeError(
                "MockProvider has no responses remaining."
            ) from exc
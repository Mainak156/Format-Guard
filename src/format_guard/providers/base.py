from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Abstract interface for all LLM providers.

    Format Guard depends only on this interface and therefore
    remains independent of any specific LLM vendor.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a raw text response from the language model.

        Args:
            prompt: Prompt sent to the model.

        Returns:
            Raw model response as a string.

        Raises:
            NotImplementedError:
                If the subclass does not implement this method.
        """

        raise NotImplementedError
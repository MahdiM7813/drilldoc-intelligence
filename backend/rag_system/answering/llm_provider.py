from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        """Generate plain text locally."""

    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: str = "") -> dict[str, Any]:
        """Generate and parse a JSON object locally."""

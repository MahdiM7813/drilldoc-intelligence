"""Ollama implementation of the RAG text/JSON provider interface."""

from __future__ import annotations

import json
from typing import Any, Callable
from urllib.request import Request, urlopen

from .llm_provider import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, config: dict[str, Any], opener: Callable[..., Any] = urlopen):
        self.base_url = str(config["ollama_base_url"]).rstrip("/")
        self.model = str(config["ollama_model"]).strip()
        if not self.model:
            raise ValueError("Ollama model is required")
        self.config = config
        self._opener = opener

    def _chat(self, prompt: str, system_prompt: str, json_mode: bool) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": float(self.config.get("temperature", 0.0)),
                "num_predict": int(self.config.get("max_new_tokens", 384)),
            },
        }
        if json_mode:
            payload["format"] = "json"
        request = Request(f"{self.base_url}/api/chat", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self._opener(request, timeout=float(self.config.get("timeout", 120.0))) as response:
                result = json.load(response)
            content = result["message"]["content"]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("Ollama chat request failed or returned an invalid response") from exc
        if not isinstance(content, str):
            raise RuntimeError("Ollama response content must be text")
        return content.strip()

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        return self._chat(prompt, system_prompt, False)

    def generate_json(self, prompt: str, system_prompt: str = "") -> dict[str, Any]:
        text = self._chat(prompt, system_prompt, True)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ollama produced malformed JSON: {text[:1200]}") from exc
        if not isinstance(value, dict):
            raise ValueError("Ollama JSON response must be an object")
        return value

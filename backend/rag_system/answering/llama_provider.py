from __future__ import annotations

import json
import threading
import time
from typing import Any

from .llm_provider import LLMProvider


class LlamaCppProvider(LLMProvider):
    instances_created = 0
    instances_lock = threading.Lock()
    def __init__(self, config: dict[str, Any]):
        from llama_cpp import Llama

        self.config = config
        self.model_path = str(config["model_path"])
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=int(config.get("n_ctx", 8192)),
            n_batch=int(config.get("n_batch", 512)),
            n_threads=int(config.get("n_threads", 8)),
            n_threads_batch=int(config.get("n_threads_batch", config.get("n_threads", 8))),
            n_gpu_layers=int(config.get("n_gpu_layers", 0)),
            verbose=False,
        )
        with self.instances_lock:
            type(self).instances_created += 1
        self.generation_lock = threading.Lock()
        self.last_generation: dict[str, Any] = {}

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        started = time.perf_counter()
        with self.generation_lock:
            response = self.model.create_chat_completion(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                max_tokens=int(self.config.get("max_new_tokens", 192)),
                temperature=float(self.config.get("temperature", 0.1)),
            )
        elapsed = time.perf_counter() - started
        choice = response["choices"][0]
        usage = response.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_generation = {
            "elapsed_seconds": elapsed,
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": completion_tokens,
            "tokens_per_second": completion_tokens / elapsed if completion_tokens and elapsed else None,
            "max_tokens": int(self.config.get("max_new_tokens", 192)),
        }
        return str(choice["message"]["content"]).strip()

    def tokenize(self, text: str) -> list[int]:
        return self.model.tokenize(text.encode("utf-8"), add_bos=False)

    def chat_prompt_tokens(self, system_prompt: str, user_prompt: str) -> int:
        rendered = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
            f"{system_prompt}<|eot_id|>\n\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>\n\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return len(self.model.tokenize(rendered.encode("utf-8"), add_bos=False))

    def generate_json(self, prompt: str, system_prompt: str = "") -> dict[str, Any]:
        text = self.generate_text(prompt, system_prompt)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Llama output did not contain JSON")
        return json.loads(text[start:end + 1])

    def close(self) -> None:
        model = self.model
        self.model = None
        del model

from __future__ import annotations

import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .llm_provider import LLMProvider


class QwenTransformersProvider(LLMProvider):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_path = str(config["model_path"])
        if not config.get("local_files_only", True):
            raise ValueError("Qwen provider must use local_files_only")
        dtype_name = config.get("dtype", "bfloat16")
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        self.device = str(config.get("device", "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False, dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def _generate(self, prompt: str, system_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        do_sample = float(self.config.get("temperature", 0.0)) > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 384)),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = float(self.config["temperature"])
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        return self._generate(prompt, system_prompt)

    def generate_json(self, prompt: str, system_prompt: str = "") -> dict[str, Any]:
        text = self._generate(prompt, system_prompt)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"Qwen output did not contain a JSON object: {text[:300]}")
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Qwen produced malformed JSON: {text[:1200]}") from exc

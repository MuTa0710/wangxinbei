"""Optional local Qwen3 four-choice adapter for M4.

The adapter is lazy: importing the project does not load weights. M4 uses the
deterministic baseline unless ``--backend qwen3`` is explicitly selected.
"""

from __future__ import annotations

import re
from typing import Any


class QwenChoiceAgent:
    def __init__(self, model_name: str, *, device: str = "auto", dtype: str = "float16") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        if dtype == "float16":
            import torch

            model_kwargs["torch_dtype"] = torch.float16
        elif dtype == "bfloat16":
            import torch

            model_kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

    def choose(self, instruction: str, evidence: str) -> tuple[str, str]:
        prompt = (
            "你是B01航迹理解判别器。只根据给定航迹事实和选项选择唯一正确的A、B、C或D。"
            "最终只输出一个大写字母，不要输出解释。\n\n"
            f"航迹事实：\n{evidence}\n\n题目与选项：\n{instruction}\n答案："
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        match = re.search(r"\b([ABCD])\b", text.upper())
        if not match:
            raise ValueError(f"Qwen3 output did not contain A/B/C/D: {text!r}")
        return match.group(1), text


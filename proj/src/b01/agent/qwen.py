"""Lazy local Qwen3 nodes for M6.

The class performs bounded JSON generation. It does not execute arbitrary code
or discover tools dynamically; the controller decides which fixed tool to call.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from b01.retrieval.baseline import BEHAVIOR_TERMS

from .schemas import Task1Decision, Task2QueryIR, model_json_schema


def _build_behavior_aliases() -> dict[str, str]:
    """Reverse map: Chinese behavior term -> canonical English name."""
    aliases: dict[str, str] = {}
    for canonical, terms in BEHAVIOR_TERMS:
        for term in terms:
            aliases.setdefault(term, canonical)
    return aliases


_BEHAVIOR_ALIASES = _build_behavior_aliases()

_BEHAVIOR_CANONICAL_SET = set(_BEHAVIOR_ALIASES.values())
_BEHAVIOR_CANONICAL_BY_LEN = tuple(sorted(_BEHAVIOR_CANONICAL_SET, key=len, reverse=True))


def _canonicalize_behavior_name(name: str) -> str:
    """Normalize a model-produced behavior name to a canonical name.

    Handles Chinese terms, case/whitespace, and the common hallucination where
    the model prefixes a canonical name with a modifier (``abrupt_climb`` ->
    ``climb``, ``long_platform_step_decel`` -> ``step_decel``).
    """
    key = name.strip().lower()
    if key in _BEHAVIOR_CANONICAL_SET:
        return key
    if key in _BEHAVIOR_ALIASES:
        return _BEHAVIOR_ALIASES[key]
    for canonical in _BEHAVIOR_CANONICAL_BY_LEN:
        if key.endswith(canonical):
            return canonical
    return key


# Few-shot examples targeting the two recurring Task2 failure modes:
# 1) echoing the input ``regions`` field into the output (extra_forbidden);
# 2) inventing ``abrupt_step_decel`` / ``long_platform_step_decel`` which are
#    not in the canonical behavior vocabulary (deceleration folds into step_decel).
_TASK2_FEW_SHOT: list[tuple[dict[str, Any], dict[str, Any]]] = [
    (
        {
            "sample_id": "demo_1",
            "text_query": "检索经过区域 A、先盘旋绕飞随后低速移动的航迹。",
            "top_k": 10,
            "regions": [{"区域编号": "区域 A"}],
        },
        {
            "sample_id": "demo_1",
            "text_query": "检索经过区域 A、先盘旋绕飞随后低速移动的航迹。",
            "top_k": 10,
            "positive_regions": ["区域 A"],
            "negative_regions": [],
            "behavior_sequence": [{"name": "orbit"}, {"name": "low_speed"}],
        },
    ),
    (
        {
            "sample_id": "demo_2",
            "text_query": "检索先急促阶梯型加速随后阶梯型减速的航迹。",
            "top_k": 10,
            "regions": [],
        },
        {
            "sample_id": "demo_2",
            "text_query": "检索先急促阶梯型加速随后阶梯型减速的航迹。",
            "top_k": 10,
            "positive_regions": [],
            "negative_regions": [],
            "behavior_sequence": [{"name": "abrupt_step_accel"}, {"name": "step_decel"}],
        },
    ),
    (
        {
            "sample_id": "demo_3",
            "text_query": "检索起始时间在2025年1月1日10:00:00至18:00:00、速度在60m/s至70m/s之间、经过区域 A 的航迹。",
            "top_k": 10,
            "regions": [{"区域编号": "区域 A"}],
        },
        {
            "sample_id": "demo_3",
            "text_query": "检索起始时间在2025年1月1日10:00:00至18:00:00、速度在60m/s至70m/s之间、经过区域 A 的航迹。",
            "top_k": 10,
            "time_start_utc": "2025-01-01T10:00:00",
            "time_end_utc": "2025-01-01T18:00:00",
            "speed_range_mps": [60, 70],
            "positive_regions": ["区域 A"],
            "negative_regions": [],
            "behavior_sequence": [],
        },
    ),
]


_TASK2_SYSTEM_PROMPT = (
    "你是B01查询解析器。只把中文查询转换为给定JSON结构。不要检索数据，不要解释，不要新增字段。"
    "行为名称必须使用schema允许的canonical名称。\n"
    "规则：输出不含 regions 字段，区域只放进 positive_regions/negative_regions（区域编号字符串，如\"区域 A\"）；"
    "\"减速\"类行为统一用 step_decel，不要自创 abrupt_step_decel 等不存在的名称。\n"
    + "\n".join(
        f"示例{i} 输入: {json.dumps(inp, ensure_ascii=False)}\n示例{i} 输出: {json.dumps(out, ensure_ascii=False)}"
        for i, (inp, out) in enumerate(_TASK2_FEW_SHOT, 1)
    )
)


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    # Qwen3 may emit a <think>...</think> reasoning block even when thinking is
    # disabled, and a truncated "<think>" with no closing marker on short
    # responses. Strip it before looking for the JSON object.
    if "<think>" in cleaned:
        if "</think>" in cleaned:
            cleaned = cleaned.rsplit("</think>", 1)[1]
        else:
            cleaned = cleaned.split("<think>", 1)[0]
        cleaned = cleaned.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S | re.I)
    candidate = fenced.group(1) if fenced else cleaned
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"model output does not contain a JSON object: {text!r}")
    return json.loads(candidate[start : end + 1])


class Qwen3Agent:
    def __init__(self, model_name: str, *, dtype: str = "float16", max_retries: int = 1) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_retries = max_retries
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        if dtype == "float16":
            import torch

            model_kwargs["torch_dtype"] = torch.float16
        elif dtype == "bfloat16":
            import torch

            model_kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        # Qwen3's default generation config enables thinking. M6 requires a
        # bounded JSON response, so disable sampling-only defaults as well.
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(self.model.generation_config, field):
                setattr(self.model.generation_config, field, None)

    def _generate(self, system: str, user: str, max_new_tokens: int = 256) -> tuple[dict[str, Any], str]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    enable_thinking=False,
                )
            except TypeError:
                inputs = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                )
            inputs = {"input_ids": inputs} if not hasattr(inputs, "items") else inputs
        else:
            prompt = f"System: {system}\nUser: {user}\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            return_dict_in_generate=False,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        raw = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        return _extract_json(raw), raw

    def _generate_json(
        self,
        system: str,
        user_payload: dict[str, Any],
        *,
        max_new_tokens: int,
        validate: Callable[[dict[str, Any]], Any],
    ) -> tuple[Any, str]:
        """Generate and validate, retrying with the error as feedback on failure.

        Retries are bounded by ``max_retries``; when every attempt fails the last
        exception is re-raised so the controller falls back to the deterministic
        implementation.
        """
        last_error: Exception | None = None
        payload = dict(user_payload)
        for _ in range(self.max_retries + 1):
            user = json.dumps(payload, ensure_ascii=False)
            data, raw = self._generate(system, user, max_new_tokens=max_new_tokens)
            try:
                return validate(data), raw
            except Exception as exc:  # noqa: BLE001 - bounded retry, fallback upstream
                last_error = exc
                error_text = str(exc)
                if len(error_text) > 800:
                    error_text = error_text[:800] + "…"
                payload["previous_attempt_error"] = error_text
                payload["retry_note"] = "上一次输出不符合要求，请严格按 required_schema 重新输出完整、合法的 JSON，不要解释，不要新增字段。"
        assert last_error is not None
        raise last_error

    @staticmethod
    def _canonicalize_regions(data: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
        """Map model outputs such as ``A`` to the official ``区域 A`` label."""

        aliases: dict[str, str] = {}
        for region in sample.get("regions", []):
            official = str(region.get("区域编号", region.get("region_id", "")))
            short = official.replace("区域", "").strip()
            for alias in (official, official.replace(" ", ""), short, short.upper(), short.lower()):
                if alias:
                    aliases[alias] = official
        for field in ("positive_regions", "negative_regions"):
            values = data.get(field, [])
            data[field] = [aliases.get(str(value), str(value)) for value in values]
        return data

    @staticmethod
    def _canonicalize_behaviors(data: dict[str, Any]) -> dict[str, Any]:
        """Map Chinese terms and hallucinated names in ``name`` to canonical names."""
        sequence = data.get("behavior_sequence")
        if not isinstance(sequence, list):
            return data
        for item in sequence:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                item["name"] = _canonicalize_behavior_name(item["name"])
        return data

    def parse_task2(self, sample: dict[str, Any]) -> tuple[Task2QueryIR, str]:
        schema = json.dumps(model_json_schema(Task2QueryIR), ensure_ascii=False)
        payload = {
            "sample_id": sample["sample_id"],
            "text_query": sample["text_query"],
            "top_k": sample.get("top_k", 10),
            "regions": sample.get("regions", []),
            "required_schema": schema,
        }

        def validate(data: dict[str, Any]) -> Task2QueryIR:
            data = self._canonicalize_regions(data, sample)
            data = self._canonicalize_behaviors(data)
            return Task2QueryIR.model_validate(data)

        return self._generate_json(
            _TASK2_SYSTEM_PROMPT,
            payload,
            max_new_tokens=512,
            validate=validate,
        )

    def judge_task1(self, instruction: str, evidence: dict[str, Any]) -> tuple[Task1Decision, str]:
        schema = json.dumps(model_json_schema(Task1Decision), ensure_ascii=False)
        payload = {"instruction": instruction, "evidence": evidence, "required_schema": schema}
        return self._generate_json(
            "你是B01航迹语义判别器。只能依据提供的事实选择A、B、C或D。输出符合schema的JSON，不要虚构事实。",
            payload,
            max_new_tokens=128,
            validate=Task1Decision.model_validate,
        )

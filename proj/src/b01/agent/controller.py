"""M6 fixed Agent workflow and deterministic fallback policy."""

from __future__ import annotations

from typing import Any

from b01.submission.schema import make_task1_record, make_task2_record

from .schemas import Task2QueryIR
from .tools import B01Tools


class B01AgentController:
    """Run one bounded workflow over the official normalized samples."""

    def __init__(self, tools: B01Tools, *, backend: str = "deterministic", qwen: Any = None) -> None:
        if backend not in {"deterministic", "qwen3"}:
            raise ValueError(f"unsupported backend: {backend}")
        if backend == "qwen3" and qwen is None:
            raise ValueError("qwen3 backend requires a Qwen3Agent instance")
        self.tools = tools
        self.backend = backend
        self.qwen = qwen

    def run_task1(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        for sample in self.tools.segment_rows:
            evidence = self.tools.task1_segment_facts(sample)
            fallback_choice, fallback_scores = self.tools.task1_segment_fallback(sample)
            choice, raw, used_fallback, error = fallback_choice, None, True, None
            if self.backend == "qwen3":
                try:
                    decision, raw = self.qwen.judge_task1(sample["text_instruction"], evidence)
                    choice, used_fallback = decision.choice, False
                except Exception as exc:  # bounded per-sample fallback
                    error = f"{type(exc).__name__}: {exc}"
            records.append(make_task1_record(sample["sample_id"], choice))
            facts.append({"sample_id": sample["sample_id"], "task": "segment", "facts": evidence})
            diagnostics.append(
                {
                    "sample_id": sample["sample_id"],
                    "task": "segment",
                    "backend": self.backend,
                    "used_fallback": used_fallback,
                    "fallback_choice": fallback_choice,
                    "fallback_scores": fallback_scores,
                    "choice": choice,
                    "raw_model_output": raw,
                    "error": error,
                }
            )
        for sample in self.tools.region_rows:
            evidence = self.tools.task1_region_facts(sample)
            fallback_choice, fallback_scores, _ = self.tools.task1_region_fallback(sample)
            choice, raw, used_fallback, error = fallback_choice, None, True, None
            if self.backend == "qwen3":
                try:
                    decision, raw = self.qwen.judge_task1(sample["text_instruction"], evidence)
                    choice, used_fallback = decision.choice, False
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            records.append(make_task1_record(sample["sample_id"], choice))
            facts.append(
                {
                    "sample_id": sample["sample_id"],
                    "task": "region",
                    "facts": evidence,
                }
            )
            diagnostics.append(
                {
                    "sample_id": sample["sample_id"],
                    "task": "region",
                    "backend": self.backend,
                    "used_fallback": used_fallback,
                    "fallback_choice": fallback_choice,
                    "fallback_scores": fallback_scores,
                    "choice": choice,
                    "raw_model_output": raw,
                    "error": error,
                }
            )
        return records, diagnostics, facts

    def _parse_task2(self, sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        fallback = Task2QueryIR.model_validate(
            self._deterministic_query(sample)
        )
        if self.backend != "qwen3":
            return fallback.as_retrieval_dict(), {
                "backend": "deterministic",
                "used_fallback": True,
                "error": None,
                "raw_model_output": None,
            }
        try:
            parsed, raw = self.qwen.parse_task2(sample)
            if parsed.sample_id != sample["sample_id"] or parsed.top_k != int(sample.get("top_k", 10)):
                raise ValueError("model changed sample_id or top_k")
            return parsed.as_retrieval_dict(), {
                "backend": "qwen3",
                "used_fallback": False,
                "error": None,
                "raw_model_output": raw,
            }
        except Exception as exc:
            return fallback.as_retrieval_dict(), {
                "backend": "qwen3",
                "used_fallback": True,
                "error": f"{type(exc).__name__}: {exc}",
                "raw_model_output": None,
            }

    @staticmethod
    def _deterministic_query(sample: dict[str, Any]) -> dict[str, Any]:
        from b01.retrieval.baseline import parse_query

        return parse_query(sample)

    def run_task2(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        query_ir: list[dict[str, Any]] = []
        for sample in self.tools.queries:
            parsed, parse_diag = self._parse_task2(sample)
            results, retrieval_diag = self.tools.task2_retrieve(parsed)
            ids = [item["航迹编号"] for item in results]
            records.append(make_task2_record(sample["sample_id"], ids, top_k=int(sample.get("top_k", 10))))
            query_ir.append({"sample_id": sample["sample_id"], **parsed})
            diagnostics.append(
                {
                    "sample_id": sample["sample_id"],
                    "parse": parse_diag,
                    "retrieval": retrieval_diag,
                    "result_count": len(results),
                }
            )
        return records, diagnostics, query_ir

    def run(self) -> dict[str, Any]:
        task1_records, task1_diag, task1_facts = self.run_task1()
        task2_records, task2_diag, query_ir = self.run_task2()
        return {
            "records": task1_records + task2_records,
            "task1_diagnostics": task1_diag,
            "task1_facts": task1_facts,
            "task2_diagnostics": task2_diag,
            "query_ir": query_ir,
        }

"""Official B01 submission records and validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


TASK1_OPTION_KEYS = ("选项", "结果")


def make_task1_record(sample_id: str, option: str, *, field_name: str = "选项") -> dict[str, Any]:
    if field_name not in TASK1_OPTION_KEYS:
        raise ValueError(f"Unsupported task1 output field: {field_name}")
    option = option.strip().upper()
    if option not in {"A", "B", "C", "D"}:
        raise ValueError(f"task1 option must be A/B/C/D, got {option!r}")
    return {"样本编号": sample_id, field_name: option}


def make_task2_record(sample_id: str, trajectory_ids: Iterable[str], *, top_k: int | None = None) -> dict[str, Any]:
    ids = list(trajectory_ids)
    if top_k is not None and len(ids) > top_k:
        ids = ids[:top_k]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate trajectory id in task2 result for {sample_id}")
    return {
        "样本编号": sample_id,
        "结果": [{"序号": index, "航迹编号": trajectory_id} for index, trajectory_id in enumerate(ids, start=1)],
    }


def validate_submission_records(
    records: Iterable[dict[str, Any]],
    expected_sample_ids: Iterable[str],
    *,
    task1_field_name: str = "选项",
) -> dict[str, Any]:
    expected = list(expected_sample_ids)
    expected_set = set(expected)
    seen: set[str] = set()
    errors: list[str] = []
    task1_count = 0
    task2_count = 0
    for line_number, record in enumerate(records, start=1):
        sample_id = record.get("样本编号")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"line {line_number}: missing 样本编号")
            continue
        if sample_id in seen:
            errors.append(f"line {line_number}: duplicate 样本编号 {sample_id}")
        seen.add(sample_id)
        if sample_id.startswith(("seg_", "region_")):
            task1_count += 1
            if task1_field_name not in record or record[task1_field_name] not in {"A", "B", "C", "D"}:
                errors.append(f"line {line_number}: invalid task1 option")
        elif sample_id.startswith("retrieval_"):
            task2_count += 1
            results = record.get("结果")
            if not isinstance(results, list):
                errors.append(f"line {line_number}: task2 结果 must be a list")
            else:
                expected_order = list(range(1, len(results) + 1))
                actual_order = [item.get("序号") for item in results if isinstance(item, dict)]
                ids = [item.get("航迹编号") for item in results if isinstance(item, dict)]
                if actual_order != expected_order:
                    errors.append(f"line {line_number}: task2 序号 is not continuous")
                if len(ids) != len(set(ids)):
                    errors.append(f"line {line_number}: duplicate task2 航迹编号")
        else:
            errors.append(f"line {line_number}: unknown sample prefix {sample_id}")

    missing = sorted(expected_set - seen)
    unexpected = sorted(seen - expected_set)
    errors.extend(f"missing sample {sample_id}" for sample_id in missing)
    errors.extend(f"unexpected sample {sample_id}" for sample_id in unexpected)
    return {
        "valid": not errors,
        "errors": errors,
        "expected_count": len(expected),
        "actual_count": len(seen),
        "task1_count": task1_count,
        "task2_count": task2_count,
    }

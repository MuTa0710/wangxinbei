#!/usr/bin/env python3
"""Run the bounded M6 Agent workflow and produce a validated submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from b01.agent.controller import B01AgentController  # noqa: E402
from b01.agent.qwen import Qwen3Agent  # noqa: E402
from b01.agent.tools import B01Tools  # noqa: E402
from b01.io.jsonl import read_jsonl, write_jsonl  # noqa: E402
from b01.submission.schema import validate_submission_records  # noqa: E402


SCHEMA_VERSION = "m6.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_ids(cache_dir: Path) -> list[str]:
    files = (
        "task1_segment_normalized.jsonl",
        "task1_region_normalized.jsonl",
        "task2_queries_normalized.jsonl",
    )
    return [row["sample_id"] for name in files for row in read_jsonl(cache_dir / name)]


def _structure(records: list[dict[str, Any]]) -> dict[str, Any]:
    task1 = task2 = 0
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        sample_id = record.get("样本编号", "")
        if sample_id.startswith(("seg_", "region_")):
            if record.get("选项") not in {"A", "B", "C", "D"}:
                errors.append(f"line {index}: invalid task1 option")
            else:
                task1 += 1
        elif sample_id.startswith("retrieval_"):
            result = record.get("结果")
            if not isinstance(result, list):
                errors.append(f"line {index}: invalid task2 result")
            else:
                order = [item.get("序号") for item in result]
                ids = [item.get("航迹编号") for item in result]
                if order != list(range(1, len(result) + 1)):
                    errors.append(f"line {index}: non-continuous task2 order")
                if len(ids) != len(set(ids)):
                    errors.append(f"line {index}: duplicate task2 ids")
                task2 += 1
        else:
            errors.append(f"line {index}: unknown sample id")
    return {"task1": task1, "task2": task2, "errors": errors}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tools = B01Tools(cache_dir, args.m3_dir.resolve() if args.m3_dir else None)
    qwen = Qwen3Agent(args.model_name, dtype=args.dtype, max_retries=args.max_retries) if args.backend == "qwen3" else None
    controller = B01AgentController(tools, backend=args.backend, qwen=qwen)
    result = controller.run()

    submission = output_dir / "traj_result.jsonl"
    records = result["records"]
    write_jsonl(submission, records)
    expected = _expected_ids(cache_dir)
    validation = validate_submission_records(records, expected, task1_field_name="选项")
    structure = _structure(records)
    write_jsonl(output_dir / "task1_diagnostics.jsonl", result["task1_diagnostics"])
    write_jsonl(output_dir / "task1_facts.jsonl", result["task1_facts"])
    write_jsonl(output_dir / "task2_diagnostics.jsonl", result["task2_diagnostics"])
    write_jsonl(output_dir / "query_ir.jsonl", result["query_ir"])
    fallback_task1 = sum(item["used_fallback"] for item in result["task1_diagnostics"])
    fallback_task2 = sum(item["parse"]["used_fallback"] for item in result["task2_diagnostics"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now().astimezone().isoformat().replace("+00:00", "Z"),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "backend": args.backend,
        "model_name": args.model_name,
        "policy": {
            "max_model_calls_per_task1_sample": 1 + args.max_retries,
            "max_model_calls_per_task2_sample": 1 + args.max_retries,
            "deterministic_tools": ["task1_facts", "task1_fallback", "task2_retrieve", "submission_validator"],
            "fallback_enabled": True,
        },
        "counts": {
            "records": len(records),
            "task1": len(result["task1_diagnostics"]),
            "task2": len(result["task2_diagnostics"]),
            "task1_fallback": fallback_task1,
            "task2_parser_fallback": fallback_task2,
        },
        "validation": validation,
        "structure": structure,
        "artifacts": {
            "submission": str(submission),
            "submission_sha256": _sha256(submission),
        },
    }
    (output_dir / "m6_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "m1")
    parser.add_argument("--m3-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "m3")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "m6")
    parser.add_argument("--backend", choices=("deterministic", "qwen3"), default="deterministic")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max-retries", type=int, default=1, help="max retries per sample with validation-error feedback")
    args = parser.parse_args()
    if args.backend == "qwen3" and not args.model_name:
        parser.error("--model-name is required with --backend qwen3")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["validation"]["valid"] and not report["structure"]["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

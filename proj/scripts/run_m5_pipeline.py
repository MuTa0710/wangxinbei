#!/usr/bin/env python3
"""Run M3/M4, merge official submission records, and produce the M5 report."""

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
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from build_m3_retrieval import build as build_m3  # noqa: E402
from build_m4_task1 import build as build_m4  # noqa: E402
from b01.io.jsonl import read_jsonl, write_jsonl  # noqa: E402
from b01.submission.schema import validate_submission_records  # noqa: E402


SCHEMA_VERSION = "m5.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_ids(cache_dir: Path) -> list[str]:
    segment_ids = [row["sample_id"] for row in read_jsonl(cache_dir / "task1_segment_normalized.jsonl")]
    region_ids = [row["sample_id"] for row in read_jsonl(cache_dir / "task1_region_normalized.jsonl")]
    retrieval_ids = [row["sample_id"] for row in read_jsonl(cache_dir / "task2_queries_normalized.jsonl")]
    return segment_ids + region_ids + retrieval_ids


def _merge_records(m4_dir: Path, m3_dir: Path, output_file: Path, expected_ids: list[str]) -> dict[str, Any]:
    task1 = {row["样本编号"]: {"样本编号": row["样本编号"], "选项": row["选项"]} for row in read_jsonl(m4_dir / "task1_baseline_results.jsonl")}
    task2 = {
        row["样本编号"]: {"样本编号": row["样本编号"], "结果": row["结果"]}
        for row in read_jsonl(m3_dir / "task2_baseline_results.jsonl")
    }
    missing = [sample_id for sample_id in expected_ids if sample_id not in task1 and sample_id not in task2]
    if missing:
        raise ValueError(f"Cannot merge missing sample IDs: {missing[:5]}")
    records = [task1.get(sample_id, task2.get(sample_id)) for sample_id in expected_ids]
    count = write_jsonl(output_file, records)
    return {"records": count, "task1": len(task1), "task2": len(task2), "missing": missing}


def _structure_checks(records: list[dict[str, Any]]) -> dict[str, Any]:
    task1_valid = 0
    task2_valid = 0
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        sample_id = record.get("样本编号", "")
        if sample_id.startswith(("seg_", "region_")):
            if record.get("选项") in {"A", "B", "C", "D"}:
                task1_valid += 1
            else:
                errors.append(f"line {index}: invalid task1 option")
        elif sample_id.startswith("retrieval_"):
            result = record.get("结果")
            if isinstance(result, list) and [item.get("序号") for item in result] == list(range(1, len(result) + 1)):
                ids = [item.get("航迹编号") for item in result]
                if len(ids) == len(set(ids)):
                    task2_valid += 1
                else:
                    errors.append(f"line {index}: duplicate task2 ids")
            else:
                errors.append(f"line {index}: invalid task2 result")
        else:
            errors.append(f"line {index}: unknown sample id")
    return {"task1_valid": task1_valid, "task2_valid": task2_valid, "errors": errors}


def _synthetic_metrics(m2_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    boundary_file = m2_dir / "boundary_validation.json"
    boundary = json.loads(boundary_file.read_text(encoding="utf-8")) if boundary_file.exists() else {"passed": False}
    structure = _structure_checks(records)
    return {
        "m2_boundary_checks_passed": bool(boundary.get("passed")),
        "task1_output_parse_rate": structure["task1_valid"] / max(1, sum(record["样本编号"].startswith(("seg_", "region_")) for record in records)),
        "task2_output_structure_rate": structure["task2_valid"] / max(1, sum(record["样本编号"].startswith("retrieval_") for record in records)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    cache_dir = args.cache_dir.resolve()
    m3_dir = args.work_dir.resolve() / "m3"
    m4_dir = args.work_dir.resolve() / "m4"
    m2_dir = args.work_dir.resolve() / "m2"
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse:
        m3_manifest = json.loads((m3_dir / "manifest.json").read_text(encoding="utf-8"))
        m4_manifest = json.loads((m4_dir / "manifest.json").read_text(encoding="utf-8"))
    else:
        m3_manifest = build_m3(cache_dir, m3_dir)
        m4_manifest = build_m4(cache_dir, m4_dir, backend=args.backend, model_name=args.model_name)

    expected_ids = _official_ids(cache_dir)
    submission_file = output_dir / "traj_result.jsonl"
    merge = _merge_records(m4_dir, m3_dir, submission_file, expected_ids)
    records = list(read_jsonl(submission_file))
    validation = validate_submission_records(records, expected_ids, task1_field_name="选项")
    structure = _structure_checks(records)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now().astimezone().isoformat().replace("+00:00", "Z"),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "backend": args.backend,
        "reuse_caches": args.reuse,
        "stages": {"m3": m3_manifest, "m4": m4_manifest},
        "merge": merge,
        "validation": validation,
        "structure_checks": structure,
        "synthetic_metrics": _synthetic_metrics(m2_dir, records),
        "artifacts": {
            "submission": str(submission_file),
            "submission_sha256": _sha256(submission_file),
        },
    }
    (output_dir / "m5_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "m1")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "m5")
    parser.add_argument("--backend", choices=("deterministic", "qwen3"), default="deterministic")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--reuse", action="store_true", help="Reuse existing M3/M4 manifests and outputs")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

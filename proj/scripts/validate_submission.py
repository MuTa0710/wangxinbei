#!/usr/bin/env python3
"""Validate a B01 traj_result.jsonl against the stage1 sample IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from b01.io.jsonl import read_jsonl  # noqa: E402
from b01.submission.schema import validate_submission_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--task1-field", default="选项")
    args = parser.parse_args()

    segment = next(args.data_root.rglob("task1/01_segment_questions.jsonl"))
    release_root = segment.parent.parent
    expected_ids = [
        sample["样本编号"]
        for relative in (
            "task1/01_segment_questions.jsonl",
            "task1/02_region_questions.jsonl",
            "task2/01_retrieval_questions.jsonl",
        )
        for sample in read_jsonl(release_root / relative)
    ]
    records = list(read_jsonl(args.submission))
    report = validate_submission_records(records, expected_ids, task1_field_name=args.task1_field)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the M1 normalized trajectory and region-event caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from b01.data.normalize import normalize_trajectory  # noqa: E402
from b01.geometry.regions import region_event_summary  # noqa: E402
from b01.io.jsonl import read_jsonl, write_jsonl  # noqa: E402


SCHEMA_VERSION = "m1.0"


def find_release_root(data_root: Path) -> Path:
    if (data_root / "task1" / "01_segment_questions.jsonl").exists():
        return data_root
    matches = list(data_root.rglob("task1/01_segment_questions.jsonl"))
    if not matches:
        raise FileNotFoundError(f"Cannot find stage1 release under {data_root}")
    return matches[0].parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def task1_trajectory(sample: dict[str, Any]) -> tuple[str, str | None, list[dict[str, Any]]]:
    data = sample["航迹数据"]
    return data["航迹编号"], data.get("起始时间"), data["点位序列"]


def build_task1_cache(release_root: Path, output_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    segment_rows: list[dict[str, Any]] = []
    segment_path = release_root / "task1" / "01_segment_questions.jsonl"
    for sample in read_jsonl(segment_path):
        trajectory_id, start_time, points = task1_trajectory(sample)
        normalized = normalize_trajectory(trajectory_id, start_time, points, task="task1")
        segment_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample["样本编号"],
                "trajectory": normalized,
                "text_instruction": sample["文本指令"],
            }
        )
    counts["task1_segment"] = write_jsonl(output_dir / "task1_segment_normalized.jsonl", segment_rows)

    region_rows: list[dict[str, Any]] = []
    region_path = release_root / "task1" / "02_region_questions.jsonl"
    for sample in read_jsonl(region_path):
        trajectory_id, start_time, points = task1_trajectory(sample)
        normalized = normalize_trajectory(trajectory_id, start_time, points, task="task1")
        events = [region_event_summary(normalized, region) for region in sample.get("区域", [])]
        region_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample["样本编号"],
                "trajectory": normalized,
                "regions": sample.get("区域", []),
                "region_events": events,
                "text_instruction": sample["文本指令"],
            }
        )
    counts["task1_region"] = write_jsonl(output_dir / "task1_region_normalized.jsonl", region_rows)
    return counts


def load_task2_tracks(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"航迹编号", "时间戳", "经度", "纬度", "高度", "速度"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing task2 columns: {sorted(missing)}")
        for row in reader:
            grouped[row["航迹编号"]].append(row)
    return grouped


def build_task2_cache(release_root: Path, output_dir: Path) -> dict[str, int]:
    track_path = release_root / "task2" / "02_tracks.csv"
    grouped = load_task2_tracks(track_path)
    normalized_rows: list[dict[str, Any]] = []
    fact_rows: list[dict[str, Any]] = []
    for trajectory_id in sorted(grouped):
        normalized = normalize_trajectory(trajectory_id, None, grouped[trajectory_id], task="task2")
        normalized_rows.append(normalized)
        summary = normalized["summary"]
        fact_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "trajectory_id": trajectory_id,
                "start_time_utc": normalized["start_time_utc"],
                "end_time_utc": summary.get("end_time_utc"),
                "summary": summary,
                "quality": normalized["summary"]["quality"],
            }
        )
    query_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample["样本编号"],
            "regions": sample.get("区域", []),
            "text_query": sample["文本查询"],
            "top_k": int(sample["top_k"]),
        }
        for sample in read_jsonl(release_root / "task2" / "01_retrieval_questions.jsonl")
    ]
    counts = {
        "task2_tracks": write_jsonl(output_dir / "tracks_normalized.jsonl", normalized_rows),
        "task2_facts": write_jsonl(output_dir / "track_facts.jsonl", fact_rows),
        "task2_queries": write_jsonl(output_dir / "task2_queries_normalized.jsonl", query_rows),
    }
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "m1")
    args = parser.parse_args()
    release_root = find_release_root(args.data_root)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = [
        release_root / "README.md",
        release_root / "task1" / "01_segment_questions.jsonl",
        release_root / "task1" / "02_region_questions.jsonl",
        release_root / "task2" / "01_retrieval_questions.jsonl",
        release_root / "task2" / "02_tracks.csv",
    ]
    missing = [str(path) for path in source_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing stage1 files: {missing}")

    counts = {}
    counts.update(build_task1_cache(release_root, output_dir))
    counts.update(build_task2_cache(release_root, output_dir))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now().astimezone().isoformat().replace("+00:00", "Z"),
        "source_root": str(release_root),
        "source_sha256": {str(path.relative_to(release_root)): sha256(path) for path in source_files},
        "counts": counts,
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

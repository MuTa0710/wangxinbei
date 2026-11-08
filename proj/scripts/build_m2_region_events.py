#!/usr/bin/env python3
"""Build the M2 region-event cache and run deterministic boundary checks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from b01.data.normalize import normalize_trajectory  # noqa: E402
from b01.geometry.regions import region_event_summary, summarize_regions  # noqa: E402
from b01.io.jsonl import read_jsonl, write_jsonl  # noqa: E402


SCHEMA_VERSION = "m2.0"


def _square() -> dict[str, Any]:
    return {
        "区域编号": "test",
        "类型": "多边形",
        "坐标": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
    }


def _trajectory(raw_points: list[dict[str, Any]]) -> dict[str, Any]:
    return normalize_trajectory(
        "traj_m2_test",
        "2025-01-01T00:00:00Z",
        raw_points,
        task="task1",
    )


def _point(timestamp: float, longitude: float, latitude: float = 5.0) -> dict[str, Any]:
    return {
        "时间戳": timestamp,
        "经度": longitude,
        "纬度": latitude,
        "高度": 1000.0,
        "速度": 100.0,
        "航向角": 90.0,
    }


def run_boundary_checks() -> dict[str, Any]:
    region = _square()
    cases = {
        "crossing": [_point(0.0, -1.0), _point(10.0, 11.0)],
        "starts_inside": [_point(0.0, 5.0), _point(10.0, 11.0)],
        "ends_inside": [_point(0.0, -1.0), _point(10.0, 5.0)],
        "reentry": [_point(0.0, -1.0), _point(10.0, 5.0), _point(20.0, 11.0), _point(30.0, 5.0), _point(40.0, 11.0)],
    }
    summaries = {
        name: region_event_summary(_trajectory(points), region)
        for name, points in cases.items()
    }
    checks = {
        "crossing": (
            len(summaries["crossing"]["entry_events"]) == 1
            and len(summaries["crossing"]["exit_events"]) == 1
            and summaries["crossing"]["outside_to_inside_count"] == 1
        ),
        "starts_inside": (
            summaries["starts_inside"]["start_inside"]
            and len(summaries["starts_inside"]["entry_events"]) == 0
            and len(summaries["starts_inside"]["exit_events"]) == 1
        ),
        "ends_inside": (
            summaries["ends_inside"]["end_inside"]
            and len(summaries["ends_inside"]["entry_events"]) == 1
            and len(summaries["ends_inside"]["exit_events"]) == 0
            and summaries["ends_inside"]["dwell_intervals"][-1]["open_end"]
        ),
        "reentry": (
            summaries["reentry"]["outside_to_inside_count"] == 2
            and len(summaries["reentry"]["exit_events"]) == 2
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summaries": summaries,
    }


def build_cache(cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    source = cache_dir / "task1_region_normalized.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"M1 cache not found: {source}")
    rows: list[dict[str, Any]] = []
    total_regions = total_entries = total_exits = visited_regions = open_intervals = 0
    for row in read_jsonl(source):
        trajectory = row["trajectory"]
        regions = row.get("regions", [])
        result = summarize_regions(trajectory, regions)
        summaries = result["regions"]
        total_regions += len(summaries)
        total_entries += sum(len(summary["entry_events"]) for summary in summaries)
        total_exits += sum(len(summary["exit_events"]) for summary in summaries)
        visited_regions += sum(bool(summary["entry_events"] or summary["start_inside"]) for summary in summaries)
        open_intervals += sum(
            sum(interval["open_end"] for interval in summary["dwell_intervals"])
            for summary in summaries
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": row["sample_id"],
                "trajectory_id": trajectory["trajectory_id"],
                "start_time_utc": trajectory.get("start_time_utc"),
                "region_summaries": summaries,
                "entry_sequence": result["entry_sequence"],
                "text_instruction": row.get("text_instruction", ""),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "task1_region_events.jsonl"
    count = write_jsonl(output_file, rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now().astimezone().isoformat().replace("+00:00", "Z"),
        "source": str(source),
        "outputs": [output_file.name],
        "counts": {
            "samples": count,
            "regions": total_regions,
            "visited_regions": visited_regions,
            "entry_events": total_entries,
            "exit_events": total_exits,
            "open_dwell_intervals": open_intervals,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "m1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "m2")
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    validation = {"passed": True, "skipped": args.skip_checks}
    if not args.skip_checks:
        validation = run_boundary_checks()
        (args.output / "boundary_validation.json").parent.mkdir(parents=True, exist_ok=True)
        (args.output / "boundary_validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not validation["passed"]:
            print(json.dumps(validation, ensure_ascii=False, indent=2))
            return 1

    manifest = build_cache(args.cache_dir.resolve(), args.output.resolve())
    manifest["boundary_validation"] = {"passed": validation["passed"], "skipped": validation.get("skipped", False)}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

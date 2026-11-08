#!/usr/bin/env python3
"""Build the M3 structured Task2 retrieval baseline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from b01.geometry.regions import _polygon, region_event_summary  # noqa: E402
from b01.io.jsonl import read_jsonl, write_jsonl  # noqa: E402
from b01.retrieval.baseline import parse_query, rank_candidates, trajectory_features  # noqa: E402


SCHEMA_VERSION = "m3.0"


def _bbox_intersects(summary: dict[str, Any], region: dict[str, Any]) -> bool:
    geometry = _polygon(region)
    min_x, min_y, max_x, max_y = geometry.bounds
    bbox = summary["bbox"]
    return not (
        bbox["max_lon"] < min_x or bbox["min_lon"] > max_x
        or bbox["max_lat"] < min_y or bbox["min_lat"] > max_y
    )


def _region_hit(trajectory: dict[str, Any], region: dict[str, Any]) -> bool:
    if not _bbox_intersects(trajectory["summary"], region):
        return False
    summary = region_event_summary(trajectory, region)
    return bool(summary["start_inside"] or summary["end_inside"] or summary["entry_events"] or summary["exit_events"])


def _load_inputs(cache_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    facts = list(read_jsonl(cache_dir / "track_facts.jsonl"))
    tracks = list(read_jsonl(cache_dir / "tracks_normalized.jsonl"))
    queries = list(read_jsonl(cache_dir / "task2_queries_normalized.jsonl"))
    if len(facts) != len(tracks):
        raise ValueError(f"Facts/tracks count mismatch: {len(facts)} != {len(tracks)}")
    return facts, tracks, queries


def build(cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    facts, tracks, queries = _load_inputs(cache_dir)
    facts_by_id = {row["trajectory_id"]: row for row in facts}
    tracks_by_id = {row["trajectory_id"]: row for row in tracks}
    region_by_id: dict[str, dict[str, Any]] = {}
    for query in queries:
        for region in query.get("regions", []):
            region_by_id[str(region["区域编号"])] = region
    region_ids = sorted(region_by_id)

    feature_by_id = {
        trajectory_id: trajectory_features(track)
        for trajectory_id, track in tracks_by_id.items()
    }
    region_hits_by_id: dict[str, dict[str, bool]] = {}
    for index, (trajectory_id, trajectory) in enumerate(tracks_by_id.items(), start=1):
        region_hits_by_id[trajectory_id] = {
            region_id: _region_hit(trajectory, region_by_id[region_id])
            for region_id in region_ids
        }
        if index % 250 == 0:
            print(f"region hit cache: {index}/{len(tracks_by_id)}", file=sys.stderr)

    parsed_queries: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for sample in queries:
        parsed = parse_query(sample)
        parsed_queries.append({"schema_version": SCHEMA_VERSION, **parsed})
        results, diagnostic = rank_candidates(facts, feature_by_id, parsed, region_hits_by_id)
        result_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "样本编号": parsed["sample_id"],
                "结果": results,
            }
        )
        diagnostics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": parsed["sample_id"],
                "candidate_count": diagnostic["candidate_count"],
                "rejected_count": diagnostic["rejected_count"],
                "top_k": parsed["top_k"],
                "ranked_candidates": diagnostic["ranked_candidates"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "query_ir.jsonl": write_jsonl(output_dir / "query_ir.jsonl", parsed_queries),
        "task2_baseline_results.jsonl": write_jsonl(output_dir / "task2_baseline_results.jsonl", result_rows),
        "retrieval_diagnostics.jsonl": write_jsonl(output_dir / "retrieval_diagnostics.jsonl", diagnostics),
        "region_hits.jsonl": write_jsonl(
            output_dir / "region_hits.jsonl",
            [
                {"schema_version": SCHEMA_VERSION, "trajectory_id": trajectory_id, "region_hits": hits}
                for trajectory_id, hits in sorted(region_hits_by_id.items())
            ],
        ),
    }
    empty_results = sum(not row["结果"] for row in result_rows)
    short_results = sum(len(row["结果"]) < min(int(query["top_k"]), len(facts)) for row, query in zip(result_rows, queries))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now().astimezone().isoformat().replace("+00:00", "Z"),
        "inputs": {"tracks": len(tracks), "facts": len(facts), "queries": len(queries), "regions": region_ids},
        "outputs": files,
        "results": {"empty": empty_results, "shorter_than_top_k": short_results},
        "notes": [
            "Numeric/time/region conditions are hard filters.",
            "Behavior phrases are matched by deterministic trajectory features and used for ranking.",
            "This is an inspectable baseline; no language model or external service is called.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "m1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "m3")
    args = parser.parse_args()
    manifest = build(args.cache_dir.resolve(), args.output.resolve())
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

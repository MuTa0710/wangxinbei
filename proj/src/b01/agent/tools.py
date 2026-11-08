"""Deterministic tools exposed to the M6 Agent controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from b01.geometry.regions import _polygon, region_event_summary, summarize_regions
from b01.io.jsonl import read_jsonl
from b01.retrieval.baseline import rank_candidates, trajectory_features
from b01.task1.baseline import choose_region, choose_segment, trajectory_fact_summary


class B01Tools:
    """A small, fixed tool registry backed by the M2/M3 implementations."""

    def __init__(self, cache_dir: Path, m3_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir
        self.m3_dir = m3_dir
        self.segment_rows = list(read_jsonl(cache_dir / "task1_segment_normalized.jsonl"))
        self.region_rows = list(read_jsonl(cache_dir / "task1_region_normalized.jsonl"))
        self.facts = list(read_jsonl(cache_dir / "track_facts.jsonl"))
        self.tracks = list(read_jsonl(cache_dir / "tracks_normalized.jsonl"))
        self.queries = list(read_jsonl(cache_dir / "task2_queries_normalized.jsonl"))
        self.facts_by_id = {row["trajectory_id"]: row for row in self.facts}
        self.tracks_by_id = {row["trajectory_id"]: row for row in self.tracks}
        self.feature_by_id = {
            trajectory_id: trajectory_features(track)
            for trajectory_id, track in self.tracks_by_id.items()
        }
        self.region_by_id: dict[str, dict[str, Any]] = {}
        for query in self.queries:
            for region in query.get("regions", []):
                region_id = str(region.get("区域编号", region.get("region_id", "")))
                self.region_by_id[region_id] = region
        self.region_hits_by_id = self._build_region_hits()

    def _build_region_hits(self) -> dict[str, dict[str, bool]]:
        # Reuse the M3 cache when available. It is generated deterministically
        # and avoids recalculating polygon intersections for every M6 run.
        if self.m3_dir is not None:
            cache_file = self.m3_dir / "region_hits.jsonl"
            if cache_file.exists():
                rows = list(read_jsonl(cache_file))
                if rows:
                    return {row["trajectory_id"]: row["region_hits"] for row in rows}
        region_ids = sorted(self.region_by_id)
        return {
            trajectory_id: {
                region_id: self._region_hit(track, self.region_by_id[region_id])
                for region_id in region_ids
            }
            for trajectory_id, track in self.tracks_by_id.items()
        }

    @staticmethod
    def _region_hit(trajectory: dict[str, Any], region: dict[str, Any]) -> bool:
        geometry = _polygon(region)
        min_x, min_y, max_x, max_y = geometry.bounds
        bbox = trajectory["summary"]["bbox"]
        if (
            bbox["max_lon"] < min_x
            or bbox["min_lon"] > max_x
            or bbox["max_lat"] < min_y
            or bbox["min_lat"] > max_y
        ):
            return False
        summary = region_event_summary(trajectory, region)
        return bool(summary["start_inside"] or summary["end_inside"] or summary["entry_events"] or summary["exit_events"])

    def task1_segment_facts(self, sample: dict[str, Any]) -> dict[str, Any]:
        return trajectory_fact_summary(sample["trajectory"])

    def task1_region_facts(self, sample: dict[str, Any]) -> dict[str, Any]:
        trajectory = sample["trajectory"]
        return summarize_regions(trajectory, sample.get("regions", []))

    def task1_segment_fallback(self, sample: dict[str, Any]) -> tuple[str, dict[str, float]]:
        return choose_segment(sample["trajectory"], sample["text_instruction"])

    def task1_region_fallback(self, sample: dict[str, Any]) -> tuple[str, dict[str, float], dict[str, Any]]:
        return choose_region(sample["trajectory"], sample.get("regions", []), sample["text_instruction"])

    def task2_retrieve(self, parsed_query: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return rank_candidates(
            self.facts,
            self.feature_by_id,
            parsed_query,
            self.region_hits_by_id,
        )

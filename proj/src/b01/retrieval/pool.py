"""M8 rule-side candidate pool: similarity over the 2,000-track library.

This module is intentionally independent from the M8 model runtime: it can build
the pool and render candidate cards without loading a model, so the pipeline can
be tested with ``--backend rule`` first.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from b01.io.jsonl import read_jsonl, write_jsonl
from b01.retrieval.baseline import behavior_score, hard_filter, trajectory_features


DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")

# Weights for the similarity terms.  Only constraints present in the query are
# activated; the final score is a weighted average of the active terms.
WEIGHTS = {
    "time": 2.0,
    "altitude": 1.5,
    "speed": 1.5,
    "region": 2.5,
    "behavior": 2.0,
}

SAME_DATE_CREDIT = 0.8
QUALITY_PENALTY = 0.05


def _parse_dt(value: str) -> datetime:
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def query_date(text: str) -> datetime | None:
    match = DATE_RE.search(text or "")
    if not match:
        return None
    return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=timezone.utc)


def _closeness(value: float, low: float, high: float, width: float) -> float:
    """1.0 inside [low, high]; linear decay outside with a floor of 0."""

    if low <= value <= high:
        return 1.0
    margin = (low - value) if value < low else (value - high)
    return max(0.0, 1.0 - margin / max(width, 1.0))


def _delta_label(seconds: float) -> str:
    sign = "+" if seconds >= 0 else "-"
    value = abs(seconds)
    if value < 3600:
        return f"{sign}{value:.0f}s"
    return f"{sign}{value / 3600.0:.1f}h"


def _format_start(value: str) -> str:
    parsed = _parse_dt(value)
    if parsed.second or parsed.microsecond:
        return parsed.strftime("%m-%d %H:%M:%S")
    return parsed.strftime("%m-%d %H:%M")


def _short_region(region_id: str) -> str:
    return str(region_id).replace("区域", "").strip()


def _active_terms(ir: dict[str, Any]) -> list[str]:
    active: list[str] = []
    if ir.get("time_start_utc") and ir.get("time_end_utc"):
        active.append("time")
    if ir.get("altitude_range_m"):
        active.append("altitude")
    if ir.get("speed_range_mps"):
        active.append("speed")
    if ir.get("positive_regions"):
        active.append("region")
    if ir.get("behavior_sequence"):
        active.append("behavior")
    return active


def _time_score(
    start: datetime,
    ir: dict[str, Any],
    qdate: datetime | None,
) -> tuple[float, bool, str | None]:
    """Return (score, same_date_batch, fail_label)."""

    low_raw, high_raw = ir.get("time_start_utc"), ir.get("time_end_utc")
    if not low_raw or not high_raw:
        return 0.5, False, None
    low, high = _parse_dt(low_raw), _parse_dt(high_raw)
    if low <= start <= high:
        return 1.0, False, None
    if qdate is not None and start.date() == qdate.date():
        return SAME_DATE_CREDIT, True, _delta_label((start - high).total_seconds())
    if qdate is not None:
        return 0.0, False, "other_date"
    hours_out = min(abs((start - low).total_seconds()), abs((start - high).total_seconds())) / 3600.0
    return max(0.0, 1.0 - hours_out / 24.0), False, _delta_label((start - high).total_seconds())


def _score_track(
    fact: dict[str, Any],
    feature: dict[str, Any],
    region_hits: dict[str, bool],
    ir: dict[str, Any],
    qdate: datetime | None,
) -> tuple[float | None, dict[str, Any], int]:
    """Score one track.  ``None`` means excluded by a hard veto (negative region)."""

    negative = ir.get("negative_regions") or []
    if any(region_hits.get(region_id, False) for region_id in negative):
        return None, {}, 0

    terms: list[tuple[str, float]] = []
    fail: dict[str, Any] = {}

    start = _parse_dt(fact["start_time_utc"])
    time_score, same_date_batch, time_fail = _time_score(start, ir, qdate)
    if ir.get("time_start_utc") and ir.get("time_end_utc"):
        terms.append(("time", time_score))
        if time_fail:
            fail["time"] = time_fail

    if ir.get("altitude_range_m"):
        low, high = [float(v) for v in ir["altitude_range_m"]]
        value = float(feature["altitude_median_m"])
        terms.append(("altitude", _closeness(value, low, high, max(high - low, 500.0))))
        if value < low or value > high:
            fail["alt"] = f"{value - min(max(value, low), high):+.0f}m"

    if ir.get("speed_range_mps"):
        low, high = [float(v) for v in ir["speed_range_mps"]]
        value = float(feature["speed_median_mps"])
        terms.append(("speed", _closeness(value, low, high, max(high - low, 20.0))))
        if value < low or value > high:
            fail["spd"] = f"{value - min(max(value, low), high):+.1f}m/s"

    positive = ir.get("positive_regions") or []
    if positive:
        hit_count = sum(1 for region_id in positive if region_hits.get(region_id, False))
        terms.append(("region", hit_count / len(positive)))
        missing = [region_id for region_id in positive if not region_hits.get(region_id, False)]
        if missing:
            fail["reg"] = [_short_region(region_id) for region_id in missing]

    behavior = ir.get("behavior_sequence") or []
    if behavior:
        raw_score, matched = behavior_score(feature, ir)
        terms.append(("behavior", min(1.0, float(raw_score))))
        unmatched = [item["name"] for item in matched if not item.get("matched")]
        if unmatched:
            fail["beh"] = unmatched

    if not terms:
        score = 0.5
    else:
        total_weight = sum(WEIGHTS[name] for name, _ in terms)
        score = sum(WEIGHTS[name] * value for name, value in terms) / total_weight

    if fact.get("quality", {}).get("has_quality_issue"):
        score -= QUALITY_PENALTY

    return max(0.0, min(1.0, score)), fail, 1 if same_date_batch else 0


def _card(
    rank: int,
    candidate: dict[str, Any],
    fact: dict[str, Any],
    feature: dict[str, Any],
    region_hits: dict[str, bool],
    query_regions: Iterable[str],
) -> dict[str, Any]:
    deltas_alt = feature.get("altitude_window_deltas_m") or [0.0]
    deltas_speed = feature.get("speed_window_deltas_mps") or [0.0]
    regions = {
        _short_region(region_id): 1 if region_hits.get(region_id, False) else 0
        for region_id in query_regions
    }
    return {
        "id": candidate["trajectory_id"],
        "start": _format_start(fact["start_time_utc"]),
        "dur": int(round(float(fact["summary"].get("duration_seconds") or 0.0))),
        "alt": int(round(float(feature["altitude_median_m"]))),
        "spd": round(float(feature["speed_median_mps"]), 1),
        "spd_std": round(float(feature["speed_std_mps"]), 1),
        "dAlt": f"{round(max(deltas_alt)):+d}/{round(min(deltas_alt)):+d}",
        "dSpd": f"{max(deltas_speed):+.1f}/{min(deltas_speed):+.1f}",
        "turn": int(round(float(feature["total_turn_deg"]))),
        "circ": round(float(feature["circularity"]), 1),
        "reg": regions,
        "fail": candidate["fail"],
        "db": candidate["db"],
        "q": 1 if fact.get("quality", {}).get("has_quality_issue") else 0,
    }


@dataclass
class Pool:
    sample_id: str
    top_k: int
    target_size: int
    candidates: list[dict[str, Any]]
    cards: list[dict[str, Any]]

    @property
    def ids(self) -> list[str]:
        return [candidate["trajectory_id"] for candidate in self.candidates]

    def to_record(self, limit: int | None = None) -> dict[str, Any]:
        ids = self.ids[: limit if limit is not None else len(self.ids)]
        return {
            "样本编号": self.sample_id,
            "结果": [{"序号": index, "航迹编号": trajectory_id} for index, trajectory_id in enumerate(ids, start=1)],
        }


def build_pool(
    sample: dict[str, Any],
    ir: dict[str, Any],
    facts: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    region_hits: dict[str, dict[str, bool]],
    *,
    multiplier: float = 2.0,
) -> Pool:
    sample_id = str(sample.get("sample_id", sample.get("样本编号")))
    top_k = int(sample.get("top_k", 10))
    target_size = max(1, int(round(multiplier * top_k)))
    facts_by_id = {fact["trajectory_id"]: fact for fact in facts}
    qdate = query_date(str(sample.get("text_query", sample.get("文本查询", ""))))
    query_regions = list(ir.get("positive_regions") or []) + list(ir.get("negative_regions") or [])

    scored: list[dict[str, Any]] = []
    strict_ids: set[str] = set()
    for fact in facts:
        trajectory_id = fact["trajectory_id"]
        feature = features[trajectory_id]
        hits = region_hits.get(trajectory_id, {})
        score, fail, same_date_batch = _score_track(fact, feature, hits, ir, qdate)
        if score is None:
            continue
        passed, _ = hard_filter(fact, ir, hits, feature)
        if passed:
            strict_ids.add(trajectory_id)
        scored.append(
            {
                "trajectory_id": trajectory_id,
                "score": score,
                "fail": fail,
                "db": same_date_batch,
                "strict": passed,
            }
        )

    scored.sort(key=lambda item: (-item["score"], item["trajectory_id"]))
    selected = scored[:target_size]

    # If hard-pass candidates fit inside the pool, make sure none of them is
    # missing (replace the lowest-scoring relaxed entry).
    if len(strict_ids) <= target_size:
        selected_ids = {item["trajectory_id"] for item in selected}
        missing = [item for item in scored if item["strict"] and item["trajectory_id"] not in selected_ids]
        if missing:
            replaceable = [
                index for index in range(len(selected) - 1, -1, -1)
                if not selected[index]["strict"]
            ]
            for candidate, index in zip(missing, replaceable):
                selected[index] = candidate
            selected.sort(key=lambda item: (-item["score"], item["trajectory_id"]))

    cards = [
        _card(index, candidate, facts_by_id[candidate["trajectory_id"]], features[candidate["trajectory_id"]],
              region_hits.get(candidate["trajectory_id"], {}), query_regions)
        for index, candidate in enumerate(selected, start=1)
    ]
    return Pool(sample_id=sample_id, top_k=top_k, target_size=target_size, candidates=selected, cards=cards)


def load_resources(cache_dir: Path, m3_dir: Path, features_cache: Path) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, bool]], dict[str, dict[str, Any]]
]:
    facts = list(read_jsonl(Path(cache_dir) / "track_facts.jsonl"))
    facts_by_id = {fact["trajectory_id"]: fact for fact in facts}

    if Path(features_cache).exists():
        features = {row["trajectory_id"]: row for row in read_jsonl(Path(features_cache))}
    else:
        features = {}
        for track in read_jsonl(Path(cache_dir) / "tracks_normalized.jsonl"):
            features[track["trajectory_id"]] = trajectory_features(track)
        Path(features_cache).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            Path(features_cache),
            [{"trajectory_id": trajectory_id, **feature} for trajectory_id, feature in sorted(features.items())],
        )

    region_hits = {
        row["trajectory_id"]: row["region_hits"] for row in read_jsonl(Path(m3_dir) / "region_hits.jsonl")
    }
    queries = {str(row.get("sample_id", row.get("样本编号"))): row for row in read_jsonl(Path(cache_dir) / "task2_queries_normalized.jsonl")}
    return facts, features, region_hits, queries

"""Deterministic Task2 query parsing, filtering and ranking for M3.

This module intentionally keeps the first retrieval baseline inspectable. It
does not call a language model: numeric/time/region constraints are parsed into
an intermediate representation, while behavior phrases are matched against
features calculated from the normalized trajectory.
"""

from __future__ import annotations

import math
import re
import statistics
from datetime import datetime, time, timedelta, timezone
from typing import Any, Iterable

from b01.geometry.regions import region_event_summary


TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")
DATE_PATTERN = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
RANGE_PATTERN = re.compile(
    r"(高度|速度)在\s*([-+]?\d+(?:\.\d+)?)\s*(?:m/s|m)?\s*至\s*([-+]?\d+(?:\.\d+)?)\s*(?:m/s|m)?\s*(?:之间|范围内)?"
)

PERIODS: dict[str, tuple[int, int]] = {
    "凌晨": (0, 6),
    "早晨": (6, 9),
    "上午": (6, 12),
    "正午": (11, 13),
    "中午": (11, 13),
    "下午": (12, 18),
    "黄昏": (17, 19),
    "傍晚": (17, 19),
    "夜间": (18, 24),
    "深夜": (0, 6),
}

BEHAVIOR_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("large_climb", ("大仰角爬升", "大落差阶梯型上升", "陡峭阶梯型上升")),
    ("climb", ("缓慢爬升", "平缓爬升", "急速爬升", "爬升")),
    ("large_descent", ("大落差阶梯型下降", "陡峭阶梯型下降")),
    ("descent", ("下降", "减速")),
    ("gentle_step_accel", ("柔和阶梯型加速", "平缓阶梯型加速")),
    ("abrupt_step_accel", ("急促阶梯型加速", "顿挫阶梯型加速", "陡峭阶梯型加速")),
    ("long_platform_step_accel", ("长平台阶梯型加速",)),
    ("step_accel", ("阶梯型加速", "加速")),
    ("step_decel", ("阶梯型减速", "减速")),
    ("orbit", ("盘旋绕飞", "绕飞", "盘旋")),
    ("speed_hold", ("速度保持", "精准速度保持")),
    ("cruise", ("精准巡航", "巡航")),
    ("low_speed", ("低速移动",)),
    ("approach", ("进近",)),
    ("periodic_speed", ("微波速度周期波动",)),
)


def _parse_dt(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_clock(match: re.Match[str]) -> int:
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3) or 0)
    return hour * 3600 + minute * 60 + second


def _date_from_query(query: str) -> datetime | None:
    match = DATE_PATTERN.search(query)
    if not match:
        return None
    return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=timezone.utc)


def _time_window(query: str) -> tuple[datetime | None, datetime | None]:
    date = _date_from_query(query)
    if date is None:
        return None, None
    explicit = re.search(
        r"(\d{1,2}:\d{2}(?::\d{2})?)\s*至\s*(\d{1,2}:\d{2}(?::\d{2})?)",
        query,
    )
    if explicit:
        start = _parse_clock(TIME_PATTERN.fullmatch(explicit.group(1)))
        end = _parse_clock(TIME_PATTERN.fullmatch(explicit.group(2)))
        return date + timedelta(seconds=start), date + timedelta(seconds=end)
    periods = [(term, window) for term, window in PERIODS.items() if term in query]
    if periods:
        start_hour, end_hour = periods[0][1]
        return date + timedelta(hours=start_hour), date + timedelta(hours=end_hour)
    clocks = list(TIME_PATTERN.finditer(query))
    if clocks:
        start = _parse_clock(clocks[0])
        return date + timedelta(seconds=start), date + timedelta(seconds=start + 1)
    return date, date + timedelta(days=1)


def _ranges(query: str) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    altitude: tuple[float, float] | None = None
    speed: tuple[float, float] | None = None
    for match in RANGE_PATTERN.finditer(query):
        bounds = (float(match.group(2)), float(match.group(3)))
        if match.group(1) == "高度":
            altitude = bounds
        else:
            speed = bounds
    return altitude, speed


def _region_mentions(query: str, regions: Iterable[dict[str, Any]]) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    negative: list[str] = []
    for region in regions:
        region_id = str(region.get("区域编号", region.get("region_id", "")))
        short = region_id.replace("区域", "").strip()
        # Include both the complete label and its letter for robust matching.
        mentions = (region_id, short)
        positions = [query.find(value) for value in mentions if value and query.find(value) >= 0]
        if not positions:
            continue
        position = min(positions)
        context = query[max(0, position - 12):position]
        if "不满足" in context or "不经过" in context or "未经过" in context or "不满足经过" in context:
            negative.append(region_id)
        else:
            positive.append(region_id)
    return list(dict.fromkeys(positive)), list(dict.fromkeys(negative))


def _behavior_sequence(query: str) -> list[dict[str, Any]]:
    found: list[tuple[int, int, str, str]] = []
    for canonical, terms in BEHAVIOR_TERMS:
        for term in terms:
            position = query.find(term)
            if position >= 0:
                found.append((position, position + len(term), canonical, term))
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    # Remove generic phrases nested inside a longer phrase, such as
    # "加速" inside "长平台阶梯型加速".
    selected: list[tuple[int, int, str, str]] = []
    for item in found:
        position, end, canonical, term = item
        if any(position >= left and end <= right for left, right, _, _ in selected):
            continue
        selected = [existing for existing in selected if not (existing[0] >= position and existing[1] <= end)]
        selected.append(item)
    selected.sort(key=lambda item: item[0])
    result = [
        {"name": canonical, "term": term, "char_position": position}
        for position, _, canonical, term in selected
    ]
    return result


def parse_query(sample: dict[str, Any]) -> dict[str, Any]:
    query = str(sample.get("text_query", sample.get("文本查询", "")))
    start, end = _time_window(query)
    altitude, speed = _ranges(query)
    positive_regions, negative_regions = _region_mentions(query, sample.get("regions", sample.get("区域", [])))
    return {
        "sample_id": sample.get("sample_id", sample.get("样本编号")),
        "text_query": query,
        "top_k": int(sample.get("top_k", 10)),
        "time_start_utc": start.isoformat().replace("+00:00", "Z") if start else None,
        "time_end_utc": end.isoformat().replace("+00:00", "Z") if end else None,
        "altitude_range_m": altitude,
        "speed_range_mps": speed,
        "positive_regions": positive_regions,
        "negative_regions": negative_regions,
        "behavior_sequence": _behavior_sequence(query),
        "flags": {
            "low_speed": "低速" in query,
            "very_high_altitude": "超高空" in query,
            "negation": "不满足" in query or "不经过" in query,
        },
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    middle = (len(values) - 1) / 2.0
    denominator = sum((index - middle) ** 2 for index in range(len(values))) or 1.0
    return sum((index - middle) * (value - _mean(values)) for index, value in enumerate(values)) / denominator


def _window(values: list[float], index: int, count: int = 8) -> list[float]:
    if not values:
        return []
    left = index * len(values) // count
    right = (index + 1) * len(values) // count
    return values[left:max(left + 1, right)]


def trajectory_features(trajectory: dict[str, Any]) -> dict[str, Any]:
    points = trajectory.get("points", [])
    altitudes = [float(point["altitude_m"]) for point in points]
    speeds = [float(point["speed_mps"]) for point in points]
    alt_slope = [_slope([float(x["altitude_m"]) for x in _window(points, i)]) for i in range(8)]
    speed_slope = [_slope([float(x["speed_mps"]) for x in _window(points, i)]) for i in range(8)]
    alt_window_mean = [_mean([float(x["altitude_m"]) for x in _window(points, i)]) for i in range(8)]
    speed_window_mean = [_mean([float(x["speed_mps"]) for x in _window(points, i)]) for i in range(8)]
    deltas_alt = [alt_window_mean[i + 1] - alt_window_mean[i] for i in range(7)]
    deltas_speed = [speed_window_mean[i + 1] - speed_window_mean[i] for i in range(7)]
    bearings = [float(point["bearing_derived_deg"]) for point in points if point.get("bearing_derived_deg") is not None]
    turn_deltas = [((b - a + 180.0) % 360.0) - 180.0 for a, b in zip(bearings, bearings[1:])]
    total_turn = sum(abs(value) for value in turn_deltas)
    path_distance = sum(float(point.get("distance_m") or 0.0) for point in points)
    if len(points) >= 2:
        lon_delta = float(points[-1]["longitude"]) - float(points[0]["longitude"])
        lat_delta = float(points[-1]["latitude"]) - float(points[0]["latitude"])
        straight_distance = math.hypot(lon_delta, lat_delta) * 111_000.0
    else:
        straight_distance = 0.0
    circularity = path_distance / max(straight_distance, 1.0)
    speed_std = statistics.pstdev(speeds) if len(speeds) > 1 else 0.0
    alt_std = statistics.pstdev(altitudes) if len(altitudes) > 1 else 0.0
    sign_changes = sum(
        1 for left, right in zip(deltas_speed, deltas_speed[1:])
        if left * right < 0 and abs(left) > 1.0 and abs(right) > 1.0
    )
    behavior_flags = {
        "climb": max(deltas_alt, default=0.0) > 100.0,
        "large_climb": max(deltas_alt, default=0.0) > 500.0,
        "descent": min(deltas_alt, default=0.0) < -100.0,
        "large_descent": min(deltas_alt, default=0.0) < -500.0,
        "step_accel": max(deltas_speed, default=0.0) > 5.0,
        "gentle_step_accel": 2.0 < max(deltas_speed, default=0.0) <= 10.0,
        "abrupt_step_accel": max(deltas_speed, default=0.0) > 10.0,
        "long_platform_step_accel": max(deltas_speed, default=0.0) > 5.0 and speed_std < 80.0,
        "step_decel": min(deltas_speed, default=0.0) < -5.0,
        "orbit": circularity >= 2.0 or (total_turn >= 1000.0 and path_distance > 20_000.0),
        "speed_hold": (max(speeds, default=0.0) - min(speeds, default=0.0)) <= 3.0 or speed_std <= 1.5,
        "cruise": speed_std <= 8.0 and alt_std <= 80.0,
        "low_speed": _mean(speeds) <= 80.0,
        "approach": min(deltas_alt, default=0.0) < -100.0 and min(deltas_speed, default=0.0) < -2.0,
        "periodic_speed": sign_changes >= 3 and speed_std >= 3.0,
    }
    # A coarse stage sequence supports ordered behavior matching.
    stages: list[str] = []
    for alt_delta, speed_delta in zip(deltas_alt, deltas_speed):
        if alt_delta > 100:
            stages.append("climb")
        elif alt_delta < -100:
            stages.append("descent")
        elif speed_delta > 3:
            stages.append("accel")
        elif speed_delta < -3:
            stages.append("decel")
        else:
            stages.append("stable")
    return {
        "point_count": len(points),
        "altitude_mean_m": _mean(altitudes),
        "altitude_min_m": min(altitudes, default=0.0),
        "altitude_max_m": max(altitudes, default=0.0),
        "speed_mean_mps": _mean(speeds),
        "speed_median_mps": statistics.median(speeds) if speeds else 0.0,
        "speed_min_mps": min(speeds, default=0.0),
        "speed_max_mps": max(speeds, default=0.0),
        "altitude_median_m": statistics.median(altitudes) if altitudes else 0.0,
        "speed_std_mps": speed_std,
        "altitude_std_m": alt_std,
        "altitude_window_means_m": alt_window_mean,
        "speed_window_means_mps": speed_window_mean,
        "altitude_window_deltas_m": deltas_alt,
        "speed_window_deltas_mps": deltas_speed,
        "total_turn_deg": total_turn,
        "circularity": circularity,
        "behavior_flags": behavior_flags,
        "stages": stages,
    }


def _time_match(fact: dict[str, Any], parsed: dict[str, Any]) -> bool:
    start = _parse_dt(fact["start_time_utc"])
    lower = parsed.get("time_start_utc")
    upper = parsed.get("time_end_utc")
    return (lower is None or start >= _parse_dt(lower)) and (upper is None or start <= _parse_dt(upper))


def _range_match(value: float, bounds: tuple[float, float] | None) -> bool:
    return bounds is None or bounds[0] <= value <= bounds[1]


def hard_filter(
    fact: dict[str, Any], parsed: dict[str, Any], region_hits: dict[str, bool],
    features: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    summary = fact["summary"]
    reasons: list[str] = []
    if not _time_match(fact, parsed):
        reasons.append("time")
    representative_altitude = (features or {}).get("altitude_median_m", summary["altitude_m"]["mean"])
    representative_speed = (features or {}).get("speed_median_mps", summary["speed_mps"]["mean"])
    if parsed.get("altitude_range_m") and not _range_match(representative_altitude, parsed["altitude_range_m"]):
        reasons.append("altitude")
    if parsed.get("speed_range_mps") and not _range_match(representative_speed, parsed["speed_range_mps"]):
        reasons.append("speed")
    for region_id in parsed.get("positive_regions", []):
        if not region_hits.get(region_id, False):
            reasons.append(f"missing:{region_id}")
    for region_id in parsed.get("negative_regions", []):
        if region_hits.get(region_id, False):
            reasons.append(f"forbidden:{region_id}")
    if parsed.get("flags", {}).get("low_speed") and summary["speed_mps"]["mean"] > 100.0:
        reasons.append("low_speed")
    if parsed.get("flags", {}).get("very_high_altitude") and summary["altitude_m"]["mean"] < 8000.0:
        reasons.append("very_high_altitude")
    return not reasons, reasons


def behavior_score(features: dict[str, Any], parsed: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    flags = features["behavior_flags"]
    sequence = parsed.get("behavior_sequence", [])
    if not sequence:
        return 0.0, []
    matched: list[dict[str, Any]] = []
    stage_position = 0
    for behavior in sequence:
        name = behavior["name"]
        matched_flag = bool(flags.get(name, False))
        stage_name = ""
        if name in {"climb", "large_climb"}:
            stage_name = "climb"
        elif name in {"descent", "large_descent"}:
            stage_name = "descent"
        elif name in {"step_accel", "gentle_step_accel", "abrupt_step_accel", "long_platform_step_accel"}:
            stage_name = "accel"
        elif name == "step_decel":
            stage_name = "decel"
        if stage_name:
            positions = [index for index, stage in enumerate(features["stages"]) if stage == stage_name and index >= stage_position]
            if positions:
                stage_position = positions[0] + 1
                matched_flag = True
        matched.append({"name": name, "matched": matched_flag, "term": behavior["term"]})
    score = sum(1.0 if item["matched"] else 0.0 for item in matched) / len(matched)
    if all(item["matched"] for item in matched):
        score += 0.25
    return score, matched


def rank_candidates(
    facts: list[dict[str, Any]], feature_by_id: dict[str, dict[str, Any]], parsed: dict[str, Any],
    region_hits_by_id: dict[str, dict[str, bool]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    rejected = 0
    for fact in facts:
        trajectory_id = fact["trajectory_id"]
        passed, reasons = hard_filter(
            fact, parsed, region_hits_by_id.get(trajectory_id, {}), feature_by_id[trajectory_id]
        )
        if not passed:
            rejected += 1
            continue
        score, matched = behavior_score(feature_by_id[trajectory_id], parsed)
        candidates.append(
            {
                "trajectory_id": trajectory_id,
                "score": score,
                "behavior_match": matched,
                "quality_issue": fact.get("quality", {}).get("has_quality_issue", False),
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["quality_issue"], item["trajectory_id"]))
    top_k = min(int(parsed["top_k"]), len(candidates))
    results = [
        {"序号": index + 1, "航迹编号": item["trajectory_id"]}
        for index, item in enumerate(candidates[:top_k])
    ]
    return results, {
        "candidate_count": len(candidates),
        "rejected_count": rejected,
        "behavior_sequence": parsed.get("behavior_sequence", []),
        "ranked_candidates": candidates[:top_k],
    }


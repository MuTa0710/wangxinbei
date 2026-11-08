"""Deterministic Task1 four-choice baseline for M4.

The region subtask is checked against the M2 geometric facts. The segment
subtask compares option stage phrases with equal-duration trajectory summaries.
The implementation is deliberately inspectable and provides a stable fallback
when Qwen3 weights are not present in the runtime.
"""

from __future__ import annotations

import math
import re
import statistics
from datetime import datetime
from typing import Any, Iterable


OPTION_RE = re.compile(r"(?m)^\s*([ABCD])\s*[\.、]\s*(.+?)(?=\n\s*[ABCD]\s*[\.、]|\Z)", re.S)
TIME_TEXT_RE = re.compile(r"(\d{1,2})时(\d{1,2})分(\d+(?:\.\d+)?)秒")
REGION_HEADER_RE = re.compile(r"区域\s*([A-Za-z])\s*[：:]")


def parse_options(instruction: str) -> dict[str, str]:
    options = {match.group(1): " ".join(match.group(2).split()) for match in OPTION_RE.finditer(instruction)}
    if set(options) != {"A", "B", "C", "D"}:
        raise ValueError(f"Expected A/B/C/D options, got {sorted(options)}")
    return options


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    middle = (len(values) - 1) / 2.0
    denominator = sum((index - middle) ** 2 for index in range(len(values))) or 1.0
    return sum((index - middle) * (value - _mean(values)) for index, value in enumerate(values)) / denominator


def _stage(points: list[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    left = index * len(points) // count
    right = (index + 1) * len(points) // count
    return points[left:max(left + 1, right)]


def _turn_delta(left: float, right: float) -> float:
    return (right - left + 180.0) % 360.0 - 180.0


def stage_summaries(trajectory: dict[str, Any], stage_count: int) -> list[dict[str, Any]]:
    points = trajectory.get("points", [])
    summaries: list[dict[str, Any]] = []
    for index in range(stage_count):
        values = _stage(points, index, stage_count)
        altitudes = [float(point["altitude_m"]) for point in values]
        speeds = [float(point["speed_mps"]) for point in values]
        bearings = [float(point["bearing_derived_deg"]) for point in values if point.get("bearing_derived_deg") is not None]
        turns = [_turn_delta(left, right) for left, right in zip(bearings, bearings[1:])]
        path_distance = sum(float(point.get("distance_m") or 0.0) for point in values)
        if len(values) >= 2:
            lon_delta = float(values[-1]["longitude"]) - float(values[0]["longitude"])
            lat_delta = float(values[-1]["latitude"]) - float(values[0]["latitude"])
            displacement = math.hypot(lon_delta, lat_delta) * 111_000.0
        else:
            displacement = 0.0
        speed_deltas = [right - left for left, right in zip(speeds, speeds[1:])]
        altitude_deltas = [right - left for left, right in zip(altitudes, altitudes[1:])]
        summaries.append(
            {
                "index": index + 1,
                "point_count": len(values),
                "altitude_mean_m": _mean(altitudes),
                "altitude_first_m": altitudes[0] if altitudes else 0.0,
                "altitude_last_m": altitudes[-1] if altitudes else 0.0,
                "altitude_min_m": min(altitudes, default=0.0),
                "altitude_max_m": max(altitudes, default=0.0),
                "altitude_slope": _slope(altitudes),
                "altitude_range_m": max(altitudes, default=0.0) - min(altitudes, default=0.0),
                "altitude_sign_changes": sum(
                    1 for left, right in zip(altitude_deltas, altitude_deltas[1:])
                    if left * right < 0 and abs(left) > 1.0 and abs(right) > 1.0
                ),
                "speed_mean_mps": _mean(speeds),
                "speed_first_mps": speeds[0] if speeds else 0.0,
                "speed_last_mps": speeds[-1] if speeds else 0.0,
                "speed_min_mps": min(speeds, default=0.0),
                "speed_max_mps": max(speeds, default=0.0),
                "speed_slope": _slope(speeds),
                "speed_range_mps": max(speeds, default=0.0) - min(speeds, default=0.0),
                "speed_sign_changes": sum(
                    1 for left, right in zip(speed_deltas, speed_deltas[1:])
                    if left * right < 0 and abs(left) > 0.2 and abs(right) > 0.2
                ),
                "speed_std_mps": statistics.pstdev(speeds) if len(speeds) > 1 else 0.0,
                "path_distance_m": path_distance,
                "displacement_m": displacement,
                "directness": displacement / max(path_distance, 1.0),
                "turn_abs_deg": sum(abs(value) for value in turns),
                "turn_sign_changes": sum(
                    1 for left, right in zip(turns, turns[1:]) if left * right < 0 and abs(left) > 2 and abs(right) > 2
                ),
            }
        )
    return summaries


def _trend(summary: dict[str, Any], dimension: str) -> str:
    if dimension == "altitude":
        delta = summary["altitude_last_m"] - summary["altitude_first_m"]
        value_range = summary["altitude_range_m"]
        sign_changes = summary["altitude_sign_changes"]
        threshold = max(50.0, value_range * 0.12)
    else:
        delta = summary["speed_last_mps"] - summary["speed_first_mps"]
        value_range = summary["speed_range_mps"]
        sign_changes = summary["speed_sign_changes"]
        threshold = max(2.0, value_range * 0.12)
    if sign_changes >= 3 and value_range > threshold:
        return "periodic"
    if delta > threshold:
        return "up"
    if delta < -threshold:
        return "down"
    return "stable"


def _spatial(summary: dict[str, Any]) -> str:
    if summary["path_distance_m"] <= 1.0:
        return "stationary"
    directness = summary["directness"]
    if summary["turn_sign_changes"] >= 3 and directness < 0.45:
        return "orbit"
    if directness >= 0.72:
        return "straight"
    if directness < 0.35:
        return "runway"
    return "cruise"


def classify_stage(summary: dict[str, Any]) -> dict[str, str]:
    return {
        "altitude": _trend(summary, "altitude"),
        "speed": _trend(summary, "speed"),
        "spatial": _spatial(summary),
    }


def _stage_phrase_score(phrase: str, summary: dict[str, Any]) -> float:
    labels = classify_stage(summary)
    score = 0.0
    # Spatial state.
    if any(term in phrase for term in ("八字", "盘旋", "绕飞")):
        score += 3.0 if labels["spatial"] == "orbit" else -2.0
    if "直线" in phrase:
        score += 3.0 if labels["spatial"] == "straight" else -1.5
    if "跑道巡逻" in phrase:
        score += 3.0 if labels["spatial"] == "runway" else -1.0
    if "往返巡航" in phrase:
        score += 2.0 if labels["spatial"] in {"cruise", "runway"} else -1.0
    if "巡航" in phrase and "跑道" not in phrase and "往返" not in phrase:
        score += 2.0 if labels["spatial"] == "cruise" else -0.5
    if "滑行" in phrase:
        score += 2.0 if summary["speed_mean_mps"] < 30 else -1.0
    # Vertical state.
    if any(term in phrase for term in ("周期波动", "高度周期")):
        score += 2.5 if labels["altitude"] == "periodic" else -1.0
    elif "爬升" in phrase or "上升" in phrase:
        score += 2.5 if labels["altitude"] == "up" else -1.0
    elif "下降" in phrase:
        score += 2.5 if labels["altitude"] == "down" else -1.0
    if "巡航保持" in phrase or "定高" in phrase:
        score += 2.0 if labels["altitude"] == "stable" else -1.0
    # Speed state.
    if "速度周期波动" in phrase:
        score += 2.5 if labels["speed"] == "periodic" else -1.0
    elif "加速" in phrase:
        score += 2.5 if labels["speed"] == "up" else -1.0
    elif "减速" in phrase:
        score += 2.5 if labels["speed"] == "down" else -1.0
    if "速度保持" in phrase or "定速" in phrase:
        score += 2.0 if labels["speed"] == "stable" else -1.0
    # Joint phrases should agree with all explicit dimensions.
    if "平飞" in phrase:
        score += 1.0 if labels["altitude"] == "stable" else -0.5
    return score


def _option_stage_phrases(option: str) -> list[str]:
    text = option.replace("：", ":")
    # Strip a leading explanation and split on the separators used in stage1.
    text = re.sub(r"^.*?(?=(?:阶段\s*1|先|减速|加速|速度保持|高度周期|巡航保持))", "", text)
    parts = re.split(r"\s*(?:，随后|，最后|最后|随后|→|；|;|,|。)\s*", text)
    parts = [part.strip(" ：:") for part in parts if part.strip(" ：:")]
    # A phrase can still contain a stage prefix; remove it for matching.
    return [re.sub(r"^阶段\s*\d+\s*[:：]?\s*", "", part).strip() for part in parts]


def segment_option_scores(trajectory: dict[str, Any], instruction: str) -> dict[str, float]:
    options = parse_options(instruction)
    scores: dict[str, float] = {}
    for letter, option in options.items():
        phrases = _option_stage_phrases(option)
        summaries = stage_summaries(trajectory, len(phrases))
        if len(phrases) != len(summaries):
            scores[letter] = -999.0
            continue
        scores[letter] = sum(_stage_phrase_score(phrase, summary) for phrase, summary in zip(phrases, summaries))
    return scores


def _seconds_of_day(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second + parsed.microsecond / 1_000_000


def _text_times(text: str) -> list[float]:
    values: list[float] = []
    for match in TIME_TEXT_RE.finditer(text):
        values.append(int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3)))
    return values


def _close_lists(actual: list[float], expected: list[float], tolerance: float = 0.35) -> tuple[int, int]:
    if len(actual) != len(expected):
        return 0, max(len(actual), len(expected))
    matched = sum(abs(left - right) <= tolerance for left, right in zip(actual, expected))
    return matched, len(expected)


def _region_sections(option: str) -> dict[str, str]:
    headers = list(REGION_HEADER_RE.finditer(option))
    sections: dict[str, str] = {}
    for index, match in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else option.find("区域进入顺序")
        if end < 0:
            end = len(option)
        sections[match.group(1).lower()] = option[match.end():end]
    return sections


def region_option_score(option: str, regions: list[dict[str, Any]], event_result: dict[str, Any]) -> float:
    sections = _region_sections(option)
    expected_by_id = {str(item["region_id"]).lower(): item for item in event_result["regions"]}
    score = 0.0
    for region in regions:
        region_id = str(region.get("区域编号", region.get("region_id", ""))).replace("区域", "").strip().lower()
        section = sections.get(region_id, "")
        expected = expected_by_id.get(region_id)
        if expected is None:
            continue
        expected_visited = bool(expected["start_inside"] or expected["entry_events"] or expected["exit_events"])
        option_visited = "未到访" not in section
        score += 3.0 if option_visited == expected_visited else -3.0
        count_match = re.search(r"从区外进入\s*(\d+)\s*次", section)
        if count_match:
            score += 3.0 if int(count_match.group(1)) == expected["outside_to_inside_count"] else -2.0
        dwell_match = re.search(r"累计停留\s*([\d.]+)\s*秒", section)
        if dwell_match:
            score += 2.5 if abs(float(dwell_match.group(1)) - expected["total_dwell_seconds"]) <= 0.8 else -1.5
        entry_part = section.split("离开", 1)[0]
        exit_part = section.split("离开", 1)[1] if "离开" in section else ""
        exit_part = exit_part.split("累计", 1)[0]
        expected_entries = [_seconds_of_day(item["time_utc"]) for item in expected["entry_events"] if item.get("time_utc")]
        expected_exits = [_seconds_of_day(item["time_utc"]) for item in expected["exit_events"] if item.get("time_utc")]
        matched, total = _close_lists(_text_times(entry_part), expected_entries)
        score += 3.0 if total == 0 and matched == 0 else 2.5 * matched / total
        matched, total = _close_lists(_text_times(exit_part), expected_exits)
        score += 3.0 if total == 0 and matched == 0 else 2.5 * matched / total
    sequence_match = re.search(r"区域进入顺序为(.+?)(?:。|$)", option)
    if sequence_match:
        actual_sequence = [item.lower() for item in re.findall(r"区域\s*([A-Za-z])", sequence_match.group(1))]
        expected_sequence = [str(item["region_id"]).replace("区域", "").strip().lower() for item in event_result["entry_sequence"]]
        if actual_sequence == expected_sequence:
            score += 6.0
        elif actual_sequence and expected_sequence:
            score += 6.0 * sum(left == right for left, right in zip(actual_sequence, expected_sequence)) / max(len(actual_sequence), len(expected_sequence))
        elif not actual_sequence and not expected_sequence:
            score += 6.0
        else:
            score -= 3.0
    return score


def choose_segment(trajectory: dict[str, Any], instruction: str) -> tuple[str, dict[str, float]]:
    scores = segment_option_scores(trajectory, instruction)
    choice = max(sorted(scores), key=lambda letter: scores[letter])
    return choice, scores


def choose_region(trajectory: dict[str, Any], regions: list[dict[str, Any]], instruction: str) -> tuple[str, dict[str, float], dict[str, Any]]:
    options = parse_options(instruction)
    # Import lazily to keep the module usable in environments without geometry dependencies.
    from b01.geometry.regions import summarize_regions

    event_result = summarize_regions(trajectory, regions)
    scores = {letter: region_option_score(option, regions, event_result) for letter, option in options.items()}
    choice = max(sorted(scores), key=lambda letter: scores[letter])
    return choice, scores, event_result


def trajectory_fact_summary(trajectory: dict[str, Any], stage_count: int = 4) -> dict[str, Any]:
    stages = stage_summaries(trajectory, stage_count)
    return {
        "trajectory_id": trajectory.get("trajectory_id"),
        "point_count": trajectory.get("summary", {}).get("point_count", len(trajectory.get("points", []))),
        "duration_seconds": trajectory.get("summary", {}).get("duration_seconds", 0.0),
        "stages": [{**summary, "labels": classify_stage(summary)} for summary in stages],
    }


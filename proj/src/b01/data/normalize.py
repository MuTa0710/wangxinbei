"""Normalize task1 and task2 trajectories into a common representation."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

EARTH_RADIUS_M = 6_371_000.0


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and return an aware UTC datetime."""

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, (lon1, lat1, lon2, lat2))
    d_lon = lon2_r - lon1_r
    d_lat = lat2_r - lat1_r
    a = math.sin(d_lat / 2.0) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(d_lon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, (lon1, lat1, lon2, lat2))
    d_lon = lon2_r - lon1_r
    x = math.sin(d_lon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angle_delta_deg(current: float, previous: float) -> float:
    """Return the signed shortest difference in [-180, 180)."""

    return (current - previous + 180.0) % 360.0 - 180.0


def _point_from_task1(raw: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "timestamp_raw": float(raw["时间戳"]),
        "longitude": float(raw["经度"]),
        "latitude": float(raw["纬度"]),
        "altitude_m": float(raw["高度"]),
        "speed_mps": float(raw["速度"]),
        "heading_deg": float(raw["航向角"]) if _finite(raw.get("航向角")) else None,
    }


def _point_from_task2(raw: dict[str, str], index: int) -> dict[str, Any]:
    timestamp = parse_utc(raw["时间戳"])
    return {
        "index": index,
        "timestamp_utc": iso_utc(timestamp),
        "timestamp_dt": timestamp,
        "longitude": float(raw["经度"]),
        "latitude": float(raw["纬度"]),
        "altitude_m": float(raw["高度"]),
        "speed_mps": float(raw["速度"]),
        "heading_deg": None,
    }


def _quality_flags(points: list[dict[str, Any]], *, task: str, large_gap_s: float) -> dict[str, Any]:
    timestamp_values = [
        float(point["timestamp_raw"]) if task == "task1" else point["timestamp_dt"].timestamp()
        for point in points
    ]
    duplicate_timestamps = sum(count - 1 for count in Counter(timestamp_values).values() if count > 1)
    nonfinite_count = 0
    negative_altitude_count = 0
    zero_speed_count = 0
    for point in points:
        for key in ("longitude", "latitude", "altitude_m", "speed_mps"):
            if not _finite(point[key]):
                nonfinite_count += 1
        if point["altitude_m"] < 0:
            negative_altitude_count += 1
        if point["speed_mps"] == 0:
            zero_speed_count += 1
    gaps = [b - a for a, b in zip(timestamp_values, timestamp_values[1:]) if b >= a]
    return {
        "task": task,
        "point_count": len(points),
        "duplicate_timestamp_count": duplicate_timestamps,
        "nonfinite_value_count": nonfinite_count,
        "negative_altitude_count": negative_altitude_count,
        "zero_speed_count": zero_speed_count,
        "large_gap_count": sum(gap > large_gap_s for gap in gaps),
        "max_gap_seconds": max(gaps, default=0.0),
        "has_quality_issue": any(
            value > 0
            for key, value in {
                "duplicate_timestamp_count": duplicate_timestamps,
                "nonfinite_value_count": nonfinite_count,
                "negative_altitude_count": negative_altitude_count,
                "large_gap_count": sum(gap > large_gap_s for gap in gaps),
            }.items()
        ),
    }


def _add_derived_features(points: list[dict[str, Any]], *, task: str) -> None:
    if not points:
        return
    first_time = points[0]["timestamp_raw"] if task == "task1" else points[0]["timestamp_dt"].timestamp()
    previous_heading: float | None = None
    for point in points:
        current_time = point["timestamp_raw"] if task == "task1" else point["timestamp_dt"].timestamp()
        point["relative_seconds"] = current_time - first_time
        point["dt_seconds"] = None
        point["distance_m"] = None
        point["bearing_derived_deg"] = None
        point["altitude_rate_mps"] = None
        point["acceleration_mps2"] = None
        point["heading_delta_deg"] = None
        heading = point.get("heading_deg")
        if heading is not None and previous_heading is not None:
            point["heading_delta_deg"] = angle_delta_deg(heading, previous_heading)
        if heading is not None:
            previous_heading = heading

    for previous, current in zip(points, points[1:]):
        previous_time = previous["timestamp_raw"] if task == "task1" else previous["timestamp_dt"].timestamp()
        current_time = current["timestamp_raw"] if task == "task1" else current["timestamp_dt"].timestamp()
        dt = current_time - previous_time
        current["dt_seconds"] = dt
        current["distance_m"] = haversine_m(
            previous["longitude"], previous["latitude"], current["longitude"], current["latitude"]
        )
        current["bearing_derived_deg"] = bearing_deg(
            previous["longitude"], previous["latitude"], current["longitude"], current["latitude"]
        )
        if dt > 0:
            current["altitude_rate_mps"] = (current["altitude_m"] - previous["altitude_m"]) / dt
            current["acceleration_mps2"] = (current["speed_mps"] - previous["speed_mps"]) / dt


def summarize_trajectory(points: list[dict[str, Any]], quality: dict[str, Any], *, task: str) -> dict[str, Any]:
    if not points:
        return {"point_count": 0, "quality": quality}
    times = [point["relative_seconds"] for point in points]
    distances = [point["distance_m"] for point in points if point["distance_m"] is not None]
    altitudes = [point["altitude_m"] for point in points]
    speeds = [point["speed_mps"] for point in points]
    lons = [point["longitude"] for point in points]
    lats = [point["latitude"] for point in points]
    summary: dict[str, Any] = {
        "point_count": len(points),
        "duration_seconds": max(times) - min(times),
        "bbox": {"min_lon": min(lons), "min_lat": min(lats), "max_lon": max(lons), "max_lat": max(lats)},
        "altitude_m": {"min": min(altitudes), "max": max(altitudes), "mean": sum(altitudes) / len(altitudes)},
        "speed_mps": {"min": min(speeds), "max": max(speeds), "mean": sum(speeds) / len(speeds)},
        "total_distance_m": sum(distances),
        "quality": quality,
    }
    if task == "task2":
        summary["start_time_utc"] = points[0]["timestamp_utc"]
        summary["end_time_utc"] = points[-1]["timestamp_utc"]
    return summary


def normalize_trajectory(
    trajectory_id: str,
    start_time: str | None,
    raw_points: Iterable[dict[str, Any]],
    *,
    task: str,
    large_gap_s: float = 300.0,
) -> dict[str, Any]:
    """Normalize one trajectory while retaining raw values and quality signals."""

    if task not in {"task1", "task2"}:
        raise ValueError(f"Unsupported task: {task}")
    points = (
        [_point_from_task1(raw, index) for index, raw in enumerate(raw_points)]
        if task == "task1"
        else [_point_from_task2(raw, index) for index, raw in enumerate(raw_points)]
    )
    if task == "task1":
        points.sort(key=lambda point: (point["timestamp_raw"], point["index"]))
        start_utc = parse_utc(start_time) if start_time else None
    else:
        points.sort(key=lambda point: (point["timestamp_dt"], point["index"]))
        start_utc = parse_utc(start_time) if start_time else (points[0]["timestamp_dt"] if points else None)
    quality = _quality_flags(points, task=task, large_gap_s=large_gap_s)
    _add_derived_features(points, task=task)
    for point in points:
        point.pop("timestamp_dt", None)
    normalized: dict[str, Any] = {
        "schema_version": "m1.0",
        "task": task,
        "trajectory_id": trajectory_id,
        "start_time_utc": iso_utc(start_utc) if start_utc else None,
        "points": points,
        "summary": summarize_trajectory(points, quality, task=task),
    }
    return normalized

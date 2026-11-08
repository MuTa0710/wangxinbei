"""Deterministic polygon/trajectory event calculations for M2.

The event policy follows the B01 stage1 README:

* points are assumed to be time sorted by the M1 normalizer;
* the polygon boundary is treated as inside (``covers`` semantics);
* an initial in-region state is not counted as a new entry;
* an in-region state at the trajectory end has no fabricated exit event;
* dwell time is clipped to the observed trajectory interval;
* entry order across regions is stable by time, then input region order.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry


_EPS = 1e-10


def _polygon(region: dict[str, Any]) -> BaseGeometry:
    coordinates = region.get("坐标", region.get("边界"))
    if not coordinates or len(coordinates) < 3:
        raise ValueError(f"Region {region!r} has fewer than three coordinates")
    polygon: BaseGeometry = Polygon(coordinates)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise ValueError(f"Region {region!r} produces an empty geometry")
    return polygon


def _region_id(region: dict[str, Any]) -> str:
    return str(region.get("区域编号", region.get("region_id", "")))


def _point_time(point: dict[str, Any]) -> float:
    return float(point["relative_seconds"])


def _time_utc(trajectory: dict[str, Any], seconds: float) -> str | None:
    start = trajectory.get("start_time_utc")
    if not start:
        return None
    text = str(start)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed.astimezone(timezone.utc) + timedelta(seconds=float(seconds))).isoformat().replace("+00:00", "Z")


def _geometry_points(geometry: BaseGeometry) -> list[Point]:
    """Flatten point-like pieces of a Shapely intersection geometry."""

    if geometry.is_empty:
        return []
    geometry_type = geometry.geom_type
    if geometry_type == "Point":
        return [geometry]  # type: ignore[list-item]
    if geometry_type in {"MultiPoint", "GeometryCollection", "MultiLineString"}:
        points: list[Point] = []
        for item in geometry.geoms:  # type: ignore[union-attr]
            points.extend(_geometry_points(item))
        return points
    if geometry_type in {"LineString", "LinearRing"}:
        coords = list(geometry.coords)  # type: ignore[attr-defined]
        if not coords:
            return []
        return [Point(coords[0]), Point(coords[-1])]
    return []


def _intersection_fractions(segment: LineString, polygon: BaseGeometry) -> list[float]:
    if segment.length <= _EPS:
        return []
    boundary = segment.intersection(polygon.boundary)
    fractions: list[float] = []
    for point in _geometry_points(boundary):
        fraction = segment.project(point, normalized=True)
        if -_EPS <= fraction <= 1.0 + _EPS:
            fractions.append(max(0.0, min(1.0, fraction)))
    values = sorted(fractions)
    unique: list[float] = []
    for value in values:
        if not unique or abs(value - unique[-1]) > _EPS:
            unique.append(value)
    return unique


def _event_time(t0: float, t1: float, fraction: float) -> float:
    return t0 + (t1 - t0) * max(0.0, min(1.0, fraction))


def _segment_transitions(
    previous: dict[str, Any],
    current: dict[str, Any],
    polygon: BaseGeometry,
    previous_inside: bool,
    current_inside: bool,
) -> list[tuple[float, str]]:
    """Return (fraction, type) transitions along one trajectory segment."""

    t0, t1 = _point_time(previous), _point_time(current)
    if t1 <= t0:
        return []
    segment = LineString(
        [(float(previous["longitude"]), float(previous["latitude"])),
         (float(current["longitude"]), float(current["latitude"]))]
    )
    fractions = [0.0, *_intersection_fractions(segment, polygon), 1.0]
    fractions = sorted(set(round(value, 12) for value in fractions))
    interval_states: list[bool] = []
    for left, right in zip(fractions, fractions[1:]):
        middle = (left + right) / 2.0
        point = segment.interpolate(middle, normalized=True)
        interval_states.append(bool(polygon.covers(point)))

    transitions: list[tuple[float, str]] = []
    comparisons: list[tuple[bool, bool, float]] = []
    if interval_states:
        comparisons.append((previous_inside, interval_states[0], fractions[0]))
        comparisons.extend(
            (left_state, right_state, fractions[index])
            for index, (left_state, right_state) in enumerate(
                zip(interval_states, interval_states[1:]), start=1
            )
        )
        comparisons.append((interval_states[-1], current_inside, fractions[-1]))
    for left_state, right_state, boundary_fraction in comparisons:
        if left_state == right_state:
            continue
        transitions.append((boundary_fraction, "entry" if right_state else "exit"))
    return transitions


def _empty_summary(region_id: str) -> dict[str, Any]:
    return {
        "region_id": region_id,
        "start_inside": False,
        "end_inside": False,
        "entry_events": [],
        "exit_events": [],
        "dwell_intervals": [],
        "outside_to_inside_count": 0,
        "total_dwell_seconds": 0.0,
        "event_order": [],
    }


def region_event_summary(trajectory: dict[str, Any], region: dict[str, Any]) -> dict[str, Any]:
    """Calculate in-range entry, exit and dwell facts for one polygon.

    ``dwell_intervals`` may end at the trajectory end without an exit event;
    such an interval has ``open_end=True``. This preserves the distinction
    between an observed exit and truncation at the data boundary.
    """

    region_id = _region_id(region)
    points = list(trajectory.get("points", []))
    if not points:
        return _empty_summary(region_id)
    polygon = _polygon(region)
    inside_flags = [
        bool(polygon.covers(Point(float(point["longitude"]), float(point["latitude"]))))
        for point in points
    ]
    start_inside, end_inside = inside_flags[0], inside_flags[-1]
    raw_events: list[dict[str, Any]] = []
    for previous, current, previous_inside, current_inside in zip(
        points, points[1:], inside_flags, inside_flags[1:]
    ):
        for fraction, event_type in _segment_transitions(
            previous, current, polygon, previous_inside, current_inside
        ):
            seconds = _event_time(_point_time(previous), _point_time(current), fraction)
            raw_events.append(
                {
                    "time_seconds": seconds,
                    "time_utc": _time_utc(trajectory, seconds),
                    "interpolated": fraction not in (0.0, 1.0),
                    "type": event_type,
                }
            )

    raw_events.sort(key=lambda event: (event["time_seconds"], 0 if event["type"] == "exit" else 1))
    entry_events = [
        {key: value for key, value in event.items() if key != "type"}
        for event in raw_events if event["type"] == "entry"
    ]
    exit_events = [
        {key: value for key, value in event.items() if key != "type"}
        for event in raw_events if event["type"] == "exit"
    ]

    intervals: list[dict[str, Any]] = []
    active_start = _point_time(points[0]) if start_inside else None
    for event in raw_events:
        event_time = float(event["time_seconds"])
        if event["type"] == "entry":
            if active_start is None:
                active_start = event_time
        elif active_start is not None:
            intervals.append(
                {
                    "start_seconds": active_start,
                    "end_seconds": event_time,
                    "duration_seconds": max(0.0, event_time - active_start),
                    "open_end": False,
                    "start_utc": _time_utc(trajectory, active_start),
                    "end_utc": _time_utc(trajectory, event_time),
                }
            )
            active_start = None
    if active_start is not None:
        end_time = _point_time(points[-1])
        intervals.append(
            {
                "start_seconds": active_start,
                "end_seconds": end_time,
                "duration_seconds": max(0.0, end_time - active_start),
                "open_end": True,
                "start_utc": _time_utc(trajectory, active_start),
                "end_utc": _time_utc(trajectory, end_time),
            }
        )

    return {
        "region_id": region_id,
        "start_inside": start_inside,
        "end_inside": end_inside,
        "entry_events": entry_events,
        "exit_events": exit_events,
        "dwell_intervals": intervals,
        "outside_to_inside_count": len(entry_events),
        "total_dwell_seconds": sum(item["duration_seconds"] for item in intervals),
        "event_order": [
            {key: value for key, value in event.items() if key != "type"} | {"type": event["type"]}
            for event in raw_events
        ],
    }


def region_entry_sequence(
    trajectory: dict[str, Any], regions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return all entry events across regions in time/input order."""

    entries: list[tuple[float, int, dict[str, Any]]] = []
    for region_index, region in enumerate(regions):
        summary = region_event_summary(trajectory, region)
        for entry in summary["entry_events"]:
            entries.append((float(entry["time_seconds"]), region_index, {
                "region_id": summary["region_id"],
                **entry,
            }))
    entries.sort(key=lambda item: (item[0], item[1]))
    return [item[2] | {"sequence_index": index + 1} for index, item in enumerate(entries)]


def summarize_regions(trajectory: dict[str, Any], regions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return per-region summaries and the cross-region entry sequence."""

    region_list = list(regions)
    summaries = [region_event_summary(trajectory, region) for region in region_list]
    return {
        "regions": summaries,
        "entry_sequence": region_entry_sequence(trajectory, region_list),
    }

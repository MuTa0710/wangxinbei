#!/usr/bin/env python3
"""Build the M4 Task1 four-choice baseline and fact caches."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from b01.io.jsonl import read_jsonl, write_jsonl  # noqa: E402
from b01.submission.schema import make_task1_record  # noqa: E402
from b01.task1.baseline import choose_region, choose_segment, trajectory_fact_summary  # noqa: E402


SCHEMA_VERSION = "m4.0"


def build(cache_dir: Path, output_dir: Path, *, backend: str = "deterministic", model_name: str | None = None) -> dict[str, Any]:
    segment_rows = list(read_jsonl(cache_dir / "task1_segment_normalized.jsonl"))
    region_rows = list(read_jsonl(cache_dir / "task1_region_normalized.jsonl"))
    results: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    qwen_agent = None
    if backend == "qwen3":
        if not model_name:
            raise ValueError("--model-name is required with --backend qwen3")
        from b01.task1.qwen_choice import QwenChoiceAgent

        qwen_agent = QwenChoiceAgent(model_name)

    for row in segment_rows:
        trajectory = row["trajectory"]
        fact = trajectory_fact_summary(trajectory)
        if qwen_agent is None:
            choice, scores = choose_segment(trajectory, row["text_instruction"])
            raw_model_output = None
        else:
            evidence = json.dumps(fact, ensure_ascii=False)
            choice, raw_model_output = qwen_agent.choose(row["text_instruction"], evidence)
            scores = {}
        results.append(make_task1_record(row["sample_id"], choice))
        diagnostics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": row["sample_id"],
                "task": "segment",
                "backend": backend,
                "choice": choice,
                "scores": scores,
                "raw_model_output": raw_model_output,
            }
        )
        facts.append({"schema_version": SCHEMA_VERSION, "sample_id": row["sample_id"], "task": "segment", "facts": fact})

    for row in region_rows:
        trajectory = row["trajectory"]
        choice, scores, event_result = choose_region(trajectory, row.get("regions", []), row["text_instruction"])
        results.append(make_task1_record(row["sample_id"], choice))
        diagnostics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": row["sample_id"],
                "task": "region",
                "backend": "deterministic_geometry",
                "choice": choice,
                "scores": scores,
            }
        )
        facts.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": row["sample_id"],
                "task": "region",
                "facts": {"trajectory": trajectory_fact_summary(trajectory), "regions": event_result["regions"], "entry_sequence": event_result["entry_sequence"]},
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "task1_baseline_results.jsonl": write_jsonl(output_dir / "task1_baseline_results.jsonl", results),
        "task1_diagnostics.jsonl": write_jsonl(output_dir / "task1_diagnostics.jsonl", diagnostics),
        "task1_facts.jsonl": write_jsonl(output_dir / "task1_facts.jsonl", facts),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now().astimezone().isoformat().replace("+00:00", "Z"),
        "backend": backend,
        "inputs": {"segment_samples": len(segment_rows), "region_samples": len(region_rows)},
        "outputs": outputs,
        "results": {"records": len(results), "options": {letter: sum(row["选项"] == letter for row in results) for letter in "ABCD"}},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "m1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "m4")
    parser.add_argument("--backend", choices=("deterministic", "qwen3"), default="deterministic")
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()
    manifest = build(args.cache_dir.resolve(), args.output.resolve(), backend=args.backend, model_name=args.model_name)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

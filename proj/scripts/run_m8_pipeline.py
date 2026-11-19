#!/usr/bin/env python3
"""Run the M8 rule-pool -> Qwen3 -> structured-answer pipeline.

Examples (inside the container)::

    # rule-only run (no GPU): builds pools and emits the rule-order submission
    python scripts/run_m8_pipeline.py --backend rule --limit 5

    # Qwen3 on a few samples
    python scripts/run_m8_pipeline.py --backend qwen3 --limit 3

    # full task2 with Qwen3
    python scripts/run_m8_pipeline.py --backend qwen3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from b01.agent.m8_runtime import M8Pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rule", "qwen3"), default="rule")
    parser.add_argument("--prompt-style", choices=("copy", "judge"), default="copy",
                        help="copy: model must reproduce the rule-pool order; judge: model may reorder/select")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "m1")
    parser.add_argument("--m3-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "m3")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "m8")
    parser.add_argument("--model-name", default="/models/Qwen3-4B")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--dump-prompts", action="store_true")
    parser.add_argument("--refresh-features", action="store_true")
    args = parser.parse_args()

    if args.refresh_features:
        cache = args.output / "cache" / "features.jsonl"
        if cache.exists():
            cache.unlink()

    pipeline = M8Pipeline(
        cache_dir=args.cache_dir,
        m3_dir=args.m3_dir,
        output_dir=args.output,
        backend=args.backend,
        prompt_style=args.prompt_style,
        model_name=args.model_name,
        dtype=args.dtype,
        multiplier=args.multiplier,
        max_new_tokens=args.max_new_tokens,
        retries=args.retries,
        limit=args.limit,
        sample_id=args.sample_id,
        dump_prompts=args.dump_prompts,
    )
    report = pipeline.run()
    print(json.dumps({key: report[key] for key in ("backend", "count", "sources", "avg_pool_size", "avg_returned", "elapsed_seconds", "validation")}, ensure_ascii=False, indent=2))
    print(f"[m8] output: {args.output}", file=sys.stderr)
    return 0 if report["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

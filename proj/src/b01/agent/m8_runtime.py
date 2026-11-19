"""M8 pipeline: rule pool -> Qwen3 -> structured official answer.

The pipeline is deliberately one-shot: the rule side builds a Top-2K candidate
pool, the model receives the query plus candidate cards and returns the final
official ``traj_result.jsonl`` record.  The runtime validates the record and
falls back to the rule order when the model output is invalid.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from b01.agent.m7_protocol import ProtocolError, strict_json_load
from b01.agent.m7_runtime import AgentModel
from b01.io.jsonl import read_jsonl, write_jsonl
from b01.retrieval.baseline import parse_query
from b01.retrieval.pool import Pool, build_pool, load_resources, query_date
from b01.submission.schema import validate_submission_records

SCHEMA_VERSION = "m8.0"
SAME_DATE_CREDIT = 0.8


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_system_prompt(style: str = "copy") -> str:
    if style == "copy":
        return """你是 B01 赛题 task2 的结构化输出器。候选池已经由规则按相似度从高到低排好，你不需要重新检索，也不需要重排。

## 输入
- 原始查询：中文自然语言，仅用于确认候选池与题意一致。
- 样本编号：输出时必须原样照抄。
- top_k：本次输出条数上限。
- 候选池：按相关性从高到低排列的航迹编号列表，每行 `序号. 航迹编号`。

## 输出（唯一合法格式）
直接按候选池顺序输出前 top_k 条候选，不要跳过、不要改序，只输出一个 JSON 对象：
{"样本编号":"<原样照抄>","结果":[{"序号":1,"航迹编号":"traj_XXXXXX"},{"序号":2,"航迹编号":"traj_YYYYYY"}]}
- 序号从 1 开始连续递增；
- 航迹编号必须来自候选池且不重复；
- 不输出 why、picks、rank、confidence 等额外字段；
- 禁止 markdown 代码块、禁止前后解释文字。"""
    return """你是 B01 赛题 task2 的检索裁决器。你只能从给定候选池中挑选航迹并排序，直接输出最终结果；不能引入池外航迹，不能编造编号。

## 输入
- 原始查询：中文自然语言，包含时间/高度/速度/区域/行为顺序等约束。
- 样本编号：输出时必须原样照抄。
- top_k：本次最多输出条数。
- 规则解析：规则引擎对查询的结构化理解，可能过严或漏项，仅供参考。
- 候选池：一行一个 JSON 卡片，按规则相似度从高到低排列。字段：
  id=航迹编号；start=起始时间(UTC，月-日 时:分[:秒])；dur=时长(秒)；
  alt=高度中位数(m)；spd=速度中位数(m/s)；spd_std=速度标准差；
  dAlt=分段高度变化最大正/负值(m)；dSpd=分段速度变化最大正/负值(m/s)；
  turn=航向累计变化(度)；circ=路径/首尾位移比(盘旋>2)；
  reg=查询涉及区域的命中(1/0)；fail=未满足的约束及偏差(空=全部满足)；
  db=1 表示同日期批次；q=1 表示数据质量异常。

## 判定规则
1. 候选池已经按规则相似度从高到低排序，这是强先验。默认直接按池中顺序取前 top_k 条。
2. 只有当某条候选与查询明显冲突（例如 fail 含更重约束、速度/高度明显不符、日期矛盾）时才跳过它，并从池中后续补位。
3. 禁止把 fail 更多的候选排到 fail 更少的候选前面；同级别内保持池中相对顺序，禁止为了“体现判断”而大幅重排。
4. 日期题：db=1 的同日候选通常优于 other_date 候选；除非它明显违反查询主体约束，否则不要把它排到 other_date 候选后面。
5. 行为强弱看证据：急促看 dSpd 大，平缓看 dSpd 小；大落差/大仰角看 dAlt；盘旋看 turn/circ；速度保持看 spd_std。
6. 结果按相关度从高到低排列，长度不超过 top_k；候选确实不够时宁可少给，不要凑数。

## 输出（唯一合法格式）
只输出一个 JSON 对象，禁止 markdown 代码块、禁止前后解释文字：
{"样本编号":"<原样照抄>","结果":[{"序号":1,"航迹编号":"traj_XXXXXX"},{"序号":2,"航迹编号":"traj_XXXXXX"}]}
- 序号从 1 开始连续递增；
- 航迹编号必须来自候选池，不能重复；
- 不能输出 why、picks、rank、confidence 等额外字段。"""


def compact_ir(sample: dict[str, Any], ir: dict[str, Any]) -> dict[str, Any]:
    qdate = query_date(str(sample.get("text_query", sample.get("文本查询", ""))))
    return {
        "date": qdate.strftime("%Y-%m-%d") if qdate else None,
        "time": [ir.get("time_start_utc"), ir.get("time_end_utc")]
        if ir.get("time_start_utc") and ir.get("time_end_utc")
        else None,
        "alt": ir.get("altitude_range_m"),
        "spd": ir.get("speed_range_mps"),
        "pos": ir.get("positive_regions") or [],
        "neg": ir.get("negative_regions") or [],
        "beh": [item["name"] for item in (ir.get("behavior_sequence") or [])],
        "same_date_credit": SAME_DATE_CREDIT,
    }


def build_user_message(
    sample: dict[str, Any],
    ir: dict[str, Any],
    pool: Pool,
    *,
    error: str | None = None,
    prompt_style: str = "judge",
) -> str:
    sample_id = str(sample.get("sample_id", sample.get("样本编号")))
    top_k = int(sample.get("top_k", 10))
    if prompt_style == "copy":
        candidate_block = "# 候选池（按相关性从高到低，共 {} 条）\n{}".format(
            len(pool.ids),
            "\n".join(f"{index}. {trajectory_id}" for index, trajectory_id in enumerate(pool.ids, start=1)),
        )
    else:
        cards = "\n".join(json.dumps(card, ensure_ascii=False, separators=(",", ":")) for card in pool.cards)
        candidate_block = f"# 候选池（共 {len(pool.cards)} 条，按规则相似度降序）\n{cards}"
    message = f"""# 查询
{sample.get('text_query', sample.get('文本查询', ''))}

# 样本编号
{sample_id}

# top_k
{top_k}

# 规则解析（可能不准确，仅供参考）
{json.dumps(compact_ir(sample, ir), ensure_ascii=False, separators=(',', ':'))}

{candidate_block}

# 输出
只输出一个 JSON 对象，结构必须为：
{{"样本编号":"{sample_id}","结果":[{{"序号":1,"航迹编号":"<候选池中的航迹编号>"}},...]}}
要求：序号从 1 连续；航迹编号必须来自候选池且不重复；按相关度降序；长度不超过 {top_k}；不要解释、不要 markdown。"""
    if error:
        message += f"\n\n# 上一次输出错误\n{error}\n请严格按上面的格式重新输出完整 JSON。"
    return message


def validate_final(payload: dict[str, Any], sample_id: str, top_k: int, pool_ids: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("output must be a JSON object")
    if set(payload) != {"样本编号", "结果"}:
        raise ValueError(f"top-level keys must be exactly 样本编号/结果, got {sorted(payload)}")
    if str(payload["样本编号"]) != sample_id:
        raise ValueError(f"样本编号 mismatch: expected {sample_id!r}, got {payload['样本编号']!r}")
    results = payload["结果"]
    if not isinstance(results, list):
        raise ValueError("结果 must be a list")
    if not 1 <= len(results) <= top_k:
        raise ValueError(f"结果 length must be 1..{top_k}, got {len(results)}")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict) or set(item) != {"序号", "航迹编号"}:
            raise ValueError(f"results[{index}] must have exactly 序号/航迹编号")
        if item["序号"] != index:
            raise ValueError(f"results[{index}] 序号 must be {index}, got {item['序号']!r}")
        trajectory_id = str(item["航迹编号"])
        if trajectory_id not in pool_ids:
            raise ValueError(f"航迹编号 {trajectory_id!r} is not in the candidate pool")
        if trajectory_id in seen:
            raise ValueError(f"duplicate 航迹编号 {trajectory_id!r}")
        seen.add(trajectory_id)
        normalized.append({"序号": index, "航迹编号": trajectory_id})
    return {"样本编号": sample_id, "结果": normalized}


class M8Pipeline:
    def __init__(
        self,
        *,
        cache_dir: Path,
        m3_dir: Path,
        output_dir: Path,
        backend: str = "rule",
        prompt_style: str = "copy",
        model_name: str = "/models/Qwen3-4B",
        dtype: str = "float16",
        multiplier: float = 2.0,
        max_new_tokens: int = 512,
        retries: int = 1,
        limit: int | None = None,
        sample_id: str | None = None,
        dump_prompts: bool = False,
    ) -> None:
        if backend not in {"rule", "qwen3"}:
            raise ValueError(f"unsupported backend: {backend}")
        if prompt_style not in {"copy", "judge"}:
            raise ValueError(f"unsupported prompt_style: {prompt_style}")
        self.cache_dir = Path(cache_dir)
        self.m3_dir = Path(m3_dir)
        self.output_dir = Path(output_dir)
        self.backend = backend
        self.prompt_style = prompt_style
        self.dtype = dtype
        self.multiplier = multiplier
        self.max_new_tokens = max_new_tokens
        self.retries = max(0, int(retries))
        self.limit = limit
        self.sample_id = sample_id
        self.dump_prompts = dump_prompts
        self.model = AgentModel(model_name, dtype=dtype) if backend == "qwen3" else None

        features_cache = self.output_dir / "cache" / "features.jsonl"
        self.facts, self.features, self.region_hits, self.queries = load_resources(
            self.cache_dir, self.m3_dir, features_cache
        )
        self.system_prompt = build_system_prompt(self.prompt_style)

    def _select_queries(self) -> list[dict[str, Any]]:
        rows = [self.queries[key] for key in sorted(self.queries)]
        if self.sample_id:
            rows = [row for row in rows if str(row.get("sample_id", row.get("样本编号"))) == self.sample_id]
        if self.limit is not None:
            rows = rows[: self.limit]
        return rows

    def _run_model(self, sample: dict[str, Any], ir: dict[str, Any], pool: Pool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        sample_id = str(sample.get("sample_id", sample.get("样本编号")))
        top_k = int(sample.get("top_k", 10))
        pool_ids = set(pool.ids)
        attempts: list[dict[str, Any]] = []
        error: str | None = None
        record: dict[str, Any] | None = None
        for attempt in range(self.retries + 1):
            started = time.time()
            user = build_user_message(sample, ir, pool, error=error, prompt_style=self.prompt_style)
            raw = self.model.generate(self.system_prompt, user, max_new_tokens=self.max_new_tokens)  # type: ignore[union-attr]
            item: dict[str, Any] = {"attempt": attempt + 1, "raw": raw, "elapsed_ms": round((time.time() - started) * 1000, 1)}
            try:
                payload = strict_json_load(raw)
                candidate_record = validate_final(payload, sample_id, top_k, pool_ids)
                if self.prompt_style == "copy":
                    expected_len = min(top_k, len(pool.ids))
                    if len(candidate_record["结果"]) != expected_len:
                        raise ValueError(
                            f"copy mode: expected {expected_len} results, got {len(candidate_record['结果'])}"
                        )
                    expected = pool.ids[: len(candidate_record["结果"])]
                    actual = [item["航迹编号"] for item in candidate_record["结果"]]
                    if actual != expected:
                        raise ValueError(
                            "copy mode: output must follow the candidate-pool order; "
                            f"expected {expected[:5]}..., got {actual[:5]}..."
                        )
                record = candidate_record
                item["ok"] = True
                attempts.append(item)
                break
            except (ProtocolError, ValueError, KeyError) as exc:
                error = str(exc)
                item["ok"] = False
                item["error"] = error[:800]
                attempts.append(item)
        if record is None:
            record = pool.to_record(limit=min(top_k, len(pool.ids)))
            return record, attempts
        return record, attempts

    def run(self) -> dict[str, Any]:
        started_all = time.time()
        rows = self._select_queries()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trace_dir = self.output_dir / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)

        pool_rows: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        prompt_rows: list[dict[str, Any]] = []
        sources: dict[str, int] = {"rule": 0, "model": 0, "fallback": 0}
        strict_json_fail = 0
        attempt_count = 0
        per_sample: list[dict[str, Any]] = []

        for sample in rows:
            started = time.time()
            sample_id = str(sample.get("sample_id", sample.get("样本编号")))
            ir = parse_query(sample)
            pool = build_pool(sample, ir, self.facts, self.features, self.region_hits, multiplier=self.multiplier)
            pool_rows.append(
                {
                    "sample_id": sample_id,
                    "top_k": pool.top_k,
                    "target_size": pool.target_size,
                    "candidates": [
                        {
                            "rank": index,
                            "id": candidate["trajectory_id"],
                            "score": round(candidate["score"], 4),
                            "strict": candidate["strict"],
                            "db": candidate["db"],
                            "fail": candidate["fail"],
                        }
                        for index, candidate in enumerate(pool.candidates, start=1)
                    ],
                }
            )

            trace: dict[str, Any] = {
                "schema": SCHEMA_VERSION,
                "sample_id": sample_id,
                "backend": self.backend,
                "query": sample.get("text_query", sample.get("文本查询", "")),
                "top_k": pool.top_k,
                "pool_size": len(pool.ids),
                "attempts": [],
                "source": "rule",
            }
            if self.backend == "qwen3":
                record, attempts = self._run_model(sample, ir, pool)
                trace["attempts"] = attempts
                attempt_count += len(attempts)
                if len(attempts) > 1 or not attempts or not attempts[-1].get("ok"):
                    strict_json_fail += sum(1 for item in attempts if not item.get("ok"))
                if attempts and attempts[-1].get("ok"):
                    trace["source"] = "model"
                else:
                    trace["source"] = "fallback"
                    trace["fallback_reason"] = attempts[-1].get("error") if attempts else "no attempts"
            else:
                record = pool.to_record(limit=min(pool.top_k, len(pool.ids)))
                trace["source"] = "rule"

            sources[trace["source"]] = sources.get(trace["source"], 0) + 1
            trace["final"] = record
            trace["elapsed_seconds"] = round(time.time() - started, 3)
            records.append(record)
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "source": trace["source"],
                    "pool_size": len(pool.ids),
                    "returned": len(record["结果"]),
                    "elapsed_seconds": trace["elapsed_seconds"],
                }
            )
            (trace_dir / f"{sample_id}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if self.dump_prompts:
                prompt_rows.append(
                    {
                        "sample_id": sample_id,
                        "system": self.system_prompt,
                        "user": build_user_message(sample, ir, pool, prompt_style=self.prompt_style),
                        "raw": trace["attempts"][-1]["raw"] if trace["attempts"] else None,
                    }
                )

        write_jsonl(self.output_dir / "traj_result.jsonl", records)
        write_jsonl(self.output_dir / "pools.jsonl", pool_rows)
        if self.dump_prompts:
            write_jsonl(self.output_dir / "prompts.jsonl", prompt_rows)

        expected_ids = [str(row.get("sample_id", row.get("样本编号"))) for row in rows]
        validation = validate_submission_records(records, expected_ids)
        elapsed = round(time.time() - started_all, 2)
        report = {
            "schema": SCHEMA_VERSION,
            "backend": self.backend,
            "prompt_style": self.prompt_style,
            "model_name": getattr(self.model, "model_name", None),
            "count": len(records),
            "sources": sources,
            "strict_json_fail_attempts": strict_json_fail,
            "total_attempts": attempt_count,
            "elapsed_seconds": elapsed,
            "avg_pool_size": round(sum(row["pool_size"] for row in per_sample) / max(1, len(per_sample)), 2),
            "avg_returned": round(sum(row["returned"] for row in per_sample) / max(1, len(per_sample)), 2),
            "validation": validation,
            "per_sample": per_sample,
            "created_at_utc": _now_iso(),
        }
        (self.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

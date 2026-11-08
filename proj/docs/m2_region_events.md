# M2 区域事件引擎

M2 在 M1 标准化轨迹之上，计算轨迹与多边形区域之间的确定性事件。输入为
`artifacts/m1/task1_region_normalized.jsonl`，输出为
`artifacts/m2/task1_region_events.jsonl`。

## 事件口径

- 轨迹点按 M1 结果中的时间顺序处理；
- 多边形边界按“在区域内”处理；
- 航迹起点已经在区域内时，不计为新的进入；
- 轨迹结束时仍在区域内，不虚构离开事件；
- 未闭合的停留区间仍计算到轨迹结束时，并标记 `open_end: true`；
- 线段穿过区域边界时，按线性时间插值计算进入/离开时刻；
- 多区域进入顺序按事件时间排序，同一时刻按区域在输入中的顺序排序；
- `time_seconds` 是相对于航迹起点的秒数，`time_utc` 是由 `start_time_utc` 换算出的绝对时间。

## 输出结构

每行对应一个 Task1 区域样本：

```json
{
  "schema_version": "m2.0",
  "sample_id": "region_000001",
  "trajectory_id": "traj_region_000001",
  "start_time_utc": "2025-10-11T20:24:46Z",
  "region_summaries": [
    {
      "region_id": "a",
      "start_inside": false,
      "end_inside": false,
      "entry_events": [],
      "exit_events": [],
      "dwell_intervals": [],
      "outside_to_inside_count": 0,
      "total_dwell_seconds": 0.0,
      "event_order": []
    }
  ],
  "entry_sequence": []
}
```

`entry_sequence` 只列出进入事件，重复进入重复保留，并带有从 1 开始的
`sequence_index`。后续 Task1 区域基线可以直接使用该缓存，不需要重复进行几何计算。

## 运行

在容器内执行：

```bash
python scripts/build_m2_region_events.py \
  --cache-dir artifacts/m1 \
  --output artifacts/m2
```

脚本会先运行人工构造的边界场景校验，随后生成：

- `task1_region_events.jsonl`：区域事件缓存；
- `boundary_validation.json`：起点在区内、终点在区内、穿越、重复进入等检查；
- `manifest.json`：输入、输出和统计信息。


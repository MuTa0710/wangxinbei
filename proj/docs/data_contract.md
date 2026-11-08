# M0 数据与提交契约

## 输入

- stage1 根目录由 `DATA_ROOT` 指向，容器默认路径为 `/workspace/data`；
- task1 使用相对起始时间，task2 CSV 使用 UTC ISO 8601 时间；
- 坐标统一为 `[经度, 纬度]`；
- 高度单位为米，速度单位为 `m/s`。

## 标准化对象

所有轨迹标准化为：

- `trajectory_id`；
- `start_time_utc`；
- 按时间排序的 `points`；
- `summary`；
- `quality`。

原始点值不覆盖。重复时间、非有限值、负高度、零速度和大时间间隔只记录为质量标记。

## 提交对象

`traj_result.jsonl` 每行一个对象：

```json
{"样本编号":"seg_000001","选项":"A"}
{"样本编号":"region_000001","选项":"B"}
{"样本编号":"retrieval_000001","结果":[{"序号":1,"航迹编号":"traj_000001"}]}
```

当前 stage1 的提交示例对 task1 使用 `结果` 占位字段，但 README 要求 `选项`。实现中通过 `task1_field_name` 配置控制，默认使用 README 的 `选项`，最终提交前按官方最新通知确认。

## M1 缓存

`artifacts/m1/manifest.json` 记录 schema 版本、输入文件 SHA256、样本数量和输出文件；生成的 JSONL 缓存不提交 Git。

## M2 区域事件缓存

`artifacts/m2/task1_region_events.jsonl` 每行对应一个 Task1 区域样本，包含：

- 每个区域的 `entry_events`、`exit_events` 和 `dwell_intervals`；
- `outside_to_inside_count` 和累计停留时长；
- 相对秒数及换算后的 UTC 时间；
- 全部区域的 `entry_sequence`。

轨迹结束时仍在区域内不会生成虚构的离开事件，对应停留区间使用 `open_end: true`。

## M3 Task2 输出

`artifacts/m3/task2_baseline_results.jsonl` 每行一个检索样本，格式为：

```json
{"schema_version":"m3.0","样本编号":"retrieval_000001","结果":[{"序号":1,"航迹编号":"traj_000001"}]}
```

`artifacts/m3/query_ir.jsonl` 和 `retrieval_diagnostics.jsonl` 用于审计查询解析、过滤原因和行为排序，不直接作为提交文件。

## M4/M5 提交产物

`artifacts/m4/task1_baseline_results.jsonl` 保存 Task1 的 100 条四选一结果；
`artifacts/m5/traj_result.jsonl` 按官方样本顺序合并 Task1 和 Task2，作为当前基线的完整提交文件。
`artifacts/m5/m5_report.json` 保存结构校验、边界校验、运行耗时和提交文件哈希。

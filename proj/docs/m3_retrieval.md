# M3 Task2 结构化检索基线

M3 将 Task2 的自然语言查询转换为可执行的查询 IR，再对 2,000 条标准化航迹执行确定性过滤和排序。

## 处理流程

```text
文本查询
  -> 查询 IR
  -> 时间/高度/速度硬过滤
  -> 正向/否定区域命中过滤
  -> 行为特征匹配与顺序评分
  -> Top-K 结果
```

## 查询 IR

`query_ir.jsonl` 为每条查询保留：

- `time_start_utc`、`time_end_utc`；
- `altitude_range_m`、`speed_range_mps`；
- `positive_regions`、`negative_regions`；
- `behavior_sequence`；
- `top_k` 和否定/超高空等标志。

当前支持官方 stage1 查询中出现的日期、明确时刻范围、凌晨/深夜/中午/下午/黄昏/傍晚/夜间等时段表达，以及官方样例中的高度、速度、区域和行为短语。

## 航迹特征

M3 从 M1 轨迹点计算：

- 高度、速度的均值、中位数、范围和标准差；
- 分窗口高度/速度变化；
- 航向变化总量、路径长度与首尾位移比，用于盘旋判断；
- 爬升、下降、阶梯加速、阶梯减速、速度保持、低速、进近和周期波动等基线标志。

时间、数值和区域约束是硬过滤；行为约束先转为特征匹配分数，再用于候选排序。这样能保持确定性，也能在行为识别不完全时保留候选供后续增强模块重排。

## 输出

- `query_ir.jsonl`：50 条结构化查询；
- `region_hits.jsonl`：2,000 条航迹的 A–F 区域命中缓存；
- `retrieval_diagnostics.jsonl`：候选数、拒绝数、行为匹配和排序诊断；
- `task2_baseline_results.jsonl`：符合官方提交结构的 Task2 结果。

结果中的每个元素均为：

```json
{"序号": 1, "航迹编号": "traj_000001"}
```

返回数量不超过查询的 `top_k`。由于部分查询带日期，单日航迹数量少于 `top_k` 时会自然返回较少结果。

## 运行

```bash
python scripts/build_m3_retrieval.py \
  --cache-dir artifacts/m1 \
  --output artifacts/m3
```

当前版本是可解释的确定性基线，不调用大模型或外部服务。后续若加入 BGE-M3、reranker 或 Qwen3 查询解析，应保留该版本作为回退基线。


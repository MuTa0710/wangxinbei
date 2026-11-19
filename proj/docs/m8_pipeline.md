# M8 最小可运行 Pipeline：规则召回 → Qwen3 → 官方结果

> 目的：先把 task2 的“前置工具筛选 → 喂给模型 → 模型输出官方结构”完整跑通。
> 筛选约束、prompt 策略后续再调；本版优先保证**流程可运行、输出合法、可回退**。

## 1. 流程

```text
M1 缓存（track_facts / tracks_normalized / task2_queries）
M3 缓存（region_hits）
        │
        ▼
[1] 规则侧（b01.retrieval.pool）
  - 解析 IR（复用 M3 parse_query）
  - 负区域一票否决
  - 对 2,000 条算加权相似度（时间/高度/速度/正区域/行为）
  - 取 2 × top_k 条候选；硬过滤通过项在能放下时强制入池
  - 生成候选卡（id/start/dur/alt/spd/.../fail/db/q）
        │
        ▼
[2] 模型侧（b01.agent.m8_runtime）
  - system prompt + 用户查询 + IR 回显 + 候选卡 JSONL
  - 本地 Qwen3-4B 单次前向（no thinking，greedy，≤512 new tokens）
  - 输出 {"样本编号": "...", "结果": [{"序号":1,"航迹编号":"traj_..."}]}
        │
        ▼
[3] Runtime 校验
  - 样本编号一致、结果 1..top_k、序号连续、编号不重复且属于候选池
  - 不合法 → 错误回喂重试 1 次 → 仍不合法回退规则序
  - 官方 validate_submission_records 二次校验
        │
        ▼
artifacts/m8_final/{traj_result.jsonl, pools.jsonl, trace/, report.json}
```

## 2. 代码位置

| 文件 | 作用 |
|---|---|
| `src/b01/retrieval/pool.py` | 相似度打分、Top-2K 池、候选卡、特征缓存 |
| `src/b01/agent/m8_runtime.py` | prompt、Qwen3 调用、官方结构校验、回退、trace/report |
| `scripts/run_m8_pipeline.py` | CLI |

复用：`b01.retrieval.baseline`、`b01.agent.m7_runtime.AgentModel`、`b01.agent.m7_protocol.strict_json_load`、`b01.submission.schema`。

## 3. 运行

在 `b01-dev` 容器内：

```bash
cd /workspace/repo/proj

# 0) 规则后端（不需要 GPU）：先确认池/格式/校验都通
python scripts/run_m8_pipeline.py --backend rule --output artifacts/m8_rule

# 1) 小样本模型冒烟
python scripts/run_m8_pipeline.py --backend qwen3 --prompt-style copy --limit 3 \
  --model-name /models/Qwen3-4B --output artifacts/m8_smoke

# 2) 全量 50 题（copy 模式：模型按池顺序输出，runtime 校验；非法则回退规则序）
python scripts/run_m8_pipeline.py --backend qwen3 --prompt-style copy \
  --model-name /models/Qwen3-4B --output artifacts/m8_final

# 3) 评审模式（允许模型重排/取舍；目前 4B 效果不稳定，仅作对照）
python scripts/run_m8_pipeline.py --backend qwen3 --prompt-style judge \
  --model-name /models/Qwen3-4B --output artifacts/m8_judge
```

常用参数：`--limit`、`--sample-id`、`--multiplier`（池 = multiplier×top_k）、`--retries`、`--dump-prompts`、`--refresh-features`。

首次运行会生成 `artifacts/<out>/cache/features.jsonl`（约 2.6 MB），后续直接复用；改特征逻辑时加 `--refresh-features` 或删除该文件。

## 4. 当前结果（2026-11-18，RTX 4090）

| 运行 | 样本 | 模型完成 | fallback | 池均值 | 返回均值 | 耗时 | 模型尝试 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `--backend rule` | 50 | — | 0 | 24.56 | 12.28 | ~1 s（缓存后） | — |
| `qwen3 --prompt-style judge` | 50 | 50 | 0 | 24.56 | 12.28 | 491 s | 51 次（1 次重试） |
| **`qwen3 --prompt-style copy`（`artifacts/m8_final`）** | **50** | **50** | **0** | **24.56** | **12.28** | **472 s** | **50 次（0 重试）** |

要点：

- 规则后端 50/50 合法，平均池 24.56、平均返回 12.28；
- `judge` 模式下模型会主动重排/跳选，实测对代理真值的 Jaccard 0.441、date-batch recall@10 0.777，**低于规则序的 0.545 / 0.905**；所以 v0 默认用 `copy` 模式，先保证流程稳定，模型判断能力留到筛选模块/事件特征调好后再放开；
- `copy` 模式对模型输出做严格校验：必须与候选池顺序一致、长度 = min(top_k, 池大小)；不一致重试 1 次，仍失败回退规则序。最终 **50/50 与规则序完全一致**，`validate_submission_records` 全部通过；
- copy 模式给模型的候选池只渲染为“序号 + 航迹编号”的列表（不渲染证据字段），prompt 更短，模型 50/50 一次通过、0 重试；这也是复制任务更稳的原因。

## 5. 已知限制 / 下一步

1. 筛选模块目前是加权相似度 + 2×池，无日期行为题仍有大量并列（见设计文档 3.3/3.6），后续换成事件级行为特征；
2. `copy` 模式只是把模型当结构化输出器，模型判断能力没有发挥；等筛选质量上来后切 `judge`；
3. `judge` 模式对 4B 偏难：实测对代理真值的 Jaccard 0.441、date-batch recall@10 0.777，低于规则序 0.545/0.905，需要更完整的事件级证据和更强的顺序约束；
4. task2 输出目前是独立文件；合并 task1 时由 M5 统一生成 `traj_result.jsonl`。

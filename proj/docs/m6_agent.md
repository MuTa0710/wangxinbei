# M6 受控 Agent 工作流

## 目标

M6 在 M0-M5 确定性底座上增加两个受控的 Qwen3 节点：

1. Task1 节点根据 M2/M4 事实摘要选择 A/B/C/D。
2. Task2 节点将自然语言问题解析为标准 Query IR，再交给 M3 检索工具。

区域几何、轨迹统计、硬过滤、Top-K 排序、提交文件生成和校验仍由固定 Python 工具执行。

## 工作流

```text
样本
  -> AgentController
      -> Task1: facts -> Qwen3 judge -> schema validation -> error-feedback retry -> deterministic fallback
      -> Task2: Qwen3 parser -> Query IR validation -> error-feedback retry -> M3 retrieval -> output validation
  -> traj_result.jsonl + trace files + m6_report.json
```

每个 Task1/Task2 样本最多调用 Qwen3 `1 + max_retries` 次（默认 1 次重试，即最多 2 次）：首次输出校验失败时，把校验错误回喂给模型重试一次，仍失败才回退。模型输出不允许直接写入最终提交，必须通过 Pydantic schema 和官方提交校验。解析失败、字段非法或模型异常时，使用现有 M4/M3 确定性实现。

## 运行

使用确定性后端复现 M5 结果：

```bash
python scripts/run_m6_agent.py --backend deterministic
```

使用本地 Qwen3：

```bash
python scripts/run_m6_agent.py \
  --backend qwen3 \
  --model-name /models/Qwen3-4B \
  --dtype float16
```

输出目录默认是 `artifacts/m6/`，包含：

- `traj_result.jsonl`：可提交文件
- `task1_diagnostics.jsonl`：Task1 模型输出、回退原因和选择结果
- `task2_diagnostics.jsonl`：Task2 解析、检索和回退信息
- `query_ir.jsonl`：最终送入 M3 的 Query IR
- `m6_report.json`：运行配置、回退数量、哈希和校验结果

## 设计边界

- 不把 483k 级别的原始轨迹点直接交给模型。
- 不使用动态工具发现、无限 ReAct 循环或多 Agent 对话；错误回馈重试次数有上限（`--max-retries`，默认 1），不构成无限循环。
- Qwen3 只负责语义解析和判断，数值计算和排序使用确定性代码。
- `artifacts/m6/` 属于运行产物，继续由 `.gitignore` 忽略；模型权重不提交 Git。

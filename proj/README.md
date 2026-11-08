# B01 多模态数据语义理解项目环境

本目录用于提交 Git，并由 Ubuntu 20.04 Docker 容器拉取。当前只提供比赛环境文档和依赖声明，不包含业务代码。

## 环境选择

| 项目 | 固定选择 | 依据 |
|---|---|---|
| 容器系统 | Ubuntu 20.04 | B01 赛题环境要求 Ubuntu 20.04+ |
| Python | 3.12 | B01 赛题指定 Python 3.12 |
| CUDA | 12.1 runtime | 满足 B01 的 CUDA 12+ 要求，并与 PyTorch 2.3.0 官方 wheel 对应 |
| PyTorch | 2.3.0 | B01 要求 PyTorch >=2.0.0 且 <=2.3.0 |
| torchvision | 0.18.0 | 与 PyTorch 2.3.0 官方匹配 |
| Transformers | 4.51.3 | Qwen3 官方模型卡要求至少 4.51.0 |
| Tokenizers | 0.21.1 | 与 Transformers 4.51.3 配套固定 |
| Accelerate | 1.6.0 | 支持本地 `device_map` 和混合精度推理 |
| GPU | NVIDIA GPU，支持 CUDA 12.1 | Docker 运行时需要宿主机 NVIDIA 驱动和 NVIDIA Container Toolkit |

PyTorch 2.3.0 / torchvision 0.18.0 / CUDA 12.1 的安装方式采用 PyTorch 官方 wheel 源。B01 没有要求 torchaudio，本项目暂不安装。

## 目录说明

    proj/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── .dockerignore
    ├── .gitignore
    ├── README.md
    └── docs/
        └── environment.md

后续开发代码、数据、模型权重和运行输出应另行加入，不要把大体积数据集和模型权重直接提交到 Git。

当前 Compose 会将宿主机仓库根目录挂载到容器的 `/workspace/repo`，将 stage1 数据只读挂载到 `/workspace/data`，容器工作目录为 `/workspace/repo/proj`。因此可以直接在容器内执行 Git 操作，代码修改会同步到宿主机工作树。

## 构建镜像

在项目根目录执行：

    docker build -t b01-dev:pt230-cu121 .

若 Docker Hub 在当前网络不可达，可先使用镜像代理拉取并保留 Dockerfile 中的官方标签：

    docker pull docker.m.daocloud.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04
    docker tag docker.m.daocloud.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04 nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

## 启动容器

需要宿主机已安装 NVIDIA 驱动和 NVIDIA Container Toolkit：

    docker compose run --rm b01-dev

或直接运行：

    docker run --rm -it --gpus all \
      -v "$PWD:/workspace/proj" \
      b01-dev:pt230-cu121

Windows PowerShell 可使用：

    docker compose run --rm b01-dev

启动长期开发容器：

    docker compose up -d b01-dev
    docker compose exec b01-dev bash

容器内 Git 已配置宿主机 Clash HTTP 代理 `host.docker.internal:1081`。如宿主机代理端口或 Docker Desktop 配置变化，应同步修改 `docker-compose.yml`。

## M1 数据底座

在容器内生成标准化轨迹和区域事件缓存：

    python scripts/build_m1_cache.py --data-root /workspace/data

输出位于 `artifacts/m1/`，包括输入文件 SHA256 清单、task1 标准化样本、task2 查询、2,000 条航迹标准化数据和航迹事实摘要。`artifacts/` 已加入 Git 忽略，不提交生成缓存。

提交文件可用以下命令校验：

    python scripts/validate_submission.py /workspace/repo/proj/artifacts/traj_result.jsonl

## M2 区域事件引擎

在 M1 缓存基础上生成 Task1 区域事件和跨区域进入顺序：

    python scripts/build_m2_region_events.py --cache-dir artifacts/m1 --output artifacts/m2

M2 不把航迹结束时仍在区域内的状态虚构成离开事件，并保留未闭合停留区间的
`open_end` 标记。运行前会检查穿越、起点在区内、终点在区内和重复进入等边界场景。
具体输出字段见 `docs/m2_region_events.md`。

## M3 Task2 结构化检索基线

在 M1 航迹缓存和 M2 几何引擎基础上生成 Task2 查询 IR、区域命中缓存和 Top-K 结果：

    python scripts/build_m3_retrieval.py --cache-dir artifacts/m1 --output artifacts/m3

M3 先执行时间、高度、速度和区域硬过滤，再使用轨迹统计特征匹配行为顺序并排序。
输出说明见 `docs/m3_retrieval.md`。

## M4/M5 Task1 与统一提交

生成 Task1 四选一结果：

    python scripts/build_m4_task1.py --cache-dir artifacts/m1 --output artifacts/m4 --backend deterministic

运行 M3、M4 并合并为官方提交文件，同时执行结构校验：

    python scripts/run_m5_pipeline.py --cache-dir artifacts/m1 --work-dir artifacts --output artifacts/m5

输出为 `artifacts/m5/traj_result.jsonl`，报告为 `artifacts/m5/m5_report.json`。M4/M5 的字段和运行方式见 `docs/m4_m5.md`。

## M6 受控 Agent 工作流

在 M0-M5 确定性底座上运行受控 Agent 层：

    python scripts/run_m6_agent.py --backend deterministic

使用本地 Qwen3 时：

    python scripts/run_m6_agent.py --backend qwen3 --model-name /models/Qwen3-4B

Qwen3 只负责 Task1 语义判断和 Task2 查询 IR 解析；M2 区域事实、M3 轨迹检索、排序、提交生成和校验仍由固定工具完成。每个样本默认最多调用两次模型（一次初试 + 一次错误回馈重试，可用 `--max-retries` 调整），解析仍失败自动回退到 M3/M4 确定性实现。输出和设计说明见 `docs/m6_agent.md`。

## 环境检查

进入容器后执行：

    python --version
    python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

预期版本：

    Python 3.12.x
    torch 2.3.0
    torchvision 0.18.0
    CUDA 12.1

Qwen3 依赖检查：

    python -c "import transformers, tokenizers, accelerate; print(transformers.__version__, tokenizers.__version__, accelerate.__version__)"

预期版本：

    transformers 4.51.3
    tokenizers 0.21.1
    accelerate 1.6.0

torch.cuda.is_available() 只有在使用 --gpus all、宿主机驱动正常且 NVIDIA Container Toolkit 配置完成时才会为 True。

## 与 B01 规则的关系

- 依赖版本满足赛题公开的 Python、PyTorch、torchvision 和 CUDA 范围。
- 复赛要求本地私有化部署，容器不依赖在线商业大模型服务。
- 当前依赖只准备通用深度学习、轨迹处理、检索和配置基础，不代表已经完成 Agent、MCP 工具或赛题算法。
- 后续增加模型和工具依赖时，必须确认国内开放权重、本地离线部署和参数规模限制。
- requirements.txt 中的版本固定用于减少开发环境漂移；若官方后续发布更具体的镜像或依赖要求，应以官方通知为准。

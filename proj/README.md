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

## 构建镜像

在项目根目录执行：

    docker build -t b01-dev:pt230-cu121 .

## 启动容器

需要宿主机已安装 NVIDIA 驱动和 NVIDIA Container Toolkit：

    docker compose run --rm b01-dev

或直接运行：

    docker run --rm -it --gpus all \
      -v "$PWD:/workspace/proj" \
      b01-dev:pt230-cu121

Windows PowerShell 可使用：

    docker compose run --rm b01-dev

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

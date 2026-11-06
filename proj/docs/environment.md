# B01 运行环境说明

## 1. 官方约束摘要

B01 赛题资料给出的环境要求：

- Python 3.12；
- PyTorch >= 2.0.0 且 <= 2.3.0；
- torchvision 与 PyTorch 版本严格匹配；
- Linux，Ubuntu 20.04+；
- CUDA 12+（使用 GPU 时）；
- 推荐内存 64GB+；
- 推荐至少 256GB 可用磁盘；
- Dockerfile 和 docker-compose.yml 为可选，但有助于环境复现。

复赛计算资源下限：

- GPU 显存不低于 64GB；
- 支持 FP16 或 BF16 推理；
- CPU 不低于 32 核；
- 系统内存不低于 128GB；
- 固态存储不低于 512GB。

上述复赛资源是赛事运行环境要求，不等于本地开发机必须立即达到同样规格。

## 2. 本项目固定版本

本项目暂定以下组合：

    Ubuntu 20.04
    Python 3.12
    CUDA 12.1 runtime
    PyTorch 2.3.0+cu121
    torchvision 0.18.0+cu121

选择 PyTorch 2.3.0 是因为它正好处于 B01 允许范围的上限，并且官方提供与 CUDA 12.1 对应的 torchvision 0.18.0 安装组合。使用上限版本可以避免后续误用 2.4 或更高版本而违反赛题约束。

## 3. GPU 与宿主机要求

容器中的 CUDA runtime 不能代替宿主机 NVIDIA 驱动。Windows 宿主机至少需要：

1. 可正常工作的 NVIDIA GPU 驱动；
2. Docker Desktop 的 WSL2 后端；
3. NVIDIA Container Toolkit 或 Docker Desktop 对 GPU 的支持；
4. 使用 docker run --gpus all 或 Compose 的 GPU 配置。

如果暂时没有 NVIDIA GPU，可先使用容器完成 Python、数据格式和 CPU 逻辑开发，但不能据此验证最终 GPU 推理性能。

## 4. 依赖分层

requirements.txt 目前分为：

- PyTorch 和 torchvision：模型运行；
- NumPy、SciPy、pandas：数据处理和数值计算；
- Shapely、pyproj：区域几何和经纬度处理；
- Transformers、Accelerate、Safetensors、Tokenizers：本地开放权重模型；
- Sentence Transformers：文本/轨迹摘要的向量检索基础；
- Pydantic、orjson、jsonlines、PyYAML、tqdm：结构化输入输出、配置和运行辅助。

MCP 服务端依赖暂不固定。等专用工具接口确定后，再单独增加并验证 MCP 包版本，避免在尚未写代码时引入无用依赖。

## 5. 离线部署注意事项

开发阶段可以联网安装依赖和下载经过许可的开放权重；比赛运行阶段应保证：

- 不调用闭源商业大模型；
- 不依赖在线推理服务；
- 模型权重和依赖可以在指定环境中本地加载；
- 输出严格符合官方要求的 traj_result.jsonl；
- 不包含针对测试样本的答案映射或特判逻辑。



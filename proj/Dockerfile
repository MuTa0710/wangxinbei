FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Etc/UTC

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       software-properties-common \
       ca-certificates \
       curl \
       git \
       build-essential \
       libglib2.0-0 \
       libgomp1 \
       libgl1 \
       libgeos-dev \
       libproj-dev \
       proj-data \
       proj-bin \
       tzdata \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.12 \
       python3.12-dev \
       python3.12-venv \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && python -m pip install --upgrade pip==24.1.2 setuptools==70.1.1 wheel==0.43.0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/proj

COPY requirements.txt README.md ./
RUN python -m pip install -r requirements.txt

CMD ["/bin/bash"]

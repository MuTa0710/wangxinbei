FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Etc/UTC

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON_VERSION=3.12.10

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
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
         libssl-dev \
         zlib1g-dev \
         libbz2-dev \
         libreadline-dev \
         libsqlite3-dev \
         libffi-dev \
         liblzma-dev \
         libncursesw5-dev \
         libgdbm-dev \
         libgdbm-compat-dev \
         libnss3-dev \
         uuid-dev \
         tk-dev \
         libexpat1-dev \
       tzdata \
      && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -o /tmp/python.tgz \
      && tar -xzf /tmp/python.tgz -C /tmp \
      && cd "/tmp/Python-${PYTHON_VERSION}" \
      && ./configure --prefix=/opt/python3.12 --with-ensurepip=install \
      && make -j"$(nproc)" \
      && make altinstall \
      && ln -sf /opt/python3.12/bin/python3.12 /usr/local/bin/python \
      && ln -sf /opt/python3.12/bin/python3.12 /usr/local/bin/python3 \
      && ln -sf /opt/python3.12/bin/pip3.12 /usr/local/bin/pip \
    && python -m pip install --upgrade pip==24.1.2 setuptools==70.1.1 wheel==0.43.0 \
      && rm -rf /tmp/python.tgz "/tmp/Python-${PYTHON_VERSION}"

WORKDIR /workspace/proj

COPY requirements.txt README.md ./
RUN python -m pip install -r requirements.txt

CMD ["/bin/bash"]

# Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip wheel setuptools && \
    pip install --no-cache-dir flash-attn==2.8.3 bitsandbytes

WORKDIR /workspace

# Week 6: vLLM Docker Image
# 
# This Dockerfile creates a production-ready container for vLLM inference
# with Qwen2.5-7B-Instruct model.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
RUN pip3 install --no-cache-dir vllm

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    huggingface-hub \
    transformers \
    fastapi \
    uvicorn

# Create directories
RUN mkdir -p /app /models

WORKDIR /app

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]


#!/bin/bash

# Week 6: Container Entrypoint Script
# Starts vLLM server with Qwen2.5-7B-Instruct

set -e

# Default model (can be overridden via environment variable)
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}

echo "======================================================================"
echo "Starting vLLM Server"
echo "======================================================================"
echo "Model: $MODEL_NAME"
echo "Port: 8000"
echo "======================================================================"
echo ""

# Start vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256


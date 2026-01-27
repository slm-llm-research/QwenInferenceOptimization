#!/bin/bash

# Week 1: Dependency Installation Script
# This script installs all required dependencies for vLLM and Qwen2.5 inference

set -e  # Exit on error

echo "================================================"
echo "Week 1: Installing Dependencies for vLLM"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch with CUDA 12.1 support..."
echo "(This may take several minutes)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "✓ PyTorch installed"
echo ""

# Install vLLM
echo "Installing vLLM..."
echo "(This may take several minutes)"
pip install vllm
echo "✓ vLLM installed"
echo ""

# Install additional dependencies
echo "Installing additional dependencies..."
pip install huggingface-hub transformers tqdm
echo "✓ Additional dependencies installed"
echo ""

echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Run: python verify_environment.py"
echo "2. Run: python download_model.py"
echo "3. Run: python baseline_inference.py"
echo ""


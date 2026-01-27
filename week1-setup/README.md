# Week 1: Introduction and Environment Preparation

## 🎯 Goals

By the end of this week, you will:
- Set up a working Python environment with vLLM and PyTorch
- Verify CUDA and GPU support
- Download the Qwen2.5-7B-Instruct model
- Successfully run your first inference test
- Understand the basic vLLM API

## 📚 What You'll Learn

- How to install vLLM and its dependencies
- CUDA version compatibility requirements
- Model loading and basic inference with vLLM
- The difference between local model files and HuggingFace Hub loading

## 🔧 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥7.0
  - Examples: T4, A10G, A100, RTX 3090, RTX 4090
  - Minimum 16GB VRAM recommended for Qwen2.5-7B
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space for model and dependencies

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+) or macOS (for CPU testing)
- **Python**: 3.9, 3.10, or 3.11
- **CUDA**: 12.1 or compatible (check with `nvcc --version` or `nvidia-smi`)

## 📖 Background: What is vLLM?

vLLM (Very Large Language Model) is a high-performance inference and serving engine for LLMs. Key innovations:

1. **PagedAttention**: Efficient memory management for the key-value (KV) cache, inspired by OS paging
2. **Continuous Batching**: Dynamically batches incoming requests without waiting for batch boundaries
3. **Optimized CUDA Kernels**: Custom kernels for faster attention and generation
4. **Easy Integration**: Drop-in replacement for HuggingFace with better performance

Compared to standard HuggingFace `transformers`, vLLM can be **2-10x faster** for inference workloads!

## 🚀 Setup Instructions

### Step 1: Check Your Environment

First, verify you have a compatible GPU and CUDA installation:

```bash
# Check CUDA version
nvcc --version

# Check GPU availability
nvidia-smi
```

**Expected Output**:
- `nvcc --version` should show CUDA 12.1 or higher
- `nvidia-smi` should display your GPU (e.g., "Tesla T4", "A100-SXM4-40GB")

If CUDA is not installed or version is < 12.1, see the [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads).

### Step 2: Create Project Environment

```bash
# Navigate to week1 directory
cd week1-setup

# Create virtual environment
python3 -m venv vllm-env

# Activate environment
source vllm-env/bin/activate  # On Linux/Mac
# Or: vllm-env\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies

Run the installation script:

```bash
# Make script executable
chmod +x install_dependencies.sh

# Run installation
./install_dependencies.sh
```

Or install manually:

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
pip install vllm

# Install additional utilities
pip install huggingface-hub
```

**This will take 5-10 minutes** depending on your internet connection.

### Step 4: Verify Installation

Run the verification script:

```bash
python verify_environment.py
```

**Expected Output**:
```
✓ Python version: 3.10.x
✓ PyTorch installed: 2.x.x
✓ CUDA available: True
✓ CUDA version: 12.1
✓ GPU detected: Tesla T4 (15GB)
✓ vLLM installed: 0.x.x

All checks passed! Your environment is ready.
```

If any check fails, review the error message and troubleshooting section below.

### Step 5: Download Qwen2.5 Model

Run the model download script:

```bash
python download_model.py
```

This will download the Qwen2.5-7B-Instruct model (~15GB) to your local cache. The script shows progress and estimated time.

**Note**: You can skip this step and let vLLM download the model on first use, but explicit download is recommended for faster subsequent runs.

### Step 6: Run Baseline Inference Test

Now for the exciting part - run your first inference!

```bash
python baseline_inference.py
```

**Expected Output**:
```
Loading model: Qwen/Qwen2.5-7B-Instruct...
Model loaded successfully!

Running test inference...
Prompt: "Hello, my name is"
Generated: " Alice, and I'm a software engineer based in San Francisco..."

✓ Inference test successful!
Time taken: 2.3 seconds
Tokens generated: 20
```

Congratulations! You've successfully run your first LLM inference with vLLM! 🎉

## 📂 Files in This Week

| File | Purpose |
|------|---------|
| `README.md` | This guide |
| `install_dependencies.sh` | Automated dependency installation |
| `verify_environment.py` | Environment verification script |
| `download_model.py` | Model download utility |
| `baseline_inference.py` | Basic inference test |
| `experiment_sampling_params.py` | **NEW**: Comprehensive parameter experiments |
| `test_custom_params.py` | **NEW**: Quick custom parameter testing |
| `compare_results.py` | **NEW**: Compare experiment results |
| `requirements.txt` | Python package dependencies |
| `results/` | Directory for saved experiment results |

## 🧪 Experiments to Try

### Quick Experiments (Modify baseline_inference.py)

Once the baseline test works, experiment with these modifications to `baseline_inference.py`:

#### 1. Different Prompts
Try various prompts to see how the model responds:
```python
prompts = [
    "Explain quantum computing in simple terms:",
    "Write a haiku about machine learning:",
    "What is the capital of France?",
]
```

#### 2. Adjust Generation Length
Modify the `max_tokens` parameter:
```python
sampling_params = SamplingParams(
    max_tokens=50,  # Try 10, 50, 100, 200
    temperature=0.7
)
```

#### 3. Temperature and Sampling
Control randomness and creativity:
```python
sampling_params = SamplingParams(
    max_tokens=50,
    temperature=0.1,  # Lower = more deterministic (try 0.1, 0.7, 1.0)
    top_p=0.9,        # Nucleus sampling
    top_k=50          # Top-k sampling
)
```

### 🔬 Advanced Experiments (NEW!)

After understanding the basics, run these comprehensive experiments:

#### Experiment 1: Parameter Testing Suite

Run systematic tests of all sampling parameters:

```bash
python experiment_sampling_params.py
```

**This will test**:
- `max_tokens`: 10, 25, 50, 100, 200
- `temperature`: 0.0, 0.3, 0.7, 1.0, 1.5
- `top_p`: 0.5, 0.7, 0.9, 0.95, 1.0
- `top_k`: 10, 30, 50, 100, 200

Results are automatically saved to `results/` folder with detailed analysis!

**Time**: ~10 minutes

#### Experiment 2: Custom Parameter Testing

Test any parameter combination interactively:

```bash
# Interactive mode (recommended)
python test_custom_params.py

# Or command line
python test_custom_params.py --prompt "Your prompt" --temperature 0.8 --max-tokens 100
```

**Great for**:
- Testing specific use cases
- Finding your ideal parameters
- Quick experimentation

#### Experiment 3: Compare Results

Analyze and compare all your experiment results:

```bash
python compare_results.py
```

**Shows**:
- Side-by-side parameter comparisons
- Recommended combinations for different tasks
- Performance statistics

### 📊 Understanding the Results

All experiments save results to JSON files in `results/` directory:

```
results/
├── sampling_experiments_20260127_143022.json
├── custom_test_20260127_143530.json
└── ...
```

You can review these anytime to understand how parameters affect output!

## 📊 Understanding the Output

When you run `baseline_inference.py`, vLLM loads the model and generates text. Here's what happens under the hood:

1. **Model Loading**: Weights are loaded into GPU memory (~14GB for Qwen2.5-7B in FP16)
2. **Tokenization**: Input text is converted to token IDs
3. **Prefill Phase**: The entire prompt is processed in parallel to build the KV cache
4. **Decode Phase**: Tokens are generated one at a time (autoregressive generation)
5. **Detokenization**: Token IDs are converted back to text

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: 
- Close other GPU applications
- Use a smaller model or a GPU with more VRAM
- Try reducing `gpu_memory_utilization` (covered in Week 3)

### Issue: "torch.cuda.is_available() returns False"
**Solution**:
- Verify CUDA installation: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version
- Check GPU drivers are up to date

### Issue: "Model download is slow or fails"
**Solution**:
- Check internet connection
- Try downloading manually: `huggingface-cli download Qwen/Qwen2.5-7B-Instruct`
- Set `HF_HUB_OFFLINE=1` if model is already cached

### Issue: "vLLM import fails"
**Solution**:
- Ensure virtual environment is activated
- Reinstall vLLM: `pip install --upgrade vllm`
- Check Python version compatibility (3.9-3.11)

## 🎓 Key Concepts Learned

- **vLLM**: High-performance LLM inference engine
- **Qwen2.5**: Open-source instruction-tuned language model
- **LLM Class**: vLLM's main interface for loading models and generating text
- **SamplingParams**: Controls generation behavior (length, randomness, etc.)
- **GPU Memory**: Models are loaded entirely into VRAM for fast inference

## 📖 Additional Reading

- [vLLM Quickstart Guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Understanding LLM Inference](https://huggingface.co/blog/llm-inference-optimization)

## ✅ Week 1 Checklist

Before moving to Week 2, ensure you have:

- [ ] Successfully installed Python 3.9+ in a virtual environment
- [ ] Installed PyTorch with CUDA support
- [ ] Installed vLLM
- [ ] Verified CUDA is accessible (`torch.cuda.is_available() == True`)
- [ ] Downloaded Qwen2.5-7B-Instruct model
- [ ] Run `baseline_inference.py` successfully
- [ ] Experimented with different prompts and parameters
- [ ] Understood the basic vLLM API

## 🔜 Next Steps

Once you've completed all tasks, you're ready for **Week 2: LLM Inference Profiling**!

In Week 2, you'll learn how to:
- Measure latency and throughput
- Benchmark different batch sizes
- Profile GPU utilization
- Establish baseline metrics for comparison

---

**Questions or Issues?** Document them for discussion or review the troubleshooting section above.


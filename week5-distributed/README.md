# Week 5: Parallelism and Distributed Inference

## 🎯 Goals

By the end of this week, you will:
- Configure tensor parallelism for multi-GPU inference
- Understand distributed inference trade-offs
- Simulate concurrent client requests
- Measure multi-GPU scalability
- Optimize for production-level throughput

## 📚 What You'll Learn

- **Tensor Parallelism (TP)**: Splitting model across GPUs on one node
- **Concurrent Request Handling**: Simulating real-world load
- **Scheduler Behavior**: How vLLM manages multiple requests
- **Scalability Analysis**: Single-GPU vs multi-GPU performance
- **Production Patterns**: Request batching and load distribution

## 🔬 Key Concepts

### Tensor Parallelism

**How it works**:
- Model layers are split across multiple GPUs
- Each GPU processes a portion of each layer
- GPUs communicate via NVLink or PCIe
- All GPUs work together on same request

**When to use**:
- Model is too large for single GPU
- Want to reduce latency per request
- Have multiple GPUs on same node

**Configuration**:
```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=4  # Use 4 GPUs
)
```

### Performance Expectations

**Ideal scaling** (4 GPUs):
- Memory: 4x capacity
- Throughput: ~3-4x (due to communication overhead)
- Latency: Similar or slightly higher (communication cost)

**Real-world scaling**:
- Depends on interconnect (NVLink > PCIe)
- Larger batches scale better
- May see 2-3x improvement with 4 GPUs

## 🚀 Running the Experiments

**⚠️ IMPORTANT**: These experiments require multi-GPU access. Options:
1. Rent multi-GPU instance on Runpod/AWS (recommended)
2. Use university/company cluster
3. Skip to Week 6 if single-GPU only (Week 6 includes alternatives)

### Experiment 1: Enable Tensor Parallelism

Test basic multi-GPU setup:

```bash
python test_tensor_parallel.py
```

**What it does**:
- Loads model with different TP sizes (1, 2, 4 GPUs)
- Verifies model loads correctly
- Runs basic inference test
- Compares memory usage across GPUs

**Key check**: All GPUs should show ~equal memory usage.

### Experiment 2: Concurrent Request Simulation

Simulate multiple clients:

```bash
python simulate_concurrent_requests.py
```

**What it does**:
- Spawns multiple async requests
- Measures throughput under load
- Tests scheduler fairness
- Compares concurrent vs batch performance

**Insight**: vLLM's continuous batching should handle both similarly.

### Experiment 3: Scalability Benchmark

Measure scaling efficiency:

```bash
python benchmark_scaling.py
```

**What it does**:
- Tests 1, 2, 4 GPUs (if available)
- Measures throughput at each scale
- Calculates scaling efficiency
- Identifies communication overhead

**Expected**: 2-3x throughput improvement with 4 GPUs.

### Experiment 4: Mixed Workload Test

Test fairness with mixed request types:

```bash
python test_mixed_workload.py
```

**What it does**:
- Sends long and short requests together
- Measures individual latencies
- Tests chunked prefill effectiveness
- Analyzes scheduler fairness

**Goal**: Short requests shouldn't wait for long ones.

### Experiment 5: Full Multi-GPU Benchmark

Run comprehensive multi-GPU benchmark:

```bash
python run_distributed_benchmark.py
```

**What it does**:
- Runs all Week 2 benchmarks on multi-GPU
- Compares with single-GPU baseline
- Generates scaling report
- Saves results for comparison

## 📂 Files in This Week

| File | Purpose |
|------|---------|
| `README.md` | This guide |
| `test_tensor_parallel.py` | Basic TP setup and verification |
| `simulate_concurrent_requests.py` | Concurrent request handling |
| `benchmark_scaling.py` | Multi-GPU scaling analysis |
| `test_mixed_workload.py` | Fairness testing |
| `run_distributed_benchmark.py` | Complete multi-GPU benchmark |
| `requirements.txt` | Additional dependencies |

## 💻 Setting Up Multi-GPU Environment

### Option 1: Runpod (Recommended for Learning)

1. **Create Runpod Account**: https://runpod.io
2. **Select GPU Pod**:
   - Community Cloud → GPU Instances
   - Choose: "4x A100 (40GB)" or "2x A100"
   - Template: PyTorch or NVIDIA CUDA
3. **Connect via SSH or Jupyter**
4. **Install dependencies**:
   ```bash
   pip install vllm torch
   ```
5. **Clone your code and run experiments**

**Cost**: ~$2-4/hour for 4x A100 (terminate after use!)

### Option 2: AWS EC2

1. **Launch instance**: `p4d.24xlarge` (8x A100) or `p3.8xlarge` (4x V100)
2. **Use Deep Learning AMI** (includes CUDA, PyTorch)
3. **Install vLLM**:
   ```bash
   pip install vllm
   ```
4. **Run experiments**

**Cost**: ~$3-10/hour depending on instance

### Option 3: Google Colab (Limited)

- Colab Pro+ offers some multi-GPU access
- Limited to shorter sessions
- Good for testing, not full benchmarks

## 🧪 Expected Results

### Typical Scaling (Qwen2.5-7B)

| GPUs | Throughput | Speedup | Efficiency |
|------|------------|---------|------------|
| 1    | 450 tok/s  | 1.0x    | 100%       |
| 2    | 800 tok/s  | 1.8x    | 90%        |
| 4    | 1400 tok/s | 3.1x    | 78%        |

**Why not 4x?**
- Communication overhead (GPU-GPU transfers)
- Synchronization costs
- PCIe bandwidth limits (if not NVLink)

## 🐛 Troubleshooting

### Issue: "No module named 'torch.distributed'"
**Solution**: Ensure PyTorch is properly installed with distributed support

### Issue: "NCCL error" or "distributed init failed"
**Solutions**:
- Check all GPUs are visible: `nvidia-smi`
- Verify NCCL is installed: `python -c "import torch; print(torch.cuda.nccl.version())"`
- Set environment: `export NCCL_DEBUG=INFO`

### Issue: GPUs have unequal memory usage
**Cause**: TP not working correctly
**Solution**: Check tensor_parallel_size matches GPU count

### Issue: Slower than single GPU
**Causes**:
- PCIe bottleneck (need NVLink)
- Batch size too small (communication dominates)
- Model too small to benefit from TP

### Issue: OOM with multi-GPU
**Why**: TP uses more memory for activations and communication buffers
**Solution**: Reduce max_num_seqs or batch size

## 💡 Optimization Tips

### Maximizing Multi-GPU Performance

1. **Use NVLink if available**:
   - Check: `nvidia-smi nvlink --status`
   - Much faster GPU-GPU communication

2. **Increase batch size**:
   - Larger batches amortize communication
   - TP benefits from parallel work

3. **Tune max_num_seqs**:
   - Can be higher with more GPUs
   - More memory available

4. **Enable chunked prefill**:
   - Better utilization with mixed workloads
   - Reduces head-of-line blocking

## 📖 Additional Resources

- [vLLM Distributed Inference](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [PyTorch Distributed Tutorial](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)

## ✅ Week 5 Checklist

Before moving to Week 6, ensure you have:

- [ ] Access to multi-GPU environment (or understanding of limitations)
- [ ] Run `test_tensor_parallel.py` successfully
- [ ] Tested concurrent requests with `simulate_concurrent_requests.py`
- [ ] Measured scaling with `benchmark_scaling.py`
- [ ] Compared multi-GPU vs single-GPU performance
- [ ] Understood scaling efficiency and overhead
- [ ] Documented your optimal multi-GPU configuration
- [ ] Noted any cost implications for production

## 💰 Cost Management

**CRITICAL**: Multi-GPU instances are expensive!

### After Each Experiment:
1. Save your results locally
2. Terminate the instance (don't just stop!)
3. Verify termination in cloud dashboard
4. Check no lingering volumes or IPs

### Runpod Teardown:
```
1. Pods → Select your pod → Terminate
2. Confirm termination
3. Verify status: "Terminated"
```

### AWS Teardown:
```bash
# List instances
aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,State.Name]'

# Terminate
aws ec2 terminate-instances --instance-ids i-xxxxx

# Verify
aws ec2 describe-instances --instance-ids i-xxxxx
```

## 🔜 Next Steps

Ready for **Week 6: Production Deployment**!

Week 6 covers:
- Containerizing your inference system
- Deploying on Runpod (serverless)
- Deploying on AWS EKS (Kubernetes)
- Production monitoring and scaling
- **Complete teardown procedures**

---

**Note**: If you don't have multi-GPU access, review the concepts and code, then proceed to Week 6 which includes single-GPU deployment options.


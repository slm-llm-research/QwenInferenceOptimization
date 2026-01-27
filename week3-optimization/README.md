# Week 3: GPU-Level Optimization and Profiling

## 🎯 Goals

By the end of this week, you will:
- Use profiling tools to identify GPU bottlenecks
- Optimize vLLM configuration parameters
- Improve throughput by tuning memory utilization
- Understand PagedAttention and KV cache efficiency
- Achieve measurable performance improvements over baseline

## 📚 What You'll Learn

- **GPU Profiling**: Using PyTorch Profiler and Nsight Systems
- **vLLM Parameters**: `max_num_seqs`, `gpu_memory_utilization`, `swap_space`
- **PagedAttention**: How vLLM manages KV cache efficiently
- **Bottleneck Analysis**: Compute-bound vs memory-bound workloads
- **Performance Tuning**: Finding optimal configurations for your hardware

## 🔬 Key Concepts

### vLLM Parameters to Optimize

#### 1. `gpu_memory_utilization` (default: 0.9)
- **What**: Percentage of GPU memory vLLM can use
- **Impact**: More memory = larger KV cache = more concurrent sequences
- **Range**: 0.7 - 0.95 (don't use 1.0 - need headroom for CUDA)
- **When to increase**: If GPU has unused memory
- **When to decrease**: If getting OOM errors

#### 2. `max_num_seqs` (default: 256)
- **What**: Maximum concurrent sequences in a batch
- **Impact**: Higher = better GPU utilization = more throughput
- **Limit**: Constrained by available GPU memory
- **Sweet spot**: As high as possible without OOM

#### 3. `swap_space` (default: 4GB per GPU)
- **What**: CPU memory for offloading KV cache when GPU is full
- **Impact**: Allows more sequences but adds latency (CPU-GPU transfer)
- **When to use**: When you need very high concurrency
- **Trade-off**: Throughput vs latency

### Compute-Bound vs Memory-Bound

**Compute-bound**:
- GPU cores are fully utilized
- Memory bandwidth not saturated
- Sign: High SM (streaming multiprocessor) utilization
- Solution: Already well-optimized

**Memory-bound**:
- Waiting on memory transfers
- GPU cores underutilized
- Sign: Low SM utilization, high memory transactions
- Solution: Batch more operations, optimize memory access patterns

### PagedAttention

vLLM's key innovation - manages KV cache like OS manages memory:
- Divides KV cache into fixed-size "pages" (blocks)
- Shares memory between sequences (e.g., for same prompt prefix)
- Eliminates fragmentation
- Enables efficient memory use

## 🚀 Running the Experiments

### Experiment 1: Profile Baseline with PyTorch Profiler

Capture detailed GPU activity:

```bash
python profile_baseline.py
```

**What it does**:
- Runs inference with PyTorch Profiler enabled
- Generates trace file in `./profiles/`
- Shows top operations by GPU time
- Identifies bottlenecks

**How to analyze**:
```bash
# View in TensorBoard
pip install torch-tb-profiler
tensorboard --logdir=./profiles

# Open browser to http://localhost:6006
```

Look for:
- Kernel occupancy (how busy GPU is)
- Memory operations (data transfers)
- Idle time (gaps = optimization opportunity)

### Experiment 2: Tune `gpu_memory_utilization`

Test different memory allocation levels:

```bash
python optimize_memory_utilization.py
```

**What it does**:
- Tests gpu_memory_utilization: 0.7, 0.8, 0.9, 0.95
- Measures throughput for each setting
- Finds optimal memory utilization for your GPU

**Expected outcome**:
- Higher values → more memory → higher throughput
- Until OOM limit is reached

### Experiment 3: Tune `max_num_seqs`

Find the optimal concurrent sequence limit:

```bash
python optimize_max_num_seqs.py
```

**What it does**:
- Tests max_num_seqs: 64, 128, 256, 512, 1024
- Measures throughput at each level
- Identifies the sweet spot before OOM

**Key insight**:
- Throughput improves until memory limit
- Beyond that, either OOM or swapping to CPU (slower)

### Experiment 4: PagedAttention Analysis

Debug KV cache efficiency:

```bash
python analyze_paged_attention.py
```

**What it does**:
- Enables vLLM's KV cache statistics
- Shows memory block utilization
- Identifies fragmentation or waste

**Optimal result**:
- >90% KV cache block utilization
- Minimal fragmentation
- Efficient memory sharing

### Experiment 5: Combined Optimization

Apply all optimizations together:

```bash
python run_optimized_benchmark.py
```

**What it does**:
- Uses best parameters from previous experiments
- Runs same benchmarks as Week 2
- Compares against baseline

**Expected improvements**:
- 1.5x - 3x throughput increase
- Better GPU utilization
- Lower latency variance

## 📂 Files in This Week

| File | Purpose |
|------|---------|
| `README.md` | This guide |
| `profile_baseline.py` | PyTorch Profiler integration |
| `optimize_memory_utilization.py` | Tune GPU memory parameter |
| `optimize_max_num_seqs.py` | Tune concurrent sequences |
| `analyze_paged_attention.py` | KV cache efficiency analysis |
| `run_optimized_benchmark.py` | Full optimized benchmark |
| `compare_results.py` | Compare Week 2 vs Week 3 results |
| `requirements.txt` | Additional dependencies |

## 📊 Understanding Profiler Output

### PyTorch Profiler Key Metrics

When viewing in TensorBoard:

1. **Overview Tab**:
   - GPU utilization over time
   - Should be >80% during generation
   - Gaps indicate idle time

2. **Operator View**:
   - Top operations by GPU time
   - Look for attention kernels (should dominate)
   - Check for unexpected CPU operations

3. **Kernel View**:
   - Individual CUDA kernels
   - Check occupancy (>50% is good)
   - Memory bandwidth utilization

### What to Look For

✅ **Good signs**:
- Consistent GPU utilization >80%
- Attention/matmul kernels dominate time
- Minimal CPU-GPU synchronization

⚠️ **Issues**:
- GPU utilization <50% → underutilized
- Many small gaps → batching issues
- High memory copy time → memory bottleneck

## 🧪 Expected Results

### Typical Improvements (Qwen2.5-7B on A100)

| Metric | Baseline (Week 2) | Optimized (Week 3) | Improvement |
|--------|-------------------|--------------------|-----------| 
| Single request | 42 tok/s | 45 tok/s | 1.07x |
| Batch 16 | 450 tok/s | 680 tok/s | 1.5x |
| Batch 32 | 500 tok/s | 850 tok/s | 1.7x |
| GPU utilization | 65% | 88% | +23% |

Your results will vary by GPU model!

## 🐛 Troubleshooting

### Issue: "CUDA out of memory" when tuning parameters
**Expected**: 
- This helps find the upper limit!
- Back off to the last working configuration
- Try smaller increments

### Issue: Profiler crashes or hangs
**Solutions**:
- Reduce profiling duration
- Profile with smaller batch
- Ensure enough disk space for traces

### Issue: No improvement from optimization
**Check**:
- Baseline was already well-optimized
- GPU might be compute-bound (less room for improvement)
- Verify parameters are actually being used

### Issue: Throughput decreases with higher max_num_seqs
**Cause**:
- Hitting memory limit → swapping to CPU
- Reduce max_num_seqs or decrease sequence length

## 💡 Optimization Strategy

### Step-by-Step Tuning Process

1. **Establish Baseline**:
   - Use Week 2 results
   - Note current throughput and GPU utilization

2. **Profile First**:
   - Run `profile_baseline.py`
   - Identify if compute or memory bound
   - Look for obvious inefficiencies

3. **Tune Memory**:
   - Run `optimize_memory_utilization.py`
   - Find highest safe value (no OOM)

4. **Tune Concurrency**:
   - Run `optimize_max_num_seqs.py`
   - Find maximum sequences supported

5. **Verify**:
   - Run `run_optimized_benchmark.py`
   - Compare against baseline
   - Document improvements

6. **Iterate**:
   - If improvement is modest, try combinations
   - Consider workload-specific tuning

## 📖 Additional Resources

- [vLLM Engine Arguments](https://docs.vllm.ai/en/latest/models/engine_args.html)
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [Understanding GPU Performance](https://developer.nvidia.com/blog/how-to-implement-performance-metrics-in-cuda-cc/)

## ✅ Week 3 Checklist

Before moving to Week 4, ensure you have:

- [ ] Run `profile_baseline.py` and viewed traces in TensorBoard
- [ ] Optimized `gpu_memory_utilization` for your GPU
- [ ] Optimized `max_num_seqs` for your workload
- [ ] Analyzed PagedAttention efficiency
- [ ] Run `run_optimized_benchmark.py` with best settings
- [ ] Compared results with Week 2 baseline using `compare_results.py`
- [ ] Documented your optimal configuration
- [ ] Achieved measurable throughput improvement

## 📝 Document Your Configuration

Save your optimal configuration for future use:

```python
# My Optimized vLLM Configuration
optimal_config = {
    "gpu_memory_utilization": 0.95,  # Your best value
    "max_num_seqs": 512,              # Your best value
    "swap_space": 4,                  # GB, if needed
}
```

You'll use this in Week 5-7!

## 🔜 Next Steps

Ready for **Week 4: Integration Week**!

Week 4 is lighter - focused on:
- Reviewing advanced vLLM features
- Guest lectures and reading
- Consolidating knowledge
- Preparing for distributed deployment

---

**Questions?** Review profiler traces or check vLLM documentation for parameter details.


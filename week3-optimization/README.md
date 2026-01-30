# Week 3: GPU-Level Optimization and Profiling

## 🎯 Goals

By the end of this week, you will:
- Use profiling tools to identify GPU bottlenecks
- Optimize vLLM configuration parameters
- Improve throughput by tuning memory utilization
- Understand PagedAttention and KV cache efficiency
- Achieve measurable performance improvements over baseline

## 📊 Week 2 Context: What We're Optimizing

Week 2 benchmarking revealed three critical bottlenecks (see [`../week2-profiling/INSIGHTS.md`](../week2-profiling/INSIGHTS.md)):

### **1. Queue Time Dominance (85%) - PRIMARY TARGET** ⚠️

Your Week 2 measurements showed:
- Queue time: **10.47s (85%)** - Requests waiting to be processed
- Generation time: **1.78s (15%)** - Actual token generation
- **Total latency (P50): 12.25s**

**This means**: Your GPU spends most of its time idle while requests queue up. This is a **configuration issue**, not a hardware limitation!

**Week 3 Solution**: Increase `max_num_seqs` to allow more concurrent processing.

### **2. Long Sequence Performance Issues**

Week 2 results:
- Very long sequences (500+ tokens): **4.4s at P95**
- Super-linear scaling: **15.8x slower** than short sequences
- Cannot meet < 3s SLA targets for long-form content

**Week 3 Solution**: Enable chunked prefill for efficient long prompt handling.

### **3. Tail Latency Gap (P95 = 3x P50)**

Week 2 distribution:
- P50: 0.850s (good!)
- P95: 2.544s (3x higher)
- P99: 4.396s (5x higher)

5% of users experience much slower responses than median users.

**Week 3 Solution**: Reducing queue time will improve tail latencies most.

### **4. Limited Request Capacity**

Current capacity:
- Throughput: **949 tokens/second**
- Request rate: **4.6 requests/second**
- Daily capacity: ~400,000 requests

**Week 3 Target**: Increase to 10-15 req/s through better GPU utilization.

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

## 🚀 Quick Start: Run All Optimizations

**Recommended**: Run the full optimization suite to automatically execute all experiments:

```bash
python run_full_optimization_suite.py
```

This will:
1. Profile your baseline
2. Optimize memory utilization
3. Optimize max_num_seqs (addresses 85% queue time!)
4. Optimize queue time directly
5. Test chunked prefill for long sequences
6. Run comprehensive benchmark with optimal settings
7. Generate before/after visualizations

**Estimated time**: 2-3 hours

---

## 🔬 Individual Experiments

You can also run optimizations individually:

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

### Experiment 3: Tune `max_num_seqs` ⭐ CRITICAL

Find the optimal concurrent sequence limit:

```bash
python optimize_max_num_seqs.py
```

**What it does**:
- Tests max_num_seqs: **256, 512, 1024, 2048, 4096** (updated for Week 2 findings!)
- Measures throughput AND queue time at each level
- Identifies optimal value to reduce 85% queue time

**Key insight**:
- **THIS IS YOUR PRIMARY BOTTLENECK** (85% queue time)
- Higher values allow more concurrent requests = less waiting
- Throughput improves until memory limit

**Week 2 Baseline**: Default (~256) caused 85% queue time
**Week 3 Goal**: Find value that reduces queue time to <50%

### Experiment 3B: Optimize Queue Time Directly ⭐ NEW

Specifically target the queue time bottleneck:

```bash
python optimize_queue_time.py
```

**What it does**:
- Tests aggressive max_num_seqs values
- Measures queue time percentage directly
- Compares against Week 2 baseline (85%)

**Target**: Reduce queue time from 85% to below 50%

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

### Experiment 5: Chunked Prefill for Long Sequences ⭐ NEW

Optimize handling of long sequences (300+ tokens):

```bash
python optimize_chunked_prefill.py
```

**What it does**:
- Tests chunked prefill with different chunk sizes
- Measures latency for 100, 200, 300, 400, 500 token sequences
- Compares against Week 2 baseline (4.4s for 500+ tokens)

**Target**: Reduce long sequence P95 from 4.4s to <3.0s

### Experiment 6: Comprehensive Optimized Benchmark

Apply all optimizations and run full benchmark suite:

```bash
python run_comprehensive_optimized_benchmark.py
```

**What it does**:
- Loads optimal parameters from all experiments
- Runs ALL Week 2 benchmarks with optimized config
- Enables direct before/after comparison

**Expected improvements**:
- 2-3x throughput increase (949 → 1900+ tok/s)
- Queue time reduction (85% → <50%)
- Long sequence improvement (4.4s → <3.0s)
- Better tail latencies (P95/P99)

### Experiment 7: Compare Week 2 vs Week 3 ⭐ NEW

Generate detailed comparison report:

```bash
python compare_week2_week3.py
```

Shows improvement percentages for all metrics.

### Experiment 8: Visualize Results ⭐ NEW

Generate before/after comparison plots:

```bash
python visualize_optimization_results.py
```

Creates visual dashboard showing optimization impact.

## 📂 Files in This Week

| File | Purpose | Priority |
|------|---------|----------|
| `README.md` | This guide | - |
| **`run_full_optimization_suite.py`** | **Master script - runs everything** | ⭐ START HERE |
| `optimize_queue_time.py` | Target 85% queue bottleneck | ⭐ HIGH |
| `optimize_max_num_seqs.py` | Tune concurrent sequences (updated) | ⭐ HIGH |
| `optimize_memory_utilization.py` | Tune GPU memory parameter (updated) | HIGH |
| `optimize_chunked_prefill.py` | Optimize long sequences | HIGH |
| `profile_baseline.py` | PyTorch Profiler integration | MEDIUM |
| `run_comprehensive_optimized_benchmark.py` | Full optimized benchmark (Week 2 replica) | ⭐ HIGH |
| `run_optimized_benchmark.py` | Simple optimized benchmark | MEDIUM |
| `compare_week2_week3.py` | Detailed before/after comparison | ⭐ HIGH |
| `visualize_optimization_results.py` | Generate comparison plots | ⭐ HIGH |
| `OPTIMIZATION_RESULTS.md` | Document your improvements | - |
| `requirements.txt` | Additional dependencies | - |

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

### Typical Improvements Based on Week 2 Findings

| Metric | Week 2 Baseline | Week 3 Target | Expected Strategy |
|--------|----------------|---------------|-------------------|
| **Queue Time %** | 85% | <50% | Increase max_num_seqs to 1024-2048 |
| **Throughput** | 949 tok/s | 1200-1900 tok/s | Combined optimizations |
| **Request Capacity** | 4.6 req/s | 10-15 req/s | Better concurrency |
| **Long Seq (500+ tok)** | 4.4s P95 | <3.0s | Chunked prefill |
| **P95 Latency** | 2.54s | <2.0s | Queue time reduction |

**Your actual results will vary by GPU model and workload!**

These targets are based on YOUR specific Week 2 measurements, not generic benchmarks.

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

### Core Optimizations
- [ ] Run `run_full_optimization_suite.py` OR individual scripts below
- [ ] ⭐ Optimized `max_num_seqs` (addresses 85% queue time bottleneck)
- [ ] ⭐ Measured queue time improvement (target: <50%)
- [ ] Optimized `gpu_memory_utilization` for your GPU
- [ ] Tested chunked prefill for long sequences
- [ ] Run `run_comprehensive_optimized_benchmark.py` with optimal settings

### Analysis & Documentation
- [ ] ⭐ Run `compare_week2_week3.py` for detailed comparison
- [ ] ⭐ Run `visualize_optimization_results.py` for plots
- [ ] Review all generated visualizations in `results/` folder
- [ ] Documented optimal configuration in `OPTIMIZATION_RESULTS.md`

### Validation
- [ ] ✅ Queue time reduced from 85% (target: <50%)
- [ ] ✅ Throughput improved (target: >1200 tok/s)
- [ ] ✅ Long sequence P95 improved (target: <3s)
- [ ] ✅ Overall P95 latency improved (target: <2s)

### Optional
- [ ] Run `profile_baseline.py` and viewed traces in TensorBoard
- [ ] Analyzed PagedAttention efficiency

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


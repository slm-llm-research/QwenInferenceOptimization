# Week 2: LLM Inference Profiling (Baseline)

## 🎯 Goals

By the end of this week, you will:
- Understand how to measure LLM inference performance
- Benchmark latency and throughput under various conditions
- Analyze the impact of batch size and sequence length
- Monitor GPU utilization during inference
- Establish baseline metrics for future comparisons

## 📚 What You'll Learn

- **Latency**: Time to complete a single inference request
- **Throughput**: Total tokens generated per second
- **Batch Processing**: How multiple requests improve GPU utilization
- **Continuous Batching**: vLLM's dynamic request scheduling
- **Performance Metrics**: How to measure and interpret inference speed

## 🔬 Key Concepts

### Latency vs Throughput

- **Latency** (ms): How long a user waits for a response
  - Important for: Interactive applications, chatbots
  - Lower is better (faster response)
  
- **Throughput** (tokens/sec): How many tokens the system generates total
  - Important for: Batch processing, high-traffic services
  - Higher is better (more work done)

### Batch Size Impact

- **Small batches (1-4)**: Low latency, underutilized GPU
- **Large batches (16-64)**: Higher throughput, but each request waits longer
- **vLLM's continuous batching**: Automatically mixes requests for optimal GPU usage

### Why Profiling Matters

Before optimizing, you need to know:
1. What is the current performance? (baseline)
2. Where are the bottlenecks? (GPU, memory, CPU)
3. How does workload affect performance? (batch size, sequence length)

## 📊 Experiments This Week

You will run 4 types of benchmarks:

1. **Single Request Latency**: Baseline speed for one prompt
2. **Batch Throughput**: Performance with multiple concurrent requests
3. **Sequence Length Impact**: How prompt/generation length affects speed
4. **GPU Utilization Analysis**: Resource usage patterns

## 🚀 Running the Experiments

### Experiment 1: Basic Latency Test

Measure how long it takes to process single requests:

```bash
python benchmark_latency.py
```

**What it does**:
- Runs inference on a single prompt
- Measures end-to-end time
- Calculates tokens per second
- Repeats multiple times for accuracy

**Expected output**:
```
Single Request Latency Test
Average latency: 1.2s
Average throughput: 41.7 tokens/sec
```

### Experiment 2: Batch Throughput Test

Test performance with multiple simultaneous requests:

```bash
python benchmark_throughput.py
```

**What it does**:
- Tests batch sizes: 1, 4, 8, 16, 32
- Measures total throughput for each batch size
- Shows GPU utilization improvement with batching

**Expected output**:
```
Batch size 1:  Throughput = 42 tokens/sec
Batch size 4:  Throughput = 150 tokens/sec (3.6x)
Batch size 8:  Throughput = 280 tokens/sec (6.7x)
Batch size 16: Throughput = 450 tokens/sec (10.7x)
```

**Key insight**: Throughput scales almost linearly with batch size up to GPU saturation!

### Experiment 3: Sequence Length Test

Analyze how input/output length affects performance:

```bash
python benchmark_sequence_length.py
```

**What it does**:
- Tests different prompt lengths: 10, 50, 200, 1000 tokens
- Tests different output lengths: 50, 100, 200 tokens
- Shows how prefill vs decode phases scale

**Expected observations**:
- Longer prompts → longer initial delay (prefill phase)
- Longer outputs → more total time but similar per-token rate
- Very long contexts may reduce throughput

### Experiment 4: Full Benchmark Suite

Run all tests and generate a comprehensive report:

```bash
python run_all_benchmarks.py
```

**What it does**:
- Runs all experiments above
- Generates a results table
- Saves results to `results/baseline_metrics.json`
- Creates visualizations (if matplotlib available)

### Experiment 5: GPU Utilization Monitoring

Monitor GPU during inference (requires separate terminal):

```bash
# Terminal 1: Start monitoring
python monitor_gpu.py

# Terminal 2: Run inference
python benchmark_throughput.py
```

**What it shows**:
- GPU utilization percentage over time
- Memory usage patterns
- Identifies idle periods (optimization opportunities)

## 📂 Files in This Week

| File | Purpose |
|------|---------|
| `README.md` | This guide |
| `benchmark_latency.py` | Single request latency tests |
| `benchmark_throughput.py` | Batch throughput tests |
| `benchmark_sequence_length.py` | Sequence length impact analysis |
| `run_all_benchmarks.py` | Run complete benchmark suite |
| `monitor_gpu.py` | Real-time GPU monitoring |
| `utils.py` | Shared utility functions |
| `requirements.txt` | Additional dependencies |
| `results/` | Output directory for results |

## 📈 Understanding Your Results

### Good Baseline Metrics (Qwen2.5-7B on A100)

- Single request latency: 1-2 seconds (50 tokens)
- Throughput at batch=16: 400-600 tokens/sec
- GPU utilization: 70-90% during generation

### What to Look For

✅ **Good signs**:
- Higher batch sizes → proportionally higher throughput
- GPU utilization >70% during generation
- Consistent latency across multiple runs

⚠️ **Potential issues**:
- GPU utilization <50% → underutilized (can optimize)
- Throughput doesn't scale with batch → memory bottleneck
- High variance in latency → system interference

## 🔍 Analysis Questions

After running experiments, answer these:

1. **How does batch size affect throughput?**
   - Plot: batch size vs tokens/sec
   - At what batch size does scaling slow down?

2. **What's the cost of longer sequences?**
   - Does a 2x longer prompt take 2x longer?
   - How does output length compare?

3. **GPU Utilization:**
   - Is the GPU fully utilized during generation?
   - Are there idle gaps between requests?

4. **Bottleneck Identification:**
   - Compute-bound (high GPU usage) or memory-bound (waiting)?
   - Does increasing batch size improve utilization?

## 🧪 Optional: Advanced Profiling

If you want to dive deeper:

### Option 1: PyTorch Profiler

```python
# Add to any benchmark script
import torch.profiler as profiler

with profiler.profile() as prof:
    outputs = llm.generate(prompts, sampling_params)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Option 2: NVIDIA nvidia-smi

Monitor GPU in real-time (separate terminal):

```bash
watch -n 0.5 nvidia-smi
```

Or log to file:

```bash
nvidia-smi dmon -s ucm -o TD > gpu_log.txt
# Run benchmark in another terminal
# Stop with Ctrl+C
```

## 📊 Saving Your Results

All benchmarks save results to `results/` directory:

```
results/
├── baseline_metrics.json          # Raw data
├── latency_results.csv            # Latency tests
├── throughput_results.csv         # Throughput tests
└── sequence_length_results.csv    # Sequence tests
```

These will be compared against optimized results in Week 3!

## 🐛 Troubleshooting

### Issue: "CUDA out of memory" with large batches
**Solution**: 
- Reduce batch size
- Decrease max_tokens
- This is expected - finding the limit is part of profiling!

### Issue: Low GPU utilization
**Expected**: 
- Single requests don't fully utilize GPU
- This is why batching helps!
- Will optimize in Week 3

### Issue: Inconsistent results
**Solutions**:
- Run more iterations (increase `num_runs`)
- Ensure no other processes using GPU
- Use `temperature=0` for deterministic generation

### Issue: Scripts run very slowly
**Check**:
- GPU is actually being used: `nvidia-smi`
- CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Not running on CPU by accident

## 🎓 Key Takeaways

After Week 2, you should understand:

1. **Baseline Performance**: Your model's default speed
2. **Batching Benefits**: Why serving multiple requests together helps
3. **Scaling Behavior**: How performance changes with load
4. **Bottlenecks**: Whether you're compute or memory limited

These insights guide Week 3's optimizations!

## 📖 Additional Resources

- [vLLM Performance Benchmark](https://docs.vllm.ai/en/latest/performance/benchmarking.html)
- [Understanding GPU Utilization](https://developer.nvidia.com/blog/how-to-optimize-gpu-utilization/)
- [LLM Inference Math](https://kipp.ly/blog/transformer-inference-arithmetic/)

## ✅ Week 2 Checklist

Before moving to Week 3, ensure you have:

- [ ] Run `benchmark_latency.py` successfully
- [ ] Run `benchmark_throughput.py` with multiple batch sizes
- [ ] Run `benchmark_sequence_length.py` to see scaling
- [ ] Completed `run_all_benchmarks.py` for full baseline
- [ ] Monitored GPU utilization with `monitor_gpu.py`
- [ ] Saved results to `results/` directory
- [ ] Identified 1-2 bottlenecks or optimization opportunities
- [ ] Understood the difference between latency and throughput

## 🔜 Next Steps

Ready for **Week 3: GPU-Level Optimization**!

In Week 3, you'll learn:
- How to use PyTorch Profiler for detailed analysis
- Tuning vLLM parameters (`max_num_seqs`, `gpu_memory_utilization`)
- PagedAttention efficiency debugging
- Reducing memory waste and improving throughput

Your baseline metrics from this week will prove the improvements!

---

**Questions?** Review this README or check the inline comments in the scripts.


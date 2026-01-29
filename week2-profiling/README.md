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

**🆕 NEW: Production-Grade Benchmarking Options**

We now offer **production-realistic versions** of both key benchmarks!

### 📊 Two Testing Approaches

**1. Systematic Testing** (Good for optimization tracking)
- Tests controlled scenarios (prompt length × generation length matrices)
- Consistent conditions for fair comparison
- Use for: Week 3 optimization comparison, understanding scaling behavior

**2. Production Testing** ⭐ (Essential for SLA validation)
- Tests realistic mixed workloads (varied use cases, sizes, patterns)
- Measures P50/P90/P95/P99 latencies
- Use for: SLA validation, capacity planning, production readiness

### 🎯 Available Benchmarks

**Latency Testing:**
| Version | Approach | Time | Command |
|---------|----------|------|---------|
| Basic | Systematic | 2-3 min | `python benchmark_latency.py` |
| Comprehensive | Systematic | 5-8 min | `python benchmark_latency_comprehensive.py` |
| **Production** ⭐ | **Production** | **4-6 min** | `python benchmark_latency_comprehensive.py --production` |

**Throughput Testing:**
| Version | Approach | Time | Command |
|---------|----------|------|---------|
| Basic | Systematic | 3-5 min | `python benchmark_throughput.py` |
| **Production** ⭐ | **Production** | **3-10 min** | `python benchmark_throughput_production.py` |

### 💡 Quick Recommendations

**For Learning (Week 2):**
```bash
# Day 1: Understand basics
python benchmark_latency.py              # 3 min
python benchmark_throughput.py           # 5 min

# Day 2: Comprehensive baseline ⭐
python benchmark_latency_comprehensive.py             # Systematic
python benchmark_latency_comprehensive.py --production  # Production
python benchmark_throughput_production.py             # Production
```

**For Production Deployment:**
```bash
# MUST run these for SLA validation:
python benchmark_latency_comprehensive.py --production
python benchmark_throughput_production.py --stress

# You need P95/P99 metrics to guarantee SLAs!
```

**For Week 3 Optimization:**
```bash
# Use systematic versions for fair before/after comparison:
python benchmark_latency_comprehensive.py      # Week 2 baseline
# ... optimize in Week 3 ...
python benchmark_latency_comprehensive.py      # Week 3 results
# → Easy to compare same test matrix!
```

### 🔑 Key Differences

**Systematic vs Production:**
- **Systematic**: "Does doubling generation length double latency?" → Optimization insights
- **Production**: "Can I meet 1.5s P95 SLA?" → Deployment readiness

**Why both matter:**
- Systematic = understand behavior, track improvements
- Production = validate real-world performance, set SLAs

**Example:**
```
Systematic test: "Average latency: 0.8s" ← Good for comparison
Production test: "P95: 1.42s, P99: 2.87s" ← Good for SLA validation
```

## 🚀 Running the Experiments

### Experiment 1: Latency Testing

Measure how long it takes to process requests:

```bash
# Option A: Basic (3 test cases, 5 runs) - 2-3 minutes
python benchmark_latency.py

# Option B: Comprehensive Systematic (9 test cases, 10 runs) - 5-8 minutes
python benchmark_latency_comprehensive.py

# Option C: Production Workload (100 mixed requests) - 4-6 minutes ⭐ NEW!
python benchmark_latency_comprehensive.py --production
```

**What each does**:
- **Basic**: Quick baseline with 3 scenarios
- **Comprehensive (Systematic)**: 3×3 matrix (prompt × generation length) for thorough coverage
- **Comprehensive (Production)**: Mixed realistic workloads (chat, code, summarization, Q&A) with P50/P90/P95/P99 analysis

**Which to use?**
- **Learning basics**: Start with basic, then try comprehensive
- **Week 2 baseline**: Use comprehensive systematic (good for Week 3 comparison)
- **SLA validation**: Use production mode (essential for P95/P99 metrics!)
- **Both recommended**: Run systematic for optimization tracking, production for SLA planning

**Expected output (systematic)**:
```
Test Case: Short prompt, 50 tokens
Average latency: 1.2s
Throughput: 41.7 tokens/sec
```

**Expected output (production)** ⭐:
```
Latency Distribution:
  P50: 0.45s  |  P90: 1.15s  |  P95: 1.42s  |  P99: 2.87s

By Use Case (P95):
  chat:    0.89s ✓
  code:    1.23s ✓
  summary: 0.67s ✓
  
Production insights: P95 < 1.5s for most use cases ✓
```

### Experiment 2: Batch Throughput Test

Test performance with multiple simultaneous requests:

```bash
# Basic version: Simple batch scaling (3-5 min)
python benchmark_throughput.py

# OR: Production version: Realistic mixed workloads (3-10 min) [RECOMMENDED for deployment]
python benchmark_throughput_production.py
```

**What it does**:
- **Basic**: Tests batch sizes (1, 4, 8, 16, 32) with uniform prompts
- **Production**: Tests realistic mixed workloads with various prompt types, lengths, and sustained load
- Shows GPU utilization improvement with batching

**Which one to use?**
- **Basic**: Great for learning batching concepts
- **Production**: Essential for SLA planning, capacity estimation, P95/P99 latency analysis
- **Both**: Run both to see theory vs practice! (recommended)
- See `THROUGHPUT_COMPARISON.md` for detailed guide

**Expected output (basic)**:
```
Batch size 1:  Throughput = 42 tokens/sec
Batch size 4:  Throughput = 150 tokens/sec (3.6x)
Batch size 16: Throughput = 450 tokens/sec (10.7x)
```

**Expected output (production)**:
```
Mixed Workload: 380 tokens/sec
P50 latency: 0.45s  |  P95 latency: 1.23s  |  P99 latency: 2.87s
Sustained load: 5 req/s stable, breaking point at ~12 req/s
```

**Key insight**: 
- Basic: Throughput scales with batch size!
- Production: Real workload is heterogeneous - P95/P99 matter for SLAs!

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
| `README.md` | This guide (you are here!) |
| **Latency Benchmarks** | |
| `benchmark_latency.py` | Basic latency (3 scenarios, quick) |
| `benchmark_latency_comprehensive.py` | 📊 **ENHANCED!** Systematic OR Production modes:<br>• `--standard`: 9-case matrix<br>• `--production`: 100 mixed requests with P95/P99 ⭐ |
| **Throughput Benchmarks** | |
| `benchmark_throughput.py` | Basic batch throughput (simple scaling) |
| `benchmark_throughput_production.py` | 🏭 **NEW!** Production workload (mixed requests, sustained load, stress test) |
| **Guides & Documentation** | |
| `BENCHMARKING_GUIDE.md` | Detailed latency testing strategies |
| `THROUGHPUT_COMPARISON.md` | Batch vs production throughput explained |
| `QUICK_COMPARISON.md` | TL;DR decision trees and visuals |
| `WHATS_NEW.md` | Complete overview of enhancements |
| **Other Tools** | |
| `benchmark_sequence_length.py` | Sequence length impact analysis |
| `run_all_benchmarks.py` | Run complete benchmark suite |
| `monitor_gpu.py` | Real-time GPU monitoring |
| `utils.py` | Shared utility functions |
| `requirements.txt` | Additional dependencies |
| `results/` | Output directory for all results |

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

### Performance Fundamentals
1. **Baseline Performance**: Your model's default speed and capacity
2. **Batching Benefits**: Why serving multiple requests together helps GPU utilization
3. **Scaling Behavior**: How performance changes with load and request size
4. **Bottlenecks**: Whether you're compute-bound or memory-bound

### Production Readiness
5. **Tail Latency Matters**: P95/P99 > average for user experience and SLAs
6. **Mixed Workloads**: Real traffic is heterogeneous (chat + code + long form)
7. **Capacity Limits**: Your system has a breaking point - know it before production!
8. **Use Case Variance**: Different request types have different performance profiles

### Two Testing Approaches
- **Systematic**: Controlled tests for optimization comparison
- **Production**: Realistic tests for SLA validation and capacity planning

**These insights guide Week 3's optimizations and deployment decisions!**

## 📖 Additional Resources

- [vLLM Performance Benchmark](https://docs.vllm.ai/en/latest/performance/benchmarking.html)
- [Understanding GPU Utilization](https://developer.nvidia.com/blog/how-to-optimize-gpu-utilization/)
- [LLM Inference Math](https://kipp.ly/blog/transformer-inference-arithmetic/)

## ✅ Week 2 Checklist

Before moving to Week 3, ensure you have:

### Minimum Requirements (Choose Your Path)

**Path A: Quick Learning** (15 min total)
- [ ] Run `benchmark_latency.py` 
- [ ] Run `benchmark_throughput.py`
- [ ] Run `benchmark_sequence_length.py`
- [ ] Saved results to `results/` directory

**Path B: Thorough Baseline** ⭐ RECOMMENDED (25 min total)
- [ ] Run `benchmark_latency_comprehensive.py` (systematic)
- [ ] Run `benchmark_latency_comprehensive.py --production` (realistic)
- [ ] Run `benchmark_throughput_production.py` (realistic)
- [ ] Run `benchmark_sequence_length.py`

**Path C: Production-Ready** (30 min total)
- [ ] All benchmarks from Path B
- [ ] Run `benchmark_throughput_production.py --stress` (find limits)
- [ ] Documented P95/P99 targets for SLA planning
- [ ] Validated latency stability over time

### Understanding & Analysis (All Paths)
- [ ] Understand difference between systematic vs production testing
- [ ] Know what P50/P90/P95/P99 percentiles mean
- [ ] Identified which use cases have highest latency
- [ ] Documented baseline metrics for Week 3 comparison
- [ ] Optional: Monitored GPU utilization with `monitor_gpu.py`

### Key Concepts Learned
- [ ] Why average latency ≠ P95 latency (tail latency matters!)
- [ ] How batching improves throughput
- [ ] That real workloads are heterogeneous (varied sizes/types)
- [ ] What your system's capacity limits are
- [ ] Which metrics matter for SLA validation

### Decision Helper

**Not sure which path to choose?**

- **Student on budget** → Path A (quick)
- **Serious learner** → Path B (recommended) ⭐
- **Planning production deployment** → Path C (complete)
- **ML Engineer** → Path B minimum, Path C preferred
- **Researcher** → Path C + stress modes

## 🎓 What You've Learned

By completing Week 2, you now understand:

### Core Concepts
- **Latency vs Throughput**: Individual request speed vs total system capacity
- **Batching Benefits**: How processing multiple requests together improves GPU utilization
- **Tail Latency**: Why P95/P99 matter more than averages for user experience
- **Mixed Workloads**: Real production is heterogeneous (not uniform like benchmarks)

### Two Testing Philosophies

**Systematic Testing** (Matrix approach):
- Controlled variables (prompt length × generation length)
- Good for: Understanding behavior, comparing optimizations
- Example: "Doubling output length increases latency by 1.8x"

**Production Testing** (Realistic approach):
- Mixed use cases, varied sizes, realistic patterns
- Good for: SLA validation, capacity planning
- Example: "95% of chat requests complete in < 1.2s"

### Key Metrics You Can Now Measure

1. **Single-request latency** (user experience)
2. **Batch throughput** (system capacity)
3. **P50/P90/P95/P99 latencies** (SLA compliance)
4. **Per-use-case performance** (optimization priorities)
5. **Breaking points** (capacity limits)
6. **Sustained load capacity** (production readiness)

### Production Readiness Insights

From your production benchmarks, you now know:
- ✅ What SLA targets you can realistically meet
- ✅ Which use cases need optimization (high P95)
- ✅ Your system's request/second capacity
- ✅ Where the breaking point is (when latency degrades)
- ✅ If performance is stable over time

This data is **critical** for Week 3 optimization and eventual deployment!

## 📊 Results Summary Template

Document your findings like this:

```markdown
## My Week 2 Baseline Results

**Hardware**: Tesla T4 (16GB VRAM)
**Date**: 2026-01-29

### Systematic Tests
- Short prompt, 50 tokens: 1.12s avg (σ=0.04s)
- Medium prompt, 100 tokens: 2.34s avg (σ=0.07s)
- Long prompt, 200 tokens: 4.67s avg (σ=0.12s)

### Production Tests
**Latency Distribution**:
- P50: 0.45s | P95: 1.42s | P99: 2.87s

**By Use Case (P95)**:
- Chat (quick): 0.89s ✓ Good
- Chat (detailed): 2.15s ⚠️ Needs optimization
- Code (simple): 1.01s ✓ Good
- Code (complex): 1.78s ✓ Acceptable

**Capacity**:
- Sustained: 5 req/s stable
- Breaking point: ~12 req/s

**SLA Analysis**:
- Can meet 1.5s P95 for chat/code ✓
- Detailed responses exceed 2s (optimize in Week 3)

### Week 3 Goals
1. Optimize long prompt processing
2. Target: P95 < 1.5s for all use cases
3. Increase sustained capacity to 8-10 req/s
```

## 🔜 Next Steps

Ready for **Week 3: GPU-Level Optimization**!

In Week 3, you'll learn:
- How to use PyTorch Profiler for detailed analysis
- Tuning vLLM parameters (`max_num_seqs`, `gpu_memory_utilization`)
- PagedAttention efficiency debugging
- Reducing memory waste and improving throughput

**Your baseline metrics from this week will prove the improvements!**

Use your **production test results** to prioritize optimizations:
- High P95 latency → optimize that use case first
- Low throughput → tune batching parameters  
- Breaking point too low → increase concurrency limits

---

**Questions?** Review this README or check the inline comments in the scripts.

**Need help choosing tests?** See the decision helpers in the checklist above!


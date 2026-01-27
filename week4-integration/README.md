# Week 4: Guest Lectures and Explorations (Integration Week)

## 🎯 Goals

This is an integration week - a lighter week designed to:
- Consolidate knowledge from Weeks 1-3
- Explore advanced vLLM features
- Review documentation and research papers
- Prepare for distributed deployment in Week 5
- Update project documentation

## 📚 What You'll Learn

- Advanced vLLM features (speculative decoding, chunked prefill)
- Multi-GPU and distributed inference concepts
- Industry best practices for LLM deployment
- Production considerations and trade-offs

## 🔬 This Week's Activities

### Activity 1: Review and Document Your Progress

Take stock of what you've learned so far:

1. **Create a summary document** (`my_learnings.md`):
   - Baseline metrics from Week 2
   - Optimizations from Week 3
   - Performance improvements achieved
   - Key insights and surprises

2. **Update code documentation**:
   - Add comments to any unclear sections
   - Document your optimal configuration
   - Note any issues or workarounds

### Activity 2: Explore Advanced vLLM Features

Read about and optionally experiment with:

#### Chunked Prefill
- **What**: Breaks long prompt processing into chunks
- **Why**: Allows decoding to interleave with prefill for better fairness
- **Enable**: `enable_chunked_prefill=True`
- **Benefit**: Reduces head-of-line blocking for mixed workloads

#### Speculative Decoding
- **What**: Uses a smaller "draft" model to speed up generation
- **Why**: Draft model proposes tokens, main model verifies
- **Requirement**: Need a compatible smaller model
- **Speedup**: 1.5-2x for certain workloads

#### Prefix Caching
- **What**: Reuses KV cache for common prompt prefixes
- **Why**: Saves computation when prompts share beginnings
- **Use case**: System prompts, few-shot examples
- **Benefit**: Reduces latency for repeated prefixes

### Activity 3: Read Research and Documentation

Recommended reading (1-2 hours):

1. **vLLM Paper**: ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180)
   - Understand PagedAttention algorithm
   - Learn about continuous batching
   - See benchmark comparisons

2. **vLLM Documentation Deep Dive**:
   - [Engine Arguments](https://docs.vllm.ai/en/latest/models/engine_args.html)
   - [Distributed Inference](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
   - [Performance Best Practices](https://docs.vllm.ai/en/latest/performance/benchmarking.html)

3. **Industry Blogs** (optional):
   - How companies scale LLM inference
   - Real-world deployment architectures
   - Cost optimization strategies

### Activity 4: Plan Multi-GPU Strategy

Prepare for Week 5 by understanding distributed inference:

**Tensor Parallelism (TP)**:
- Splits model layers across GPUs on same node
- All GPUs work on same request
- Good for: Large models, low latency
- Communication: High bandwidth (NVLink preferred)

**Pipeline Parallelism (PP)**:
- Different layers on different GPUs
- Sequential processing through pipeline
- Good for: Very large models
- Trade-off: Bubble overhead

**Data Parallelism**:
- Complete model copy on each GPU/node
- Different requests to different copies
- Good for: High throughput, many requests
- Load balancing: Need request router

For Week 5, we'll focus on **Tensor Parallelism** as it's best for single-node multi-GPU.

### Activity 5: Experiment with vLLM Server Mode

Try running vLLM as an API server (preparation for Week 6):

```bash
python explore_server_mode.py
```

This script demonstrates:
- Launching vLLM as an OpenAI-compatible API
- Sending requests via HTTP
- Server configuration options

## 📂 Files in This Week

| File | Purpose |
|------|---------|
| `README.md` | This guide |
| `my_learnings.md` | Template for documenting your progress |
| `explore_server_mode.py` | Demo vLLM API server |
| `reading_list.md` | Curated resources and papers |
| `distributed_planning.md` | Notes on multi-GPU strategy |

## 📖 Learning Resources

### Official Documentation
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)

### Research Papers
- vLLM PagedAttention Paper (2023)
- FlashAttention: Fast and Memory-Efficient Exact Attention (2022)
- DeepSpeed Inference: Enabling Efficient Inference (2022)

### Blog Posts and Tutorials
- "LLM Inference at Scale" - industry practices
- "GPU Optimization for Transformers" - NVIDIA blogs
- "Cost-Effective LLM Serving" - various cloud providers

## 🧪 Optional Experiments

If you want hands-on practice this week:

### Experiment 1: Test Chunked Prefill

Compare latency with and without chunked prefill for mixed workloads:

```python
# Without chunked prefill
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# With chunked prefill
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enable_chunked_prefill=True,
    max_num_batched_tokens=2048  # Chunk size
)
```

Test with mix of short and long prompts.

### Experiment 2: Profile Memory Patterns

Use `nvidia-smi` to observe memory patterns:

```bash
# Terminal 1
watch -n 0.1 nvidia-smi

# Terminal 2
python ../week3-optimization/run_optimized_benchmark.py
```

Observe:
- Memory allocation patterns
- Peak usage vs sustained usage
- Memory fragmentation

### Experiment 3: Test Different Sampling Strategies

Experiment with generation parameters:

```python
# Greedy (deterministic)
SamplingParams(temperature=0.0)

# Sampling (creative)
SamplingParams(temperature=0.9, top_p=0.95, top_k=50)

# Beam search (quality)
SamplingParams(use_beam_search=True, best_of=4)
```

Compare output quality and speed.

## 📝 Documentation Templates

### my_learnings.md Template

```markdown
# My LLM Inference Optimization Journey

## Week 1-2: Baseline
- Initial throughput: ___ tokens/sec
- GPU model: ___
- Key challenge: ___

## Week 3: Optimization
- Optimal gpu_memory_utilization: ___
- Optimal max_num_seqs: ___
- Final throughput: ___ tokens/sec
- Improvement: ___x

## Key Insights
1. ___
2. ___
3. ___

## Questions for Week 5+
1. ___
2. ___
```

### distributed_planning.md Template

```markdown
# Multi-GPU Strategy for Week 5

## Hardware Available
- Number of GPUs: ___
- GPU Model: ___
- GPU Memory: ___ GB each
- Inter-GPU Link: ___ (NVLink/PCIe)

## Strategy
- Parallelism type: Tensor Parallelism
- Reason: ___
- Expected speedup: ___

## Concerns
- ___
```

## ✅ Week 4 Checklist

This week is flexible, but aim to complete:

- [ ] Documented progress in `my_learnings.md`
- [ ] Read the vLLM PagedAttention paper (at least abstract and introduction)
- [ ] Reviewed vLLM documentation on distributed inference
- [ ] Experimented with server mode using `explore_server_mode.py`
- [ ] Planned multi-GPU strategy for Week 5
- [ ] Reviewed and improved code documentation from Weeks 1-3
- [ ] Understood tensor parallelism concept
- [ ] Optional: Tried one advanced feature (chunked prefill, etc.)

## 🎓 Key Concepts Reinforced

By the end of Week 4, you should have solid understanding of:

1. **vLLM Architecture**: PagedAttention, continuous batching, memory management
2. **Performance Tuning**: Parameter relationships and trade-offs
3. **Distributed Concepts**: TP vs PP vs DP, when to use each
4. **Production Readiness**: What it takes to deploy at scale

## 🔜 Next Steps

Week 5 is where we scale out! You'll:
- Configure tensor parallelism on multi-GPU
- Test distributed inference performance
- Simulate concurrent client requests
- Compare single-GPU vs multi-GPU throughput

Make sure you understand the basics of distributed inference before starting Week 5.

---

**Questions or Want to Dive Deeper?**

Use this week to explore topics that interest you most. No code deliverables required - focus on understanding and preparation for the final weeks.


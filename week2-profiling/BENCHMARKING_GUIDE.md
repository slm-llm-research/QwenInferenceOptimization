# Week 2: Benchmarking Strategy Guide

## 🤔 Which Benchmark Should I Use?

### Quick Reference Table

| Benchmark | Test Cases | Runs/Case | Time | Best For |
|-----------|-----------|-----------|------|----------|
| **benchmark_latency.py** | 3 | 5 | ~2-3 min | Quick baseline, first pass |
| **benchmark_latency_comprehensive.py** (standard) | 9 | 10 | ~5-8 min | Thorough baseline, recommended |
| **benchmark_latency_comprehensive.py** (stress) | 12 | 15 | ~10-15 min | Publication-quality, edge cases |

## 🎯 Decision Framework

### Use **benchmark_latency.py** (Original) When:

✅ You're just starting and want quick feedback  
✅ You're paying for GPU time by the hour (cost-conscious)  
✅ You're doing rapid iteration and will refine later  
✅ You only care about ballpark numbers  
✅ You're following the course on a tight schedule  

**Pros:**
- Fast execution (~2-3 minutes)
- Low cost
- Good enough for relative comparisons
- Easy to run multiple times

**Cons:**
- Limited coverage (misses long prompts)
- Lower statistical confidence (only 5 runs)
- May miss edge cases

### Use **benchmark_latency_comprehensive.py --standard** When:

✅ You want proper baseline metrics for Week 3 comparisons  
✅ You need statistical confidence in your numbers  
✅ You're documenting results for a report/blog  
✅ You want to understand edge cases (long prompts)  
✅ **RECOMMENDED for most learners**  

**Pros:**
- 3x3 matrix covers all important scenarios
- 10 runs gives good statistical confidence
- Tests long prompts (common in production)
- Detailed analysis output
- Still reasonably fast (~5-8 minutes)

**Cons:**
- Takes 2-3x longer than basic version
- More GPU time = more cost

### Use **benchmark_latency_comprehensive.py --stress** When:

✅ You're preparing results for publication/presentation  
✅ You noticed high variance and need more samples  
✅ You're testing very long contexts (200+ tokens)  
✅ You're validating critical production configurations  
✅ You have free/unlimited GPU access  

**Pros:**
- Maximum coverage (12 scenarios)
- High statistical confidence (15 runs)
- Tests extreme edge cases
- Best for documentation

**Cons:**
- Time consuming (~10-15 minutes)
- Higher GPU cost
- Overkill for most learning purposes

## 📊 Statistical Considerations

### Why More Runs Matter

**5 runs** (basic):
- Standard error: ~45% of true std dev
- Good for: Relative comparisons
- Risk: May miss variance patterns

**10 runs** (standard):
- Standard error: ~32% of true std dev
- Good for: Confident baselines
- Sweet spot: Cost vs accuracy

**15+ runs** (stress):
- Standard error: ~26% of true std dev
- Good for: Publication quality
- Diminishing returns after 15

### Coefficient of Variation (CoV)

The comprehensive version reports CoV = (StdDev / Mean) × 100%

**Interpreting CoV:**
- **< 5%**: Excellent consistency ✅
- **5-10%**: Good consistency ✅
- **10-20%**: Moderate variance ⚠️
- **> 20%**: High variance - investigate! ⚠️

If you see high CoV, you should:
1. Run more iterations
2. Ensure no other processes are using GPU
3. Use temperature=0 for deterministic generation
4. Check for thermal throttling

## 🔬 Test Coverage Analysis

### What Each Version Tests

#### Basic (3 cases):
```
Prompt Length:  Short  Short  Medium
Generation:      20     50     100
```
**Coverage:** ~40% of real-world scenarios

#### Standard (9 cases):
```
                Short   Medium   Long
Generation 20:    ✓       ✓      ✗
Generation 50:    ✓       ✓      ✓
Generation 100:   ✓       ✓      ✓
Generation 200:   ✗       ✗      ✓
```
**Coverage:** ~75% of real-world scenarios

#### Stress (12 cases):
```
                Short   Medium   Long
Generation 10:    ✓       ✗      ✗
Generation 20:    ✓       ✓      ✗
Generation 50:    ✓       ✓      ✓
Generation 100:   ✓       ✓      ✓
Generation 200:   ✗       ✓      ✓
Generation 300:   ✗       ✗      ✓
```
**Coverage:** ~95% of real-world scenarios

## 💰 Cost Considerations

Assuming GPU rental at **$1.50/hour** (e.g., A100):

| Version | Time | Cost | Cost/Data Point |
|---------|------|------|-----------------|
| Basic | 2.5 min | $0.06 | $0.020 |
| Standard | 7 min | $0.18 | $0.020 |
| Stress | 13 min | $0.33 | $0.028 |

**Key Insight:** Standard mode gives you 3x more data for 3x the cost - **same cost-per-datapoint!**

## 🎓 My Recommendation (As Your Tutor)

### For Week 2 (Baseline):
**Use: `benchmark_latency_comprehensive.py` (standard mode)**

**Reasoning:**
1. You'll run this **once** to establish baseline
2. The extra 5 minutes is worth the confidence
3. You'll compare against this in Week 3 - better baseline = clearer improvement signal
4. Long prompt testing is important (often used in production)
5. Statistical rigor pays off when making optimization decisions

### Run this command:
```bash
cd week2-profiling
python benchmark_latency_comprehensive.py
# or explicitly: python benchmark_latency_comprehensive.py --mode standard
```

### For Quick Iteration/Debugging:
**Use: `benchmark_latency.py`**

When you're:
- Testing if setup works
- Debugging code changes
- Validating fixes
- On a very tight budget

### For Final Validation:
**Use: `--stress` mode**

After Week 3 optimization, run stress mode once to:
- Validate improvements hold across all scenarios
- Generate publication-quality comparison
- Document final results

## 📈 Example Workflow

```bash
# Week 2, Day 1: Quick validation that everything works
python benchmark_latency.py                        # 3 min ✓

# Week 2, Day 2: Proper baseline (MAIN BASELINE)
python benchmark_latency_comprehensive.py          # 7 min ✓✓✓

# Week 2, Day 3: Throughput and sequence length tests
python benchmark_throughput.py                      # 8 min ✓
python benchmark_sequence_length.py                 # 10 min ✓

# Week 3: Optimization experimentation
python ../week3-optimization/optimize_memory_utilization.py
python ../week3-optimization/optimize_max_num_seqs.py

# Week 3, Final: Compare optimized vs baseline
python benchmark_latency_comprehensive.py --stress  # 13 min ✓
python compare_week2_vs_week3.py                    # generates comparison
```

## 🔧 Custom Testing

You can also customize the comprehensive benchmark:

```bash
# Run 20 iterations for maximum confidence
python benchmark_latency_comprehensive.py --runs 20

# Quick mode (3 cases, 5 runs - same as original)
python benchmark_latency_comprehensive.py --quick

# Standard mode (9 cases, 10 runs - recommended)
python benchmark_latency_comprehensive.py --standard
# or just:
python benchmark_latency_comprehensive.py

# Stress mode (12 cases, 15 runs - most thorough)
python benchmark_latency_comprehensive.py --stress
```

## 📝 Documenting Your Results

Whichever version you use, document:

```markdown
## My Week 2 Baseline

**Hardware:** Tesla T4 (16GB)
**Benchmark:** comprehensive (standard mode)
**Date:** 2026-01-29

### Key Metrics:
- Short prompt, 50 tokens: 1.23s (σ=0.04s, CoV=3.2%)
- Medium prompt, 100 tokens: 2.45s (σ=0.08s, CoV=3.3%)
- Long prompt, 200 tokens: 4.89s (σ=0.15s, CoV=3.1%)

### Observations:
- Consistent performance (CoV < 5%)
- Linear scaling with generation length
- Prompt length has moderate impact on total time
```

## ❓ FAQ

**Q: Can I use the basic version for Week 2 and comprehensive for Week 3 comparison?**  
A: Yes, but it's better to use the same benchmark for both. Use comprehensive for both baseline and post-optimization.

**Q: The benchmark takes too long. Can I reduce runs?**  
A: Yes! Use `--runs 3` for quick testing, but note the lower confidence in results.

**Q: Should I run all three benchmarks (latency, throughput, sequence)?**  
A: Yes! They test different aspects:
- Latency: Single request speed
- Throughput: Batch processing capability
- Sequence: Scaling behavior

**Q: How do I know if my results are good?**  
A: Compare against GPU specs and similar setups. Qwen2.5-7B on A100 should achieve ~40-50 tokens/sec for single requests.

## 🎯 Summary

- **Learning/Budget constrained:** Use basic version
- **Serious baseline (recommended):** Use comprehensive standard mode
- **Research/Publication:** Use comprehensive stress mode
- **Always document:** Which version, hardware, and results

Remember: The goal is not perfection, but **consistent measurement** so you can see improvements in Week 3! 🚀


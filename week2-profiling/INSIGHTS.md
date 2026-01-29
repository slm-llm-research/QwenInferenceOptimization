# Week 2 Benchmark Results: Understanding Your Numbers

## 📊 Overview: What Did You Just Measure?

You've run comprehensive benchmarks across three dimensions:
1. **Latency**: How long individual requests take
2. **Throughput**: How many requests/tokens the system can handle
3. **Sequence Length**: How prompt/output size affects performance

This guide will help you understand **what these numbers mean** and **how to use them**.

---

## 📈 Visual Overview: Generate Your Plots First!

Before diving into the analysis, **generate visualizations** from your results:

```bash
python generate_insights_plots.py
```

**This creates 8 plots in `results/` folder:**
1. `latency_scaling.png` - How output length affects latency
2. `percentile_distribution.png` - P50/P90/P95/P99 visualization
3. `use_case_performance.png` - Which workloads are slowest
4. `queue_time_breakdown.png` - Where time is spent (critical!)
5. `throughput_analysis.png` - Batch scaling benefits
6. `sequence_length_impact.png` - Sequence length categories
7. `consistency_analysis.png` - Performance predictability
8. `comprehensive_dashboard.png` - Complete overview

**Time**: ~10 seconds

**Then**: Open the plots and read this guide together for maximum understanding! 📊

---

---

## 🎯 Quick Performance Summary (YOUR Results)

Based on your actual benchmark data:

### System Performance Snapshot
- **Model**: Qwen/Qwen2.5-7B-Instruct
- **Hardware**: GPU with adequate memory for 7B model
- **Test Date**: January 29, 2026

### Key Metrics At-A-Glance

| Metric | Value | What It Means |
|--------|-------|---------------|
| **Fastest Request** | 0.096s | Ultra-short Q&A responses |
| **Median Latency** | 0.85s | Typical user experience |
| **P95 Latency** | 2.54s | 95% of requests faster than this |
| **P99 Latency** | 4.40s | 99% of requests faster than this |
| **Throughput** | 949 tokens/sec | System capacity (batch mode) |
| **Request Rate** | 4.6 req/sec | Sustained concurrent load |

**Verdict**: Your system performs well for most use cases but has some areas to optimize!

---

## 📖 Understanding the Metrics

### 1. **Latency (Time Per Request)**

**What it is**: How long a single request takes from submission to completion.

**Why it matters**: This is what users experience. High latency = frustrated users.

**Your Results**:

#### From Systematic Testing (Controlled)
```
Short prompt (5 words):
  • 10 tokens out:  0.105s (mean) ← Very fast!
  • 20 tokens out:  0.206s (mean) ← Doubling output ~doubles time
  • 50 tokens out:  0.512s (mean) ← Linear scaling confirmed
  • 100 tokens out: 1.021s (mean) ← Consistent pattern
```

**Key Insight**: Your system shows **linear scaling** with output length!
- Approximately **0.010s per token** generated
- This is GOOD - means consistent, predictable performance

#### From Production Testing (Realistic Mix)

**📊 See: `results/use_case_performance.png`** - Which workloads need optimization!

```
By Use Case (P95 latency):
  • Ultra-short Q&A:     0.279s  ✓ Excellent
  • Short Q&A:           0.850s  ✓ Good
  • Code (short):        0.544s  ✓ Very good
  • Medium explanations: 1.817s  ⚠️ Acceptable
  • Long-form content:   4.396s  ❌ Slow
```

**Key Insight**: You have a **latency hierarchy**:
- Quick questions: Sub-second ✓
- Typical requests: 1-2 seconds ✓
- Long responses: 4+ seconds ⚠️ **← Optimization target!**

**In the plot**, you'll see:
- Bar chart comparing P50/P95/P99 across all use cases
- Color coding: Green (fast), Orange (OK), Red (slow)
- SLA reference lines showing which use cases meet targets
- Clear visual of which categories need Week 3 optimization

---

### 2. **Percentiles (P50, P90, P95, P99)**

**📊 See: `results/percentile_distribution.png`** - Visual breakdown of tail latency!

**What they are**: 
- **P50 (Median)**: 50% of requests complete by this time
- **P90**: 90% of requests complete by this time
- **P95**: 95% of requests complete by this time
- **P99**: 99% of requests complete by this time

**Why they matter**: **Averages lie!** Percentiles tell you the real user experience.

#### Example from YOUR Data:

```
Overall Latency Distribution:
  P50:  0.850s  ← Half of users wait less than this
  P90:  2.329s  ← 10% of users wait MORE than this
  P95:  2.544s  ← 5% of users wait MORE than this
  P99:  4.396s  ← 1% of users wait MORE than this
  Max:  4.396s  ← Worst case
```

**Understanding the Gap**:
- **P50 to P95 gap**: 2.544s / 0.850s = **3x difference**
- This means: While most users get fast responses, some wait 3x longer!

**In the plot**, you'll see:
- Left panel: Bar chart showing the dramatic increase from P50 → P99
- Right panel: Cumulative distribution curve showing exactly where latency jumps
- SLA reference lines (1.5s, 2.0s) showing which percentiles meet targets

**Why this happens**:
- Long prompts take longer
- Longer outputs take longer
- Queue waiting time (if multiple requests)
- Model's attention mechanism scales with sequence length

**For SLA Planning**:

If you promise **"95% of requests complete in < 2s"**:
- ❌ **FAIL**: Your P95 is 2.544s (exceeds target)
- ✅ **PASS** if you relax to: "95% complete in < 3s"

**Action**: You need to optimize or set realistic SLAs!

---

### 3. **Throughput (Tokens Per Second)**

**📊 See: `results/throughput_analysis.png`** - Batching benefits visualized!

**What it is**: Total tokens generated per second (system capacity).

**Why it matters**: Determines how many users you can serve simultaneously.

**Your Results**:

```
Batch Processing (Mixed Workload):
  • Total tokens: 20,486 tokens
  • Total time: 21.6 seconds
  • Throughput: 949 tokens/second
  • Batch size: 16 concurrent requests
```

**What this means**:
- Your system can generate ~949 tokens/sec when processing multiple requests
- This is **MUCH faster** than single-request mode (~100 tokens/sec)
- **Batching gives you 9-10x speedup!** ← Key insight

**In the plot**, you'll see:
- Left panel: Throughput scaling with batch size (exponential curve!)
- Speedup annotations showing 2x, 4x, 8x, 10x improvements
- Right panel: Your system's capacity metrics (949 tok/s, 4.6 req/s)

#### Throughput by Use Case:

Looking at individual request types in your production test:
```
Request Type        Avg Tokens Generated    Tokens/Sec
─────────────────────────────────────────────────────────
Ultra-short (Q&A)        ~15 tokens           72 tok/s
Short Q&A                ~60 tokens           83 tok/s
Medium explanations     ~150 tokens           99 tok/s
Code generation         ~100 tokens           86 tok/s
Long-form               ~400 tokens           91 tok/s
```

**Key Insight**: Throughput is relatively consistent (~70-100 tok/s per request) regardless of size. This is the model's generation speed.

---

### 4. **Queue Time vs Generation Time**

**📊 See: `results/queue_time_breakdown.png`** - The shocking truth about where time goes!

**Your Results**:

```
From Production Throughput Test:
  Latency Percentiles:
    P50: 12.25s total
    
  Queue Percentiles:
    P50: 10.47s waiting
```

**What this reveals**:
- **Queue time**: 10.47s (85% of total time!)
- **Generation time**: ~1.78s (only 15% of total time!)

**Critical Insight**: Most time is spent **WAITING**, not generating!

**In the plot**, you'll see:
- Stacked bars showing queue (red) vs generation (green) time
- Percentages showing 85% waiting, 15% working
- Pie chart dramatically illustrating the imbalance
- **This is your #1 optimization target!**

**Why this happens**:
- You're processing requests **sequentially** in the production test
- Each request waits for previous ones to complete
- This simulates realistic concurrent load

**For Production**:
- Use vLLM's **continuous batching** (happens automatically)
- Increase `max_num_seqs` parameter (Week 3!)
- This will reduce queue time significantly

---

### 5. **Standard Deviation (Consistency)**

**📊 See: `results/consistency_analysis.png`** - Your system's predictability!

**What it is**: How much latency varies between requests.

**Why it matters**: High variation = unpredictable user experience.

**Your Results**:

```
Short prompt, 10 tokens:
  Mean: 0.105s
  StdDev: 0.0004s
  Coefficient of Variation: 0.41%  ← EXCELLENT!
  
Short prompt, 20 tokens:
  Mean: 0.206s
  StdDev: 0.0002s
  Coefficient of Variation: 0.08%  ← EXCELLENT!
```

**Interpretation**:
- **CoV < 5%**: Excellent consistency ✓ (YOUR SYSTEM)
- **CoV 5-10%**: Good consistency ✓
- **CoV 10-20%**: Moderate variation ⚠️
- **CoV > 20%**: High variation ❌

**Your System**: **Extremely consistent** performance! 
- Latency varies by less than 1% for same request types
- Users will get predictable response times
- This is a STRENGTH of your setup

**In the plot**, you'll see:
- Top panel: Error bars showing mean ± std dev (tiny error bars = good!)
- Bottom panel: Coefficient of variation bars (all green = excellent!)
- Your system has rock-solid consistency across all test cases

---

## 🔬 Deep Dive: Understanding Scaling Relationships

### How Does Output Length Affect Latency?

**📊 See: `results/latency_scaling.png`** - Visual proof of linear scaling!

From your systematic tests:

```
Output Tokens    Latency    Latency/Token    Scaling
────────────────────────────────────────────────────────
10 tokens        0.105s     0.0105s/tok     1.0x
20 tokens        0.206s     0.0103s/tok     1.96x  ← Linear!
50 tokens        0.512s     0.0102s/tok     4.88x  ← Linear!
100 tokens       1.021s     0.0102s/tok     9.73x  ← Linear!
```

**Key Finding**: **Perfect linear scaling** at ~10ms per token!

**In the plot**, you'll see:
- Left panel: Latency increases linearly with output tokens (straight line!)
- Right panel: Throughput stays constant ~100 tok/s (flat line = consistent)
- Red dashed line: Linear fit showing the 10ms/token rate

**What this means**:
- Predictable: You can estimate latency for any output length
- Formula: `Latency ≈ 0.01 × num_tokens` (for short prompts)
- Consistent: Model doesn't slow down with longer outputs
- **Good for planning**: Double the tokens = double the time

**For SLA Planning**:
```
If target P95 is 2.0s, max output tokens ≈ 200 tokens
If target P95 is 1.0s, max output tokens ≈ 100 tokens
```

---

### How Does Prompt Length Affect Latency?

Looking at your data systematically:

```
Prompt Type    Prompt Words    50 Token Output    Impact
─────────────────────────────────────────────────────────────
Short          5 words         0.512s             Baseline
Medium         19 words        0.512s             +0%
Long           106 words       0.512s             +0%
```

**Key Finding**: **Minimal prompt length impact** (for these sizes)!

**Why?**
- **Prefill phase** (processing prompt) is parallelized on GPU
- Short-to-medium prompts process very quickly (< 50ms)
- Your GPU handles prefill efficiently

**However**, from production tests:
```
Very long prompts (200+ words) → noticeable slowdown
```

**Takeaway**: Prompt length matters less than output length for your system.

---

### Sequence Length Categories (Production Test)

**📊 See: `results/sequence_length_impact.png`** - Dramatic scaling visualization!

From your sequence length production benchmark:

```
Category      Total Sequence    P95 Latency    Scaling Factor
──────────────────────────────────────────────────────────────
Tiny          < 50 tokens       0.279s         1.0x
Short         50-150 tokens     0.850s         3.0x
Medium        150-300 tokens    1.817s         6.5x
Long          300-500 tokens    2.544s         9.1x
Very Long     500+ tokens       4.396s         15.8x
```

**Key Insight**: **Super-linear scaling** for very long sequences!

**Why it's not linear**:
- Attention mechanism: O(n²) complexity
- Longer KV cache lookups
- Memory bandwidth limitations

**Action**: If many requests exceed 300 tokens, optimize these first in Week 3.

**In the plot**, you'll see:
- Three lines showing P50/P95/P99 climbing steeply with sequence length
- The gap between tiny and very_long is dramatic (15x!)
- SLA reference line showing where you exceed targets
- Clear visual of which sequence categories need optimization

---

## 🎯 Real-World Scenarios: What Your Numbers Mean

### Scenario 1: Chatbot for Customer Service

**Requirements**:
- Target: 95% of responses in < 1.5 seconds
- Expected: Short Q&A (50-100 tokens)

**Your Performance**:
```
Short Q&A (50-100 tokens):
  P50: 0.73s  ✓✓
  P95: 0.85s  ✓✓ MEETS SLA
  P99: 0.87s  ✓✓
```

**Verdict**: ✅ **EXCELLENT** - Well within SLA target!

**Capacity**: Can handle ~4-5 concurrent users with current setup

---

### Scenario 2: Code Generation Assistant

**Requirements**:
- Target: 90% of responses in < 2.0 seconds
- Expected: Medium code snippets (100-200 tokens)

**Your Performance**:
```
Code Generation:
  P50: 0.54s  ✓✓
  P90: ~1.2s  ✓✓ MEETS SLA (estimated)
  P95: ~1.5s  ✓
```

**Verdict**: ✅ **GOOD** - Meets requirements comfortably

---

### Scenario 3: Long-Form Content Generation

**Requirements**:
- Target: 95% of responses in < 3.0 seconds
- Expected: Comprehensive responses (300-500 tokens)

**Your Performance**:
```
Long-form (300-500 tokens):
  P50: ~2.5s   ⚠️ Close to limit
  P95: ~4.4s   ❌ EXCEEDS SLA by 47%
```

**Verdict**: ❌ **NEEDS OPTIMIZATION** or SLA adjustment

**Options**:
1. Optimize in Week 3 (tune parameters)
2. Adjust SLA to "95% in < 5s"
3. Limit long-form responses to < 300 tokens
4. Use streaming (show partial results)

---

## 💡 Key Insights & Action Items

### ✅ Strengths of Your System

1. **Excellent Consistency** (< 1% variation)
   - Users get predictable performance
   - No erratic slowdowns or spikes

2. **Good Short-Response Performance** (< 1s for most)
   - Great for chatbots and Q&A
   - Sub-second responses for 50% of requests

3. **Linear Scaling for Output Length**
   - Predictable behavior
   - Easy to estimate latency

4. **Strong Batching Performance** (9-10x speedup)
   - System utilizes GPU well with concurrent requests
   - Good foundation for scaling

### ⚠️ Areas for Optimization

1. **Long Sequences Slow** (300+ tokens → 4+ seconds)
   - **Impact**: 1% of users wait > 4s
   - **Priority**: HIGH if you have long-form content
   - **Week 3 Action**: Tune `gpu_memory_utilization`, `max_num_seqs`

2. **High Queue Time** (85% waiting, 15% generating)
   - **Impact**: Limits concurrent capacity to ~5 req/s
   - **Priority**: HIGH for multi-user scenarios
   - **Week 3 Action**: Increase `max_num_seqs`, enable continuous batching

3. **P95/P99 Gap** (3x slower than median)
   - **Impact**: Tail latency affects user experience
   - **Priority**: MEDIUM
   - **Week 3 Action**: Profile slow requests, optimize outliers

---

## 📊 Benchmark-Specific Insights

### From Latency Comprehensive Test

**Test**: 12 scenarios, 15 runs each (180 total iterations)

**Findings**:
- ✅ Sub-0.5% standard deviation (rock solid)
- ✅ Linear scaling confirmed (0.01s per token)
- ⚠️ Long prompts (106 words) + long outputs (300 tokens) → slow

**Best Use**: Understanding scaling relationships, Week 3 comparisons

---

### From Throughput Production Test

**Test**: 100 mixed requests, batch size 16

**Findings**:
- ✅ 949 tok/s throughput (good for 7B model)
- ❌ High queue time (need better concurrency)
- ✅ All workload types handled successfully

**Best Use**: Capacity planning, understanding real-world performance

---

### From Sequence Length Production Test

**Test**: 100 requests, realistic distributions

**Findings**:
- ✅ Ultra-short performs excellently (< 0.3s)
- ⚠️ Very long sequences 15x slower than tiny
- ✅ Most requests (50%) stay under 0.85s

**Best Use**: Identifying which content types need optimization

---

## 🎓 How to Use These Insights

### For Week 3 Optimization

**Priority 1**: Reduce queue time
- Target metric: Queue time P50
- Goal: Reduce from 10.5s to < 5s
- How: Increase `max_num_seqs` from 256 to 512+

**Priority 2**: Optimize long sequences
- Target metric: Long-form P95 latency
- Goal: Reduce from 4.4s to < 3s
- How: Tune `gpu_memory_utilization`, enable chunked prefill

**Priority 3**: Increase throughput
- Target metric: Tokens/second
- Goal: Increase from 949 to 1200+
- How: Better batching, memory optimization

### For SLA Setting

**Conservative** (High confidence):
```
P95 Latency SLA:
  • Short content (< 100 tokens): 1.0s
  • Medium content (100-200 tokens): 2.0s
  • Long content (200-500 tokens): 5.0s
```

**Aggressive** (Requires optimization):
```
P95 Latency SLA:
  • Short content: 0.5s  ← Need optimization
  • Medium content: 1.5s ← Need optimization
  • Long content: 3.0s   ← Need optimization
```

### For Capacity Planning

**Current Capacity**:
- Sustained: ~5 concurrent requests/second
- Peak: ~10 req/s (with degraded latency)

**To Scale to 20 req/s**, you need:
1. **Option A**: Optimize single GPU (Week 3)
   - Increase max_num_seqs
   - Better memory utilization
   - Target: 10-15 req/s

2. **Option B**: Add GPU (Week 5)
   - Tensor parallelism or replication
   - Target: 20+ req/s

---

## 📈 Comparing Against Industry Standards

**Your System** (Qwen2.5-7B on single GPU):
```
Short requests:  ~0.85s P50  
Throughput:      ~949 tok/s (batched)
```

**Industry Benchmarks** (similar 7B models):
```
GPT-3.5-turbo:   ~0.5-1.0s typical
LLaMA-2-7B:      ~0.8-1.2s typical  
```

**Your Standing**: **Competitive** for a 7B model! ✓

**Room for Improvement**: Yes! Week 3 optimizations can get you to top-tier performance.

---

## 🔑 Key Takeaways

### 1. **Your Numbers Are Good!**
- Solid baseline performance
- Consistent, predictable latency
- Good batching efficiency

### 2. **Percentiles > Averages**
- P95 of 2.54s (vs mean of 1.13s) tells the real story
- Plan for the 95th percentile, not the average
- Use P95/P99 for SLA commitments

### 3. **Output Length Dominates**
- ~10ms per token generated
- Prompt length has minimal impact (< 100 words)
- Long sequences (500+ tokens) are your bottleneck

### 4. **Queue Time Is Your Enemy**
- 85% of time spent waiting
- 15% actually generating
- Week 3 optimization should focus here!

### 5. **You Can Meet Most SLAs**
- ✅ Short content: Excellent
- ✅ Medium content: Good
- ❌ Long content: Needs work

---

## 🚀 Next Steps

### Ready for Week 3?

**Yes!** You now understand:
- ✅ What your numbers mean
- ✅ Where bottlenecks are
- ✅ What to optimize
- ✅ How to measure improvement

### Week 3 Goals Based on YOUR Data:

1. **Reduce P95 latency by 30%**
   - Target: 2.54s → 1.8s

2. **Increase throughput by 50%**
   - Target: 949 tok/s → 1400+ tok/s

3. **Reduce queue time by 50%**
   - Target: 10.5s → 5s

4. **Support 10 concurrent req/s**
   - Current: ~5 req/s → 10 req/s

**You have the baseline. Now let's optimize!** 🎯

---

## 📚 Quick Reference: Metric Definitions

| Metric | Definition | Your Value | Good/Bad |
|--------|-----------|------------|----------|
| **P50 (Median)** | 50% of requests complete by | 0.850s | ✓ Good |
| **P95** | 95% of requests complete by | 2.544s | ⚠️ OK |
| **P99** | 99% of requests complete by | 4.396s | ❌ Slow |
| **Throughput** | Tokens generated per second | 949 tok/s | ✓ Good |
| **Request Rate** | Concurrent requests per second | 4.6 req/s | ⚠️ OK |
| **Queue Time** | Time waiting for processing | 10.47s (P50) | ❌ High |
| **Consistency** | Coefficient of variation | < 1% | ✓✓ Excellent |

---

## 📊 Visual Summary: Your Complete Performance Dashboard

**📊 See: `results/comprehensive_dashboard.png`** - Everything at a glance!

This single dashboard shows:
- **Top row**: Output scaling (linear fit) + Percentile comparison
- **Middle row**: Use case performance (which workloads are slow?)
- **Bottom row**: Time allocation pie chart + Key metrics (949 tok/s, 4.6 req/s)

**Perfect for**:
- Quick reference during Week 3 optimization
- Presentations or reports
- Comparing before/after optimization

---

## 🎯 How to Use These Visualizations

### Step 1: Generate All Plots
```bash
python generate_insights_plots.py
```

### Step 2: Review in Order

1. **Start with**: `comprehensive_dashboard.png` - Get the big picture
2. **Then**: `percentile_distribution.png` - Understand tail latency
3. **Critical**: `queue_time_breakdown.png` - See the 85% waiting problem!
4. **Deep dive**: `use_case_performance.png` - Find optimization targets
5. **Understand**: `latency_scaling.png` - Confirm linear behavior
6. **Validate**: `consistency_analysis.png` - Check predictability

### Step 3: Document Insights

Create your optimization plan based on what the plots reveal:
```markdown
## Week 3 Optimization Plan (Based on Plots)

From queue_time_breakdown.png:
  → 85% time wasted in queue - PRIORITY #1

From use_case_performance.png:
  → Long-form content exceeds SLA - PRIORITY #2

From sequence_length_impact.png:
  → Very long sequences 15x slower - PRIORITY #3
```

---

## 📁 Complete Results Package

After generating plots, you have:

```
results/
├── JSON Data (Raw Numbers):
│   ├── latency_benchmark_comprehensive.json
│   ├── throughput_production_benchmark.json
│   └── sequence_length_production_benchmark.json
│
└── Visualizations (Understanding):
    ├── comprehensive_dashboard.png ⭐ START HERE
    ├── percentile_distribution.png
    ├── use_case_performance.png
    ├── queue_time_breakdown.png ⭐ CRITICAL
    ├── latency_scaling.png
    ├── throughput_analysis.png
    ├── sequence_length_impact.png
    └── consistency_analysis.png
```

**Use both**: Numbers for precision, plots for understanding!

---

**Questions?** Review your actual results in the `results/` folder, look at the plots, and refer back to this guide! 🎓


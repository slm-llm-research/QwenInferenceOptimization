"""
Week 2: Sequence Length Benchmark (Production-Enhanced)

This script analyzes how prompt length and output length affect inference performance.

SYSTEMATIC MODE (default):
- Controlled tests with fixed prompt/output lengths
- Good for understanding scaling relationships
- Tests: 10, 50, 100, 200 word prompts × various output lengths

PRODUCTION MODE (--production): ⭐ NEW!
- Mixed realistic workload with varied sequence lengths
- Tests real-world distributions (short questions, long contexts, etc.)
- Per-length-category latency analysis (P50/P95/P99)
- Identifies which sequence ranges perform best/worst

Key Concepts:
- Prefill phase: Processing the input prompt (can be parallelized)
- Decode phase: Generating tokens one-by-one (sequential)
- Longer prompts → longer prefill time
- Longer outputs → more decode steps

Usage:
    python benchmark_sequence_length.py              # Systematic testing
    python benchmark_sequence_length.py --production  # Production workload ⭐
"""

import sys
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    print_benchmark_header,
    save_results,
    print_gpu_memory,
    format_duration,
    create_results_dir,
)


@dataclass
class SequenceRequest:
    """Represents a request with specific sequence characteristics"""
    request_id: int
    category: str
    prompt: str
    expected_output_tokens: int
    prompt_length: int


class ProductionSequenceGenerator:
    """Generates realistic production workloads with varied sequence lengths"""
    
    # Real-world prompt/output combinations
    SEQUENCE_PATTERNS = {
        "ultra_short": [
            # Quick factual questions (short prompt, short output)
            ("What is Python?", 15),
            ("Define REST API", 20),
            ("Who invented TCP/IP?", 15),
            ("What does CPU stand for?", 10),
            ("Name three sorting algorithms", 25),
        ],
        "short_qa": [
            # Short questions, medium answers
            ("Explain the difference between GET and POST requests", 60),
            ("How does virtual memory work?", 80),
            ("What are the benefits of using Docker?", 70),
            ("Describe the MVC pattern briefly", 65),
            ("What is test-driven development?", 75),
        ],
        "medium_explain": [
            # Medium prompts, medium-long outputs
            ("Explain how machine learning models are trained. Cover data preprocessing, "
             "training loops, and validation.", 150),
            ("Describe the software development lifecycle from requirements gathering to "
             "deployment and maintenance.", 140),
            ("What are microservices? Discuss their advantages and challenges compared to "
             "monolithic architectures.", 160),
            ("Explain database indexing. How does it improve query performance and what "
             "are the trade-offs?", 130),
        ],
        "long_context": [
            # Long prompts with specific context
            ("Given the following system architecture: A React frontend communicates with a "
             "Node.js backend via REST API. The backend uses PostgreSQL for data storage and "
             "Redis for caching. User sessions are stored in Redis with 24-hour expiry. "
             "What are the potential bottlenecks in this system and how would you optimize it?", 200),
            ("You have a legacy monolithic application with 500K lines of code, tightly coupled "
             "components, and no test coverage. The database is MySQL with multiple large tables "
             "without proper indexing. How would you approach modernizing this system?", 250),
        ],
        "code_short": [
            # Code generation - short
            ("Write a Python function to reverse a string", 50),
            ("Create a function to check if a number is prime", 60),
            ("Implement binary search in Python", 70),
        ],
        "code_long": [
            # Code generation - longer with explanation
            ("Implement a LRU cache in Python with O(1) operations. Include get, put methods "
             "and detailed comments explaining the approach.", 200),
            ("Create a RESTful API endpoint in Flask for user authentication. Include password "
             "hashing, JWT token generation, and error handling. Add comprehensive docstrings.", 250),
        ],
        "summarization": [
            # Summarization tasks (medium input, short output)
            ("Summarize the following article about cloud computing trends: [Long article text "
             "would go here about serverless, edge computing, multi-cloud strategies, cost optimization, "
             "security considerations, and future predictions...] Provide 5 key bullet points.", 100),
        ],
        "long_form": [
            # Comprehensive responses
            ("Write a comprehensive tutorial on getting started with Kubernetes. Include: "
             "1) What is Kubernetes and why use it, 2) Core concepts (pods, services, deployments), "
             "3) Setting up a local cluster, 4) Deploying your first application, 5) Best practices. "
             "Make it beginner-friendly with examples.", 400),
        ],
    }
    
    # Production distribution (based on real chatbot/assistant usage)
    PRODUCTION_DISTRIBUTION = {
        "ultra_short": 0.20,     # 20% quick questions
        "short_qa": 0.25,        # 25% short Q&A
        "medium_explain": 0.25,  # 25% medium explanations
        "long_context": 0.10,    # 10% long context
        "code_short": 0.10,      # 10% short code
        "code_long": 0.05,       # 5% long code
        "summarization": 0.03,   # 3% summarization
        "long_form": 0.02,       # 2% long-form content
    }
    
    def generate_request(self, request_id: int) -> SequenceRequest:
        """Generate a single realistic sequence request"""
        
        # Select category based on production distribution
        category = random.choices(
            list(self.PRODUCTION_DISTRIBUTION.keys()),
            weights=list(self.PRODUCTION_DISTRIBUTION.values())
        )[0]
        
        # Select random prompt/output for this category
        prompt, expected_tokens = random.choice(self.SEQUENCE_PATTERNS[category])
        
        # Add variance (±15%)
        variance = int(expected_tokens * 0.15)
        expected_tokens = random.randint(
            expected_tokens - variance, 
            expected_tokens + variance
        )
        
        return SequenceRequest(
            request_id=request_id,
            category=category,
            prompt=prompt,
            expected_output_tokens=expected_tokens,
            prompt_length=len(prompt.split()),
        )
    
    def generate_workload(self, num_requests: int) -> List[SequenceRequest]:
        """Generate full production workload"""
        return [self.generate_request(i) for i in range(num_requests)]


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles"""
    if not values:
        return {}
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    return {
        "p50": sorted_values[int(n * 0.50)],
        "p90": sorted_values[int(n * 0.90)],
        "p95": sorted_values[int(n * 0.95)],
        "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": sum(sorted_values) / n,
    }


def categorize_sequence_length(prompt_words: int, output_tokens: int) -> str:
    """Categorize request by total sequence length"""
    total = prompt_words + output_tokens
    
    if total < 50:
        return "tiny"
    elif total < 150:
        return "short"
    elif total < 300:
        return "medium"
    elif total < 500:
        return "long"
    else:
        return "very_long"


def run_production_sequence_benchmark(llm, num_requests: int = 100):
    """
    Run production-realistic sequence length benchmark.
    
    Tests varied sequence lengths matching real-world usage patterns.
    Critical for understanding performance across different use cases.
    """
    from vllm import SamplingParams
    
    print(f"\n{'='*70}")
    print(f"🏭 Production Sequence Length Benchmark")
    print(f"{'='*70}")
    print(f"   Testing {num_requests} requests with realistic sequence distributions...")
    
    generator = ProductionSequenceGenerator()
    
    # Generate workload
    print("\n   📋 Generating production workload...")
    requests = generator.generate_workload(num_requests)
    
    # Show composition
    category_counts = defaultdict(int)
    for req in requests:
        category_counts[req.category] += 1
    
    print("\n   📊 Workload Composition:")
    for cat, count in sorted(category_counts.items()):
        pct = (count / num_requests) * 100
        print(f"      {cat:20s}: {count:3d} requests ({pct:5.1f}%)")
    
    # Process requests
    print(f"\n   🔄 Processing requests...")
    
    results = []
    start_time = time.time()
    
    for i, req in enumerate(requests):
        sampling_params = SamplingParams(
            max_tokens=req.expected_output_tokens,
            temperature=0.0,
        )
        
        req_start = time.perf_counter()
        outputs = llm.generate([req.prompt], sampling_params)
        req_end = time.perf_counter()
        
        latency = req_end - req_start
        tokens_generated = len(outputs[0].outputs[0].text.split())
        
        # Categorize by total sequence length
        seq_category = categorize_sequence_length(req.prompt_length, tokens_generated)
        
        results.append({
            "request_id": req.request_id,
            "category": req.category,
            "sequence_category": seq_category,
            "prompt_length": req.prompt_length,
            "output_tokens": tokens_generated,
            "total_sequence": req.prompt_length + tokens_generated,
            "latency": latency,
            "throughput": tokens_generated / latency if latency > 0 else 0,
        })
        
        # Progress
        if (i + 1) % 10 == 0 or i == len(requests) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"      Progress: {i+1}/{num_requests} ({rate:.1f} req/s)", end="\r")
    
    total_time = time.time() - start_time
    print(f"\n   ✅ Completed in {format_duration(total_time)}")
    
    # Analyze results
    all_latencies = [r["latency"] for r in results]
    overall_percentiles = calculate_percentiles(all_latencies)
    
    # Group by category
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["latency"])
    
    # Group by sequence length category
    by_seq_length = defaultdict(list)
    for r in results:
        by_seq_length[r["sequence_category"]].append(r["latency"])
    
    # Analyze prompt length impact
    by_prompt_length = defaultdict(list)
    for r in results:
        if r["prompt_length"] < 20:
            bucket = "0-20"
        elif r["prompt_length"] < 50:
            bucket = "20-50"
        elif r["prompt_length"] < 100:
            bucket = "50-100"
        else:
            bucket = "100+"
        by_prompt_length[bucket].append(r["latency"])
    
    return {
        "num_requests": num_requests,
        "total_time": total_time,
        "overall_percentiles": overall_percentiles,
        "by_category": {
            cat: calculate_percentiles(lats)
            for cat, lats in by_category.items()
        },
        "by_sequence_length": {
            cat: calculate_percentiles(lats)
            for cat, lats in by_seq_length.items()
        },
        "by_prompt_length": {
            bucket: calculate_percentiles(lats)
            for bucket, lats in by_prompt_length.items()
        },
        "all_results": results,
    }


def generate_prompt(target_words: int) -> str:
    """Generate a prompt with approximately target_words words"""
    base_text = "The quick brown fox jumps over the lazy dog. "
    base_words = len(base_text.split())
    
    repetitions = target_words // base_words
    remainder = target_words % base_words
    
    prompt = base_text * repetitions
    if remainder > 0:
        prompt += " ".join(base_text.split()[:remainder])
    
    return prompt.strip() + " Please continue this text:"


def benchmark_prompt_length(llm, prompt_words: int, output_tokens: int = 50):
    """
    Benchmark with a specific prompt length.
    
    Args:
        llm: vLLM LLM instance
        prompt_words: Approximate number of words in prompt
        output_tokens: Number of tokens to generate
    
    Returns:
        Benchmark results
    """
    from vllm import SamplingParams
    
    prompt = generate_prompt(prompt_words)
    actual_words = len(prompt.split())
    
    sampling_params = SamplingParams(
        max_tokens=output_tokens,
        temperature=0.0,
    )
    
    print(f"\n📝 Prompt length: ~{actual_words} words")
    print(f"   Output length: {output_tokens} tokens")
    
    # Warmup
    _ = llm.generate([prompt], sampling_params)
    
    # Measure
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    end = time.perf_counter()
    
    elapsed = end - start
    generated_tokens = len(outputs[0].outputs[0].text.split())
    
    print(f"   ⏱️  Time: {format_duration(elapsed)}")
    print(f"   📊 Throughput: {generated_tokens/elapsed:.1f} tokens/sec")
    
    return {
        "prompt_words": actual_words,
        "output_tokens": output_tokens,
        "elapsed_time": elapsed,
        "generated_tokens": generated_tokens,
        "throughput": generated_tokens / elapsed,
    }


def benchmark_output_length(llm, prompt: str, output_tokens: int):
    """
    Benchmark with a specific output length.
    
    Args:
        llm: vLLM LLM instance
        prompt: Input prompt
        output_tokens: Number of tokens to generate
    
    Returns:
        Benchmark results
    """
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        max_tokens=output_tokens,
        temperature=0.0,
    )
    
    print(f"\n📝 Output length: {output_tokens} tokens")
    
    # Warmup
    _ = llm.generate([prompt], sampling_params)
    
    # Measure
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    end = time.perf_counter()
    
    elapsed = end - start
    generated_tokens = len(outputs[0].outputs[0].text.split())
    
    print(f"   ⏱️  Time: {format_duration(elapsed)}")
    print(f"   📊 Throughput: {generated_tokens/elapsed:.1f} tokens/sec")
    print(f"   ⚡ Time per token: {elapsed/generated_tokens*1000:.1f}ms")
    
    return {
        "output_tokens": output_tokens,
        "elapsed_time": elapsed,
        "generated_tokens": generated_tokens,
        "throughput": generated_tokens / elapsed,
        "time_per_token": elapsed / generated_tokens if generated_tokens > 0 else 0,
    }


def run_sequence_length_benchmarks():
    """Run sequence length impact benchmarks"""
    
    print_benchmark_header("Week 2: Sequence Length Benchmark")
    
    # Import vLLM
    try:
        from vllm import LLM
    except ImportError:
        print("❌ Error: vLLM not installed. Run: pip install vllm")
        sys.exit(1)
    
    create_results_dir()
    
    # Load model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\n📦 Loading model: {model_name}")
    
    try:
        llm = LLM(model=model_name, trust_remote_code=True)
        print("   ✅ Model loaded")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        sys.exit(1)
    
    print_gpu_memory()
    
    # Experiment 1: Varying prompt length
    print("\n" + "="*70)
    print("EXPERIMENT 1: Impact of Prompt Length")
    print("="*70)
    print("(Fixed output: 50 tokens)")
    
    prompt_lengths = [10, 50, 100, 200]
    prompt_results = []
    
    for prompt_words in prompt_lengths:
        result = benchmark_prompt_length(llm, prompt_words, output_tokens=50)
        prompt_results.append(result)
        time.sleep(0.5)
    
    # Print summary
    print("\n📊 Prompt Length Summary:")
    print("-" * 70)
    print(f"{'Prompt Words':<15} {'Time (s)':<12} {'Throughput':<20}")
    print("-" * 70)
    for r in prompt_results:
        print(f"{r['prompt_words']:<15} {r['elapsed_time']:<12.3f} {r['throughput']:<20.1f}")
    
    # Experiment 2: Varying output length
    print("\n" + "="*70)
    print("EXPERIMENT 2: Impact of Output Length")
    print("="*70)
    print("(Fixed prompt: ~20 words)")
    
    test_prompt = "Write a comprehensive essay about the future of artificial intelligence:"
    output_lengths = [25, 50, 100, 200]
    output_results = []
    
    for output_tokens in output_lengths:
        result = benchmark_output_length(llm, test_prompt, output_tokens)
        output_results.append(result)
        time.sleep(0.5)
    
    # Print summary
    print("\n📊 Output Length Summary:")
    print("-" * 70)
    print(f"{'Output Tokens':<15} {'Time (s)':<12} {'ms/token':<15} {'Throughput':<15}")
    print("-" * 70)
    for r in output_results:
        print(f"{r['output_tokens']:<15} {r['elapsed_time']:<12.3f} "
              f"{r['time_per_token']*1000:<15.1f} {r['throughput']:<15.1f}")
    
    # Save results
    save_results("sequence_length_benchmark.json", {
        "model": model_name,
        "prompt_length_experiment": prompt_results,
        "output_length_experiment": output_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    
    # Analysis
    print("\n" + "="*70)
    print("💡 Key Observations:")
    print("="*70)
    
    # Analyze prompt length scaling
    if len(prompt_results) >= 2:
        first = prompt_results[0]
        last = prompt_results[-1]
        time_increase = last["elapsed_time"] / first["elapsed_time"]
        prompt_increase = last["prompt_words"] / first["prompt_words"]
        print(f"\n📈 Prompt Length Scaling:")
        print(f"   • {prompt_increase:.1f}x longer prompt → {time_increase:.1f}x longer time")
        if time_increase < prompt_increase * 0.5:
            print(f"   • Prefill phase is well-optimized (sub-linear scaling)")
        else:
            print(f"   • Prefill time scales roughly with prompt length")
    
    # Analyze output length scaling
    if len(output_results) >= 2:
        avg_time_per_token = sum(r["time_per_token"] for r in output_results) / len(output_results)
        std_time_per_token = (sum((r["time_per_token"] - avg_time_per_token)**2 
                                   for r in output_results) / len(output_results)) ** 0.5
        
        print(f"\n📈 Output Length Scaling:")
        print(f"   • Average time per token: {avg_time_per_token*1000:.1f}ms")
        print(f"   • Std deviation: {std_time_per_token*1000:.1f}ms")
        if std_time_per_token / avg_time_per_token < 0.1:
            print(f"   • Decode phase is very consistent (linear scaling)")
        print(f"   • Generation is autoregressive (one token at a time)")
    
    print("\n✅ Sequence length benchmark completed!")
    print("\n🔜 Next: Run 'python run_all_benchmarks.py' for complete analysis")


def main():
    parser = argparse.ArgumentParser(
        description="Sequence length benchmarking for vLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Systematic testing (controlled experiments):
  python benchmark_sequence_length.py
  
  # Production testing (realistic mixed workload):
  python benchmark_sequence_length.py --production
  python benchmark_sequence_length.py --production --requests 200
        """
    )
    
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run production mode with mixed realistic sequence lengths"
    )
    
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests for production mode (default: 100)"
    )
    
    args = parser.parse_args()
    
    if args.production:
        # Production mode
        print_benchmark_header("Week 2: Production Sequence Length Benchmark")
        
        from vllm import LLM
        
        create_results_dir()
        
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        print(f"\n📦 Loading model: {model_name}")
        
        try:
            llm = LLM(model=model_name, trust_remote_code=True)
            print("   ✅ Model loaded")
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            sys.exit(1)
        
        print_gpu_memory()
        
        # Run production benchmark
        results = run_production_sequence_benchmark(llm, num_requests=args.requests)
        
        # Print results
        print(f"\n{'='*70}")
        print(f"📊 PRODUCTION SEQUENCE LENGTH RESULTS")
        print(f"{'='*70}")
        
        print(f"\n   Overall Latency Distribution:")
        for metric, value in results['overall_percentiles'].items():
            print(f"      {metric.upper():6s}: {value:.3f}s")
        
        print(f"\n   By Use Case Category (P95 latency):")
        for cat, percentiles in sorted(results['by_category'].items()):
            p95 = percentiles['p95']
            print(f"      {cat:20s}: {p95:.3f}s")
        
        print(f"\n   By Total Sequence Length (P95 latency):")
        seq_order = ["tiny", "short", "medium", "long", "very_long"]
        for seq_cat in seq_order:
            if seq_cat in results['by_sequence_length']:
                percentiles = results['by_sequence_length'][seq_cat]
                p95 = percentiles['p95']
                mean = percentiles['mean']
                count = sum(1 for r in results['all_results'] 
                           if r['sequence_category'] == seq_cat)
                print(f"      {seq_cat:15s}: {p95:.3f}s (avg: {mean:.3f}s, n={count})")
        
        print(f"\n   By Prompt Length Range (P95 latency):")
        for bucket in ["0-20", "20-50", "50-100", "100+"]:
            if bucket in results['by_prompt_length']:
                percentiles = results['by_prompt_length'][bucket]
                p95 = percentiles['p95']
                print(f"      {bucket:15s} words: {p95:.3f}s")
        
        # Insights
        print(f"\n💡 Production Insights:")
        
        # Find slowest category
        slowest = max(results['by_category'].items(), 
                     key=lambda x: x[1]['p95'])
        print(f"   • Slowest use case: {slowest[0]} ({slowest[1]['p95']:.3f}s P95)")
        
        # Check if latency scales linearly
        seq_cats = ["tiny", "short", "medium", "long", "very_long"]
        available_cats = [c for c in seq_cats if c in results['by_sequence_length']]
        if len(available_cats) >= 3:
            first_p95 = results['by_sequence_length'][available_cats[0]]['p95']
            last_p95 = results['by_sequence_length'][available_cats[-1]]['p95']
            scaling = last_p95 / first_p95 if first_p95 > 0 else 0
            print(f"   • Sequence length scaling: ~{scaling:.1f}x latency increase")
            
            if scaling < len(available_cats):
                print(f"   • Sub-linear scaling indicates good optimization ✓")
            else:
                print(f"   • Linear+ scaling - longer sequences significantly slower")
        
        # Save results
        save_results("sequence_length_production_benchmark.json", {
            "model": model_name,
            "mode": "production",
            "num_requests": args.requests,
            "results": {
                **results,
                "all_results": results["all_results"]  # Already dicts, not dataclasses
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        
        print(f"\n💾 Results saved to: results/sequence_length_production_benchmark.json")
        print(f"\n🔜 Next: Compare with systematic tests or move to Week 3!")
        
    else:
        # Systematic mode
        run_sequence_length_benchmarks()


if __name__ == "__main__":
    main()


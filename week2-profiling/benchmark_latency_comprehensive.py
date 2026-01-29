"""
Week 2: Comprehensive Latency Benchmark (Production-Grade)

This is an ENHANCED version with production-realistic testing:

STANDARD MODE (--standard, default):
- 9 test cases (3x3 matrix: short/medium/long prompts × varied generations)
- 10 runs per case for statistical confidence
- Tests long prompts (critical for real-world use)

PRODUCTION MODE (--production): ⭐ NEW!
- Mixed workload types (chat, code, summarization, Q&A, reasoning)
- Realistic request distributions matching production patterns
- Per-use-case latency analysis
- P50/P90/P95/P99 percentiles for SLA validation
- Simulates realistic user behavior patterns
- Time-series latency tracking (spot degradation)

Use cases:
- Standard: Systematic baseline (good for optimization comparison)
- Production: Real-world validation (essential for SLA planning)
- Quick: Fast validation
- Stress: Maximum coverage

Usage:
    python benchmark_latency_comprehensive.py                    # Standard: 9 cases
    python benchmark_latency_comprehensive.py --production       # Production workload ⭐
    python benchmark_latency_comprehensive.py --stress           # Maximum coverage
    python benchmark_latency_comprehensive.py --quick            # Fast validation
"""

import sys
import time
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    print_benchmark_header,
    print_result_row,
    calculate_statistics,
    format_duration,
    save_results,
    print_gpu_memory,
    create_results_dir,
)


@dataclass
class ProductionRequest:
    """Represents a production inference request"""
    request_id: int
    workload_type: str
    prompt: str
    max_tokens: int
    expected_use_case: str


class ProductionWorkloadGenerator:
    """Generates realistic production workload patterns for latency testing"""
    
    # Production-realistic prompts by use case
    WORKLOAD_PROMPTS = {
        "chat_quick": [
            ("What is Python?", 30),
            ("How to install npm?", 25),
            ("Define recursion", 20),
            ("What's the weather API?", 25),
            ("Explain async/await", 40),
        ],
        "chat_explanation": [
            ("Explain how REST APIs work and their main principles", 100),
            ("What are the differences between SQL and NoSQL databases?", 120),
            ("Describe the Model-View-Controller (MVC) pattern", 100),
            ("How does garbage collection work in Python?", 90),
            ("Explain the difference between authentication and authorization", 80),
        ],
        "chat_detailed": [
            ("You are a technical writer. Explain Docker containers to someone new to DevOps. "
             "Cover: what they are, why they're useful, and basic concepts.", 200),
            ("Provide a comprehensive guide to getting started with React hooks. Include useState, "
             "useEffect, and useContext with examples.", 250),
            ("Explain microservices architecture. Discuss benefits, challenges, and when to use it "
             "versus monolithic architecture.", 220),
        ],
        "code_simple": [
            ("Write a Python function to check if a number is prime", 60),
            ("Create a JavaScript function to reverse a string", 50),
            ("Write SQL to find duplicate emails in a users table", 40),
            ("Python code to read a CSV file with pandas", 50),
        ],
        "code_complex": [
            ("Implement a LRU cache in Python with get and put methods. Include comments.", 150),
            ("Write a React component for a searchable dropdown with debouncing", 180),
            ("Create a Python class for a binary search tree with insert, search, and delete", 200),
            ("Implement a REST API endpoint in Flask with authentication and error handling", 160),
        ],
        "summarization": [
            ("Summarize this in 3 bullet points: [Article about cloud computing trends, "
             "serverless architectures, and edge computing...]", 60),
            ("Extract key takeaways from: [Meeting notes about Q4 planning, budget, hiring...]", 80),
            ("Provide a brief summary: [Technical documentation about API rate limiting...]", 70),
        ],
        "reasoning": [
            ("If it takes 5 machines 5 minutes to make 5 widgets, how long does it take "
             "100 machines to make 100 widgets? Explain your reasoning.", 100),
            ("A farmer has 17 sheep. All but 9 die. How many are left? Show your work.", 60),
            ("You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons? "
             "Provide step-by-step solution.", 120),
        ],
        "qa_factual": [
            ("What year was Python first released?", 15),
            ("Who invented the World Wide Web?", 20),
            ("What does API stand for?", 15),
            ("Name the author of 'Clean Code'", 15),
        ],
    }
    
    # Production distribution (based on real chatbot/coding assistant usage)
    PRODUCTION_DISTRIBUTION = {
        "chat_quick": 0.25,        # 25% quick questions
        "chat_explanation": 0.20,  # 20% medium explanations
        "chat_detailed": 0.10,     # 10% detailed responses
        "code_simple": 0.15,       # 15% simple code
        "code_complex": 0.12,      # 12% complex code
        "summarization": 0.08,     # 8% summarization
        "reasoning": 0.05,         # 5% reasoning problems
        "qa_factual": 0.05,        # 5% factual Q&A
    }
    
    def generate_request(self, request_id: int) -> ProductionRequest:
        """Generate a single realistic production request"""
        
        # Select workload type based on production distribution
        workload_type = random.choices(
            list(self.PRODUCTION_DISTRIBUTION.keys()),
            weights=list(self.PRODUCTION_DISTRIBUTION.values())
        )[0]
        
        # Select random prompt and expected length for this type
        prompt, max_tokens = random.choice(self.WORKLOAD_PROMPTS[workload_type])
        
        # Add some variance to max_tokens (±20%)
        variance = int(max_tokens * 0.2)
        max_tokens = random.randint(max_tokens - variance, max_tokens + variance)
        
        return ProductionRequest(
            request_id=request_id,
            workload_type=workload_type,
            prompt=prompt,
            max_tokens=max_tokens,
            expected_use_case=workload_type.split('_')[0],  # chat, code, etc.
        )
    
    def generate_workload(self, num_requests: int) -> List[ProductionRequest]:
        """Generate a full production workload"""
        return [self.generate_request(i) for i in range(num_requests)]


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles for SLA analysis"""
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


def benchmark_single_request(llm, prompt: str, max_tokens: int = 50, num_runs: int = 10):
    """
    Benchmark latency for a single request.
    
    Args:
        llm: vLLM LLM instance
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate
        num_runs: Number of runs for averaging
    
    Returns:
        Dictionary with benchmark results
    """
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # Deterministic for consistency
    )
    
    latencies = []
    tokens_generated_list = []
    
    print(f"\n🧪 Running {num_runs} iterations...")
    print(f"   Prompt length: ~{len(prompt.split())} words")
    print(f"   Max tokens: {max_tokens}")
    
    # Warmup run
    print("   Warmup run...", end=" ")
    _ = llm.generate([prompt], sampling_params)
    print("✓")
    
    # Measured runs
    for i in range(num_runs):
        print(f"   Run {i+1}/{num_runs}...", end=" ")
        
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        end = time.perf_counter()
        
        latency = end - start
        latencies.append(latency)
        
        # Count tokens (approximate)
        generated_text = outputs[0].outputs[0].text
        tokens_generated = len(generated_text.split())
        tokens_generated_list.append(tokens_generated)
        
        print(f"{format_duration(latency)}")
    
    # Calculate statistics
    latency_stats = calculate_statistics(latencies)
    avg_tokens = sum(tokens_generated_list) / len(tokens_generated_list)
    throughput = avg_tokens / latency_stats["mean"]
    
    return {
        "prompt": prompt,
        "prompt_length_words": len(prompt.split()),
        "max_tokens": max_tokens,
        "num_runs": num_runs,
        "latencies": latencies,
        "latency_stats": latency_stats,
        "avg_tokens_generated": avg_tokens,
        "throughput_tokens_per_sec": throughput,
    }


def generate_long_prompt(length_category: str) -> str:
    """Generate prompts of different lengths"""
    
    short_prompt = "The capital of France is"
    
    medium_prompt = (
        "Write a short story about a robot learning to paint. "
        "Include details about the robot's journey and emotions."
    )
    
    long_prompt = (
        "You are an expert AI researcher writing a comprehensive review paper. "
        "Explain the evolution of large language models from early neural networks "
        "to modern transformer architectures. Discuss the key innovations that made "
        "models like GPT, BERT, and modern LLMs possible. Include details about: "
        "1) The attention mechanism and how it revolutionized NLP "
        "2) The scaling laws discovered by researchers "
        "3) The architectural improvements like layer normalization and positional encodings "
        "4) The training methodologies including pre-training and fine-tuning "
        "5) The computational challenges and how they were addressed "
        "Please provide a thorough, academic-level explanation suitable for publication."
    )
    
    if length_category == "short":
        return short_prompt
    elif length_category == "medium":
        return medium_prompt
    else:  # long
        return long_prompt


def run_production_latency_benchmark(llm, num_requests: int = 100):
    """
    Run production-realistic latency benchmark with mixed workloads.
    
    Tests various request types and measures latency distribution.
    Critical for SLA validation and capacity planning.
    """
    print(f"\n{'='*70}")
    print(f"🏭 Production Latency Benchmark")
    print(f"{'='*70}")
    print(f"   Requests to test: {num_requests}")
    print(f"   Simulating realistic production traffic patterns...")
    
    from vllm import SamplingParams
    
    generator = ProductionWorkloadGenerator()
    
    # Generate workload
    print("\n   📋 Generating production workload...")
    requests = generator.generate_workload(num_requests)
    
    # Show workload composition
    workload_counts = defaultdict(int)
    for req in requests:
        workload_counts[req.workload_type] += 1
    
    print("\n   📊 Workload Composition:")
    for wtype, count in sorted(workload_counts.items()):
        pct = (count / num_requests) * 100
        print(f"      {wtype:20s}: {count:3d} requests ({pct:5.1f}%)")
    
    # Process requests one by one (simulating real latency)
    print(f"\n   🔄 Processing requests sequentially (measuring true latency)...")
    
    results = []
    start_time = time.time()
    
    for i, req in enumerate(requests):
        sampling_params = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=0.0,  # Deterministic
        )
        
        # Measure individual request latency
        req_start = time.perf_counter()
        outputs = llm.generate([req.prompt], sampling_params)
        req_end = time.perf_counter()
        
        latency = req_end - req_start
        tokens_generated = len(outputs[0].outputs[0].text.split())
        
        results.append({
            "request_id": req.request_id,
            "workload_type": req.workload_type,
            "use_case": req.expected_use_case,
            "prompt_length": len(req.prompt.split()),
            "max_tokens": req.max_tokens,
            "tokens_generated": tokens_generated,
            "latency": latency,
            "timestamp": req_start,
        })
        
        # Progress indicator
        if (i + 1) % 10 == 0 or i == len(requests) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"      Progress: {i+1}/{num_requests} requests "
                  f"({rate:.1f} req/s, {format_duration(elapsed)} elapsed)", end="\r")
    
    total_time = time.time() - start_time
    print(f"\n   ✅ Completed in {format_duration(total_time)}")
    
    # Calculate overall metrics
    all_latencies = [r["latency"] for r in results]
    overall_percentiles = calculate_percentiles(all_latencies)
    
    total_tokens = sum(r["tokens_generated"] for r in results)
    throughput = total_tokens / total_time
    
    # Group by workload type
    by_workload = defaultdict(list)
    for result in results:
        by_workload[result["workload_type"]].append(result["latency"])
    
    # Group by use case
    by_use_case = defaultdict(list)
    for result in results:
        by_use_case[result["use_case"]].append(result["latency"])
    
    # Time-series analysis (latency over time)
    time_buckets = defaultdict(list)
    for result in results:
        bucket = int((result["timestamp"] - results[0]["timestamp"]) / 10)  # 10s buckets
        time_buckets[bucket].append(result["latency"])
    
    return {
        "num_requests": num_requests,
        "total_time": total_time,
        "throughput": throughput,
        "request_rate": num_requests / total_time,
        "overall_percentiles": overall_percentiles,
        "by_workload_type": {
            wtype: calculate_percentiles(latencies)
            for wtype, latencies in by_workload.items()
        },
        "by_use_case": {
            use_case: calculate_percentiles(latencies)
            for use_case, latencies in by_use_case.items()
        },
        "time_buckets": {
            bucket: calculate_percentiles(latencies)
            for bucket, latencies in time_buckets.items()
        },
        "all_results": results,
    }


def run_comprehensive_benchmarks(num_runs: int = 10, mode: str = "standard"):
    """
    Run a comprehensive suite of latency benchmarks
    
    Args:
        num_runs: Number of iterations per test case
        mode: 'quick' (3 cases), 'standard' (9 cases), or 'stress' (15+ cases)
    """
    
    print_benchmark_header("Week 2: Comprehensive Latency Benchmark")
    
    # Import vLLM
    try:
        from vllm import LLM
    except ImportError:
        print("❌ Error: vLLM not installed. Run: pip install vllm")
        sys.exit(1)
    
    # Create results directory
    create_results_dir()
    
    # Load model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\n📦 Loading model: {model_name}")
    print("   (This may take 30-60 seconds...)")
    
    try:
        llm = LLM(model=model_name, trust_remote_code=True)
        print("   ✅ Model loaded")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        sys.exit(1)
    
    print_gpu_memory()
    
    # Define test matrix based on mode
    if mode == "quick":
        # Original 3 test cases
        test_cases = [
            ("short", 20),
            ("short", 50),
            ("medium", 100),
        ]
        num_runs = 5
    elif mode == "stress":
        # Extra comprehensive: all combinations + edge cases
        test_cases = [
            ("short", 10),
            ("short", 20),
            ("short", 50),
            ("short", 100),
            ("medium", 20),
            ("medium", 50),
            ("medium", 100),
            ("medium", 200),
            ("long", 50),
            ("long", 100),
            ("long", 200),
            ("long", 300),
        ]
        num_runs = 15
    else:  # standard
        # 3x3 matrix: comprehensive but practical
        test_cases = [
            ("short", 20),
            ("short", 50),
            ("short", 100),
            ("medium", 20),
            ("medium", 50),
            ("medium", 100),
            ("long", 50),
            ("long", 100),
            ("long", 200),
        ]
    
    print(f"\n🎯 Test Configuration:")
    print(f"   Mode: {mode.upper()}")
    print(f"   Test cases: {len(test_cases)}")
    print(f"   Runs per case: {num_runs}")
    print(f"   Total iterations: {len(test_cases) * num_runs}")
    estimated_time = len(test_cases) * 30  # Rough estimate: 30s per test case
    print(f"   Estimated time: ~{estimated_time//60} minutes")
    print("")
    
    all_results = []
    
    # Run benchmarks
    start_all = time.time()
    
    for i, (prompt_length, max_tokens) in enumerate(test_cases, 1):
        test_name = f"{prompt_length.capitalize()} prompt, {max_tokens} tokens"
        
        print(f"\n{'='*70}")
        print(f"Test Case {i}/{len(test_cases)}: {test_name}")
        print('='*70)
        
        prompt = generate_long_prompt(prompt_length)
        
        result = benchmark_single_request(
            llm,
            prompt,
            max_tokens,
            num_runs=num_runs
        )
        
        # Print results
        print(f"\n📊 Results:")
        stats = result["latency_stats"]
        print_result_row("Average latency", f"{stats['mean']:.3f}", "seconds")
        print_result_row("Median latency", f"{stats['median']:.3f}", "seconds")
        print_result_row("Std deviation", f"{stats['stdev']:.3f}", "seconds")
        print_result_row("Coefficient of variation", f"{(stats['stdev']/stats['mean']*100):.1f}", "%")
        print_result_row("Min latency", f"{stats['min']:.3f}", "seconds")
        print_result_row("Max latency", f"{stats['max']:.3f}", "seconds")
        print_result_row("Avg tokens generated", f"{result['avg_tokens_generated']:.1f}", "tokens")
        print_result_row("Throughput", f"{result['throughput_tokens_per_sec']:.1f}", "tokens/sec")
        
        all_results.append({
            "test_case": test_name,
            "prompt_length": prompt_length,
            "max_tokens": max_tokens,
            "result": result
        })
    
    total_time = time.time() - start_all
    
    # Save results
    save_results("latency_benchmark_comprehensive.json", {
        "model": model_name,
        "mode": mode,
        "num_runs_per_case": num_runs,
        "total_test_cases": len(test_cases),
        "total_time_seconds": total_time,
        "test_cases": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE SUMMARY")
    print("="*70)
    print(f"\n🕐 Total benchmark time: {format_duration(total_time)}")
    print(f"   Average per test case: {format_duration(total_time/len(test_cases))}")
    print("")
    
    # Summary table
    print("{:<30s} {:>12s} {:>12s} {:>14s}".format(
        "Test Case", "Latency (s)", "StdDev (s)", "Throughput"))
    print("-" * 70)
    
    for item in all_results:
        name = item["test_case"]
        latency = item["result"]["latency_stats"]["mean"]
        stdev = item["result"]["latency_stats"]["stdev"]
        throughput = item["result"]["throughput_tokens_per_sec"]
        
        # Truncate long names
        display_name = name if len(name) <= 29 else name[:26] + "..."
        print(f"{display_name:<30s} {latency:>12.3f} {stdev:>12.3f} {throughput:>11.1f} t/s")
    
    # Statistical analysis
    print("\n" + "="*70)
    print("📈 STATISTICAL ANALYSIS")
    print("="*70)
    
    # Group by prompt length
    prompt_groups = {"short": [], "medium": [], "long": []}
    for item in all_results:
        prompt_len = item["prompt_length"]
        latency = item["result"]["latency_stats"]["mean"]
        prompt_groups[prompt_len].append(latency)
    
    print("\n🔍 Average Latency by Prompt Length:")
    for prompt_len, latencies in prompt_groups.items():
        if latencies:
            avg = sum(latencies) / len(latencies)
            print(f"   {prompt_len.capitalize()}: {avg:.3f}s (n={len(latencies)} tests)")
    
    # Group by generation length
    print("\n🔍 Latency vs Generation Length:")
    gen_groups = {}
    for item in all_results:
        gen_len = item["max_tokens"]
        latency = item["result"]["latency_stats"]["mean"]
        if gen_len not in gen_groups:
            gen_groups[gen_len] = []
        gen_groups[gen_len].append(latency)
    
    for gen_len in sorted(gen_groups.keys()):
        latencies = gen_groups[gen_len]
        avg = sum(latencies) / len(latencies)
        print(f"   {gen_len} tokens: {avg:.3f}s (n={len(latencies)} tests)")
    
    print("\n✅ Comprehensive latency benchmark completed!")
    print("\n💡 Key Insights:")
    print("   • Longer prompts increase prefill time (initial processing)")
    print("   • Generation length scales linearly with decode time")
    print("   • Low standard deviation = consistent performance")
    print("   • This data will be your baseline for Week 3 optimization!")
    print(f"\n💾 Results saved to: results/latency_benchmark_comprehensive.json")
    print("\n🔜 Next: Run 'python benchmark_throughput.py' to test batching")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive latency benchmarking for vLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Systematic testing (good for optimization comparison):
  python benchmark_latency_comprehensive.py                 # Standard: 9 cases, 10 runs
  python benchmark_latency_comprehensive.py --stress        # Stress: 12 cases, 15 runs
  python benchmark_latency_comprehensive.py --quick         # Quick: 3 cases, 5 runs
  
  # Production testing (essential for SLA validation):
  python benchmark_latency_comprehensive.py --production    # 100 mixed requests ⭐
  python benchmark_latency_comprehensive.py --production --requests 200  # More thorough
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "stress", "production"],
        default="standard",
        help="Benchmark mode (default: standard)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_const",
        const="quick",
        dest="mode",
        help="Quick mode: 3 test cases, 5 runs"
    )
    
    parser.add_argument(
        "--stress",
        action="store_const",
        const="stress",
        dest="mode",
        help="Stress mode: 12 test cases, 15 runs (maximum coverage)"
    )
    
    parser.add_argument(
        "--production",
        action="store_const",
        const="production",
        dest="mode",
        help="Production mode: mixed realistic workloads with P95/P99 analysis ⭐"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override number of runs per test case (systematic modes only)"
    )
    
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests for production mode (default: 100)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "production":
        # Production mode: mixed workload testing
        print_benchmark_header("Week 2: Production Latency Benchmark")
        
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
        results = run_production_latency_benchmark(llm, num_requests=args.requests)
        
        # Print results
        print(f"\n{'='*70}")
        print(f"📊 PRODUCTION LATENCY RESULTS")
        print(f"{'='*70}")
        
        print(f"\n   Overall Performance:")
        print(f"      Requests processed: {results['num_requests']}")
        print(f"      Total time: {format_duration(results['total_time'])}")
        print(f"      Request rate: {results['request_rate']:.2f} req/s")
        print(f"      Throughput: {results['throughput']:.1f} tokens/sec")
        
        print(f"\n   Latency Distribution (ALL requests):")
        for metric, value in results['overall_percentiles'].items():
            marker = "← SLA target!" if metric == "p95" else ""
            print(f"      {metric.upper():6s}: {value:.3f}s {marker}")
        
        print(f"\n   By Use Case (P95 latency):")
        for use_case, percentiles in sorted(results['by_use_case'].items()):
            p95 = percentiles['p95']
            status = "✓" if p95 < 2.0 else "⚠️"
            print(f"      {use_case:15s}: {p95:.3f}s {status}")
        
        print(f"\n   By Workload Type (P95 latency):")
        for wtype, percentiles in sorted(results['by_workload_type'].items()):
            p95 = percentiles['p95']
            count = sum(1 for r in results['all_results'] if r['workload_type'] == wtype)
            print(f"      {wtype:20s}: {p95:.3f}s (n={count})")
        
        # Time-series stability check
        if len(results['time_buckets']) > 1:
            p95_over_time = [percentiles['p95'] for percentiles in results['time_buckets'].values()]
            p95_variance = max(p95_over_time) - min(p95_over_time)
            if p95_variance < 0.5:
                print(f"\n   ✅ Latency stability: GOOD (P95 variance: {p95_variance:.3f}s)")
            else:
                print(f"\n   ⚠️  Latency stability: VARIABLE (P95 variance: {p95_variance:.3f}s)")
                print(f"       May indicate thermal throttling or resource contention")
        
        # Save results
        save_results("latency_production_benchmark.json", {
            "model": model_name,
            "mode": "production",
            "num_requests": args.requests,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        
        print(f"\n💡 Production Insights:")
        print(f"   • P95/P99 latency is critical for SLA validation")
        print(f"   • Different use cases have different latency profiles")
        print(f"   • Use these metrics to set realistic SLA targets")
        
        if results['overall_percentiles']['p95'] > 2.0:
            print(f"   ⚠️  P95 latency > 2s may impact user experience")
            print(f"   → Consider optimization in Week 3")
        
        print(f"\n💾 Results saved to: results/latency_production_benchmark.json")
        
    else:
        # Systematic mode: matrix testing
        # Determine number of runs
        if args.runs:
            num_runs = args.runs
        elif args.mode == "quick":
            num_runs = 5
        elif args.mode == "stress":
            num_runs = 15
        else:
            num_runs = 10
        
        run_comprehensive_benchmarks(num_runs=num_runs, mode=args.mode)


if __name__ == "__main__":
    main()


"""
Week 2: Comprehensive Latency Benchmark

This is an ENHANCED version of benchmark_latency.py with more aggressive testing:
- 9 test cases (3x3 matrix: short/medium/long prompts × short/medium/long generations)
- 10 runs per case for better statistical confidence
- Long prompt testing (important for real-world scenarios)
- Optional stress mode with even more iterations

Use this when you need:
- More thorough baseline metrics
- Statistical confidence in your measurements
- Edge case analysis (very long prompts/generations)

Trade-off: Takes ~5-10 minutes vs 2-3 minutes for basic version

Usage:
    python benchmark_latency_comprehensive.py              # Standard mode
    python benchmark_latency_comprehensive.py --stress     # Extra thorough
    python benchmark_latency_comprehensive.py --quick      # Use basic version
"""

import sys
import time
import argparse
from pathlib import Path

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
  python benchmark_latency_comprehensive.py              # Standard: 9 cases, 10 runs
  python benchmark_latency_comprehensive.py --stress     # Stress: 12 cases, 15 runs
  python benchmark_latency_comprehensive.py --quick      # Quick: 3 cases, 5 runs
  python benchmark_latency_comprehensive.py --runs 20    # Custom iteration count
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "stress"],
        default="standard",
        help="Benchmark mode (default: standard)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_const",
        const="quick",
        dest="mode",
        help="Quick mode: 3 test cases, 5 runs (same as original)"
    )
    
    parser.add_argument(
        "--stress",
        action="store_const",
        const="stress",
        dest="mode",
        help="Stress mode: 12 test cases, 15 runs (most thorough)"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override number of runs per test case"
    )
    
    args = parser.parse_args()
    
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


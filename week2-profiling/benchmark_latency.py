"""
Week 2: Latency Benchmark

This script measures single-request latency - how long it takes to process
one inference request. This represents the user experience for interactive
applications like chatbots.

Metrics:
- Time per request (seconds)
- Tokens per second (throughput)
- Consistency (standard deviation)

Usage:
    python benchmark_latency.py
"""

import sys
import time
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


def benchmark_single_request(llm, prompt: str, max_tokens: int = 50, num_runs: int = 5):
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
    print(f"   Prompt: '{prompt[:50]}...'")
    print(f"   Max tokens: {max_tokens}")
    
    # Warmup run
    print("   Warmup run...")
    _ = llm.generate([prompt], sampling_params)
    
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
        "max_tokens": max_tokens,
        "num_runs": num_runs,
        "latencies": latencies,
        "latency_stats": latency_stats,
        "avg_tokens_generated": avg_tokens,
        "throughput_tokens_per_sec": throughput,
    }


def run_latency_benchmarks():
    """Run a suite of latency benchmarks"""
    
    print_benchmark_header("Week 2: Latency Benchmark")
    
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
    
    # Define test cases
    test_cases = [
        {
            "name": "Short prompt, short generation",
            "prompt": "The capital of France is",
            "max_tokens": 20,
        },
        {
            "name": "Short prompt, medium generation",
            "prompt": "Explain artificial intelligence in simple terms:",
            "max_tokens": 50,
        },
        {
            "name": "Medium prompt, medium generation",
            "prompt": "Write a short story about a robot learning to paint. Include details about the robot's journey and emotions.",
            "max_tokens": 100,
        },
    ]
    
    all_results = []
    
    # Run benchmarks
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}/{len(test_cases)}: {test_case['name']}")
        print('='*70)
        
        result = benchmark_single_request(
            llm,
            test_case["prompt"],
            test_case["max_tokens"],
            num_runs=5
        )
        
        # Print results
        print(f"\n📊 Results:")
        stats = result["latency_stats"]
        print_result_row("Average latency", f"{stats['mean']:.3f}", "seconds")
        print_result_row("Median latency", f"{stats['median']:.3f}", "seconds")
        print_result_row("Std deviation", f"{stats['stdev']:.3f}", "seconds")
        print_result_row("Min latency", f"{stats['min']:.3f}", "seconds")
        print_result_row("Max latency", f"{stats['max']:.3f}", "seconds")
        print_result_row("Avg tokens generated", f"{result['avg_tokens_generated']:.1f}", "tokens")
        print_result_row("Throughput", f"{result['throughput_tokens_per_sec']:.1f}", "tokens/sec")
        
        all_results.append({
            "test_case": test_case["name"],
            "result": result
        })
    
    # Save results
    save_results("latency_benchmark.json", {
        "model": model_name,
        "test_cases": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    
    # Print summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print("\n{:<40s} {:>12s} {:>15s}".format("Test Case", "Latency (s)", "Throughput"))
    print("-" * 70)
    
    for item in all_results:
        name = item["test_case"]
        latency = item["result"]["latency_stats"]["mean"]
        throughput = item["result"]["throughput_tokens_per_sec"]
        print(f"{name:<40s} {latency:>12.3f} {throughput:>12.1f} tok/s")
    
    print("\n✅ Latency benchmark completed!")
    print("\n💡 Key Observations:")
    print("   • Single requests show baseline latency")
    print("   • Longer generations = more total time but similar per-token rate")
    print("   • This represents best-case user experience (no queuing)")
    print("\n🔜 Next: Run 'python benchmark_throughput.py' to test batching")


def main():
    run_latency_benchmarks()


if __name__ == "__main__":
    main()


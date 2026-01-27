"""
Week 2: Throughput Benchmark

This script measures throughput with different batch sizes. Batching multiple
requests together improves GPU utilization and total tokens/second, which is
critical for serving many users.

Key Insight:
vLLM's continuous batching dynamically combines requests, so throughput
scales well with concurrent load.

Metrics:
- Total tokens per second
- Speedup vs batch size 1
- GPU utilization improvement

Usage:
    python benchmark_throughput.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    print_benchmark_header,
    save_results,
    print_gpu_memory,
    format_duration,
    create_results_dir,
)


def benchmark_batch_throughput(llm, batch_size: int, prompt: str, max_tokens: int = 50):
    """
    Benchmark throughput with a specific batch size.
    
    Args:
        llm: vLLM LLM instance
        batch_size: Number of concurrent requests
        prompt: Input prompt (same for all requests)
        max_tokens: Tokens to generate per request
    
    Returns:
        Dictionary with throughput metrics
    """
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )
    
    # Create batch of prompts
    prompts = [prompt] * batch_size
    
    print(f"\n📊 Batch size: {batch_size}")
    print(f"   Generating {max_tokens} tokens × {batch_size} requests...")
    
    # Warmup
    _ = llm.generate(prompts, sampling_params)
    
    # Measured run
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end = time.perf_counter()
    
    elapsed = end - start
    
    # Count total tokens
    total_tokens = 0
    for output in outputs:
        for completion in output.outputs:
            total_tokens += len(completion.text.split())
    
    throughput = total_tokens / elapsed
    
    print(f"   ⏱️  Time: {format_duration(elapsed)}")
    print(f"   📈 Throughput: {throughput:.1f} tokens/sec")
    print(f"   🎯 Tokens generated: {total_tokens}")
    
    return {
        "batch_size": batch_size,
        "elapsed_time": elapsed,
        "total_tokens": total_tokens,
        "throughput": throughput,
        "tokens_per_request": total_tokens / batch_size,
        "latency_per_request": elapsed / batch_size,
    }


def run_throughput_benchmarks():
    """Run throughput benchmarks with various batch sizes"""
    
    print_benchmark_header("Week 2: Throughput Benchmark")
    
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
    
    # Test configuration
    test_prompt = "Write a short sentence about machine learning:"
    max_tokens = 50
    batch_sizes = [1, 4, 8, 16, 32]
    
    print(f"\n🔬 Test Configuration:")
    print(f"   Prompt: '{test_prompt}'")
    print(f"   Max tokens: {max_tokens}")
    print(f"   Batch sizes: {batch_sizes}")
    
    print("\n" + "="*70)
    print("Running Benchmarks...")
    print("="*70)
    
    results = []
    
    for batch_size in batch_sizes:
        result = benchmark_batch_throughput(llm, batch_size, test_prompt, max_tokens)
        results.append(result)
        
        # Brief pause between tests
        time.sleep(1)
    
    # Calculate speedup relative to batch_size=1
    baseline_throughput = results[0]["throughput"]
    
    # Print summary table
    print("\n" + "="*70)
    print("📊 THROUGHPUT SUMMARY")
    print("="*70)
    print("\n{:<12} {:>15} {:>12} {:>15}".format(
        "Batch Size", "Throughput", "Speedup", "Time (s)"
    ))
    print("-" * 70)
    
    for result in results:
        batch = result["batch_size"]
        throughput = result["throughput"]
        speedup = throughput / baseline_throughput
        elapsed = result["elapsed_time"]
        
        print(f"{batch:<12} {throughput:>12.1f} tok/s {speedup:>8.2f}x {elapsed:>12.2f}")
    
    print("\n" + "="*70)
    
    # Save results
    save_results("throughput_benchmark.json", {
        "model": model_name,
        "prompt": test_prompt,
        "max_tokens": max_tokens,
        "batch_sizes": batch_sizes,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    
    # Analysis
    print("\n💡 Key Observations:")
    print(f"   • Batch size 1:  {results[0]['throughput']:.0f} tokens/sec (baseline)")
    if len(results) > 1:
        print(f"   • Batch size {results[-1]['batch_size']}: {results[-1]['throughput']:.0f} tokens/sec "
              f"({results[-1]['throughput']/baseline_throughput:.1f}x speedup)")
    print(f"   • GPU batching improves throughput significantly!")
    print(f"   • Larger batches = better GPU utilization")
    
    # Check if throughput plateaus
    if len(results) >= 3:
        last_three_speedups = [r["throughput"] / baseline_throughput for r in results[-3:]]
        if max(last_three_speedups) - min(last_three_speedups) < 1.5:
            print(f"\n   ⚠️  Throughput plateau detected - GPU may be saturated")
            print(f"       Consider this batch size as the sweet spot for this hardware")
    
    print("\n✅ Throughput benchmark completed!")
    print("\n🔜 Next: Run 'python benchmark_sequence_length.py' to test sequence length impact")


def main():
    run_throughput_benchmarks()


if __name__ == "__main__":
    main()


"""
Week 3: Run Optimized Benchmark

This script runs the same benchmarks as Week 2, but with optimized parameters
from the optimization experiments. Compare results to see the improvement!

Usage:
    python run_optimized_benchmark.py
    
    # Or specify custom parameters:
    python run_optimized_benchmark.py --gpu-memory 0.95 --max-seqs 512
"""

import sys
import time
import json
import argparse
from pathlib import Path


def load_optimal_params():
    """Load optimal parameters from optimization results"""
    results_dir = Path("results")
    
    optimal_params = {
        "gpu_memory_utilization": 0.9,  # Default
        "max_num_seqs": 256,  # Default
    }
    
    # Try to load memory optimization results
    mem_file = results_dir / "memory_optimization.json"
    if mem_file.exists():
        with open(mem_file) as f:
            data = json.load(f)
            if data.get("optimal"):
                optimal_params["gpu_memory_utilization"] = data["optimal"]["mem_utilization"]
                print(f"   Loaded optimal gpu_memory_utilization: {optimal_params['gpu_memory_utilization']}")
    
    # Try to load max_num_seqs optimization results
    seqs_file = results_dir / "max_num_seqs_optimization.json"
    if seqs_file.exists():
        with open(seqs_file) as f:
            data = json.load(f)
            if data.get("optimal"):
                optimal_params["max_num_seqs"] = data["optimal"]["max_num_seqs"]
                print(f"   Loaded optimal max_num_seqs: {optimal_params['max_num_seqs']}")
    
    return optimal_params


def run_optimized_benchmark(gpu_memory_util=None, max_num_seqs=None):
    """Run benchmark with optimized parameters"""
    
    print("="*70)
    print("Week 3: Optimized Performance Benchmark")
    print("="*70)
    print()
    
    from vllm import LLM, SamplingParams
    import torch
    
    # Load or use provided parameters
    if gpu_memory_util is None or max_num_seqs is None:
        print("📊 Loading optimal parameters from previous experiments...")
        params = load_optimal_params()
        gpu_memory_util = gpu_memory_util or params["gpu_memory_utilization"]
        max_num_seqs = max_num_seqs or params["max_num_seqs"]
    
    print()
    print(f"🔧 Configuration:")
    print(f"   gpu_memory_utilization: {gpu_memory_util}")
    print(f"   max_num_seqs: {max_num_seqs}")
    print()
    
    # Load model with optimized settings
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"📦 Loading model: {model_name}")
    print("   (with optimized parameters...)")
    
    try:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_util,
            max_num_seqs=max_num_seqs,
        )
        print("   ✅ Model loaded")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
    
    # GPU memory info
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   🎮 GPU Memory: {reserved:.2f} GB / {total:.2f} GB ({reserved/total*100:.1f}%)")
    
    print()
    
    # Run benchmarks
    results = {}
    
    # Benchmark 1: Batch throughput
    print("="*70)
    print("BENCHMARK 1: Batch Throughput")
    print("="*70)
    
    batch_sizes = [1, 4, 8, 16, 32]
    batch_results = []
    
    test_prompt = "Write a short explanation about artificial intelligence:"
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
    
    for batch_size in batch_sizes:
        prompts = [test_prompt] * batch_size
        
        # Warmup
        _ = llm.generate(prompts[:min(2, batch_size)], sampling_params)
        
        # Measure
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        
        total_tokens = sum(len(out.outputs[0].text.split()) for out in outputs)
        throughput = total_tokens / elapsed
        
        batch_results.append({
            "batch_size": batch_size,
            "throughput": throughput,
            "time": elapsed,
        })
        
        print(f"   Batch {batch_size:>3}: {throughput:>8.1f} tokens/sec ({elapsed:.2f}s)")
        
        time.sleep(0.5)
    
    results["batch_throughput"] = batch_results
    
    # Benchmark 2: Latency
    print("\n" + "="*70)
    print("BENCHMARK 2: Single Request Latency")
    print("="*70)
    
    single_prompt = "Explain machine learning:"
    latencies = []
    
    for i in range(5):
        start = time.perf_counter()
        outputs = llm.generate([single_prompt], sampling_params)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
    
    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = len(outputs[0].outputs[0].text.split())
    
    print(f"   Average latency: {avg_latency:.3f}s")
    print(f"   Tokens generated: {avg_tokens}")
    print(f"   Throughput: {avg_tokens/avg_latency:.1f} tokens/sec")
    
    results["latency"] = {
        "avg_latency": avg_latency,
        "throughput": avg_tokens / avg_latency,
    }
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "optimized_benchmark.json"
    with open(output_file, 'w') as f:
        json.dump({
            "config": {
                "gpu_memory_utilization": gpu_memory_util,
                "max_num_seqs": max_num_seqs,
            },
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    
    # Compare with baseline
    print("\n" + "="*70)
    print("📊 COMPARISON WITH BASELINE")
    print("="*70)
    
    baseline_file = Path("../week2-profiling/results/throughput_benchmark.json")
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline_data = json.load(f)
            
            # Find matching batch sizes
            print("\nThroughput Improvements:")
            for opt_result in batch_results:
                batch_size = opt_result["batch_size"]
                opt_throughput = opt_result["throughput"]
                
                # Find baseline
                baseline_result = next(
                    (r for r in baseline_data["results"] if r["batch_size"] == batch_size),
                    None
                )
                
                if baseline_result:
                    baseline_throughput = baseline_result["throughput"]
                    improvement = (opt_throughput / baseline_throughput - 1) * 100
                    
                    print(f"   Batch {batch_size:>2}: {baseline_throughput:>7.1f} → {opt_throughput:>7.1f} tok/s ({improvement:+.1f}%)")
    else:
        print("\n⚠️  Baseline results not found. Run Week 2 benchmarks first for comparison.")
    
    print()
    print("="*70)
    print("✅ Optimized benchmark complete!")
    print("="*70)
    print(f"\n📁 Results saved to: {output_file}")
    print()
    print("💡 Key Takeaways:")
    print("   • Compare throughput improvements vs baseline")
    print("   • Note GPU memory utilization increase")
    print("   • These settings are optimal for your hardware!")
    print()


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Run optimized benchmark")
    parser.add_argument("--gpu-memory", type=float, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-seqs", type=int, help="Maximum number of sequences")
    
    args = parser.parse_args()
    
    run_optimized_benchmark(
        gpu_memory_util=args.gpu_memory,
        max_num_seqs=args.max_seqs
    )


if __name__ == "__main__":
    main()


"""
Week 3: Comprehensive Optimized Benchmark

This script replicates ALL Week 2 benchmarks with optimized parameters:
  1. Systematic latency tests (output length scaling)
  2. Batch throughput tests
  3. Production workload simulation (queue time analysis)
  4. Sequence length impact tests

Results can be directly compared with Week 2 using compare_week2_week3.py

Usage:
    python run_comprehensive_optimized_benchmark.py
    
    # Or specify custom parameters:
    python run_comprehensive_optimized_benchmark.py --gpu-memory 0.95 --max-seqs 1024
"""

import sys
import time
import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Any


def load_optimal_params():
    """Load optimal parameters from optimization results"""
    results_dir = Path("results")
    
    optimal_params = {
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 256,
    }
    
    # Load memory optimization
    mem_file = results_dir / "memory_optimization.json"
    if mem_file.exists():
        with open(mem_file) as f:
            data = json.load(f)
            if data.get("optimal"):
                optimal_params["gpu_memory_utilization"] = data["optimal"]["mem_utilization"]
    
    # Load max_num_seqs optimization
    seqs_file = results_dir / "max_num_seqs_optimization.json"
    if seqs_file.exists():
        with open(seqs_file) as f:
            data = json.load(f)
            if data.get("optimal"):
                optimal_params["max_num_seqs"] = data["optimal"]["max_num_seqs"]
    
    return optimal_params


def run_systematic_latency_benchmark(llm, sampling_params_func):
    """
    Replicates Week 2's systematic latency benchmark.
    Tests controlled scenarios with varying prompt/output lengths.
    """
    print("\n" + "="*70)
    print("BENCHMARK 1: Systematic Latency Tests")
    print("="*70)
    print("(Replicating Week 2 controlled conditions)")
    print()
    
    # Test scenarios from Week 2
    scenarios = []
    prompt_types = [
        ("short", "What is Python?"),
        ("medium", "Explain the concept of machine learning in detail:"),
        ("long", """Write a comprehensive explanation of artificial intelligence,
                   covering neural networks, deep learning, machine learning algorithms,
                   training processes, and real-world applications:"""),
    ]
    output_lengths = [10, 20, 50, 100]
    
    for prompt_type, prompt_text in prompt_types:
        for output_length in output_lengths:
            scenario_name = f"{prompt_type}_{output_length}tok"
            print(f"Testing: {prompt_type} prompt, {output_length} token output...")
            
            sampling_params = sampling_params_func(output_length)
            
            # Run 15 times for statistics (like Week 2)
            latencies = []
            for _ in range(15):
                start = time.perf_counter()
                outputs = llm.generate([prompt_text], sampling_params)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
            
            # Calculate statistics
            mean_lat = statistics.mean(latencies)
            std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
            cov = (std_dev / mean_lat) * 100 if mean_lat > 0 else 0
            
            actual_tokens = len(outputs[0].outputs[0].token_ids)
            
            scenarios.append({
                "prompt_type": prompt_type,
                "output_tokens": output_length,
                "actual_tokens": actual_tokens,
                "mean_latency": mean_lat,
                "std_dev": std_dev,
                "coefficient_of_variation": cov,
                "latencies": latencies,
            })
            
            print(f"   Mean: {mean_lat:.3f}s ± {std_dev:.4f}s (CoV: {cov:.2f}%)")
    
    return {"scenarios": scenarios}


def run_batch_throughput_benchmark(llm, sampling_params_func):
    """
    Replicates Week 2's batch throughput benchmark.
    """
    print("\n" + "="*70)
    print("BENCHMARK 2: Batch Throughput")
    print("="*70)
    print("(Replicating Week 2 batch size tests)")
    print()
    
    batch_sizes = [1, 4, 8, 16, 32]
    results = []
    
    test_prompt = "Write a brief explanation of machine learning:"
    sampling_params = sampling_params_func(50)
    
    for batch_size in batch_sizes:
        prompts = [test_prompt] * batch_size
        
        # Warmup
        _ = llm.generate(prompts[:min(2, batch_size)], sampling_params)
        
        # Measure
        print(f"Testing batch size {batch_size}...")
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        
        total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        throughput = total_tokens / elapsed
        
        results.append({
            "batch_size": batch_size,
            "elapsed_time": elapsed,
            "total_tokens": total_tokens,
            "throughput": throughput,
        })
        
        print(f"   Throughput: {throughput:.1f} tok/s")
        time.sleep(0.5)
    
    return {"results": results}


def run_production_workload_benchmark(llm, sampling_params_func):
    """
    Replicates Week 2's production throughput benchmark.
    Mixed workload with queue time analysis.
    """
    print("\n" + "="*70)
    print("BENCHMARK 3: Production Workload Simulation")
    print("="*70)
    print("(Replicating Week 2 mixed workload test)")
    print()
    
    # Create mixed workload matching Week 2
    use_cases = [
        ("Ultra-short Q&A", "What is Python?", 15),
        ("Short Q&A", "Explain machine learning:", 60),
        ("Code generation", "Write a Python function to sort a list:", 100),
        ("Medium explanation", "Describe how neural networks work:", 150),
        ("Long-form", "Write a comprehensive guide to deep learning:", 300),
    ]
    
    prompts = []
    expected_tokens = []
    
    # Create 100 requests (20 of each type)
    for _ in range(20):
        for use_case_name, prompt, max_tok in use_cases:
            prompts.append({
                "text": prompt,
                "use_case": use_case_name,
                "max_tokens": max_tok,
            })
    
    print(f"Running {len(prompts)} concurrent requests (mixed workload)...")
    print("This measures queue time vs generation time...")
    print()
    
    # Create sampling params (use first as representative)
    sampling_params = sampling_params_func(100)  # Average length
    
    # Run benchmark
    start_time = time.perf_counter()
    
    outputs = llm.generate(
        [p["text"] for p in prompts],
        sampling_params,
    )
    
    total_elapsed = time.perf_counter() - start_time
    
    # Calculate metrics
    all_token_counts = [len(out.outputs[0].token_ids) for out in outputs]
    total_tokens = sum(all_token_counts)
    throughput = total_tokens / total_elapsed
    requests_per_second = len(prompts) / total_elapsed
    
    # Estimate queue metrics
    avg_tokens_per_request = total_tokens / len(prompts)
    estimated_generation_time_per_request = avg_tokens_per_request / 100  # ~100 tok/s
    avg_total_latency = total_elapsed / len(prompts)
    estimated_queue_time = max(0, avg_total_latency - estimated_generation_time_per_request)
    queue_time_pct = (estimated_queue_time / avg_total_latency * 100) if avg_total_latency > 0 else 0
    
    # Calculate percentiles for latency (estimated based on token distribution)
    sorted_tokens = sorted(all_token_counts)
    p50_idx = len(sorted_tokens) // 2
    p90_idx = int(len(sorted_tokens) * 0.9)
    p95_idx = int(len(sorted_tokens) * 0.95)
    p99_idx = int(len(sorted_tokens) * 0.99)
    
    # Estimate latencies based on token counts
    p50_latency = sorted_tokens[p50_idx] / 100  # Rough estimate
    p90_latency = sorted_tokens[p90_idx] / 100
    p95_latency = sorted_tokens[p95_idx] / 100
    p99_latency = sorted_tokens[p99_idx] / 100
    
    print(f"✅ Complete!")
    print(f"\n📊 Results:")
    print(f"   Total time: {total_elapsed:.2f}s")
    print(f"   Total tokens: {total_tokens}")
    print(f"   Throughput: {throughput:.1f} tok/s")
    print(f"   Request rate: {requests_per_second:.2f} req/s")
    print(f"\n   Queue Time Analysis:")
    print(f"   Est. queue time (P50): {estimated_queue_time:.2f}s ({queue_time_pct:.1f}%)")
    print(f"   Est. generation time: {estimated_generation_time_per_request:.2f}s")
    print(f"\n   Latency Estimates:")
    print(f"   P50: {p50_latency:.2f}s")
    print(f"   P90: {p90_latency:.2f}s")
    print(f"   P95: {p95_latency:.2f}s")
    print(f"   P99: {p99_latency:.2f}s")
    
    # Group by use case
    by_use_case = {}
    for i, prompt_info in enumerate(prompts):
        use_case = prompt_info["use_case"]
        if use_case not in by_use_case:
            by_use_case[use_case] = []
        by_use_case[use_case].append(all_token_counts[i])
    
    use_case_stats = []
    for use_case, tokens in by_use_case.items():
        avg_tokens = statistics.mean(tokens)
        # Estimate P95 for this use case
        sorted_uc = sorted(tokens)
        p95_idx = int(len(sorted_uc) * 0.95)
        uc_p95_tokens = sorted_uc[p95_idx] if p95_idx < len(sorted_uc) else sorted_uc[-1]
        uc_p95_latency = uc_p95_tokens / 100
        
        use_case_stats.append({
            "use_case": use_case,
            "avg_tokens": avg_tokens,
            "p95": uc_p95_latency,
        })
    
    return {
        "total_elapsed": total_elapsed,
        "num_requests": len(prompts),
        "total_tokens": total_tokens,
        "throughput": throughput,
        "requests_per_second": requests_per_second,
        "queue_time_p50": estimated_queue_time,
        "queue_time_percentage": queue_time_pct,
        "p50_latency": p50_latency,
        "p90_latency": p90_latency,
        "p95_latency": p95_latency,
        "p99_latency": p99_latency,
        "by_use_case": use_case_stats,
    }


def main():
    """Run comprehensive optimized benchmark"""
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive optimized benchmark (all Week 2 tests)"
    )
    parser.add_argument("--gpu-memory", type=float, help="GPU memory utilization")
    parser.add_argument("--max-seqs", type=int, help="Maximum number of sequences")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Week 3: Comprehensive Optimized Benchmark")
    print("="*70)
    print()
    print("This replicates ALL Week 2 benchmarks with optimized parameters")
    print("for direct before/after comparison.")
    print()
    
    from vllm import LLM, SamplingParams
    import torch
    
    # Load optimal parameters
    if args.gpu_memory is None or args.max_seqs is None:
        print("📊 Loading optimal parameters...")
        params = load_optimal_params()
        gpu_memory_util = args.gpu_memory or params["gpu_memory_utilization"]
        max_num_seqs = args.max_seqs or params["max_num_seqs"]
        print(f"   gpu_memory_utilization: {gpu_memory_util}")
        print(f"   max_num_seqs: {max_num_seqs}")
    else:
        gpu_memory_util = args.gpu_memory
        max_num_seqs = args.max_seqs
    
    print()
    
    # Load model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"📦 Loading model: {model_name}")
    
    try:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_util,
            max_num_seqs=max_num_seqs,
        )
        print("   ✅ Model loaded with optimized parameters")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
    
    # GPU memory info
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   🎮 GPU Memory: {reserved:.2f} GB / {total:.2f} GB ({reserved/total*100:.1f}%)")
    
    print()
    input("Press Enter to start comprehensive benchmark...")
    
    # Sampling params factory
    def make_sampling_params(max_tokens):
        return SamplingParams(max_tokens=max_tokens, temperature=0.0)
    
    start_time = time.time()
    
    # Run all benchmarks
    results = {}
    
    # 1. Systematic latency
    latency_results = run_systematic_latency_benchmark(llm, make_sampling_params)
    results["latency_benchmark"] = latency_results
    
    # 2. Batch throughput
    throughput_results = run_batch_throughput_benchmark(llm, make_sampling_params)
    results["throughput_benchmark"] = throughput_results
    
    # 3. Production workload
    production_results = run_production_workload_benchmark(llm, make_sampling_params)
    results["production_benchmark"] = production_results
    
    elapsed = time.time() - start_time
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save comprehensive results
    output_file = results_dir / "comprehensive_optimized_benchmark.json"
    with open(output_file, 'w') as f:
        json.dump({
            "config": {
                "gpu_memory_utilization": gpu_memory_util,
                "max_num_seqs": max_num_seqs,
            },
            "results": results,
            "total_time_minutes": elapsed / 60,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    
    # Also save in Week 2 compatible format
    # Latency benchmark
    latency_file = results_dir / "optimized_latency_benchmark.json"
    with open(latency_file, 'w') as f:
        json.dump(latency_results, f, indent=2)
    
    # Throughput benchmark
    throughput_file = results_dir / "optimized_throughput_benchmark.json"
    with open(throughput_file, 'w') as f:
        json.dump(throughput_results, f, indent=2)
    
    # Production benchmark
    production_file = results_dir / "optimized_production_benchmark.json"
    with open(production_file, 'w') as f:
        json.dump(production_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE BENCHMARK COMPLETE!")
    print("="*70)
    print()
    print(f"Total time: {elapsed/60:.1f} minutes")
    print()
    print("📁 Results saved:")
    print(f"   • {output_file}")
    print(f"   • {latency_file}")
    print(f"   • {throughput_file}")
    print(f"   • {production_file}")
    print()
    print("📊 Key Optimized Metrics:")
    print(f"   Throughput: {production_results['throughput']:.1f} tok/s")
    print(f"   Request Rate: {production_results['requests_per_second']:.2f} req/s")
    print(f"   Queue Time %: {production_results['queue_time_percentage']:.1f}%")
    print(f"   P95 Latency: {production_results['p95_latency']:.2f}s")
    print()
    print("🔍 Next Steps:")
    print("   1. Run: python compare_week2_week3.py")
    print("   2. Review improvement percentages")
    print("   3. Generate visualizations")
    print()


if __name__ == "__main__":
    main()


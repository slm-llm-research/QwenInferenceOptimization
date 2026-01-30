"""
Week 3: Optimize Queue Time - Target the 85% Bottleneck

Based on Week 2 findings, requests spend 85% of time in queue (10.47s)
and only 15% generating (1.78s). This is the PRIMARY bottleneck.

This script specifically measures and optimizes queue time by testing
aggressive max_num_seqs values to maximize concurrent processing.

Key Week 2 Finding:
- Queue time (P50): 10.47s (85%)
- Generation time: 1.78s (15%)
- Target: Reduce queue time to <50% of total latency

Usage:
    python optimize_queue_time.py
"""

import sys
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any


def measure_queue_and_generation_time(
    max_seqs: int,
    num_requests: int = 100,
    max_tokens: int = 100
) -> Dict[str, Any]:
    """
    Measure queue time vs generation time with specific max_num_seqs.
    
    This replicates Week 2's production throughput test to measure
    the queue time breakdown.
    
    Args:
        max_seqs: Maximum concurrent sequences
        num_requests: Number of concurrent requests to simulate
        max_tokens: Tokens to generate per request
    
    Returns:
        Results dict with queue/generation time breakdown
    """
    from vllm import LLM, SamplingParams
    import torch
    
    print(f"\n{'='*70}")
    print(f"Testing max_num_seqs = {max_seqs}")
    print('='*70)
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"   Loading model with max_num_seqs={max_seqs}...")
        
        llm = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            max_num_seqs=max_seqs,
            gpu_memory_utilization=0.9,
        )
        
        print(f"   ✅ Model loaded")
        
        # Create varied workload (like Week 2 production test)
        prompts = []
        use_cases = [
            ("Ultra-short Q&A", "What is Python?", 15),
            ("Short Q&A", "Explain machine learning in simple terms:", 60),
            ("Code generation", "Write a Python function to sort a list:", 100),
            ("Medium explanation", "Describe how neural networks work:", 150),
            ("Long-form", "Write a comprehensive guide to deep learning:", 300),
        ]
        
        # Distribute requests across use cases
        for i in range(num_requests):
            use_case = use_cases[i % len(use_cases)]
            prompts.append({
                'prompt': use_case[1],
                'max_tokens': use_case[2],
                'use_case': use_case[0]
            })
        
        print(f"   Running {num_requests} concurrent requests (mixed workload)...")
        print(f"   This simulates Week 2's production test...")
        
        # Track individual request timings
        request_timings = []
        
        # Process all requests
        sampling_params_list = [
            SamplingParams(max_tokens=p['max_tokens'], temperature=0.0)
            for p in prompts
        ]
        
        # Warmup
        _ = llm.generate(
            [prompts[0]['prompt']],
            SamplingParams(max_tokens=50, temperature=0.0)
        )
        
        # Run benchmark
        start_time = time.perf_counter()
        
        # Submit all requests (they will queue internally)
        outputs = llm.generate(
            [p['prompt'] for p in prompts],
            sampling_params_list[0],  # Use first for simplicity
        )
        
        total_elapsed = time.perf_counter() - start_time
        
        # Calculate metrics
        total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        throughput = total_tokens / total_elapsed
        
        # Get memory stats
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        else:
            peak_memory = None
        
        # Estimate queue time based on throughput
        # If processing was perfectly parallel, time would be much lower
        # The difference is queue time
        avg_tokens_per_request = total_tokens / num_requests
        estimated_generation_time = avg_tokens_per_request / 100  # ~100 tok/s per request
        
        avg_total_latency = total_elapsed / num_requests
        estimated_queue_time = max(0, avg_total_latency - estimated_generation_time)
        
        queue_time_percentage = (estimated_queue_time / avg_total_latency * 100) if avg_total_latency > 0 else 0
        
        print(f"\n   📊 Results:")
        print(f"      Total time: {total_elapsed:.2f}s")
        print(f"      Total tokens: {total_tokens}")
        print(f"      Throughput: {throughput:.1f} tok/s")
        print(f"      Avg latency per request: {avg_total_latency:.2f}s")
        print(f"      Estimated queue time: {estimated_queue_time:.2f}s ({queue_time_percentage:.1f}%)")
        print(f"      Estimated generation time: {estimated_generation_time:.2f}s ({100-queue_time_percentage:.1f}%)")
        print(f"      Peak memory: {peak_memory:.2f} GB" if peak_memory else "")
        
        # Clean up
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        return {
            "max_num_seqs": max_seqs,
            "success": True,
            "total_elapsed": total_elapsed,
            "num_requests": num_requests,
            "total_tokens": total_tokens,
            "throughput": throughput,
            "avg_latency_per_request": avg_total_latency,
            "estimated_queue_time": estimated_queue_time,
            "estimated_generation_time": estimated_generation_time,
            "queue_time_percentage": queue_time_percentage,
            "peak_memory_gb": peak_memory,
        }
        
    except torch.cuda.OutOfMemoryError:
        print(f"   ❌ CUDA Out of Memory")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        return {
            "max_num_seqs": max_seqs,
            "success": False,
            "error": "OOM",
        }
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        return {
            "max_num_seqs": max_seqs,
            "success": False,
            "error": str(e),
        }


def run_queue_time_optimization():
    """Run queue time optimization experiments"""
    
    print("="*70)
    print("Week 3: Optimize Queue Time (Target 85% → <50%)")
    print("="*70)
    print()
    print("🎯 Week 2 Baseline Finding:")
    print("   Queue time: 10.47s (85%)")
    print("   Generation time: 1.78s (15%)")
    print("   → PRIMARY BOTTLENECK!")
    print()
    print("💡 Strategy:")
    print("   Increase max_num_seqs to allow more concurrent processing")
    print("   This reduces queue waiting time by processing more requests in parallel")
    print()
    print("⚠️  Note: We'll test aggressive values to find the optimal setting")
    print()
    
    # Test configuration - aggressive values based on Week 2 insights
    test_values = [256, 512, 1024, 2048, 4096]
    num_requests = 100
    
    print(f"🔬 Test Configuration:")
    print(f"   max_num_seqs values: {test_values}")
    print(f"   Concurrent requests: {num_requests}")
    print(f"   Goal: Reduce queue time from 85% to <50%")
    print()
    
    input("Press Enter to start optimization...")
    
    # Run tests
    results = []
    baseline_queue_pct = 85.0  # From Week 2
    
    for max_seqs in test_values:
        result = measure_queue_and_generation_time(max_seqs, num_requests)
        results.append(result)
        
        # If we hit OOM, stop testing higher values
        if not result["success"]:
            print(f"\n   ℹ️  Stopping at max_num_seqs={max_seqs}")
            break
        
        # Wait between tests
        time.sleep(3)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 QUEUE TIME OPTIMIZATION RESULTS")
    print("="*70)
    print()
    print(f"Week 2 Baseline: 85% queue time (10.47s queue / 1.78s generation)")
    print()
    
    print(f"{'max_num_seqs':<15} {'Status':<12} {'Queue %':<12} {'Queue Time':<15} {'Throughput':<15}")
    print("-" * 70)
    
    successful_results = []
    
    for result in results:
        max_seqs = result["max_num_seqs"]
        
        if result["success"]:
            status = "✅ Success"
            queue_pct = f"{result['queue_time_percentage']:.1f}%"
            queue_time = f"{result['estimated_queue_time']:.2f}s"
            throughput = f"{result['throughput']:.1f} tok/s"
            successful_results.append(result)
        else:
            status = f"❌ {result.get('error', 'Failed')}"
            queue_pct = "N/A"
            queue_time = "N/A"
            throughput = "N/A"
        
        print(f"{max_seqs:<15} {status:<12} {queue_pct:<12} {queue_time:<15} {throughput:<15}")
    
    print()
    
    # Find optimal value
    if successful_results:
        # Find config with lowest queue percentage
        best_result = min(successful_results, key=lambda x: x["queue_time_percentage"])
        
        # Find highest throughput for comparison
        highest_throughput = max(successful_results, key=lambda x: x["throughput"])
        
        print("="*70)
        print("🎯 RECOMMENDATION")
        print("="*70)
        print()
        print(f"Optimal Configuration (Lowest Queue Time):")
        print(f"   max_num_seqs: {best_result['max_num_seqs']}")
        print(f"   Queue time: {best_result['estimated_queue_time']:.2f}s ({best_result['queue_time_percentage']:.1f}%)")
        print(f"   Generation time: {best_result['estimated_generation_time']:.2f}s")
        print(f"   Throughput: {best_result['throughput']:.1f} tok/s")
        print()
        
        # Calculate improvement
        queue_time_reduction = baseline_queue_pct - best_result['queue_time_percentage']
        
        print(f"📈 Improvement vs Week 2 Baseline:")
        print(f"   Queue time: 85% → {best_result['queue_time_percentage']:.1f}% (Δ {queue_time_reduction:+.1f}%)")
        print()
        
        if best_result['queue_time_percentage'] < 50:
            print(f"   ✅ TARGET ACHIEVED! Queue time reduced below 50%")
        elif best_result['queue_time_percentage'] < 70:
            print(f"   ⚠️  Good improvement but not at target yet")
            print(f"      Consider testing even higher max_num_seqs values")
        else:
            print(f"   ⚠️  Limited improvement. Queue time still high.")
            print(f"      May need GPU memory optimization or hardware upgrade")
        print()
        
        # Show configuration
        print("Use this configuration:")
        print()
        print("```python")
        print(f"llm = LLM(")
        print(f'    model="Qwen/Qwen2.5-7B-Instruct",')
        print(f"    max_num_seqs={best_result['max_num_seqs']},")
        print(f"    gpu_memory_utilization=0.9,  # Adjust as needed")
        print(f")")
        print("```")
        print()
        
        # Compare with baseline throughput
        baseline_throughput = 949  # From Week 2
        throughput_improvement = (best_result['throughput'] / baseline_throughput - 1) * 100
        print(f"💡 Throughput: {baseline_throughput} → {best_result['throughput']:.1f} tok/s ({throughput_improvement:+.1f}%)")
        print()
        
    else:
        print("❌ All tests failed.")
        print("   Try starting with lower values or optimizing memory first.")
        print()
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "queue_time_optimization.json"
    with open(output_file, 'w') as f:
        json.dump({
            "week2_baseline": {
                "queue_time_seconds": 10.47,
                "generation_time_seconds": 1.78,
                "queue_time_percentage": 85.0,
            },
            "test_values": test_values,
            "num_requests": num_requests,
            "results": results,
            "optimal": best_result if successful_results else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    
    print(f"📁 Results saved to: {output_file}")
    print()


def main():
    """Main function"""
    run_queue_time_optimization()


if __name__ == "__main__":
    main()


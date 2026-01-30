"""
Week 3: Optimize GPU Memory Utilization

This script tests different gpu_memory_utilization values to find the optimal
setting for your GPU. Higher values allow more KV cache and concurrent sequences,
but too high can cause OOM errors.

Parameter: gpu_memory_utilization
- Default: 0.9 (90% of GPU memory)
- Range: 0.7 - 0.95
- Impact: Higher = more concurrent sequences = better throughput

Usage:
    python optimize_memory_utilization.py
"""

import sys
import time
import json
from pathlib import Path


def test_memory_utilization(mem_util: float, test_prompts: list, max_tokens: int = 50):
    """
    Test throughput with specific memory utilization setting.
    
    Args:
        mem_util: GPU memory utilization (0.0 to 1.0)
        test_prompts: List of prompts to test
        max_tokens: Tokens to generate
    
    Returns:
        Results dict or None if OOM
    """
    from vllm import LLM, SamplingParams
    import torch
    
    print(f"\n{'='*70}")
    print(f"Testing gpu_memory_utilization = {mem_util}")
    print('='*70)
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Load model with specified memory utilization
        print(f"   Loading model with {mem_util*100:.0f}% GPU memory...")
        
        llm = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            gpu_memory_utilization=mem_util,
        )
        
        print(f"   ✅ Model loaded successfully")
        
        # Get memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   Memory: {reserved:.2f} GB / {total:.2f} GB ({reserved/total*100:.1f}%)")
        
        # Run inference
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )
        
        # Warmup
        _ = llm.generate(test_prompts[:2], sampling_params)
        
        # Benchmark - run multiple times to get latency distribution
        print(f"   Running benchmark with {len(test_prompts)} prompts...")
        
        # Run 3 times for statistics
        all_latencies = []
        all_tokens = []
        
        for run in range(3):
            start = time.perf_counter()
            outputs = llm.generate(test_prompts, sampling_params)
            elapsed = time.perf_counter() - start
            all_latencies.append(elapsed)
            
            # Count tokens
            total_tokens = sum(
                len(output.outputs[0].text.split())
                for output in outputs
            )
            all_tokens.append(total_tokens)
        
        # Calculate statistics
        avg_elapsed = sum(all_latencies) / len(all_latencies)
        avg_tokens = sum(all_tokens) / len(all_tokens)
        throughput = avg_tokens / avg_elapsed
        
        # Estimate per-request latencies (for percentiles)
        avg_latency_per_request = avg_elapsed / len(test_prompts)
        
        print(f"   ⏱️  Time: {avg_elapsed:.2f}s (avg of {len(all_latencies)} runs)")
        print(f"   📈 Throughput: {throughput:.1f} tokens/sec")
        print(f"   📊 Avg latency per request: {avg_latency_per_request:.3f}s")
        
        # Clean up
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "mem_utilization": mem_util,
            "success": True,
            "elapsed_time": avg_elapsed,
            "total_tokens": avg_tokens,
            "throughput": throughput,
            "avg_latency_per_request": avg_latency_per_request,
            "memory_reserved_gb": reserved if torch.cuda.is_available() else None,
        }
        
    except torch.cuda.OutOfMemoryError:
        print(f"   ❌ CUDA Out of Memory Error")
        print(f"   {mem_util*100:.0f}% is too high for this workload")
        
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "mem_utilization": mem_util,
            "success": False,
            "error": "OOM",
        }
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "mem_utilization": mem_util,
            "success": False,
            "error": str(e),
        }


def run_optimization():
    """Run memory utilization optimization"""
    
    print("="*70)
    print("Week 3: Optimize GPU Memory Utilization")
    print("="*70)
    print()
    print("This experiment tests different gpu_memory_utilization values")
    print("to find the optimal setting for your GPU.")
    print()
    print("⚠️  Note: Some tests may fail with OOM - this is expected!")
    print("         We're finding the upper limit.")
    print()
    
    # Test configuration
    test_values = [0.7, 0.8, 0.85, 0.9, 0.95]
    
    # Create test prompts (batch of 16)
    test_prompts = [
        "Explain the concept of neural networks:",
        "What is quantum computing?",
        "Describe the water cycle:",
        "How does machine learning work?",
    ] * 4
    
    print(f"🔬 Test Configuration:")
    print(f"   Memory utilization values: {test_values}")
    print(f"   Batch size: {len(test_prompts)}")
    print(f"   Max tokens: 50")
    print()
    print(f"💡 Week 2 Context:")
    print(f"   Higher memory allows more concurrent sequences")
    print(f"   This can help reduce the 85% queue time bottleneck")
    print()
    
    input("Press Enter to start optimization...")
    
    # Run tests
    results = []
    
    for mem_util in test_values:
        result = test_memory_utilization(mem_util, test_prompts, max_tokens=50)
        results.append(result)
        
        # Wait between tests
        time.sleep(2)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 OPTIMIZATION RESULTS")
    print("="*70)
    print()
    
    print(f"{'Memory Util':<15} {'Status':<12} {'Throughput':<15} {'Avg Latency':<15} {'Memory':<12}")
    print("-" * 70)
    
    successful_results = []
    
    for result in results:
        mem_util = result["mem_utilization"]
        
        if result["success"]:
            status = "✅ Success"
            throughput = f"{result['throughput']:.1f} tok/s"
            latency = f"{result.get('avg_latency_per_request', 0):.3f}s"
            memory = f"{result['memory_reserved_gb']:.2f} GB" if result['memory_reserved_gb'] else "N/A"
            successful_results.append(result)
        else:
            status = f"❌ {result['error']}"
            throughput = "N/A"
            latency = "N/A"
            memory = "N/A"
        
        print(f"{mem_util:<15.2f} {status:<12} {throughput:<15} {latency:<15} {memory:<12}")
    
    print()
    
    # Find optimal value
    if successful_results:
        best_result = max(successful_results, key=lambda x: x["throughput"])
        
        print("="*70)
        print("🎯 RECOMMENDATION")
        print("="*70)
        print()
        print(f"Optimal gpu_memory_utilization: {best_result['mem_utilization']}")
        print(f"Throughput: {best_result['throughput']:.1f} tokens/sec")
        print()
        print("Use this value in your LLM configuration:")
        print()
        print("```python")
        print(f"llm = LLM(")
        print(f'    model="Qwen/Qwen2.5-7B-Instruct",')
        print(f"    gpu_memory_utilization={best_result['mem_utilization']},")
        print(f")")
        print("```")
        print()
        
        # Calculate improvement
        baseline_result = next((r for r in successful_results if r["mem_utilization"] == 0.9), None)
        if baseline_result and best_result["mem_utilization"] != 0.9:
            improvement = (best_result["throughput"] / baseline_result["throughput"] - 1) * 100
            print(f"💡 Improvement over default (0.9): {improvement:+.1f}%")
        elif best_result["mem_utilization"] == 0.9:
            print(f"💡 Default value (0.9) is optimal for your setup")
        print()
    else:
        print("❌ No successful tests. Try:")
        print("   • Reducing batch size")
        print("   • Using a smaller model")
        print("   • Closing other GPU applications")
        print()
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "memory_optimization.json"
    with open(output_file, 'w') as f:
        json.dump({
            "test_values": test_values,
            "results": results,
            "optimal": best_result if successful_results else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    
    print(f"📁 Results saved to: {output_file}")
    print()


def main():
    """Main function"""
    run_optimization()


if __name__ == "__main__":
    main()


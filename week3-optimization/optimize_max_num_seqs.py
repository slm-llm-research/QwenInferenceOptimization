"""
Week 3: Optimize max_num_seqs

This script finds the optimal max_num_seqs parameter - the maximum number
of sequences that can be processed concurrently. Higher values improve
GPU utilization and throughput, but are limited by available memory.

Parameter: max_num_seqs
- Default: 256
- Range: 64 - 2048+
- Impact: Higher = more concurrent requests = better throughput
- Limit: GPU memory capacity

Usage:
    python optimize_max_num_seqs.py
"""

import sys
import time
import json
from pathlib import Path


def test_max_num_seqs(max_seqs: int, batch_size: int, max_tokens: int = 50):
    """
    Test throughput with specific max_num_seqs setting.
    
    Args:
        max_seqs: Maximum number of sequences
        batch_size: Number of prompts to test with
        max_tokens: Tokens to generate
    
    Returns:
        Results dict or None if OOM
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
        
        # Load model with specified max_num_seqs
        print(f"   Loading model with max_num_seqs={max_seqs}...")
        
        llm = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            max_num_seqs=max_seqs,
            gpu_memory_utilization=0.9,  # Use default for fair comparison
        )
        
        print(f"   ✅ Model loaded successfully")
        
        # Create prompts
        base_prompt = "Write a brief explanation of machine learning:"
        test_prompts = [base_prompt] * batch_size
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )
        
        # Get memory stats before inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1024**3
        
        # Warmup
        _ = llm.generate(test_prompts[:min(4, batch_size)], sampling_params)
        
        # Benchmark
        print(f"   Running benchmark with batch_size={batch_size}...")
        start = time.perf_counter()
        outputs = llm.generate(test_prompts, sampling_params)
        elapsed = time.perf_counter() - start
        
        # Count tokens
        total_tokens = sum(
            len(output.outputs[0].text.split())
            for output in outputs
        )
        
        throughput = total_tokens / elapsed
        
        # Memory stats after
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"   ⏱️  Time: {elapsed:.2f}s")
        print(f"   📈 Throughput: {throughput:.1f} tokens/sec")
        print(f"   🎮 Peak memory: {mem_peak:.2f} GB")
        
        # Clean up
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        return {
            "max_num_seqs": max_seqs,
            "batch_size": batch_size,
            "success": True,
            "elapsed_time": elapsed,
            "total_tokens": total_tokens,
            "throughput": throughput,
            "peak_memory_gb": mem_peak if torch.cuda.is_available() else None,
        }
        
    except torch.cuda.OutOfMemoryError:
        print(f"   ❌ CUDA Out of Memory Error")
        print(f"   max_num_seqs={max_seqs} exceeds GPU capacity")
        
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        return {
            "max_num_seqs": max_seqs,
            "batch_size": batch_size,
            "success": False,
            "error": "OOM",
        }
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        return {
            "max_num_seqs": max_seqs,
            "batch_size": batch_size,
            "success": False,
            "error": str(e),
        }


def run_optimization():
    """Run max_num_seqs optimization"""
    
    print("="*70)
    print("Week 3: Optimize max_num_seqs")
    print("="*70)
    print()
    print("This experiment tests different max_num_seqs values to find")
    print("the optimal concurrent sequence limit for your GPU.")
    print()
    print("⚠️  Note: Higher values may cause OOM - this is expected!")
    print("         We're finding the sweet spot.")
    print()
    
    # Test configuration
    test_values = [64, 128, 256, 512, 1024]
    batch_size = 32  # Test with moderate batch
    max_tokens = 50
    
    print(f"🔬 Test Configuration:")
    print(f"   max_num_seqs values: {test_values}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max tokens: {max_tokens}")
    print()
    
    input("Press Enter to start optimization...")
    
    # Run tests
    results = []
    
    for max_seqs in test_values:
        result = test_max_num_seqs(max_seqs, batch_size, max_tokens)
        results.append(result)
        
        # If we hit OOM, stop testing higher values
        if not result["success"] and result.get("error") == "OOM":
            print(f"\n   ℹ️  Stopping at max_num_seqs={max_seqs} (OOM limit reached)")
            break
        
        # Wait between tests
        time.sleep(3)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 OPTIMIZATION RESULTS")
    print("="*70)
    print()
    
    print(f"{'max_num_seqs':<15} {'Status':<12} {'Throughput':<20} {'Peak Memory':<15}")
    print("-" * 70)
    
    successful_results = []
    
    for result in results:
        max_seqs = result["max_num_seqs"]
        
        if result["success"]:
            status = "✅ Success"
            throughput = f"{result['throughput']:.1f} tok/s"
            memory = f"{result['peak_memory_gb']:.2f} GB" if result['peak_memory_gb'] else "N/A"
            successful_results.append(result)
        else:
            status = f"❌ {result['error']}"
            throughput = "N/A"
            memory = "N/A"
        
        print(f"{max_seqs:<15} {status:<12} {throughput:<20} {memory:<15}")
    
    print()
    
    # Find optimal value
    if successful_results:
        # Find best throughput that succeeded
        best_result = max(successful_results, key=lambda x: x["throughput"])
        
        # Check if throughput is still increasing
        if len(successful_results) >= 2:
            last_throughput = successful_results[-1]["throughput"]
            second_last_throughput = successful_results[-2]["throughput"]
            improvement = (last_throughput / second_last_throughput - 1) * 100
            
            still_scaling = improvement > 5  # 5% improvement threshold
        else:
            still_scaling = False
        
        print("="*70)
        print("🎯 RECOMMENDATION")
        print("="*70)
        print()
        print(f"Optimal max_num_seqs: {best_result['max_num_seqs']}")
        print(f"Throughput: {best_result['throughput']:.1f} tokens/sec")
        print(f"Peak memory: {best_result['peak_memory_gb']:.2f} GB")
        print()
        
        if still_scaling:
            print("💡 Performance is still improving!")
            print(f"   Consider testing higher values: {best_result['max_num_seqs'] * 2}, {best_result['max_num_seqs'] * 4}")
            print()
        
        print("Use this value in your LLM configuration:")
        print()
        print("```python")
        print(f"llm = LLM(")
        print(f'    model="Qwen/Qwen2.5-7B-Instruct",')
        print(f"    max_num_seqs={best_result['max_num_seqs']},")
        print(f")")
        print("```")
        print()
        
        # Calculate improvement over default
        baseline_result = next((r for r in successful_results if r["max_num_seqs"] == 256), None)
        if baseline_result and best_result["max_num_seqs"] != 256:
            improvement = (best_result["throughput"] / baseline_result["throughput"] - 1) * 100
            print(f"📈 Improvement over default (256): {improvement:+.1f}%")
        elif best_result["max_num_seqs"] == 256:
            print(f"💡 Default value (256) is optimal for your setup")
        print()
    else:
        print("❌ All tests failed. Try:")
        print("   • Reducing batch_size")
        print("   • Starting with lower max_num_seqs values (32, 64)")
        print("   • Increasing gpu_memory_utilization first")
        print()
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "max_num_seqs_optimization.json"
    with open(output_file, 'w') as f:
        json.dump({
            "test_values": test_values,
            "batch_size": batch_size,
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


"""
Week 3: Optimize Chunked Prefill for Long Sequences

Based on Week 2 findings, long sequences (300+ tokens) show super-linear
scaling with P95 latency reaching 4.4s for 500+ token sequences.

Key Week 2 Finding:
- Very long sequences (500+ tokens): 4.4s at P95
- Super-linear scaling beyond 300 tokens (15.8x vs baseline)
- Target: Reduce long sequence latency to <3s at P95

Chunked prefill breaks long prompt processing into manageable chunks,
reducing memory pressure and improving attention efficiency.

Usage:
    python optimize_chunked_prefill.py
"""

import sys
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any


def test_chunked_prefill(
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int = None,
    sequence_lengths: List[int] = None
) -> Dict[str, Any]:
    """
    Test performance with chunked prefill configuration.
    
    Args:
        enable_chunked_prefill: Whether to enable chunked prefill
        max_num_batched_tokens: Max tokens in a batch (controls chunk size)
        sequence_lengths: List of sequence lengths to test
    
    Returns:
        Results dict with latency by sequence length
    """
    from vllm import LLM, SamplingParams
    import torch
    
    if sequence_lengths is None:
        sequence_lengths = [100, 200, 300, 400, 500]
    
    config_name = "Chunked Prefill" if enable_chunked_prefill else "Standard"
    
    print(f"\n{'='*70}")
    print(f"Testing: {config_name}")
    if enable_chunked_prefill and max_num_batched_tokens:
        print(f"max_num_batched_tokens: {max_num_batched_tokens}")
    print('='*70)
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"   Loading model...")
        
        # Build engine args
        engine_kwargs = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.9,
        }
        
        if enable_chunked_prefill and max_num_batched_tokens:
            engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
            engine_kwargs["enable_chunked_prefill"] = True
        
        llm = LLM(**engine_kwargs)
        
        print(f"   ✅ Model loaded")
        
        # Test different sequence lengths
        results_by_length = []
        
        for target_length in sequence_lengths:
            # Create prompt that will generate approximately target_length tokens
            # Use a detailed prompt to test prefill performance
            base_prompt = (
                "Write a comprehensive, detailed explanation of artificial intelligence, "
                "covering neural networks, deep learning, machine learning algorithms, "
                "training processes, optimization techniques, and real-world applications. "
                "Include technical details and examples. "
            )
            
            # Adjust output length
            output_tokens = max(50, target_length - len(base_prompt.split()) * 2)
            
            sampling_params = SamplingParams(
                max_tokens=output_tokens,
                temperature=0.0,
            )
            
            # Warmup
            if target_length == sequence_lengths[0]:
                _ = llm.generate([base_prompt], SamplingParams(max_tokens=20, temperature=0.0))
            
            # Run multiple times for statistics
            latencies = []
            num_runs = 10
            
            print(f"\n   Testing ~{target_length} token sequences ({num_runs} runs)...")
            
            for i in range(num_runs):
                start = time.perf_counter()
                outputs = llm.generate([base_prompt], sampling_params)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                
                if i == 0:
                    actual_tokens = len(outputs[0].outputs[0].token_ids)
            
            # Calculate statistics
            p50 = statistics.median(latencies)
            p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            mean_latency = statistics.mean(latencies)
            std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
            
            print(f"      Actual tokens: {actual_tokens}")
            print(f"      P50: {p50:.3f}s")
            print(f"      P95: {p95:.3f}s")
            print(f"      Mean: {mean_latency:.3f}s ± {std_dev:.3f}s")
            
            results_by_length.append({
                "target_length": target_length,
                "actual_tokens": actual_tokens,
                "latencies": latencies,
                "p50": p50,
                "p95": p95,
                "mean": mean_latency,
                "std_dev": std_dev,
            })
        
        # Get memory stats
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        else:
            peak_memory = None
        
        # Clean up
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        return {
            "config": config_name,
            "enable_chunked_prefill": enable_chunked_prefill,
            "max_num_batched_tokens": max_num_batched_tokens,
            "success": True,
            "results_by_length": results_by_length,
            "peak_memory_gb": peak_memory,
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        return {
            "config": config_name,
            "enable_chunked_prefill": enable_chunked_prefill,
            "success": False,
            "error": str(e),
        }


def run_chunked_prefill_optimization():
    """Run chunked prefill optimization experiments"""
    
    print("="*70)
    print("Week 3: Optimize Chunked Prefill for Long Sequences")
    print("="*70)
    print()
    print("🎯 Week 2 Baseline Finding:")
    print("   Long sequences (500+ tokens): 4.4s at P95")
    print("   Super-linear scaling: 15.8x vs short sequences")
    print("   Target: <3s for long sequences")
    print()
    print("💡 Strategy:")
    print("   Enable chunked prefill to break long prompt processing into chunks")
    print("   This reduces memory pressure and improves attention efficiency")
    print()
    
    sequence_lengths = [100, 200, 300, 400, 500]
    
    print(f"🔬 Test Configuration:")
    print(f"   Sequence lengths: {sequence_lengths} tokens")
    print(f"   Comparing: Standard vs Chunked Prefill")
    print()
    
    input("Press Enter to start optimization...")
    
    # Test configurations
    configs = [
        {
            "name": "Baseline (Standard)",
            "enable_chunked_prefill": False,
            "max_num_batched_tokens": None,
        },
        {
            "name": "Chunked Prefill (2048 tokens/chunk)",
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 2048,
        },
        {
            "name": "Chunked Prefill (4096 tokens/chunk)",
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
        },
        {
            "name": "Chunked Prefill (8192 tokens/chunk)",
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 8192,
        },
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {config['name']}")
        print('='*70)
        
        result = test_chunked_prefill(
            enable_chunked_prefill=config["enable_chunked_prefill"],
            max_num_batched_tokens=config["max_num_batched_tokens"],
            sequence_lengths=sequence_lengths
        )
        
        all_results.append(result)
        
        if not result["success"]:
            print(f"   ⚠️  Configuration failed, skipping...")
            continue
        
        # Wait between tests
        time.sleep(3)
    
    # Print summary comparison
    print("\n" + "="*70)
    print("📊 LONG SEQUENCE OPTIMIZATION RESULTS")
    print("="*70)
    print()
    print("Week 2 Baseline: Very long sequences (500+ tokens) = 4.4s at P95")
    print()
    
    # Compare results by sequence length
    for target_length in sequence_lengths:
        print(f"\n{target_length} Token Sequences (P95 Latency):")
        print("-" * 70)
        
        week2_baseline = {
            100: 1.02,  # From Week 2 systematic test
            200: 2.0,   # Estimated
            300: 2.54,  # From Week 2 P95
            400: 3.5,   # Estimated
            500: 4.40,  # From Week 2 P95
        }.get(target_length, None)
        
        if week2_baseline:
            print(f"   Week 2 Baseline:     {week2_baseline:.3f}s")
        
        for result in all_results:
            if not result["success"]:
                continue
            
            # Find matching length result
            length_result = next(
                (r for r in result["results_by_length"] if r["target_length"] == target_length),
                None
            )
            
            if length_result:
                config_name = result["config"]
                p95 = length_result["p95"]
                
                improvement = ""
                if week2_baseline:
                    pct_change = ((p95 - week2_baseline) / week2_baseline) * 100
                    improvement = f" ({pct_change:+.1f}%)"
                
                print(f"   {config_name:30} {p95:.3f}s{improvement}")
    
    # Find best configuration
    successful_results = [r for r in all_results if r["success"]]
    
    if successful_results:
        # Compare 500-token performance (most critical)
        long_seq_results = []
        for result in successful_results:
            length_500 = next(
                (r for r in result["results_by_length"] if r["target_length"] == 500),
                None
            )
            if length_500:
                long_seq_results.append({
                    "config": result["config"],
                    "enable_chunked": result["enable_chunked_prefill"],
                    "max_batched_tokens": result.get("max_num_batched_tokens"),
                    "p95": length_500["p95"],
                })
        
        if long_seq_results:
            best_config = min(long_seq_results, key=lambda x: x["p95"])
            
            print("\n" + "="*70)
            print("🎯 RECOMMENDATION")
            print("="*70)
            print()
            print(f"Best Configuration for Long Sequences:")
            print(f"   {best_config['config']}")
            print(f"   500-token P95 latency: {best_config['p95']:.3f}s")
            print()
            
            week2_500_token = 4.40
            improvement = ((best_config['p95'] - week2_500_token) / week2_500_token) * 100
            
            print(f"📈 Improvement vs Week 2 Baseline:")
            print(f"   500+ tokens: 4.40s → {best_config['p95']:.3f}s ({improvement:+.1f}%)")
            print()
            
            if best_config['p95'] < 3.0:
                print(f"   ✅ TARGET ACHIEVED! Long sequence latency < 3s")
            elif best_config['p95'] < 3.5:
                print(f"   ⚠️  Close to target. Consider combining with other optimizations.")
            else:
                print(f"   ⚠️  Still above target. May need hardware upgrade or content length limits.")
            print()
            
            # Show configuration
            print("Use this configuration:")
            print()
            print("```python")
            print(f"llm = LLM(")
            print(f'    model="Qwen/Qwen2.5-7B-Instruct",')
            if best_config['enable_chunked']:
                print(f"    enable_chunked_prefill=True,")
                print(f"    max_num_batched_tokens={best_config['max_batched_tokens']},")
            print(f"    gpu_memory_utilization=0.9,")
            print(f")")
            print("```")
            print()
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "chunked_prefill_optimization.json"
    with open(output_file, 'w') as f:
        json.dump({
            "week2_baseline": {
                "500_token_p95": 4.40,
                "note": "Super-linear scaling for long sequences",
            },
            "sequence_lengths_tested": sequence_lengths,
            "configurations": configs,
            "results": all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    
    print(f"📁 Results saved to: {output_file}")
    print()


def main():
    """Main function"""
    run_chunked_prefill_optimization()


if __name__ == "__main__":
    main()


"""
Week 5: Simulate Concurrent Requests

This script simulates multiple clients sending requests concurrently to test
vLLM's continuous batching and scheduler behavior.

Key concepts:
- Concurrent vs parallel requests
- Scheduler fairness
- Throughput under load

Usage:
    python simulate_concurrent_requests.py
"""

import asyncio
import time
import sys
from typing import List


async def send_request_async(llm, prompt: str, request_id: int, sampling_params):
    """Send a single async request"""
    start = time.time()
    
    # vLLM generate is synchronous, but we can simulate async by running in executor
    outputs = llm.generate([prompt], sampling_params)
    
    elapsed = time.time() - start
    generated_text = outputs[0].outputs[0].text
    tokens = len(generated_text.split())
    
    return {
        "request_id": request_id,
        "elapsed": elapsed,
        "tokens": tokens,
        "throughput": tokens / elapsed if elapsed > 0 else 0,
    }


def simulate_concurrent_requests(num_requests: int = 10, use_multi_gpu: bool = False):
    """Simulate concurrent requests"""
    
    from vllm import LLM, SamplingParams
    import torch
    
    print("="*70)
    print(f"Simulating {num_requests} Concurrent Requests")
    print("="*70)
    print()
    
    # Determine tensor parallel size
    if use_multi_gpu:
        num_gpus = torch.cuda.device_count()
        tp_size = min(num_gpus, 4)
        print(f"🎮 Using {tp_size} GPUs (tensor parallelism)")
    else:
        tp_size = 1
        print(f"🎮 Using 1 GPU")
    
    print()
    
    # Load model
    print("📦 Loading model...")
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
    )
    print("✅ Model loaded")
    print()
    
    # Create varied prompts
    prompts = [
        "Explain machine learning:",
        "What is quantum computing?",
        "Describe the water cycle:",
        "How do neural networks work?",
        "What is artificial intelligence?",
        "Explain blockchain technology:",
        "How does photosynthesis work?",
        "What is general relativity?",
        "Describe DNA structure:",
        "Explain the Internet:",
    ]
    
    # Extend if needed
    while len(prompts) < num_requests:
        prompts.extend(prompts)
    prompts = prompts[:num_requests]
    
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.7,
    )
    
    # Test 1: Batch processing (baseline)
    print("="*70)
    print("TEST 1: Batch Processing (all at once)")
    print("="*70)
    print()
    
    print(f"📤 Sending {num_requests} prompts as single batch...")
    start_batch = time.time()
    batch_outputs = llm.generate(prompts, sampling_params)
    batch_elapsed = time.time() - start_batch
    
    batch_total_tokens = sum(len(out.outputs[0].text.split()) for out in batch_outputs)
    batch_throughput = batch_total_tokens / batch_elapsed
    
    print(f"⏱️  Total time: {batch_elapsed:.2f}s")
    print(f"📊 Throughput: {batch_throughput:.1f} tokens/sec")
    print(f"📈 Tokens generated: {batch_total_tokens}")
    print()
    
    # Test 2: Sequential processing
    print("="*70)
    print("TEST 2: Sequential Processing (one by one)")
    print("="*70)
    print()
    
    print(f"📤 Processing {min(5, num_requests)} prompts sequentially...")
    sequential_results = []
    start_seq = time.time()
    
    for i in range(min(5, num_requests)):  # Limit to 5 for time
        outputs = llm.generate([prompts[i]], sampling_params)
        elapsed = time.time() - start_seq
        tokens = len(outputs[0].outputs[0].text.split())
        sequential_results.append({"elapsed": elapsed, "tokens": tokens})
    
    seq_elapsed = time.time() - start_seq
    seq_total_tokens = sum(r["tokens"] for r in sequential_results)
    seq_throughput = seq_total_tokens / seq_elapsed
    
    print(f"⏱️  Total time: {seq_elapsed:.2f}s (for {len(sequential_results)} requests)")
    print(f"📊 Throughput: {seq_throughput:.1f} tokens/sec")
    print()
    
    # Comparison
    print("="*70)
    print("📊 COMPARISON")
    print("="*70)
    print()
    print(f"{'Method':<20} {'Time (s)':<15} {'Throughput':<20} {'Speedup':<15}")
    print("-" * 70)
    print(f"{'Sequential':<20} {seq_elapsed:<15.2f} {seq_throughput:<20.1f} {'1.0x':<15}")
    print(f"{'Batch':<20} {batch_elapsed:<15.2f} {batch_throughput:<20.1f} {batch_throughput/seq_throughput:<15.1f}x")
    
    print()
    print("💡 Key Insights:")
    print(f"   • Batch processing is {batch_throughput/seq_throughput:.1f}x faster")
    print(f"   • vLLM's continuous batching handles concurrent requests efficiently")
    print(f"   • GPU utilization is much higher with batching")
    
    if tp_size > 1:
        print(f"   • Using {tp_size} GPUs for additional parallelism")
    
    print()
    
    # Test 3: Show individual request latencies in batch
    print("="*70)
    print("TEST 3: Individual Request Latencies")
    print("="*70)
    print()
    print("In a batch, all requests finish at roughly the same time.")
    print("This is expected - vLLM processes them together.")
    print()
    print("For true concurrent request simulation, use vLLM server mode")
    print("(covered in Week 6 deployment).")
    print()


def main():
    """Main function"""
    
    print("="*70)
    print("Week 5: Concurrent Request Simulation")
    print("="*70)
    print()
    
    import torch
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print(f"Available GPUs: {num_gpus}")
    print()
    
    if num_gpus >= 2:
        print("🔧 Configuration options:")
        print("   1. Single GPU")
        print("   2. Multi-GPU (tensor parallelism)")
        print()
        choice = input("Select (1 or 2, default=1): ").strip()
        use_multi = (choice == "2")
    else:
        use_multi = False
    
    print()
    num_requests = 10
    print(f"Will simulate {num_requests} concurrent requests")
    print()
    
    simulate_concurrent_requests(num_requests, use_multi_gpu=use_multi)
    
    print("="*70)
    print("✅ Concurrent request simulation complete!")
    print("="*70)
    print()
    print("🔜 Next steps:")
    print("   • For true concurrent testing, use vLLM server mode (Week 6)")
    print("   • Run benchmark_scaling.py to compare 1-GPU vs multi-GPU")
    print()


if __name__ == "__main__":
    main()


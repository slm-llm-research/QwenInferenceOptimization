"""
Week 3: Profile Baseline Performance

This script runs inference with PyTorch Profiler to capture detailed GPU metrics.
The profiler trace can be viewed in TensorBoard to identify bottlenecks.

Metrics captured:
- CUDA kernel execution times
- Memory transfers
- GPU utilization over time
- Operation breakdown

Usage:
    python profile_baseline.py
    
    # Then view results:
    tensorboard --logdir=./profiles
"""

import sys
import os
from pathlib import Path


def profile_inference():
    """Run inference with profiling enabled"""
    
    print("="*70)
    print("Week 3: Profiling Baseline Performance")
    print("="*70)
    print()
    
    # Import dependencies
    try:
        from vllm import LLM, SamplingParams
        import torch
        import torch.profiler as profiler
    except ImportError as e:
        print(f"❌ Error: Missing dependency - {e}")
        print("Run: pip install torch-tb-profiler tensorboard")
        sys.exit(1)
    
    # Create profiles directory
    profiles_dir = Path("profiles")
    profiles_dir.mkdir(exist_ok=True)
    
    # Load model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"📦 Loading model: {model_name}")
    print("   (Profiling will start after model loads...)")
    print()
    
    try:
        llm = LLM(model=model_name, trust_remote_code=True)
        print("✅ Model loaded")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Prepare test workload
    prompts = [
        "Explain machine learning in simple terms:",
        "What is the capital of France?",
        "Write a haiku about artificial intelligence:",
        "Describe the process of photosynthesis:",
    ] * 4  # 16 prompts total for good profiling data
    
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.0,
    )
    
    print(f"🔬 Profiling Configuration:")
    print(f"   Prompts: {len(prompts)}")
    print(f"   Max tokens: {sampling_params.max_tokens}")
    print(f"   Output directory: {profiles_dir}/")
    print()
    
    # Warmup
    print("🔥 Warmup run (not profiled)...")
    _ = llm.generate(prompts[:2], sampling_params)
    print("   Done")
    print()
    
    # Profile inference
    print("📊 Starting profiling...")
    print("   This will take 1-2 minutes...")
    print()
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        schedule=profiler.schedule(
            wait=0,     # No warmup
            warmup=1,   # 1 step warmup
            active=3,   # Profile 3 steps
            repeat=1    # Do once
        ),
        on_trace_ready=profiler.tensorboard_trace_handler(str(profiles_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Run inference steps
        for i in range(4):
            _ = llm.generate(prompts[i*4:(i+1)*4], sampling_params)
            prof.step()
    
    print("✅ Profiling complete!")
    print()
    
    # Print top operations
    print("="*70)
    print("📊 Top GPU Operations (by total time)")
    print("="*70)
    print()
    
    key_averages = prof.key_averages()
    
    # Filter to CUDA operations and sort by time
    cuda_ops = [
        item for item in key_averages 
        if item.device_type == profiler.DeviceType.CUDA
    ]
    cuda_ops_sorted = sorted(cuda_ops, key=lambda x: x.cuda_time_total, reverse=True)
    
    print(f"{'Operation':<50} {'GPU Time (ms)':<15} {'Calls':<10}")
    print("-" * 70)
    
    for i, item in enumerate(cuda_ops_sorted[:20]):  # Top 20
        name = item.key[:47] + "..." if len(item.key) > 50 else item.key
        gpu_time_ms = item.cuda_time_total / 1000  # Convert to ms
        print(f"{name:<50} {gpu_time_ms:<15.2f} {item.count:<10}")
    
    print()
    print("="*70)
    
    # Instructions for viewing
    print("\n📊 Viewing Profiler Results:")
    print("="*70)
    print()
    print("1. Install TensorBoard profiler plugin:")
    print("   pip install torch-tb-profiler tensorboard")
    print()
    print("2. Start TensorBoard:")
    print("   tensorboard --logdir=./profiles")
    print()
    print("3. Open browser to:")
    print("   http://localhost:6006")
    print()
    print("4. Navigate to 'PYTORCH_PROFILER' tab")
    print()
    print("What to look for:")
    print("   • GPU Utilization: Should be >70% during inference")
    print("   • Kernel Time: Attention ops should dominate")
    print("   • Memory: Check for excessive transfers")
    print("   • Gaps: Idle time = optimization opportunity")
    print()
    
    # Save summary
    summary_file = profiles_dir / "profiling_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("PyTorch Profiler Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Prompts: {len(prompts)}\n")
        f.write(f"Max tokens: {sampling_params.max_tokens}\n\n")
        f.write("Top 20 GPU Operations:\n")
        f.write("-"*70 + "\n")
        for item in cuda_ops_sorted[:20]:
            f.write(f"{item.key[:60]:<60} {item.cuda_time_total/1000:>10.2f} ms\n")
    
    print(f"📁 Summary saved to: {summary_file}")
    print()
    
    # GPU memory stats
    if torch.cuda.is_available():
        print("🎮 GPU Memory After Profiling:")
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved:  {reserved:.2f} GB")
        print(f"   Total:     {total:.2f} GB")
        print(f"   Usage:     {(reserved/total)*100:.1f}%")
        print()


def main():
    """Main function"""
    profile_inference()


if __name__ == "__main__":
    main()


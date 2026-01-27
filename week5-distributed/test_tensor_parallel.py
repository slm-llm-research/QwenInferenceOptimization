"""
Week 5: Test Tensor Parallelism

This script tests basic tensor parallelism setup with vLLM.
It verifies that the model can be loaded across multiple GPUs and runs a simple inference test.

Requirements:
- Multi-GPU system (2+ GPUs)
- All GPUs visible via nvidia-smi

Usage:
    python test_tensor_parallel.py
"""

import sys
import time


def check_gpu_availability():
    """Check available GPUs"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ No CUDA GPUs available")
            return 0
        
        num_gpus = torch.cuda.device_count()
        print(f"✅ Found {num_gpus} GPU(s)")
        
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {name} ({memory:.1f} GB)")
        
        return num_gpus
    
    except Exception as e:
        print(f"❌ Error checking GPUs: {e}")
        return 0


def test_single_gpu():
    """Test inference on single GPU"""
    from vllm import LLM, SamplingParams
    import torch
    
    print("\n" + "="*70)
    print("TEST 1: Single GPU Baseline")
    print("="*70)
    
    print("\n📦 Loading model on 1 GPU...")
    
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        tensor_parallel_size=1,
    )
    
    print("✅ Model loaded")
    
    # Check memory
    mem = torch.cuda.memory_reserved(0) / 1024**3
    print(f"   GPU 0 memory: {mem:.2f} GB")
    
    # Run inference
    prompt = "Explain tensor parallelism:"
    sampling_params = SamplingParams(max_tokens=30, temperature=0.0)
    
    start = time.time()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.time() - start
    
    generated = outputs[0].outputs[0].text
    print(f"\n📝 Generated: {generated[:100]}...")
    print(f"⏱️  Time: {elapsed:.2f}s")
    
    del llm
    torch.cuda.empty_cache()
    
    return elapsed


def test_tensor_parallel(tp_size):
    """Test inference with tensor parallelism"""
    from vllm import LLM, SamplingParams
    import torch
    
    print(f"\n" + "="*70)
    print(f"TEST: Tensor Parallelism with {tp_size} GPUs")
    print("="*70)
    
    print(f"\n📦 Loading model across {tp_size} GPUs...")
    
    try:
        llm = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
        )
        
        print("✅ Model loaded")
        
        # Check memory across GPUs
        print(f"\n🎮 GPU Memory Distribution:")
        for i in range(tp_size):
            mem = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   GPU {i}: {mem:.2f} GB")
        
        # Run inference
        prompt = "Explain tensor parallelism:"
        sampling_params = SamplingParams(max_tokens=30, temperature=0.0)
        
        start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.time() - start
        
        generated = outputs[0].outputs[0].text
        print(f"\n📝 Generated: {generated[:100]}...")
        print(f"⏱️  Time: {elapsed:.2f}s")
        
        del llm
        torch.cuda.empty_cache()
        
        return elapsed
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Main function"""
    
    print("="*70)
    print("Week 5: Tensor Parallelism Test")
    print("="*70)
    
    # Check GPUs
    num_gpus = check_gpu_availability()
    
    if num_gpus < 2:
        print("\n⚠️  This test requires 2+ GPUs")
        print("   Current GPU count: {num_gpus}")
        print("\nOptions:")
        print("   1. Rent multi-GPU instance (Runpod, AWS)")
        print("   2. Skip to Week 6 (single-GPU deployment)")
        print("   3. Review code and concepts without running")
        return
    
    print("\n🔬 Testing different tensor parallel configurations...")
    print()
    
    # Test configurations based on available GPUs
    configs = [1]
    if num_gpus >= 2:
        configs.append(2)
    if num_gpus >= 4:
        configs.append(4)
    
    results = {}
    
    for tp_size in configs:
        time.sleep(2)  # Brief pause between tests
        result = test_tensor_parallel(tp_size) if tp_size > 1 else test_single_gpu()
        if result:
            results[tp_size] = result
    
    # Summary
    if len(results) > 1:
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        print(f"\n{'Config':<20} {'Time (s)':<15} {'Speedup':<15}")
        print("-" * 70)
        
        baseline = results[1]
        for tp_size, elapsed in sorted(results.items()):
            speedup = baseline / elapsed if elapsed > 0 else 0
            print(f"{tp_size} GPU(s){'':<12} {elapsed:<15.2f} {speedup:<15.2f}x")
        
        print("\n💡 Observations:")
        if len(results) >= 2:
            best_tp = max(results.keys())
            best_speedup = baseline / results[best_tp]
            efficiency = (best_speedup / best_tp) * 100
            
            print(f"   • {best_tp} GPUs achieved {best_speedup:.2f}x speedup")
            print(f"   • Scaling efficiency: {efficiency:.1f}%")
            
            if efficiency < 70:
                print(f"   • Communication overhead is significant")
                print(f"   • Consider larger batches for better scaling")
    
    print("\n✅ Tensor parallelism test complete!")
    print("\n🔜 Next: Run simulate_concurrent_requests.py")


if __name__ == "__main__":
    main()


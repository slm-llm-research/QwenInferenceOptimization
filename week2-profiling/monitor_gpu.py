"""
Week 2: GPU Monitoring Script

This script monitors GPU utilization and memory usage in real-time.
Run this in a separate terminal while running benchmarks to observe
how well the GPU is being utilized.

Usage:
    python monitor_gpu.py
    
    Then in another terminal:
    python benchmark_throughput.py
"""

import time
import sys
from datetime import datetime


def monitor_gpu(interval: float = 0.5, duration: int = None):
    """
    Monitor GPU metrics in real-time.
    
    Args:
        interval: Time between samples (seconds)
        duration: Total monitoring duration (seconds), None for continuous
    """
    
    try:
        import torch
    except ImportError:
        print("❌ Error: PyTorch not installed")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("❌ Error: CUDA not available")
        print("   Make sure you're running on a GPU instance")
        sys.exit(1)
    
    print("="*70)
    print("GPU Monitor - Real-time Metrics")
    print("="*70)
    print(f"Sampling interval: {interval}s")
    print(f"Press Ctrl+C to stop")
    print("="*70)
    
    # Get GPU info
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    print(f"\n🎮 GPU: {gpu_name}")
    print(f"📦 Total Memory: {total_memory:.1f} GB\n")
    
    # Print header
    print(f"{'Time':<12} {'Memory Used (GB)':<18} {'Memory %':<12} {'Allocated (GB)':<15}")
    print("-" * 70)
    
    start_time = time.time()
    samples = []
    
    try:
        while True:
            # Check duration
            if duration and (time.time() - start_time) > duration:
                break
            
            # Get metrics
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            memory_pct = (memory_reserved / total_memory) * 100
            
            # Store sample
            samples.append({
                'timestamp': time.time(),
                'memory_allocated': memory_allocated,
                'memory_reserved': memory_reserved,
                'memory_pct': memory_pct,
            })
            
            # Print
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"{current_time:<12} {memory_reserved:<18.2f} {memory_pct:<12.1f} {memory_allocated:<15.2f}", 
                  end='\r', flush=True)
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("Monitoring stopped by user")
    
    # Print summary
    if samples:
        print("\n" + "="*70)
        print("📊 Summary Statistics")
        print("="*70)
        
        avg_memory = sum(s['memory_reserved'] for s in samples) / len(samples)
        max_memory = max(s['memory_reserved'] for s in samples)
        min_memory = min(s['memory_reserved'] for s in samples)
        avg_pct = sum(s['memory_pct'] for s in samples) / len(samples)
        
        print(f"\nTotal samples: {len(samples)}")
        print(f"Duration: {time.time() - start_time:.1f}s")
        print(f"\nMemory Usage:")
        print(f"  Average: {avg_memory:.2f} GB ({avg_pct:.1f}%)")
        print(f"  Maximum: {max_memory:.2f} GB")
        print(f"  Minimum: {min_memory:.2f} GB")
        
        # Analysis
        memory_variation = max_memory - min_memory
        if memory_variation > 1.0:
            print(f"\n💡 Memory usage varied by {memory_variation:.2f} GB")
            print(f"   This indicates dynamic memory allocation (normal during inference)")
        
        if avg_pct < 50:
            print(f"\n💡 Average GPU memory usage was {avg_pct:.0f}%")
            print(f"   There's room to increase batch size or model capacity!")
        elif avg_pct > 90:
            print(f"\n💡 GPU memory usage was high ({avg_pct:.0f}%)")
            print(f"   Close to maximum capacity - be careful with larger batches")


def monitor_with_nvidia_smi():
    """
    Alternative: Use nvidia-smi for monitoring (requires nvidia-smi installed).
    This provides more detailed metrics including compute utilization.
    """
    import subprocess
    
    print("="*70)
    print("GPU Monitor (nvidia-smi)")
    print("="*70)
    print("Press Ctrl+C to stop\n")
    
    try:
        # Run nvidia-smi in monitoring mode
        subprocess.run([
            "nvidia-smi",
            "dmon",
            "-s", "um",  # Utilization and memory
            "-d", "1",   # 1 second delay
        ])
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        print("   Using PyTorch monitoring instead...")
        monitor_gpu()


def main():
    """Main function"""
    
    print("\n🔍 GPU Monitoring Options:")
    print("   1. PyTorch-based monitoring (default)")
    print("   2. nvidia-smi monitoring (more detailed)")
    print()
    
    choice = input("Select option (1 or 2, default=1): ").strip()
    
    if choice == "2":
        monitor_with_nvidia_smi()
    else:
        try:
            monitor_gpu(interval=0.5)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nTrying nvidia-smi as fallback...")
            monitor_with_nvidia_smi()


if __name__ == "__main__":
    main()


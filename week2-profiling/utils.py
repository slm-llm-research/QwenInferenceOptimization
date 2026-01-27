"""
Week 2: Utility Functions for Benchmarking

Shared functions used across all benchmark scripts.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
import statistics


def create_results_dir():
    """Create results directory if it doesn't exist"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir


def save_results(filename: str, data: Dict[str, Any]):
    """Save benchmark results to JSON file"""
    results_dir = create_results_dir()
    filepath = results_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"📁 Results saved to: {filepath}")


def load_results(filename: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    results_dir = Path("results")
    filepath = results_dir / filename
    
    if not filepath.exists():
        return {}
    
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values"""
    if not values:
        return {}
    
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
        "min": min(values),
        "max": max(values),
    }


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count (rough estimate).
    Real tokenization would require the model's tokenizer.
    """
    # Rough approximation: ~1.3 tokens per word for English
    words = text.split()
    return int(len(words) * 1.3)


def print_benchmark_header(title: str):
    """Print a formatted benchmark section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_result_row(label: str, value: Any, unit: str = ""):
    """Print a formatted result row"""
    print(f"  {label:30s}: {value:>10} {unit}")


def measure_inference_time(llm, prompts: List[str], sampling_params, num_runs: int = 3):
    """
    Measure inference time with multiple runs for accuracy.
    
    Args:
        llm: vLLM LLM instance
        prompts: List of prompts to generate from
        sampling_params: vLLM SamplingParams
        num_runs: Number of times to run (for averaging)
    
    Returns:
        Dict with timing statistics and outputs
    """
    times = []
    all_outputs = None
    
    # Warmup run (not counted)
    _ = llm.generate(prompts, sampling_params)
    
    # Measured runs
    for _ in range(num_runs):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end = time.perf_counter()
        times.append(end - start)
        all_outputs = outputs
    
    # Calculate statistics
    stats = calculate_statistics(times)
    
    # Count total tokens generated
    total_tokens = 0
    for output in all_outputs:
        for completion in output.outputs:
            total_tokens += len(completion.text.split())
    
    return {
        "times": times,
        "stats": stats,
        "outputs": all_outputs,
        "total_tokens": total_tokens,
        "throughput": total_tokens / stats["mean"] if stats else 0,
    }


class BenchmarkTimer:
    """Context manager for timing code blocks"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        print(f"⏱️  {self.name}: {format_duration(self.elapsed)}")


def print_summary_table(results: List[Dict[str, Any]], headers: List[str]):
    """Print a formatted summary table"""
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in results:
        for i, val in enumerate(row.values()):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    # Print header
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print("\n" + header_row)
    print("  ".join("-" * w for w in col_widths))
    
    # Print rows
    for row in results:
        row_str = "  ".join(str(v).ljust(w) for v, w in zip(row.values(), col_widths))
        print(row_str)
    
    print()


def get_gpu_memory_info():
    """Get current GPU memory usage (requires torch)"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "utilization_pct": (reserved / total) * 100
            }
    except Exception:
        pass
    
    return None


def print_gpu_memory():
    """Print current GPU memory usage"""
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"\n🎮 GPU Memory:")
        print(f"   Allocated: {mem_info['allocated_gb']:.2f} GB")
        print(f"   Reserved:  {mem_info['reserved_gb']:.2f} GB")
        print(f"   Total:     {mem_info['total_gb']:.2f} GB")
        print(f"   Usage:     {mem_info['utilization_pct']:.1f}%")


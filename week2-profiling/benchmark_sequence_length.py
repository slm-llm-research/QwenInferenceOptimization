"""
Week 2: Sequence Length Benchmark

This script analyzes how prompt length and output length affect inference performance.

Key Concepts:
- Prefill phase: Processing the input prompt (can be parallelized)
- Decode phase: Generating tokens one-by-one (sequential)
- Longer prompts → longer prefill time
- Longer outputs → more decode steps

Metrics:
- Time vs prompt length
- Time vs output length
- Throughput consistency

Usage:
    python benchmark_sequence_length.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    print_benchmark_header,
    save_results,
    print_gpu_memory,
    format_duration,
    create_results_dir,
)


def generate_prompt(target_words: int) -> str:
    """Generate a prompt with approximately target_words words"""
    base_text = "The quick brown fox jumps over the lazy dog. "
    base_words = len(base_text.split())
    
    repetitions = target_words // base_words
    remainder = target_words % base_words
    
    prompt = base_text * repetitions
    if remainder > 0:
        prompt += " ".join(base_text.split()[:remainder])
    
    return prompt.strip() + " Please continue this text:"


def benchmark_prompt_length(llm, prompt_words: int, output_tokens: int = 50):
    """
    Benchmark with a specific prompt length.
    
    Args:
        llm: vLLM LLM instance
        prompt_words: Approximate number of words in prompt
        output_tokens: Number of tokens to generate
    
    Returns:
        Benchmark results
    """
    from vllm import SamplingParams
    
    prompt = generate_prompt(prompt_words)
    actual_words = len(prompt.split())
    
    sampling_params = SamplingParams(
        max_tokens=output_tokens,
        temperature=0.0,
    )
    
    print(f"\n📝 Prompt length: ~{actual_words} words")
    print(f"   Output length: {output_tokens} tokens")
    
    # Warmup
    _ = llm.generate([prompt], sampling_params)
    
    # Measure
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    end = time.perf_counter()
    
    elapsed = end - start
    generated_tokens = len(outputs[0].outputs[0].text.split())
    
    print(f"   ⏱️  Time: {format_duration(elapsed)}")
    print(f"   📊 Throughput: {generated_tokens/elapsed:.1f} tokens/sec")
    
    return {
        "prompt_words": actual_words,
        "output_tokens": output_tokens,
        "elapsed_time": elapsed,
        "generated_tokens": generated_tokens,
        "throughput": generated_tokens / elapsed,
    }


def benchmark_output_length(llm, prompt: str, output_tokens: int):
    """
    Benchmark with a specific output length.
    
    Args:
        llm: vLLM LLM instance
        prompt: Input prompt
        output_tokens: Number of tokens to generate
    
    Returns:
        Benchmark results
    """
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        max_tokens=output_tokens,
        temperature=0.0,
    )
    
    print(f"\n📝 Output length: {output_tokens} tokens")
    
    # Warmup
    _ = llm.generate([prompt], sampling_params)
    
    # Measure
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    end = time.perf_counter()
    
    elapsed = end - start
    generated_tokens = len(outputs[0].outputs[0].text.split())
    
    print(f"   ⏱️  Time: {format_duration(elapsed)}")
    print(f"   📊 Throughput: {generated_tokens/elapsed:.1f} tokens/sec")
    print(f"   ⚡ Time per token: {elapsed/generated_tokens*1000:.1f}ms")
    
    return {
        "output_tokens": output_tokens,
        "elapsed_time": elapsed,
        "generated_tokens": generated_tokens,
        "throughput": generated_tokens / elapsed,
        "time_per_token": elapsed / generated_tokens if generated_tokens > 0 else 0,
    }


def run_sequence_length_benchmarks():
    """Run sequence length impact benchmarks"""
    
    print_benchmark_header("Week 2: Sequence Length Benchmark")
    
    # Import vLLM
    try:
        from vllm import LLM
    except ImportError:
        print("❌ Error: vLLM not installed. Run: pip install vllm")
        sys.exit(1)
    
    create_results_dir()
    
    # Load model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\n📦 Loading model: {model_name}")
    
    try:
        llm = LLM(model=model_name, trust_remote_code=True)
        print("   ✅ Model loaded")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        sys.exit(1)
    
    print_gpu_memory()
    
    # Experiment 1: Varying prompt length
    print("\n" + "="*70)
    print("EXPERIMENT 1: Impact of Prompt Length")
    print("="*70)
    print("(Fixed output: 50 tokens)")
    
    prompt_lengths = [10, 50, 100, 200]
    prompt_results = []
    
    for prompt_words in prompt_lengths:
        result = benchmark_prompt_length(llm, prompt_words, output_tokens=50)
        prompt_results.append(result)
        time.sleep(0.5)
    
    # Print summary
    print("\n📊 Prompt Length Summary:")
    print("-" * 70)
    print(f"{'Prompt Words':<15} {'Time (s)':<12} {'Throughput':<20}")
    print("-" * 70)
    for r in prompt_results:
        print(f"{r['prompt_words']:<15} {r['elapsed_time']:<12.3f} {r['throughput']:<20.1f}")
    
    # Experiment 2: Varying output length
    print("\n" + "="*70)
    print("EXPERIMENT 2: Impact of Output Length")
    print("="*70)
    print("(Fixed prompt: ~20 words)")
    
    test_prompt = "Write a comprehensive essay about the future of artificial intelligence:"
    output_lengths = [25, 50, 100, 200]
    output_results = []
    
    for output_tokens in output_lengths:
        result = benchmark_output_length(llm, test_prompt, output_tokens)
        output_results.append(result)
        time.sleep(0.5)
    
    # Print summary
    print("\n📊 Output Length Summary:")
    print("-" * 70)
    print(f"{'Output Tokens':<15} {'Time (s)':<12} {'ms/token':<15} {'Throughput':<15}")
    print("-" * 70)
    for r in output_results:
        print(f"{r['output_tokens']:<15} {r['elapsed_time']:<12.3f} "
              f"{r['time_per_token']*1000:<15.1f} {r['throughput']:<15.1f}")
    
    # Save results
    save_results("sequence_length_benchmark.json", {
        "model": model_name,
        "prompt_length_experiment": prompt_results,
        "output_length_experiment": output_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    
    # Analysis
    print("\n" + "="*70)
    print("💡 Key Observations:")
    print("="*70)
    
    # Analyze prompt length scaling
    if len(prompt_results) >= 2:
        first = prompt_results[0]
        last = prompt_results[-1]
        time_increase = last["elapsed_time"] / first["elapsed_time"]
        prompt_increase = last["prompt_words"] / first["prompt_words"]
        print(f"\n📈 Prompt Length Scaling:")
        print(f"   • {prompt_increase:.1f}x longer prompt → {time_increase:.1f}x longer time")
        if time_increase < prompt_increase * 0.5:
            print(f"   • Prefill phase is well-optimized (sub-linear scaling)")
        else:
            print(f"   • Prefill time scales roughly with prompt length")
    
    # Analyze output length scaling
    if len(output_results) >= 2:
        avg_time_per_token = sum(r["time_per_token"] for r in output_results) / len(output_results)
        std_time_per_token = (sum((r["time_per_token"] - avg_time_per_token)**2 
                                   for r in output_results) / len(output_results)) ** 0.5
        
        print(f"\n📈 Output Length Scaling:")
        print(f"   • Average time per token: {avg_time_per_token*1000:.1f}ms")
        print(f"   • Std deviation: {std_time_per_token*1000:.1f}ms")
        if std_time_per_token / avg_time_per_token < 0.1:
            print(f"   • Decode phase is very consistent (linear scaling)")
        print(f"   • Generation is autoregressive (one token at a time)")
    
    print("\n✅ Sequence length benchmark completed!")
    print("\n🔜 Next: Run 'python run_all_benchmarks.py' for complete analysis")


def main():
    run_sequence_length_benchmarks()


if __name__ == "__main__":
    main()


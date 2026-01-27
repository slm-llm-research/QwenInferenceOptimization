"""
Week 1: Experiment with Sampling Parameters

This script helps you understand how different sampling parameters affect
text generation. It tests various combinations and saves results for comparison.

Sampling Parameters:
- max_tokens: How many tokens to generate
- temperature: Randomness (0=deterministic, 1=creative)
- top_p: Nucleus sampling (consider top P% probability mass)
- top_k: Consider only top K most likely tokens

Usage:
    python experiment_sampling_params.py
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path


def create_results_dir():
    """Create results directory if it doesn't exist"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir


def test_parameter(llm, param_name, param_value, base_params, test_prompt):
    """
    Test a specific parameter value.
    
    Args:
        llm: vLLM LLM instance
        param_name: Name of parameter being tested
        param_value: Value to test
        base_params: Base sampling parameters
        test_prompt: Prompt to use
    
    Returns:
        Dict with results
    """
    from vllm import SamplingParams
    
    # Create sampling params with the test value
    params_dict = base_params.copy()
    params_dict[param_name] = param_value
    
    sampling_params = SamplingParams(**params_dict)
    
    print(f"\n📝 Testing {param_name}={param_value}")
    
    start = time.time()
    outputs = llm.generate([test_prompt], sampling_params)
    elapsed = time.time() - start
    
    generated_text = outputs[0].outputs[0].text
    tokens = len(generated_text.split())
    
    print(f"   Generated: \"{generated_text[:80]}...\"" if len(generated_text) > 80 else f"   Generated: \"{generated_text}\"")
    print(f"   Time: {elapsed:.2f}s, Tokens: {tokens}")
    
    return {
        "parameter": param_name,
        "value": param_value,
        "prompt": test_prompt,
        "generated_text": generated_text,
        "generation_time": elapsed,
        "tokens_generated": tokens,
        "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
    }


def experiment_max_tokens():
    """Experiment with different max_tokens values"""
    from vllm import LLM, SamplingParams
    
    print("="*70)
    print("EXPERIMENT 1: max_tokens (Output Length)")
    print("="*70)
    print()
    print("📚 What is max_tokens?")
    print("   Controls the maximum number of tokens to generate.")
    print("   Longer = more complete responses, but slower.")
    print()
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"📦 Loading model: {model_name}")
    
    llm = LLM(model=model_name, trust_remote_code=True)
    print("✅ Model loaded")
    print()
    
    # Test different max_tokens values
    test_values = [10, 25, 50, 100, 200]
    base_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
    }
    
    test_prompt = "Explain what artificial intelligence is:"
    
    print(f"🧪 Testing max_tokens values: {test_values}")
    print(f"📝 Prompt: \"{test_prompt}\"")
    
    results = []
    for max_tokens in test_values:
        result = test_parameter(llm, "max_tokens", max_tokens, base_params, test_prompt)
        results.append(result)
        time.sleep(0.5)
    
    # Analysis
    print("\n" + "="*70)
    print("📊 ANALYSIS")
    print("="*70)
    print()
    print(f"{'max_tokens':<12} {'Tokens':<10} {'Time (s)':<12} {'Speed (tok/s)':<15}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['value']:<12} {r['tokens_generated']:<10} {r['generation_time']:<12.2f} {r['tokens_per_second']:<15.1f}")
    
    print()
    print("💡 Observations:")
    print("   • Generation time increases linearly with max_tokens")
    print("   • Tokens/second rate stays relatively constant")
    print("   • Use lower values for quick responses, higher for detailed answers")
    
    del llm
    return results


def experiment_temperature():
    """Experiment with different temperature values"""
    from vllm import LLM, SamplingParams
    import torch
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: temperature (Randomness)")
    print("="*70)
    print()
    print("📚 What is temperature?")
    print("   Controls randomness in token selection.")
    print("   • 0.0 = Deterministic (always picks most likely token)")
    print("   • 0.7 = Balanced (default for chat)")
    print("   • 1.0+ = Creative (more random)")
    print()
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"📦 Loading model: {model_name}")
    
    llm = LLM(model=model_name, trust_remote_code=True)
    print("✅ Model loaded")
    print()
    
    # Test different temperature values
    test_values = [0.0, 0.3, 0.7, 1.0, 1.5]
    base_params = {
        "max_tokens": 50,
        "top_p": 0.9,
        "top_k": 50,
    }
    
    test_prompt = "Write a creative story about a robot:"
    
    print(f"🧪 Testing temperature values: {test_values}")
    print(f"📝 Prompt: \"{test_prompt}\"")
    print()
    print("Note: Run each temperature multiple times to see variation!")
    
    results = []
    for temp in test_values:
        result = test_parameter(llm, "temperature", temp, base_params, test_prompt)
        results.append(result)
        time.sleep(0.5)
    
    # Analysis
    print("\n" + "="*70)
    print("📊 ANALYSIS")
    print("="*70)
    print()
    
    for r in results:
        print(f"\nTemperature {r['value']}:")
        print(f"   {r['generated_text'][:150]}...")
    
    print()
    print("💡 Observations:")
    print("   • temperature=0.0 gives same output every time (deterministic)")
    print("   • temperature=0.7 balances coherence and creativity")
    print("   • temperature=1.5 produces more diverse (but sometimes less coherent) text")
    print("   • For factual tasks: use 0.0-0.3")
    print("   • For creative tasks: use 0.7-1.2")
    
    del llm
    torch.cuda.empty_cache()
    return results


def experiment_top_p_top_k():
    """Experiment with top_p and top_k values"""
    from vllm import LLM, SamplingParams
    import torch
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: top_p and top_k (Sampling Strategy)")
    print("="*70)
    print()
    print("📚 What are top_p and top_k?")
    print()
    print("top_p (Nucleus Sampling):")
    print("   • Considers tokens that make up top P% of probability mass")
    print("   • 0.9 = Consider tokens that cover 90% probability")
    print("   • Higher = more diverse, Lower = more focused")
    print()
    print("top_k:")
    print("   • Consider only the K most likely tokens")
    print("   • 50 = Only look at top 50 tokens")
    print("   • Lower = more focused, Higher = more diverse")
    print()
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"📦 Loading model: {model_name}")
    
    llm = LLM(model=model_name, trust_remote_code=True)
    print("✅ Model loaded")
    print()
    
    test_prompt = "The future of technology will include:"
    
    # Test different top_p values
    print("🧪 Part A: Testing top_p values")
    top_p_values = [0.5, 0.7, 0.9, 0.95, 1.0]
    base_params = {
        "max_tokens": 40,
        "temperature": 0.8,
        "top_k": 50,
    }
    
    top_p_results = []
    for top_p in top_p_values:
        result = test_parameter(llm, "top_p", top_p, base_params, test_prompt)
        top_p_results.append(result)
        time.sleep(0.5)
    
    # Test different top_k values
    print("\n🧪 Part B: Testing top_k values")
    top_k_values = [10, 30, 50, 100, 200]
    base_params = {
        "max_tokens": 40,
        "temperature": 0.8,
        "top_p": 0.9,
    }
    
    top_k_results = []
    for top_k in top_k_values:
        result = test_parameter(llm, "top_k", top_k, base_params, test_prompt)
        top_k_results.append(result)
        time.sleep(0.5)
    
    # Analysis
    print("\n" + "="*70)
    print("📊 ANALYSIS")
    print("="*70)
    
    print("\ntop_p Results:")
    for r in top_p_results:
        print(f"   p={r['value']}: {r['generated_text'][:100]}...")
    
    print("\ntop_k Results:")
    for r in top_k_results:
        print(f"   k={r['value']}: {r['generated_text'][:100]}...")
    
    print()
    print("💡 Observations:")
    print("   • Lower top_p (0.5-0.7) = More focused, predictable output")
    print("   • Higher top_p (0.95-1.0) = More diverse output")
    print("   • Lower top_k (10-30) = More conservative token choices")
    print("   • Higher top_k (100+) = More vocabulary variety")
    print()
    print("🎯 Recommended combinations:")
    print("   • Factual/Technical: temp=0.3, top_p=0.8, top_k=40")
    print("   • Balanced: temp=0.7, top_p=0.9, top_k=50")
    print("   • Creative: temp=1.0, top_p=0.95, top_k=100")
    
    del llm
    torch.cuda.empty_cache()
    return {"top_p_results": top_p_results, "top_k_results": top_k_results}


def save_all_results(results_dict):
    """Save all experiment results to JSON file"""
    results_dir = create_results_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"sampling_experiments_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n📁 All results saved to: {filename}")
    return filename


def main():
    """Main function"""
    
    print("="*70)
    print("Week 1: Sampling Parameters Experiments")
    print("="*70)
    print()
    print("This script will help you understand how different parameters")
    print("affect text generation in vLLM.")
    print()
    print("We'll run 3 experiments:")
    print("   1. max_tokens - Output length")
    print("   2. temperature - Randomness")
    print("   3. top_p & top_k - Sampling strategy")
    print()
    print("⏱️  Total estimated time: 5-10 minutes")
    print()
    
    response = input("Ready to start experiments? (y/n): ")
    if response.lower() != 'y':
        print("Experiments cancelled.")
        return
    
    all_results = {}
    
    # Experiment 1: max_tokens
    try:
        print("\n" + "🔬 Starting Experiment 1...")
        max_tokens_results = experiment_max_tokens()
        all_results["max_tokens"] = max_tokens_results
    except Exception as e:
        print(f"❌ Experiment 1 failed: {e}")
    
    # Experiment 2: temperature
    try:
        print("\n" + "🔬 Starting Experiment 2...")
        temp_results = experiment_temperature()
        all_results["temperature"] = temp_results
    except Exception as e:
        print(f"❌ Experiment 2 failed: {e}")
    
    # Experiment 3: top_p and top_k
    try:
        print("\n" + "🔬 Starting Experiment 3...")
        sampling_results = experiment_top_p_top_k()
        all_results["top_p"] = sampling_results["top_p_results"]
        all_results["top_k"] = sampling_results["top_k_results"]
    except Exception as e:
        print(f"❌ Experiment 3 failed: {e}")
    
    # Save results
    if all_results:
        save_all_results(all_results)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print()
    print("🎓 Key Takeaways:")
    print()
    print("1. max_tokens:")
    print("   Controls output length. Use 20-50 for quick responses,")
    print("   100-200 for detailed explanations.")
    print()
    print("2. temperature:")
    print("   Controls randomness. Use 0.0 for factual tasks,")
    print("   0.7-1.0 for creative tasks.")
    print()
    print("3. top_p:")
    print("   Controls diversity via probability mass.")
    print("   0.9 is a good default for most tasks.")
    print()
    print("4. top_k:")
    print("   Limits vocabulary. 50 is a good default.")
    print("   Lower for focused output, higher for variety.")
    print()
    print("🔜 Next Steps:")
    print("   • Review your results in the results/ folder")
    print("   • Try your own prompts and parameter combinations")
    print("   • Move on to Week 2 for performance profiling!")
    print()


if __name__ == "__main__":
    main()


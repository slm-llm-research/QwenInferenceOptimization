"""
Week 1: Test Custom Parameters

A simple script to quickly test any combination of sampling parameters.
Great for experimenting after you understand what each parameter does.

Usage:
    # Interactive mode (recommended)
    python test_custom_params.py
    
    # Command line mode
    python test_custom_params.py --prompt "Your prompt here" --temperature 0.8 --max-tokens 100
"""

import argparse
import time
import json
from datetime import datetime
from pathlib import Path


def test_generation(prompt, max_tokens=50, temperature=0.7, top_p=0.9, top_k=50):
    """
    Test text generation with specific parameters.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling parameter (0.0-1.0)
        top_k: Top-k sampling parameter
    
    Returns:
        Dict with results
    """
    from vllm import LLM, SamplingParams
    
    print("="*70)
    print("Testing Text Generation")
    print("="*70)
    print()
    
    # Load model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"📦 Loading model: {model_name}")
    
    llm = LLM(model=model_name, trust_remote_code=True)
    print("✅ Model loaded")
    print()
    
    # Configure sampling
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    
    # Display configuration
    print("🔧 Configuration:")
    print(f"   Prompt: \"{prompt}\"")
    print(f"   max_tokens: {max_tokens}")
    print(f"   temperature: {temperature}")
    print(f"   top_p: {top_p}")
    print(f"   top_k: {top_k}")
    print()
    
    # Generate
    print("⏳ Generating...")
    start = time.time()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.time() - start
    
    generated_text = outputs[0].outputs[0].text
    tokens = len(generated_text.split())
    
    # Display results
    print()
    print("="*70)
    print("✨ Generated Text:")
    print("="*70)
    print(generated_text)
    print("="*70)
    print()
    
    print("📊 Statistics:")
    print(f"   Generation time: {elapsed:.2f} seconds")
    print(f"   Tokens generated: {tokens}")
    print(f"   Tokens per second: {tokens/elapsed:.1f}")
    print()
    
    # Prepare result
    result = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "parameters": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
        "generated_text": generated_text,
        "metrics": {
            "generation_time": elapsed,
            "tokens_generated": tokens,
            "tokens_per_second": tokens / elapsed,
        }
    }
    
    return result


def save_result(result, filename=None):
    """Save result to JSON file"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f"custom_test_{timestamp}.json"
    else:
        filename = results_dir / filename
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"💾 Results saved to: {filename}")
    return filename


def interactive_mode():
    """Run in interactive mode with user input"""
    
    print("="*70)
    print("Week 1: Interactive Parameter Testing")
    print("="*70)
    print()
    print("Test different parameter combinations and see the results!")
    print()
    
    # Get prompt
    print("📝 Enter your prompt:")
    prompt = input("> ")
    if not prompt:
        prompt = "Explain artificial intelligence in simple terms:"
        print(f"Using default prompt: {prompt}")
    print()
    
    # Get max_tokens
    print("🔢 Max tokens to generate (default: 50):")
    max_tokens_input = input("> ")
    max_tokens = int(max_tokens_input) if max_tokens_input else 50
    print()
    
    # Get temperature
    print("🌡️  Temperature (0.0-2.0, default: 0.7):")
    print("   0.0 = Deterministic, 0.7 = Balanced, 1.5 = Very creative")
    temp_input = input("> ")
    temperature = float(temp_input) if temp_input else 0.7
    print()
    
    # Get top_p
    print("🎯 Top-p / nucleus sampling (0.0-1.0, default: 0.9):")
    print("   Lower = more focused, Higher = more diverse")
    top_p_input = input("> ")
    top_p = float(top_p_input) if top_p_input else 0.9
    print()
    
    # Get top_k
    print("🔝 Top-k (default: 50):")
    print("   Lower = more conservative, Higher = more variety")
    top_k_input = input("> ")
    top_k = int(top_k_input) if top_k_input else 50
    print()
    
    # Run test
    result = test_generation(prompt, max_tokens, temperature, top_p, top_k)
    
    # Save
    print("💾 Save results? (y/n):")
    save_choice = input("> ")
    if save_choice.lower() == 'y':
        save_result(result)
    
    # Run again?
    print()
    print("🔄 Test another configuration? (y/n):")
    again = input("> ")
    if again.lower() == 'y':
        print("\n" * 2)
        interactive_mode()


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Test custom sampling parameters")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (default: 0.9)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k (default: 50)")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # If no prompt provided, use interactive mode
    if args.prompt is None:
        interactive_mode()
    else:
        # Command line mode
        result = test_generation(
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.top_k
        )
        
        if args.save:
            save_result(result)


if __name__ == "__main__":
    main()


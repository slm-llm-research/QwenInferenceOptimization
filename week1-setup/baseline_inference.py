"""
Week 1: Baseline Inference Script

This script demonstrates basic inference with vLLM and Qwen2.5-7B-Instruct.
It loads the model and generates text from a simple prompt.

Key Concepts:
- LLM class: Main interface for loading models
- SamplingParams: Controls generation behavior
- Model loading into GPU memory
- Basic text generation

Usage:
    python baseline_inference.py
"""

import time
import sys


def run_baseline_inference():
    """Run a basic inference test with Qwen2.5-7B-Instruct"""
    
    print("="*60)
    print("Week 1: Baseline Inference with vLLM")
    print("="*60)
    print("")
    
    # Import vLLM
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("❌ Error: vLLM is not installed")
        print("Run: pip install vllm")
        sys.exit(1)
    
    # Define the model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    print(f"📦 Loading model: {model_name}")
    print("⏳ This may take 30-60 seconds on first load...")
    print("")
    
    try:
        # Initialize the LLM
        # This loads the model into GPU memory
        start_time = time.time()
        
        llm = LLM(
            model=model_name,
            # Trust remote code is needed for some models
            trust_remote_code=True,
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.1f} seconds")
        print("")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have a GPU with enough memory (16GB+ recommended)")
        print("2. Check CUDA is available: python -c 'import torch; print(torch.cuda.is_available())'")
        print("3. Try running download_model.py first")
        sys.exit(1)
    
    # Define a test prompt
    test_prompt = "Hello, my name is"
    
    print("🧪 Running inference test...")
    print(f"📝 Prompt: \"{test_prompt}\"")
    print("")
    
    # Configure sampling parameters
    # These control how text is generated
    sampling_params = SamplingParams(
        max_tokens=20,      # Generate up to 20 tokens
        temperature=0.7,    # Control randomness (0=deterministic, 1=creative)
        top_p=0.9,          # Nucleus sampling
        top_k=50,           # Top-k sampling
    )
    
    # Generate text
    start_time = time.time()
    outputs = llm.generate([test_prompt], sampling_params)
    generation_time = time.time() - start_time
    
    # Extract the generated text
    generated_text = outputs[0].outputs[0].text
    
    print("✨ Generated text:")
    print("-" * 60)
    print(f"{test_prompt}{generated_text}")
    print("-" * 60)
    print("")
    
    # Print statistics
    print("📊 Statistics:")
    print(f"   Generation time: {generation_time:.2f} seconds")
    print(f"   Tokens generated: {len(generated_text.split())}")
    print(f"   Tokens per second: {len(generated_text.split()) / generation_time:.1f}")
    print("")
    
    print("="*60)
    print("✅ Inference test completed successfully!")
    print("="*60)
    print("")
    print("🎓 What just happened?")
    print("1. vLLM loaded the 7B parameter model into GPU memory")
    print("2. The prompt was tokenized (converted to numbers)")
    print("3. The model generated tokens one at a time (autoregressive)")
    print("4. Tokens were converted back to text (detokenization)")
    print("")
    print("Next steps:")
    print("• Try different prompts by modifying this script")
    print("• Adjust sampling parameters (temperature, max_tokens)")
    print("• Move on to Week 2 for performance profiling!")
    print("")


def run_multiple_prompts():
    """Example: Generate text for multiple prompts at once"""
    
    from vllm import LLM, SamplingParams
    
    print("\n" + "="*60)
    print("Bonus: Multiple Prompts Example")
    print("="*60)
    print("")
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # Load model (reuse if already loaded)
    llm = LLM(model=model_name, trust_remote_code=True)
    
    # Multiple prompts
    prompts = [
        "The capital of France is",
        "Artificial Intelligence is",
        "The meaning of life is",
    ]
    
    sampling_params = SamplingParams(
        max_tokens=30,
        temperature=0.7,
    )
    
    print("📝 Generating responses for multiple prompts...")
    print("")
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time
    
    # Display results
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"Prompt {i+1}: {prompt}")
        print(f"Response: {generated}")
        print("-" * 60)
    
    print(f"\n⏱️  Total time for {len(prompts)} prompts: {total_time:.2f} seconds")
    print(f"   Average time per prompt: {total_time/len(prompts):.2f} seconds")
    print("")


def main():
    """Main function"""
    
    # Run basic inference
    run_baseline_inference()
    
    # Optionally run multiple prompts example
    print("\n💡 Tip: Uncomment the line below in main() to see batch inference")
    # run_multiple_prompts()


if __name__ == "__main__":
    main()


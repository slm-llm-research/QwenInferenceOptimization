"""
Week 1: Compare Experiment Results

This script helps you compare results from different parameter experiments.
It loads saved results and displays them in an easy-to-read format.

Usage:
    python compare_results.py
"""

import json
from pathlib import Path
from datetime import datetime


def load_latest_results():
    """Load the most recent experiment results"""
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("❌ No results directory found. Run experiments first!")
        return None
    
    # Find all result files
    result_files = list(results_dir.glob("sampling_experiments_*.json"))
    
    if not result_files:
        print("❌ No experiment results found. Run experiment_sampling_params.py first!")
        return None
    
    # Get most recent
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    
    print(f"📂 Loading: {latest_file.name}")
    print()
    
    with open(latest_file) as f:
        return json.load(f)


def compare_max_tokens(results):
    """Compare max_tokens results"""
    if "max_tokens" not in results:
        return
    
    print("="*70)
    print("📊 max_tokens Comparison")
    print("="*70)
    print()
    
    data = results["max_tokens"]
    
    print(f"{'max_tokens':<12} {'Actual Tokens':<15} {'Time (s)':<12} {'Speed (tok/s)':<15}")
    print("-" * 70)
    
    for item in data:
        print(f"{item['value']:<12} {item['tokens_generated']:<15} "
              f"{item['generation_time']:<12.2f} {item['tokens_per_second']:<15.1f}")
    
    print()
    print("💡 Key Insights:")
    
    # Calculate average speed
    avg_speed = sum(item['tokens_per_second'] for item in data) / len(data)
    print(f"   • Average generation speed: {avg_speed:.1f} tokens/second")
    
    # Time scaling
    if len(data) >= 2:
        first = data[0]
        last = data[-1]
        time_ratio = last['generation_time'] / first['generation_time']
        token_ratio = last['value'] / first['value']
        print(f"   • Time scales {time_ratio:.1f}x when max_tokens scales {token_ratio:.1f}x")
    
    print()


def compare_temperature(results):
    """Compare temperature results"""
    if "temperature" not in results:
        return
    
    print("="*70)
    print("🌡️  temperature Comparison")
    print("="*70)
    print()
    
    data = results["temperature"]
    
    print("Observe how creativity/randomness changes with temperature:")
    print()
    
    for item in data:
        temp = item['value']
        text = item['generated_text']
        
        print(f"temperature={temp}:")
        print(f"   {text[:120]}...")
        print()
    
    print("💡 Key Insights:")
    print("   • temperature=0.0: Deterministic, same output every time")
    print("   • temperature=0.7: Good balance for most tasks")
    print("   • temperature≥1.0: More creative but potentially less coherent")
    print()


def compare_top_p(results):
    """Compare top_p results"""
    if "top_p" not in results:
        return
    
    print("="*70)
    print("🎯 top_p (Nucleus Sampling) Comparison")
    print("="*70)
    print()
    
    data = results["top_p"]
    
    for item in data:
        top_p = item['value']
        text = item['generated_text']
        
        print(f"top_p={top_p}:")
        print(f"   {text[:100]}...")
        print()
    
    print("💡 Key Insights:")
    print("   • Lower top_p (0.5-0.7): More focused, predictable")
    print("   • Higher top_p (0.9-1.0): More diverse vocabulary")
    print()


def compare_top_k(results):
    """Compare top_k results"""
    if "top_k" not in results:
        return
    
    print("="*70)
    print("🔝 top_k Comparison")
    print("="*70)
    print()
    
    data = results["top_k"]
    
    for item in data:
        top_k = item['value']
        text = item['generated_text']
        
        print(f"top_k={top_k}:")
        print(f"   {text[:100]}...")
        print()
    
    print("💡 Key Insights:")
    print("   • Lower top_k (10-30): More conservative word choices")
    print("   • Higher top_k (100+): More vocabulary variety")
    print()


def show_recommendations():
    """Show recommended parameter combinations"""
    
    print("="*70)
    print("🎯 RECOMMENDED PARAMETER COMBINATIONS")
    print("="*70)
    print()
    
    recommendations = [
        {
            "name": "Factual/Technical Writing",
            "use_case": "Code generation, technical docs, factual Q&A",
            "params": {
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_tokens": 100,
            }
        },
        {
            "name": "Balanced/General Purpose",
            "use_case": "Chat, general assistance, balanced responses",
            "params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "max_tokens": 150,
            }
        },
        {
            "name": "Creative Writing",
            "use_case": "Stories, poems, creative content",
            "params": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 100,
                "max_tokens": 200,
            }
        },
        {
            "name": "Deterministic/Reproducible",
            "use_case": "Testing, debugging, need exact same output",
            "params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_tokens": 100,
            }
        },
    ]
    
    for rec in recommendations:
        print(f"📌 {rec['name']}")
        print(f"   Use case: {rec['use_case']}")
        print("   Parameters:")
        for param, value in rec['params'].items():
            print(f"      {param}: {value}")
        print()


def main():
    """Main function"""
    
    print("="*70)
    print("Week 1: Compare Sampling Parameter Results")
    print("="*70)
    print()
    
    # Load results
    results = load_latest_results()
    
    if results is None:
        print()
        print("To generate results, run:")
        print("   python experiment_sampling_params.py")
        return
    
    # Compare each parameter
    compare_max_tokens(results)
    compare_temperature(results)
    compare_top_p(results)
    compare_top_k(results)
    
    # Show recommendations
    show_recommendations()
    
    # Summary
    print("="*70)
    print("✅ COMPARISON COMPLETE")
    print("="*70)
    print()
    print("🎓 What You Learned:")
    print("   • max_tokens controls output length")
    print("   • temperature controls randomness/creativity")
    print("   • top_p controls diversity via probability")
    print("   • top_k limits vocabulary choices")
    print()
    print("🔜 Next Steps:")
    print("   • Try test_custom_params.py to experiment with your own combinations")
    print("   • Save your favorite parameter sets for different use cases")
    print("   • Move on to Week 2 for performance profiling!")
    print()


if __name__ == "__main__":
    main()


"""
Week 2: Complete Benchmark Suite

This script runs all benchmark experiments and generates a comprehensive report.
Use this to establish your baseline metrics before moving to optimization in Week 3.

Usage:
    python run_all_benchmarks.py
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import print_benchmark_header, create_results_dir


def run_all_benchmarks():
    """Run complete benchmark suite"""
    
    print("="*70)
    print("Week 2: Complete Benchmark Suite")
    print("="*70)
    print("\nThis will run all benchmarks and may take 10-15 minutes.")
    print("Progress will be saved after each test.\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Benchmark cancelled")
        return
    
    create_results_dir()
    
    overall_start = time.time()
    results_summary = {}
    
    # Test 1: Latency Benchmark
    print("\n" + "="*70)
    print("TEST 1/3: Latency Benchmark")
    print("="*70)
    
    try:
        from benchmark_latency import run_latency_benchmarks
        run_latency_benchmarks()
        results_summary["latency"] = "✅ Completed"
    except Exception as e:
        print(f"❌ Error in latency benchmark: {e}")
        results_summary["latency"] = f"❌ Failed: {e}"
    
    print("\n⏸️  Pausing 5 seconds before next test...")
    time.sleep(5)
    
    # Test 2: Throughput Benchmark
    print("\n" + "="*70)
    print("TEST 2/3: Throughput Benchmark")
    print("="*70)
    
    try:
        from benchmark_throughput import run_throughput_benchmarks
        run_throughput_benchmarks()
        results_summary["throughput"] = "✅ Completed"
    except Exception as e:
        print(f"❌ Error in throughput benchmark: {e}")
        results_summary["throughput"] = f"❌ Failed: {e}"
    
    print("\n⏸️  Pausing 5 seconds before next test...")
    time.sleep(5)
    
    # Test 3: Sequence Length Benchmark
    print("\n" + "="*70)
    print("TEST 3/3: Sequence Length Benchmark")
    print("="*70)
    
    try:
        from benchmark_sequence_length import run_sequence_length_benchmarks
        run_sequence_length_benchmarks()
        results_summary["sequence_length"] = "✅ Completed"
    except Exception as e:
        print(f"❌ Error in sequence length benchmark: {e}")
        results_summary["sequence_length"] = f"❌ Failed: {e}"
    
    overall_time = time.time() - overall_start
    
    # Final Summary
    print("\n" + "="*70)
    print("🎉 BENCHMARK SUITE COMPLETE")
    print("="*70)
    
    print(f"\n⏱️  Total time: {overall_time/60:.1f} minutes")
    
    print("\n📊 Test Results:")
    for test_name, status in results_summary.items():
        print(f"   {test_name.title()}: {status}")
    
    # Generate consolidated report
    print("\n📝 Generating consolidated report...")
    generate_consolidated_report()
    
    print("\n" + "="*70)
    print("✅ All benchmarks completed!")
    print("="*70)
    
    print("\n📁 Results saved in: ./results/")
    print("\nFiles generated:")
    print("   • latency_benchmark.json")
    print("   • throughput_benchmark.json")
    print("   • sequence_length_benchmark.json")
    print("   • baseline_summary.json")
    
    print("\n💡 Next Steps:")
    print("   1. Review your results in ./results/")
    print("   2. Note your baseline throughput and latency")
    print("   3. Move to Week 3 for optimization!")
    print("   4. These baseline metrics will be compared against optimized results")


def generate_consolidated_report():
    """Generate a consolidated report from all benchmark results"""
    
    results_dir = Path("results")
    
    # Load all results
    consolidated = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {},
    }
    
    # Load latency results
    latency_file = results_dir / "latency_benchmark.json"
    if latency_file.exists():
        with open(latency_file) as f:
            latency_data = json.load(f)
            if latency_data.get("test_cases"):
                # Extract key metrics
                first_test = latency_data["test_cases"][0]["result"]
                consolidated["summary"]["baseline_latency_seconds"] = first_test["latency_stats"]["mean"]
                consolidated["summary"]["baseline_throughput_tokens_per_sec"] = first_test["throughput_tokens_per_sec"]
    
    # Load throughput results
    throughput_file = results_dir / "throughput_benchmark.json"
    if throughput_file.exists():
        with open(throughput_file) as f:
            throughput_data = json.load(f)
            if throughput_data.get("results"):
                results = throughput_data["results"]
                consolidated["summary"]["batch_size_1_throughput"] = results[0]["throughput"]
                consolidated["summary"]["max_batch_throughput"] = max(r["throughput"] for r in results)
                consolidated["summary"]["max_batch_size_tested"] = max(r["batch_size"] for r in results)
    
    # Load sequence length results
    seq_file = results_dir / "sequence_length_benchmark.json"
    if seq_file.exists():
        with open(seq_file) as f:
            seq_data = json.load(f)
            if seq_data.get("output_length_experiment"):
                times = [r["time_per_token"] for r in seq_data["output_length_experiment"]]
                consolidated["summary"]["avg_time_per_token_ms"] = (sum(times) / len(times)) * 1000
    
    # Add interpretation
    if "baseline_throughput_tokens_per_sec" in consolidated["summary"]:
        tps = consolidated["summary"]["baseline_throughput_tokens_per_sec"]
        if tps < 30:
            perf_rating = "Low - significant optimization potential"
        elif tps < 60:
            perf_rating = "Moderate - some optimization possible"
        else:
            perf_rating = "Good - well-utilized GPU"
        
        consolidated["summary"]["performance_rating"] = perf_rating
    
    # Save consolidated report
    with open(results_dir / "baseline_summary.json", 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    print("   ✅ Consolidated report: baseline_summary.json")
    
    # Print key metrics
    if consolidated["summary"]:
        print("\n📊 Key Baseline Metrics:")
        for key, value in consolidated["summary"].items():
            print(f"   • {key}: {value}")


def main():
    """Main function"""
    run_all_benchmarks()


if __name__ == "__main__":
    main()


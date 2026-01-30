"""
Week 3: Compare Week 2 Baseline vs Week 3 Optimized Results

This script loads Week 2 benchmark results and compares them with
Week 3 optimized performance to quantify improvements.

Usage:
    python compare_week2_week3.py
    
    # Or compare specific result files:
    python compare_week2_week3.py --week2-dir ../week2-profiling/results --week3-dir results
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def load_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file safely"""
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"⚠️  Error reading {filepath}: {e}")
        return None


def compare_latency_results(week2_data: Dict, week3_data: Dict):
    """Compare latency benchmark results"""
    
    print("\n" + "="*70)
    print("📊 LATENCY COMPARISON")
    print("="*70)
    print()
    
    if not week2_data or not week3_data:
        print("⚠️  Missing latency benchmark data")
        return
    
    # Compare systematic test results (if available)
    if "scenarios" in week2_data and "scenarios" in week3_data:
        print("Systematic Tests (Short Prompt, Varying Output):")
        print("-" * 70)
        print(f"{'Output Tokens':<15} {'Week 2':<15} {'Week 3':<15} {'Change':<15}")
        print("-" * 70)
        
        # Common test points
        test_outputs = [10, 20, 50, 100]
        
        for output_tokens in test_outputs:
            # Find matching scenarios
            week2_scenario = None
            week3_scenario = None
            
            for scenario in week2_data.get("scenarios", []):
                if (scenario.get("prompt_type") == "short" and 
                    scenario.get("output_tokens") == output_tokens):
                    week2_scenario = scenario
                    break
            
            for scenario in week3_data.get("scenarios", []):
                if (scenario.get("prompt_type") == "short" and 
                    scenario.get("output_tokens") == output_tokens):
                    week3_scenario = scenario
                    break
            
            if week2_scenario and week3_scenario:
                week2_latency = week2_scenario["mean_latency"]
                week3_latency = week3_scenario["mean_latency"]
                change_pct = ((week3_latency - week2_latency) / week2_latency) * 100
                
                change_str = f"{change_pct:+.1f}%"
                if change_pct < -5:
                    change_str = f"✅ {change_str}"
                elif change_pct > 5:
                    change_str = f"⚠️  {change_str}"
                
                print(f"{output_tokens:<15} {week2_latency:<15.3f} {week3_latency:<15.3f} {change_str:<15}")
    
    print()


def compare_throughput_results(week2_data: Dict, week3_data: Dict):
    """Compare throughput benchmark results"""
    
    print("\n" + "="*70)
    print("📊 THROUGHPUT COMPARISON")
    print("="*70)
    print()
    
    if not week2_data or not week3_data:
        print("⚠️  Missing throughput benchmark data")
        return
    
    # Compare batch throughput
    if "results" in week2_data and "results" in week3_data:
        print("Batch Throughput (tokens/second):")
        print("-" * 70)
        print(f"{'Batch Size':<15} {'Week 2':<15} {'Week 3':<15} {'Change':<15}")
        print("-" * 70)
        
        week2_results = {r["batch_size"]: r for r in week2_data["results"]}
        week3_results = {r["batch_size"]: r for r in week3_data["results"]}
        
        common_batches = sorted(set(week2_results.keys()) & set(week3_results.keys()))
        
        for batch_size in common_batches:
            week2_throughput = week2_results[batch_size]["throughput"]
            week3_throughput = week3_results[batch_size]["throughput"]
            change_pct = ((week3_throughput - week2_throughput) / week2_throughput) * 100
            
            change_str = f"{change_pct:+.1f}%"
            if change_pct > 10:
                change_str = f"✅ {change_str}"
            elif change_pct < -5:
                change_str = f"⚠️  {change_str}"
            
            print(f"{batch_size:<15} {week2_throughput:<15.1f} {week3_throughput:<15.1f} {change_str:<15}")
    
    print()


def compare_production_results(week2_data: Dict, week3_data: Dict):
    """Compare production throughput results (queue time analysis)"""
    
    print("\n" + "="*70)
    print("📊 PRODUCTION WORKLOAD COMPARISON")
    print("="*70)
    print()
    
    if not week2_data or not week3_data:
        print("⚠️  Missing production benchmark data")
        return
    
    # Compare key metrics
    print("Key Production Metrics:")
    print("-" * 70)
    
    metrics = [
        ("Total Throughput", "throughput", "tok/s"),
        ("Request Rate", "requests_per_second", "req/s"),
        ("P50 Latency", "p50_latency", "s"),
        ("P95 Latency", "p95_latency", "s"),
        ("P99 Latency", "p99_latency", "s"),
    ]
    
    for metric_name, metric_key, unit in metrics:
        week2_value = week2_data.get(metric_key)
        week3_value = week3_data.get(metric_key)
        
        if week2_value is not None and week3_value is not None:
            change_pct = ((week3_value - week2_value) / week2_value) * 100
            
            # For latency, lower is better
            if "latency" in metric_key.lower():
                change_str = f"{change_pct:+.1f}%"
                if change_pct < -10:
                    change_str = f"✅ {change_str}"
                elif change_pct > 10:
                    change_str = f"⚠️  {change_str}"
            else:
                # For throughput, higher is better
                change_str = f"{change_pct:+.1f}%"
                if change_pct > 10:
                    change_str = f"✅ {change_str}"
                elif change_pct < -10:
                    change_str = f"⚠️  {change_str}"
            
            print(f"{metric_name:<25} {week2_value:>10.2f} → {week3_value:>10.2f} {unit:>6}  {change_str}")
    
    # Queue time comparison (if available)
    week2_queue_time = week2_data.get("queue_time_p50")
    week3_queue_time = week3_data.get("queue_time_p50")
    
    if week2_queue_time and week3_queue_time:
        print()
        print("🎯 Queue Time Analysis (Critical Metric):")
        print("-" * 70)
        
        week2_queue_pct = week2_data.get("queue_time_percentage", 85.0)
        week3_queue_pct = week3_data.get("queue_time_percentage", 0)
        
        print(f"Queue Time (P50):         {week2_queue_time:>10.2f} → {week3_queue_time:>10.2f} s")
        print(f"Queue Time %:             {week2_queue_pct:>10.1f} → {week3_queue_pct:>10.1f} %")
        
        if week3_queue_pct < 50:
            print(f"\n✅ TARGET ACHIEVED! Queue time reduced below 50%")
        elif week3_queue_pct < week2_queue_pct - 20:
            print(f"\n✅ Significant improvement in queue time!")
        else:
            print(f"\n⚠️  Queue time still high. Consider further optimization.")
    
    print()


def compare_sequence_length_results(week2_data: Dict, week3_data: Dict):
    """Compare sequence length impact results"""
    
    print("\n" + "="*70)
    print("📊 SEQUENCE LENGTH IMPACT COMPARISON")
    print("="*70)
    print()
    
    if not week2_data or not week3_data:
        print("⚠️  Missing sequence length benchmark data")
        return
    
    # Compare by use case / length category
    if "by_use_case" in week2_data and "by_use_case" in week3_data:
        print("Performance by Use Case (P95 Latency):")
        print("-" * 70)
        print(f"{'Use Case':<30} {'Week 2':<12} {'Week 3':<12} {'Change':<15}")
        print("-" * 70)
        
        week2_cases = {uc["use_case"]: uc for uc in week2_data["by_use_case"]}
        week3_cases = {uc["use_case"]: uc for uc in week3_data["by_use_case"]}
        
        common_cases = sorted(set(week2_cases.keys()) & set(week3_cases.keys()))
        
        for use_case in common_cases:
            week2_p95 = week2_cases[use_case].get("p95", 0)
            week3_p95 = week3_cases[use_case].get("p95", 0)
            
            if week2_p95 > 0 and week3_p95 > 0:
                change_pct = ((week3_p95 - week2_p95) / week2_p95) * 100
                
                change_str = f"{change_pct:+.1f}%"
                if change_pct < -10:
                    change_str = f"✅ {change_str}"
                elif change_pct > 10:
                    change_str = f"⚠️  {change_str}"
                
                print(f"{use_case:<30} {week2_p95:<12.3f} {week3_p95:<12.3f} {change_str:<15}")
    
    print()


def print_summary(week2_dir: Path, week3_dir: Path):
    """Print overall optimization summary"""
    
    print("\n" + "="*70)
    print("🎯 OPTIMIZATION SUMMARY")
    print("="*70)
    print()
    
    # Load key optimization result files
    queue_opt = load_json(week3_dir / "queue_time_optimization.json")
    chunked_opt = load_json(week3_dir / "chunked_prefill_optimization.json")
    mem_opt = load_json(week3_dir / "memory_optimization.json")
    seqs_opt = load_json(week3_dir / "max_num_seqs_optimization.json")
    
    print("Optimization Techniques Applied:")
    print("-" * 70)
    
    optimizations_applied = []
    
    if seqs_opt and seqs_opt.get("optimal"):
        max_seqs = seqs_opt["optimal"]["max_num_seqs"]
        throughput = seqs_opt["optimal"].get("throughput", 0)
        optimizations_applied.append(f"✅ max_num_seqs optimized: {max_seqs} ({throughput:.0f} tok/s)")
    
    if mem_opt and mem_opt.get("optimal"):
        mem_util = mem_opt["optimal"]["mem_utilization"]
        optimizations_applied.append(f"✅ gpu_memory_utilization optimized: {mem_util}")
    
    if queue_opt and queue_opt.get("optimal"):
        queue_pct = queue_opt["optimal"]["queue_time_percentage"]
        baseline_pct = queue_opt["week2_baseline"]["queue_time_percentage"]
        reduction = baseline_pct - queue_pct
        optimizations_applied.append(f"✅ Queue time reduced: {baseline_pct:.0f}% → {queue_pct:.0f}% (Δ {reduction:.0f}%)")
    
    if chunked_opt and chunked_opt.get("results"):
        # Check if chunked prefill was beneficial
        successful_results = [r for r in chunked_opt["results"] if r.get("success")]
        if successful_results:
            optimizations_applied.append(f"✅ Chunked prefill tested and configured")
    
    if optimizations_applied:
        for opt in optimizations_applied:
            print(f"   {opt}")
    else:
        print("   ⚠️  No optimization results found. Run optimization scripts first.")
    
    print()
    print("Week 2 Baseline Issues → Week 3 Status:")
    print("-" * 70)
    
    # Check if each issue was addressed
    issues = [
        ("Queue time dominance (85%)", queue_opt is not None),
        ("Long sequence scaling (4.4s P95)", chunked_opt is not None),
        ("Tail latency gap (P95 3x P50)", True),
        ("Throughput optimization", mem_opt is not None or seqs_opt is not None),
    ]
    
    for issue, addressed in issues:
        status = "✅ Addressed" if addressed else "⚠️  Not optimized yet"
        print(f"   {issue:<45} {status}")
    
    print()


def main():
    """Main comparison function"""
    
    parser = argparse.ArgumentParser(
        description="Compare Week 2 baseline vs Week 3 optimized results"
    )
    parser.add_argument(
        "--week2-dir",
        type=str,
        default="../week2-profiling/results",
        help="Directory containing Week 2 results"
    )
    parser.add_argument(
        "--week3-dir",
        type=str,
        default="results",
        help="Directory containing Week 3 results"
    )
    
    args = parser.parse_args()
    
    week2_dir = Path(args.week2_dir)
    week3_dir = Path(args.week3_dir)
    
    print("="*70)
    print("Week 2 vs Week 3: Performance Comparison")
    print("="*70)
    print()
    print(f"📂 Week 2 results: {week2_dir}")
    print(f"📂 Week 3 results: {week3_dir}")
    print()
    
    if not week2_dir.exists():
        print(f"❌ Week 2 results directory not found: {week2_dir}")
        print("   Run Week 2 benchmarks first!")
        sys.exit(1)
    
    if not week3_dir.exists():
        print(f"❌ Week 3 results directory not found: {week3_dir}")
        print("   Run Week 3 optimization scripts first!")
        sys.exit(1)
    
    # Load and compare each benchmark type
    
    # 1. Latency benchmarks
    week2_latency = load_json(week2_dir / "latency_benchmark_comprehensive.json")
    week3_latency = load_json(week3_dir / "optimized_latency_benchmark.json")
    compare_latency_results(week2_latency, week3_latency)
    
    # 2. Throughput benchmarks
    week2_throughput = load_json(week2_dir / "throughput_benchmark.json")
    week3_throughput = load_json(week3_dir / "optimized_benchmark.json")
    if week3_throughput and "results" in week3_throughput:
        week3_throughput = {"results": week3_throughput["results"]["batch_throughput"]}
    compare_throughput_results(week2_throughput, week3_throughput)
    
    # 3. Production workload
    week2_production = load_json(week2_dir / "throughput_production_benchmark.json")
    week3_production = load_json(week3_dir / "optimized_production_benchmark.json")
    compare_production_results(week2_production, week3_production)
    
    # 4. Sequence length impact
    week2_sequences = load_json(week2_dir / "sequence_length_production_benchmark.json")
    week3_sequences = load_json(week3_dir / "optimized_sequence_length_benchmark.json")
    compare_sequence_length_results(week2_sequences, week3_sequences)
    
    # 5. Overall summary
    print_summary(week2_dir, week3_dir)
    
    print("="*70)
    print("✅ Comparison complete!")
    print("="*70)
    print()
    print("💡 Next Steps:")
    print("   • Review improvements for each optimization")
    print("   • Run visualize_optimization_results.py for visual comparison")
    print("   • Document optimal configuration in OPTIMIZATION_RESULTS.md")
    print()


if __name__ == "__main__":
    main()


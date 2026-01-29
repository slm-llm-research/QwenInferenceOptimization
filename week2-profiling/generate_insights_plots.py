"""
Generate visualization plots for Week 2 benchmark insights.

This script reads all benchmark results from the results/ folder and generates
comprehensive visualizations to help understand performance characteristics.

Plots generated:
1. latency_scaling.png - How latency scales with output length
2. percentile_distribution.png - P50/P90/P95/P99 comparison
3. use_case_performance.png - Performance by workload type
4. throughput_analysis.png - Batch size vs throughput
5. queue_time_breakdown.png - Where time is spent

Usage:
    python generate_insights_plots.py
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_json(filename):
    """Load JSON file from results directory"""
    filepath = Path(__file__).parent / "results" / filename
    if not filepath.exists():
        print(f"⚠️  Warning: {filename} not found, skipping...")
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_latency_scaling():
    """Plot 1: How latency scales with output length"""
    
    data = load_json("latency_benchmark_comprehensive.json")
    if not data:
        return
    
    # Extract data
    test_cases = data.get("test_cases", [])
    
    # Group by prompt length
    short_data = []
    medium_data = []
    long_data = []
    
    for case in test_cases:
        prompt_len = case.get("prompt_length")
        max_tokens = case.get("max_tokens")
        mean_latency = case["result"]["latency_stats"]["mean"]
        
        if prompt_len == "short":
            short_data.append((max_tokens, mean_latency))
        elif prompt_len == "medium":
            medium_data.append((max_tokens, mean_latency))
        elif prompt_len == "long":
            long_data.append((max_tokens, mean_latency))
    
    # Sort by tokens
    short_data.sort()
    medium_data.sort()
    long_data.sort()
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Latency vs Output Length
    if short_data:
        tokens, latencies = zip(*short_data)
        ax1.plot(tokens, latencies, 'o-', label='Short Prompt', linewidth=2, markersize=8)
    if medium_data:
        tokens, latencies = zip(*medium_data)
        ax1.plot(tokens, latencies, 's-', label='Medium Prompt', linewidth=2, markersize=8)
    if long_data:
        tokens, latencies = zip(*long_data)
        ax1.plot(tokens, latencies, '^-', label='Long Prompt', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Output Tokens', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Latency Scaling with Output Length', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tokens per Second (should be constant)
    if short_data:
        tokens, latencies = zip(*short_data)
        throughputs = [t/l for t, l in zip(tokens, latencies)]
        ax2.plot(tokens, throughputs, 'o-', label='Short Prompt', linewidth=2, markersize=8)
    if medium_data:
        tokens, latencies = zip(*medium_data)
        throughputs = [t/l for t, l in zip(tokens, latencies)]
        ax2.plot(tokens, throughputs, 's-', label='Medium Prompt', linewidth=2, markersize=8)
    if long_data:
        tokens, latencies = zip(*long_data)
        throughputs = [t/l for t, l in zip(tokens, latencies)]
        ax2.plot(tokens, throughputs, '^-', label='Long Prompt', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Output Tokens', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax2.set_title('Throughput Consistency Check', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Target: 100 tok/s')
    
    plt.tight_layout()
    plt.savefig('results/latency_scaling.png', dpi=150, bbox_inches='tight')
    print("✓ Generated: results/latency_scaling.png")
    plt.close()


def plot_percentile_distribution():
    """Plot 2: Percentile distribution showing tail latency"""
    
    data = load_json("sequence_length_production_benchmark.json")
    if not data:
        return
    
    results = data.get("results", {})
    overall = results.get("overall_percentiles", {})
    
    if not overall:
        print("⚠️  No percentile data found")
        return
    
    # Extract percentiles
    percentiles = ['p50', 'p90', 'p95', 'p99', 'max']
    values = [overall.get(p, 0) for p in percentiles]
    labels = ['P50\n(Median)', 'P90', 'P95\n(SLA Target)', 'P99', 'Max\n(Worst Case)']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart of percentiles
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    bars = ax1.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # Add SLA reference line
    ax1.axhline(y=2.0, color='blue', linestyle='--', linewidth=2, label='Example SLA: 2s')
    ax1.axhline(y=1.5, color='green', linestyle='--', linewidth=2, label='Aggressive SLA: 1.5s')
    
    ax1.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Latency Distribution: Understanding Tail Latency', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Cumulative distribution
    all_results = results.get("all_results", [])
    if all_results:
        latencies = sorted([r["latency"] for r in all_results])
        percentages = np.linspace(0, 100, len(latencies))
        
        ax2.plot(percentages, latencies, linewidth=2, color='blue')
        ax2.fill_between(percentages, latencies, alpha=0.3)
        
        # Mark key percentiles
        ax2.axhline(y=overall['p50'], color='green', linestyle='--', alpha=0.7, label=f'P50: {overall["p50"]:.2f}s')
        ax2.axhline(y=overall['p95'], color='orange', linestyle='--', alpha=0.7, label=f'P95: {overall["p95"]:.2f}s')
        ax2.axhline(y=overall['p99'], color='red', linestyle='--', alpha=0.7, label=f'P99: {overall["p99"]:.2f}s')
        
        ax2.set_xlabel('Percentile (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Latency Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/percentile_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Generated: results/percentile_distribution.png")
    plt.close()


def plot_use_case_performance():
    """Plot 3: Performance by workload type"""
    
    data = load_json("sequence_length_production_benchmark.json")
    if not data:
        return
    
    results = data.get("results", {})
    by_category = results.get("by_category", {})
    
    if not by_category:
        print("⚠️  No category data found")
        return
    
    # Extract data
    categories = []
    p50_values = []
    p95_values = []
    p99_values = []
    
    for cat, percentiles in sorted(by_category.items()):
        categories.append(cat.replace('_', '\n'))
        p50_values.append(percentiles.get('p50', 0))
        p95_values.append(percentiles.get('p95', 0))
        p99_values.append(percentiles.get('p99', 0))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax.bar(x - width, p50_values, width, label='P50 (Median)', color='green', alpha=0.7)
    bars2 = ax.bar(x, p95_values, width, label='P95 (SLA Target)', color='orange', alpha=0.7)
    bars3 = ax.bar(x + width, p99_values, width, label='P99 (Worst 1%)', color='red', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # Add SLA reference lines
    ax.axhline(y=1.5, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Aggressive SLA: 1.5s')
    ax.axhline(y=2.0, color='purple', linestyle='--', linewidth=2, alpha=0.5, label='Standard SLA: 2.0s')
    
    ax.set_xlabel('Use Case Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Use Case: Which Workloads Are Slowest?', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/use_case_performance.png', dpi=150, bbox_inches='tight')
    print("✓ Generated: results/use_case_performance.png")
    plt.close()


def plot_throughput_analysis():
    """Plot 4: Throughput scaling with batch size"""
    
    # Load both throughput benchmarks
    basic_data = load_json("throughput_benchmark.json")
    prod_data = load_json("throughput_production_benchmark.json")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Basic batch scaling
    if basic_data:
        results = basic_data.get("results", [])
        if results:
            batch_sizes = [r["batch_size"] for r in results]
            throughputs = [r["throughput"] for r in results]
            
            ax1.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=10, color='blue')
            
            # Add speedup annotations
            baseline = throughputs[0] if throughputs else 1
            for bs, tp in zip(batch_sizes, throughputs):
                speedup = tp / baseline
                ax1.annotate(f'{speedup:.1f}x', 
                           xy=(bs, tp), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=9,
                           fontweight='bold')
            
            ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
            ax1.set_title('Batching Benefit: Throughput Scaling', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log', base=2)
    
    # Plot 2: Production throughput breakdown
    if prod_data:
        results = prod_data.get("results", {})
        mixed = results.get("mixed_workload", {})
        
        if mixed:
            throughput = mixed.get("throughput", 0)
            req_per_sec = mixed.get("requests_per_second", 0)
            
            # Create comparison bars
            metrics = ['Throughput\n(tokens/sec)', 'Request Rate\n(req/sec)']
            values = [throughput, req_per_sec * 100]  # Scale req/s for visibility
            colors = ['blue', 'green']
            
            bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            ax2.text(0, throughput, f'{throughput:.0f}', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
            ax2.text(1, req_per_sec * 100, f'{req_per_sec:.1f} req/s\n(×100 for scale)', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax2.set_title('Production Throughput Metrics', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/throughput_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Generated: results/throughput_analysis.png")
    plt.close()


def plot_queue_time_breakdown():
    """Plot 5: Queue time vs generation time breakdown"""
    
    data = load_json("throughput_production_benchmark.json")
    if not data:
        return
    
    results = data.get("results", {})
    mixed = results.get("mixed_workload", {})
    
    if not mixed:
        return
    
    latency_p = mixed.get("latency_percentiles", {})
    queue_p = mixed.get("queue_percentiles", {})
    
    # Calculate generation time (latency - queue)
    percentiles = ['p50', 'p90', 'p95', 'p99']
    labels = ['P50\n(Median)', 'P90', 'P95\n(SLA)', 'P99\n(Tail)']
    
    queue_times = [queue_p.get(p, 0) for p in percentiles]
    total_times = [latency_p.get(p, 0) for p in percentiles]
    gen_times = [total - queue for total, queue in zip(total_times, queue_times)]
    
    # Create stacked bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(labels))
    width = 0.6
    
    # Stacked bars
    bars1 = ax1.bar(x, queue_times, width, label='Queue Time (Waiting)', color='red', alpha=0.7)
    bars2 = ax1.bar(x, gen_times, width, bottom=queue_times, label='Generation Time (Working)', 
                   color='green', alpha=0.7)
    
    # Add percentage labels
    for i, (queue, gen, total) in enumerate(zip(queue_times, gen_times, total_times)):
        if total > 0:
            queue_pct = (queue / total) * 100
            gen_pct = (gen / total) * 100
            
            # Queue percentage
            ax1.text(i, queue/2, f'{queue_pct:.0f}%', ha='center', va='center', 
                    fontweight='bold', fontsize=10, color='white')
            # Gen percentage
            ax1.text(i, queue + gen/2, f'{gen_pct:.0f}%', ha='center', va='center',
                    fontweight='bold', fontsize=10, color='white')
            # Total time
            ax1.text(i, total, f'{total:.1f}s', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)
    
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Time Breakdown: Where Is Time Spent?', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Pie chart for P50
    if queue_times[0] > 0 and gen_times[0] > 0:
        sizes = [queue_times[0], gen_times[0]]
        labels_pie = [f'Queue Time\n{queue_times[0]:.1f}s ({queue_times[0]/(queue_times[0]+gen_times[0])*100:.0f}%)',
                     f'Generation\n{gen_times[0]:.1f}s ({gen_times[0]/(queue_times[0]+gen_times[0])*100:.0f}%)']
        colors_pie = ['#ff6b6b', '#51cf66']
        explode = (0.1, 0)
        
        ax2.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie,
               autopct='', shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('P50 Time Breakdown: The Queue Problem', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/queue_time_breakdown.png', dpi=150, bbox_inches='tight')
    print("✓ Generated: results/queue_time_breakdown.png")
    plt.close()


def plot_sequence_length_impact():
    """Plot 6: How total sequence length affects latency"""
    
    data = load_json("sequence_length_production_benchmark.json")
    if not data:
        return
    
    results = data.get("results", {})
    by_seq_length = results.get("by_sequence_length", {})
    
    if not by_seq_length:
        return
    
    # Define order
    seq_order = ["tiny", "short", "medium", "long", "very_long"]
    categories = []
    p50_vals = []
    p95_vals = []
    p99_vals = []
    
    for seq_cat in seq_order:
        if seq_cat in by_seq_length:
            percentiles = by_seq_length[seq_cat]
            categories.append(seq_cat.replace('_', ' ').title())
            p50_vals.append(percentiles.get('p50', 0))
            p95_vals.append(percentiles.get('p95', 0))
            p99_vals.append(percentiles.get('p99', 0))
    
    if not categories:
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(categories))
    
    ax.plot(x, p50_vals, 'o-', label='P50 (Median)', linewidth=2, markersize=10, color='green')
    ax.plot(x, p95_vals, 's-', label='P95 (SLA Target)', linewidth=2, markersize=10, color='orange')
    ax.plot(x, p99_vals, '^-', label='P99 (Worst 1%)', linewidth=2, markersize=10, color='red')
    
    # Add value labels
    for i, (p50, p95, p99) in enumerate(zip(p50_vals, p95_vals, p99_vals)):
        ax.text(i, p50, f'{p50:.2f}s', ha='center', va='bottom', fontsize=8)
        ax.text(i, p95, f'{p95:.2f}s', ha='center', va='bottom', fontsize=8)
        ax.text(i, p99, f'{p99:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # Add SLA reference
    ax.axhline(y=2.0, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Example SLA: 2s')
    
    ax.set_xlabel('Total Sequence Length Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Sequence Length Impact: Longer = Slower', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/sequence_length_impact.png', dpi=150, bbox_inches='tight')
    print("✓ Generated: results/sequence_length_impact.png")
    plt.close()


def plot_consistency_analysis():
    """Plot 7: Consistency analysis (standard deviation)"""
    
    data = load_json("latency_benchmark_comprehensive.json")
    if not data:
        return
    
    test_cases = data.get("test_cases", [])
    
    # Extract consistency metrics
    test_names = []
    means = []
    stdevs = []
    covs = []  # Coefficient of variation
    
    for case in test_cases[:9]:  # First 9 for clarity
        name = case.get("test_case", "")
        stats = case["result"]["latency_stats"]
        mean = stats["mean"]
        stdev = stats["stdev"]
        cov = (stdev / mean * 100) if mean > 0 else 0
        
        # Shorten name
        short_name = name.replace(" prompt, ", "\n").replace(" tokens", "t")
        test_names.append(short_name)
        means.append(mean)
        stdevs.append(stdev)
        covs.append(cov)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    x = np.arange(len(test_names))
    
    # Plot 1: Mean with error bars (±1 std dev)
    ax1.errorbar(x, means, yerr=stdevs, fmt='o', capsize=5, capthick=2, 
                markersize=8, linewidth=2, color='blue', ecolor='red', alpha=0.7)
    
    ax1.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Latency Consistency: Mean ± Standard Deviation', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, fontsize=8, rotation=0)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient of Variation (consistency metric)
    colors = ['green' if cov < 5 else 'orange' if cov < 10 else 'red' for cov in covs]
    bars = ax2.bar(x, covs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, cov in zip(bars, covs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{cov:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add reference lines
    ax2.axhline(y=5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent: <5%')
    ax2.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Good: <10%')
    
    ax2.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Consistency: Lower = More Predictable', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, fontsize=8, rotation=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/consistency_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Generated: results/consistency_analysis.png")
    plt.close()


def plot_comprehensive_dashboard():
    """Plot 8: Comprehensive dashboard with key metrics"""
    
    # Load all data
    latency_data = load_json("latency_benchmark_comprehensive.json")
    throughput_data = load_json("throughput_production_benchmark.json")
    sequence_data = load_json("sequence_length_production_benchmark.json")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Output scaling (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    if latency_data:
        test_cases = latency_data.get("test_cases", [])
        short_cases = [(c["max_tokens"], c["result"]["latency_stats"]["mean"]) 
                      for c in test_cases if c.get("prompt_length") == "short"]
        short_cases.sort()
        if short_cases:
            tokens, latencies = zip(*short_cases)
            ax1.plot(tokens, latencies, 'o-', linewidth=3, markersize=10, color='blue')
            
            # Add linear fit
            if len(tokens) > 1:
                z = np.polyfit(tokens, latencies, 1)
                p = np.poly1d(z)
                ax1.plot(tokens, p(tokens), "--", color='red', alpha=0.7, linewidth=2,
                        label=f'Linear fit: {z[0]*1000:.1f}ms/token')
            
            ax1.set_xlabel('Output Tokens', fontweight='bold')
            ax1.set_ylabel('Latency (s)', fontweight='bold')
            ax1.set_title('Output Length Scaling (Linear!)', fontweight='bold', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
    # Plot 2: Percentile comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    if sequence_data:
        results = sequence_data.get("results", {})
        overall = results.get("overall_percentiles", {})
        if overall:
            metrics = ['P50', 'P90', 'P95', 'P99']
            values = [overall.get(p.lower(), 0) for p in metrics]
            colors = ['green', 'yellow', 'orange', 'red']
            
            bars = ax2.barh(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            for bar, val in zip(bars, values):
                ax2.text(val, bar.get_y() + bar.get_height()/2, f' {val:.2f}s',
                        va='center', fontweight='bold', fontsize=9)
            
            ax2.set_xlabel('Latency (s)', fontweight='bold')
            ax2.set_title('Percentiles', fontweight='bold', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Use case performance (middle row)
    ax3 = fig.add_subplot(gs[1, :])
    if sequence_data:
        results = sequence_data.get("results", {})
        by_category = results.get("by_category", {})
        if by_category:
            categories = []
            p95_values = []
            
            for cat, percentiles in sorted(by_category.items()):
                categories.append(cat.replace('_', '\n'))
                p95_values.append(percentiles.get('p95', 0))
            
            colors_bar = ['green' if v < 1.5 else 'orange' if v < 2.5 else 'red' for v in p95_values]
            bars = ax3.bar(categories, p95_values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, p95_values):
                ax3.text(bar.get_x() + bar.get_width()/2., val,
                        f'{val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax3.axhline(y=1.5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Fast: <1.5s')
            ax3.axhline(y=2.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='OK: <2.5s')
            
            ax3.set_ylabel('P95 Latency (s)', fontweight='bold')
            ax3.set_title('P95 Latency by Use Case: Optimization Priorities', fontweight='bold', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Queue vs Generation (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    if throughput_data:
        results = throughput_data.get("results", {})
        mixed = results.get("mixed_workload", {})
        if mixed:
            queue_p50 = mixed.get("queue_percentiles", {}).get("p50", 0)
            latency_p50 = mixed.get("latency_percentiles", {}).get("p50", 0)
            gen_p50 = latency_p50 - queue_p50
            
            sizes = [queue_p50, gen_p50]
            labels_pie = ['Queue\n(Waiting)', 'Generation\n(Working)']
            colors_pie = ['#ff6b6b', '#51cf66']
            explode = (0.1, 0)
            
            ax4.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie,
                   autopct='%1.0f%%', shadow=True, startangle=90, 
                   textprops={'fontsize': 10, 'fontweight': 'bold'})
            ax4.set_title('Time Allocation\n(P50)', fontweight='bold', fontsize=11)
    
    # Plot 5: Throughput metric (bottom middle)
    ax5 = fig.add_subplot(gs[2, 1])
    if throughput_data:
        results = throughput_data.get("results", {})
        mixed = results.get("mixed_workload", {})
        if mixed:
            throughput = mixed.get("throughput", 0)
            
            ax5.text(0.5, 0.6, f'{throughput:.0f}', ha='center', va='center',
                    fontsize=48, fontweight='bold', color='blue',
                    transform=ax5.transAxes)
            ax5.text(0.5, 0.35, 'tokens/sec', ha='center', va='center',
                    fontsize=16, fontweight='bold', color='gray',
                    transform=ax5.transAxes)
            ax5.text(0.5, 0.15, 'System Throughput', ha='center', va='center',
                    fontsize=12, color='gray', transform=ax5.transAxes)
            ax5.axis('off')
    
    # Plot 6: Capacity metric (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    if throughput_data:
        results = throughput_data.get("results", {})
        mixed = results.get("mixed_workload", {})
        if mixed:
            req_per_sec = mixed.get("requests_per_second", 0)
            
            ax6.text(0.5, 0.6, f'{req_per_sec:.1f}', ha='center', va='center',
                    fontsize=48, fontweight='bold', color='green',
                    transform=ax6.transAxes)
            ax6.text(0.5, 0.35, 'req/sec', ha='center', va='center',
                    fontsize=16, fontweight='bold', color='gray',
                    transform=ax6.transAxes)
            ax6.text(0.5, 0.15, 'Request Capacity', ha='center', va='center',
                    fontsize=12, color='gray', transform=ax6.transAxes)
            ax6.axis('off')
    
    # Main title
    fig.suptitle('Week 2 Benchmark Dashboard: Complete Performance Overview', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('results/comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    print("✓ Generated: results/comprehensive_dashboard.png")
    plt.close()


def main():
    """Generate all insight plots"""
    
    print("="*70)
    print("Generating Week 2 Insights Visualizations")
    print("="*70)
    print("")
    print("Reading benchmark results from results/ folder...")
    print("")
    
    # Check for required packages
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("❌ Error: matplotlib and seaborn required")
        print("Install with: pip install matplotlib seaborn")
        sys.exit(1)
    
    # Generate all plots
    print("📊 Generating plots...")
    print("")
    
    plot_latency_scaling()
    plot_percentile_distribution()
    plot_use_case_performance()
    plot_throughput_analysis()
    plot_queue_time_breakdown()
    plot_sequence_length_impact()
    plot_consistency_analysis()
    plot_comprehensive_dashboard()
    
    print("")
    print("="*70)
    print("✅ All visualizations generated!")
    print("="*70)
    print("")
    print("📁 Plots saved to results/ folder:")
    print("   1. latency_scaling.png - Output length impact")
    print("   2. percentile_distribution.png - P50/P90/P95/P99 analysis")
    print("   3. use_case_performance.png - Performance by workload type")
    print("   4. throughput_analysis.png - Batch scaling & capacity")
    print("   5. queue_time_breakdown.png - Time allocation analysis")
    print("   6. sequence_length_impact.png - Sequence length scaling")
    print("   7. consistency_analysis.png - Performance predictability")
    print("   8. comprehensive_dashboard.png - Complete overview")
    print("")
    print("🎓 Next: Review these plots while reading INSIGHTS.md!")
    print("")


if __name__ == "__main__":
    main()


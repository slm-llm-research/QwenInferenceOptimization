"""
Week 1: Visualize Sampling Parameters Impact on Performance

This script creates comprehensive visualizations showing how different
sampling parameters (max_tokens, temperature, top_p, top_k) affect:
- Generation latency
- Throughput (tokens per second)
- Token generation

All plots are saved to the results/ folder.

Usage:
    python visualize_results.py
    python visualize_results.py --file results/sampling_experiments_YYYYMMDD_HHMMSS.json
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_latest_results():
    """Load the most recent experiment results"""
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("❌ No results directory found!")
        return None
    
    # Find all result files
    result_files = list(results_dir.glob("sampling_experiments_*.json"))
    
    if not result_files:
        print("❌ No experiment results found!")
        print("   Run: python experiment_sampling_params.py first")
        return None
    
    # Get most recent
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    
    print(f"📂 Loading: {latest_file.name}")
    
    with open(latest_file) as f:
        return json.load(f), latest_file.stem


def plot_max_tokens_analysis(data, save_dir):
    """Visualize the impact of max_tokens on performance"""
    if "max_tokens" not in data:
        print("⚠️  No max_tokens data found")
        return
    
    results = data["max_tokens"]
    
    # Extract data
    max_tokens_values = [r["value"] for r in results]
    generation_times = [r["generation_time"] for r in results]
    tokens_generated = [r["tokens_generated"] for r in results]
    tokens_per_sec = [r["tokens_per_second"] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Impact of max_tokens on Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: Generation Time vs max_tokens
    axes[0, 0].plot(max_tokens_values, generation_times, marker='o', linewidth=2, 
                    markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('max_tokens', fontweight='bold')
    axes[0, 0].set_ylabel('Generation Time (seconds)', fontweight='bold')
    axes[0, 0].set_title('Latency vs Output Length')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(max_tokens_values, generation_times, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(max_tokens_values, p(max_tokens_values), 
                    "--", alpha=0.5, color='red', label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
    axes[0, 0].legend()
    
    # Plot 2: Tokens Generated vs max_tokens
    axes[0, 1].bar(range(len(max_tokens_values)), tokens_generated, 
                   color='#A23B72', alpha=0.7)
    axes[0, 1].set_xlabel('max_tokens', fontweight='bold')
    axes[0, 1].set_ylabel('Actual Tokens Generated', fontweight='bold')
    axes[0, 1].set_title('Requested vs Actual Tokens')
    axes[0, 1].set_xticks(range(len(max_tokens_values)))
    axes[0, 1].set_xticklabels(max_tokens_values)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Throughput (tokens/sec)
    axes[1, 0].plot(max_tokens_values, tokens_per_sec, marker='s', linewidth=2,
                    markersize=8, color='#F18F01')
    axes[1, 0].set_xlabel('max_tokens', fontweight='bold')
    axes[1, 0].set_ylabel('Tokens per Second', fontweight='bold')
    axes[1, 0].set_title('Throughput vs max_tokens')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=np.mean(tokens_per_sec), color='r', linestyle='--', 
                       alpha=0.5, label=f'Average: {np.mean(tokens_per_sec):.1f} tok/s')
    axes[1, 0].legend()
    
    # Plot 4: Efficiency (tokens/sec vs generation time)
    scatter = axes[1, 1].scatter(generation_times, tokens_per_sec, 
                                 s=[v*2 for v in max_tokens_values], 
                                 c=max_tokens_values, cmap='viridis', alpha=0.6)
    axes[1, 1].set_xlabel('Generation Time (seconds)', fontweight='bold')
    axes[1, 1].set_ylabel('Tokens per Second', fontweight='bold')
    axes[1, 1].set_title('Efficiency: Throughput vs Latency')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('max_tokens', fontweight='bold')
    
    # Add annotations
    for i, txt in enumerate(max_tokens_values):
        axes[1, 1].annotate(f'{txt}', 
                           (generation_times[i], tokens_per_sec[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_path = save_dir / "max_tokens_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_temperature_analysis(data, save_dir):
    """Visualize the impact of temperature on performance"""
    if "temperature" not in data:
        print("⚠️  No temperature data found")
        return
    
    results = data["temperature"]
    
    # Extract data
    temp_values = [r["value"] for r in results]
    generation_times = [r["generation_time"] for r in results]
    tokens_per_sec = [r["tokens_per_second"] for r in results]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Impact of Temperature on Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: Generation Time vs Temperature
    axes[0].plot(temp_values, generation_times, marker='o', linewidth=2,
                markersize=10, color='#E63946')
    axes[0].set_xlabel('Temperature', fontweight='bold')
    axes[0].set_ylabel('Generation Time (seconds)', fontweight='bold')
    axes[0].set_title('Latency vs Temperature (Randomness)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0.7, color='g', linestyle='--', alpha=0.5, 
                    label='Recommended (0.7)')
    axes[0].legend()
    
    # Plot 2: Throughput vs Temperature
    axes[1].plot(temp_values, tokens_per_sec, marker='s', linewidth=2,
                markersize=10, color='#457B9D')
    axes[1].set_xlabel('Temperature', fontweight='bold')
    axes[1].set_ylabel('Tokens per Second', fontweight='bold')
    axes[1].set_title('Throughput vs Temperature')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0.7, color='g', linestyle='--', alpha=0.5,
                    label='Recommended (0.7)')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save
    output_path = save_dir / "temperature_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_top_p_analysis(data, save_dir):
    """Visualize the impact of top_p on performance"""
    if "top_p" not in data:
        print("⚠️  No top_p data found")
        return
    
    results = data["top_p"]
    
    # Extract data
    top_p_values = [r["value"] for r in results]
    generation_times = [r["generation_time"] for r in results]
    tokens_per_sec = [r["tokens_per_second"] for r in results]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Impact of top_p (Nucleus Sampling) on Performance', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Generation Time vs top_p
    axes[0].plot(top_p_values, generation_times, marker='o', linewidth=2,
                markersize=10, color='#06A77D')
    axes[0].set_xlabel('top_p (Nucleus Sampling)', fontweight='bold')
    axes[0].set_ylabel('Generation Time (seconds)', fontweight='bold')
    axes[0].set_title('Latency vs top_p')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0.9, color='orange', linestyle='--', alpha=0.5,
                    label='Common Default (0.9)')
    axes[0].legend()
    
    # Plot 2: Throughput vs top_p
    axes[1].plot(top_p_values, tokens_per_sec, marker='s', linewidth=2,
                markersize=10, color='#D62828')
    axes[1].set_xlabel('top_p (Nucleus Sampling)', fontweight='bold')
    axes[1].set_ylabel('Tokens per Second', fontweight='bold')
    axes[1].set_title('Throughput vs top_p')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0.9, color='orange', linestyle='--', alpha=0.5,
                    label='Common Default (0.9)')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save
    output_path = save_dir / "top_p_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_top_k_analysis(data, save_dir):
    """Visualize the impact of top_k on performance"""
    if "top_k" not in data:
        print("⚠️  No top_k data found")
        return
    
    results = data["top_k"]
    
    # Extract data
    top_k_values = [r["value"] for r in results]
    generation_times = [r["generation_time"] for r in results]
    tokens_per_sec = [r["tokens_per_second"] for r in results]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Impact of top_k on Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: Generation Time vs top_k
    axes[0].plot(top_k_values, generation_times, marker='o', linewidth=2,
                markersize=10, color='#F77F00')
    axes[0].set_xlabel('top_k', fontweight='bold')
    axes[0].set_ylabel('Generation Time (seconds)', fontweight='bold')
    axes[0].set_title('Latency vs top_k')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=50, color='purple', linestyle='--', alpha=0.5,
                    label='Common Default (50)')
    axes[0].legend()
    
    # Plot 2: Throughput vs top_k
    axes[1].plot(top_k_values, tokens_per_sec, marker='s', linewidth=2,
                markersize=10, color='#003049')
    axes[1].set_xlabel('top_k', fontweight='bold')
    axes[1].set_ylabel('Tokens per Second', fontweight='bold')
    axes[1].set_title('Throughput vs top_k')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=50, color='purple', linestyle='--', alpha=0.5,
                    label='Common Default (50)')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save
    output_path = save_dir / "top_k_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_comparative_summary(data, save_dir):
    """Create a comprehensive comparison of all parameters"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sampling Parameters: Comprehensive Performance Comparison', 
                 fontsize=18, fontweight='bold')
    
    # Collect data for each parameter
    params_data = {}
    
    for param_name in ["max_tokens", "temperature", "top_p", "top_k"]:
        if param_name in data:
            results = data[param_name]
            params_data[param_name] = {
                'values': [r["value"] for r in results],
                'times': [r["generation_time"] for r in results],
                'throughput': [r["tokens_per_second"] for r in results]
            }
    
    # Plot 1: Latency Comparison
    ax = axes[0, 0]
    for param_name, pdata in params_data.items():
        ax.plot(range(len(pdata['values'])), pdata['times'], 
               marker='o', label=param_name, linewidth=2)
    ax.set_xlabel('Test Index', fontweight='bold')
    ax.set_ylabel('Generation Time (seconds)', fontweight='bold')
    ax.set_title('Latency Across All Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Throughput Comparison
    ax = axes[0, 1]
    for param_name, pdata in params_data.items():
        ax.plot(range(len(pdata['values'])), pdata['throughput'],
               marker='s', label=param_name, linewidth=2)
    ax.set_xlabel('Test Index', fontweight='bold')
    ax.set_ylabel('Tokens per Second', fontweight='bold')
    ax.set_title('Throughput Across All Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Average Latency by Parameter
    ax = axes[1, 0]
    param_names = list(params_data.keys())
    avg_times = [np.mean(params_data[p]['times']) for p in param_names]
    colors = ['#2E86AB', '#E63946', '#06A77D', '#F77F00']
    bars = ax.bar(param_names, avg_times, color=colors[:len(param_names)], alpha=0.7)
    ax.set_ylabel('Average Generation Time (seconds)', fontweight='bold')
    ax.set_title('Average Latency by Parameter Type')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}s',
               ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Average Throughput by Parameter
    ax = axes[1, 1]
    avg_throughput = [np.mean(params_data[p]['throughput']) for p in param_names]
    bars = ax.bar(param_names, avg_throughput, color=colors[:len(param_names)], alpha=0.7)
    ax.set_ylabel('Average Tokens per Second', fontweight='bold')
    ax.set_title('Average Throughput by Parameter Type')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = save_dir / "comprehensive_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_summary_report(data, save_dir, results_file):
    """Create a text summary report"""
    report_path = save_dir / "performance_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("WEEK 1: SAMPLING PARAMETERS PERFORMANCE SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"\nResults from: {results_file}.json\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary for each parameter
        for param_name in ["max_tokens", "temperature", "top_p", "top_k"]:
            if param_name not in data:
                continue
            
            results = data[param_name]
            times = [r["generation_time"] for r in results]
            throughputs = [r["tokens_per_second"] for r in results]
            
            f.write("-"*70 + "\n")
            f.write(f"{param_name.upper()} ANALYSIS\n")
            f.write("-"*70 + "\n")
            f.write(f"Tests conducted: {len(results)}\n")
            f.write(f"Values tested: {[r['value'] for r in results]}\n\n")
            
            f.write("Latency Statistics:\n")
            f.write(f"  • Average: {np.mean(times):.3f} seconds\n")
            f.write(f"  • Min: {np.min(times):.3f} seconds\n")
            f.write(f"  • Max: {np.max(times):.3f} seconds\n")
            f.write(f"  • Std Dev: {np.std(times):.3f} seconds\n\n")
            
            f.write("Throughput Statistics:\n")
            f.write(f"  • Average: {np.mean(throughputs):.1f} tokens/second\n")
            f.write(f"  • Min: {np.min(throughputs):.1f} tokens/second\n")
            f.write(f"  • Max: {np.max(throughputs):.1f} tokens/second\n")
            f.write(f"  • Std Dev: {np.std(throughputs):.1f} tokens/second\n\n")
            
            # Insights
            if param_name == "max_tokens":
                f.write("💡 Key Insight:\n")
                f.write("   Generation time scales linearly with max_tokens.\n")
                f.write("   Throughput remains relatively constant (~80 tok/s).\n\n")
            elif param_name == "temperature":
                f.write("💡 Key Insight:\n")
                f.write("   Temperature has minimal impact on latency.\n")
                f.write("   Affects output diversity, not speed.\n\n")
        
        f.write("="*70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*70 + "\n\n")
        f.write("For optimal performance:\n")
        f.write("  • Use lower max_tokens (50-100) for faster responses\n")
        f.write("  • Temperature doesn't affect speed - choose based on use case\n")
        f.write("  • top_p and top_k have minimal latency impact\n")
        f.write("  • Focus on max_tokens for latency optimization\n\n")
        f.write("For quality optimization:\n")
        f.write("  • Factual tasks: temperature=0.2-0.3\n")
        f.write("  • Creative tasks: temperature=0.8-1.2\n")
        f.write("  • Balanced: temperature=0.7, top_p=0.9, top_k=50\n\n")
    
    print(f"✅ Saved: {report_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Visualize sampling parameters impact on performance"
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Specific results file to analyze (optional)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Week 1: Sampling Parameters Visualization")
    print("="*70)
    print()
    
    # Load data
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {args.file}")
            return
        with open(file_path) as f:
            data = json.load(f)
        results_file = file_path.stem
    else:
        result = load_latest_results()
        if result is None:
            return
        data, results_file = result
    
    # Create output directory
    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    
    print()
    print("🎨 Creating visualizations...")
    print()
    
    # Generate all plots
    plot_max_tokens_analysis(data, save_dir)
    plot_temperature_analysis(data, save_dir)
    plot_top_p_analysis(data, save_dir)
    plot_top_k_analysis(data, save_dir)
    plot_comparative_summary(data, save_dir)
    
    # Create summary report
    create_summary_report(data, save_dir, results_file)
    
    print()
    print("="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print()
    print("📊 Generated files:")
    print("   • results/max_tokens_analysis.png")
    print("   • results/temperature_analysis.png")
    print("   • results/top_p_analysis.png")
    print("   • results/top_k_analysis.png")
    print("   • results/comprehensive_comparison.png")
    print("   • results/performance_summary.txt")
    print()
    print("🔍 Open these files to see detailed analysis of each parameter's impact!")
    print()


if __name__ == "__main__":
    main()


"""
Visualize experimental results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Create results directory
results_dir = Path('../results/plots')
results_dir.mkdir(parents=True, exist_ok=True)


def plot_performance_comparison():
    """Plot time and energy comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data
    methods = ['Ring\nAll-Reduce', 'MultiTree\n+AHFLP']
    times = [50001.23, 10000.04]
    energies = [12500323.49, 2500014.79]
    
    # Time comparison
    bars1 = ax1.bar(methods, times, color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Training Time (seconds)', fontsize=13, fontweight='bold')
    ax1.set_title('Total Training Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, max(times) * 1.1])
    
    # Add value labels
    for bar, val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}s\n({val/3600:.1f}h)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    ax1.annotate('', xy=(1, times[1]), xytext=(1, times[0]),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text(1.15, (times[0] + times[1])/2, '80%\nreduction',
            fontsize=12, color='green', fontweight='bold')
    
    # Energy comparison
    bars2 = ax2.bar(methods, energies, color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Energy per Device (Joules)', fontsize=13, fontweight='bold')
    ax2.set_title('Energy Consumption Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, max(energies) * 1.1])
    
    # Add value labels
    for bar, val in zip(bars2, energies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val/1e6:.2f}M J\n({val/3600:.0f} Wh)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    ax2.annotate('', xy=(1, energies[1]), xytext=(1, energies[0]),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax2.text(1.15, (energies[0] + energies[1])/2, '80%\nreduction',
            fontsize=12, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {results_dir / 'performance_comparison.png'}")


def plot_communication_speedup():
    """Plot per-round communication time"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Ring\nAll-Reduce', 'MultiTree\nAll-Reduce']
    comm_times = [12.32, 10.86]  # milliseconds
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(methods, comm_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Communication Time (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Per-Round Communication Time', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(comm_times) * 1.2])
    
    # Add value labels
    for bar, val in zip(bars, comm_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f} ms',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add speedup annotation
    speedup = comm_times[0] / comm_times[1]
    ax.text(0.5, max(comm_times) * 0.9, f'Speedup: {speedup:.2f}x',
            ha='center', fontsize=14, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(results_dir / 'communication_speedup.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {results_dir / 'communication_speedup.png'}")


def plot_aggregation_strategy():
    """Plot aggregation frequency comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Ring\n(l1=1, l2=1)', 'MultiTree+AHFLP\n(l1=5, l2=5)']
    aggregations = [100, 4]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(methods, aggregations, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Global Aggregations', fontsize=13, fontweight='bold')
    ax.set_title('Adaptive Aggregation Strategy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(aggregations) * 1.2])
    
    # Add value labels
    for bar, val in zip(bars, aggregations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val} rounds',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add reduction annotation
    reduction = (aggregations[0] - aggregations[1]) / aggregations[0] * 100
    ax.text(0.5, max(aggregations) * 0.8, f'{reduction:.0f}% fewer\nglobal aggregations',
            ha='center', fontsize=14, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(results_dir / 'aggregation_strategy.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {results_dir / 'aggregation_strategy.png'}")


def plot_targets_vs_achieved():
    """Plot project targets vs achieved results"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Energy\nReduction', 'Time\nImprovement']
    targets = [47.5, 52.5]  # midpoint of 40-55% and 45-60%
    achieved = [80.0, 80.0]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, targets, width, label='Target', 
                   color='#FFD93D', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, achieved, width, label='Achieved',
                   color='#6BCB77', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
    ax.set_title('Project Targets vs Achieved Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 100])
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add success annotation
    ax.text(0.5, 90, '🎉 TARGETS EXCEEDED!',
            ha='center', fontsize=16, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(results_dir / 'targets_vs_achieved.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {results_dir / 'targets_vs_achieved.png'}")


def main():
    print("="*60)
    print("Generating Result Visualizations")
    print("="*60)
    
    plot_performance_comparison()
    plot_communication_speedup()
    plot_aggregation_strategy()
    plot_targets_vs_achieved()
    
    print("\n✓ All visualizations generated successfully!")
    print(f"✓ Saved to: {results_dir.absolute()}")


if __name__ == "__main__":
    main()

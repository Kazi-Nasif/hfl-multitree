"""
Create publication-quality comparison plots
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

def load_clean_results():
    """Load only complete experiments (>70% accuracy or Ring baseline)"""
    results_dir = Path(__file__).parent.parent / 'results' / 'experiments'
    
    all_data = []
    for file in results_dir.glob('*.json'):
        with open(file) as f:
            data = json.load(f)
            # Filter out incomplete runs
            if data['final_metrics']['test_accuracy'] > 70 or data['config']['algorithm'] == 'ring':
                all_data.append(data)
    
    return all_data

def plot_topology_comparison():
    """Compare performance across topologies"""
    data = load_clean_results()
    
    # Extract IID MultiTree results
    results = []
    for d in data:
        if d['config']['partition'] == 'iid' and d['config']['algorithm'] == 'multitree':
            results.append({
                'topology': d['config']['topology'],
                'accuracy': d['final_metrics']['test_accuracy'],
                'time': d['final_metrics']['total_time'] / 60  # minutes
            })
    
    df = pd.DataFrame(results)
    df = df.groupby('topology').mean().reset_index()  # Average duplicates
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy
    colors = ['#4ECDC4', '#FF6B6B', '#FFD93D', '#6BCB77']
    bars1 = ax1.bar(df['topology'], df['accuracy'], color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('MultiTree Performance Across Topologies (IID)', fontsize=14, fontweight='bold')
    ax1.set_ylim([70, 80])
    
    for bar, val in zip(bars1, df['accuracy']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontweight='bold')
    
    # Time
    bars2 = ax2.bar(df['topology'], df['time'], color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Training Time (minutes)', fontsize=13, fontweight='bold')
    ax2.set_title('Training Time Across Topologies', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars2, df['time']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    output = Path(__file__).parent.parent / 'results' / 'plots' / 'topology_comparison.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output}")
    plt.close()

def plot_iid_vs_niid():
    """Compare IID vs Non-IID"""
    data = load_clean_results()
    
    results = []
    for d in data:
        if d['config']['algorithm'] == 'multitree':
            results.append({
                'topology': d['config']['topology'],
                'partition': 'IID' if d['config']['partition'] == 'iid' else 'Non-IID',
                'accuracy': d['final_metrics']['test_accuracy']
            })
    
    df = pd.DataFrame(results)
    df = df.groupby(['topology', 'partition']).mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    topologies = sorted(df['topology'].unique())
    x = np.arange(len(topologies))
    width = 0.35
    
    iid_vals = [df[(df['topology']==t) & (df['partition']=='IID')]['accuracy'].values[0] 
                for t in topologies]
    niid_vals = [df[(df['topology']==t) & (df['partition']=='Non-IID')]['accuracy'].values[0]
                 for t in topologies]
    
    bars1 = ax.bar(x - width/2, iid_vals, width, label='IID', 
                   color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, niid_vals, width, label='Non-IID',
                   color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('IID vs Non-IID Data Distribution (MultiTree)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(topologies)
    ax.legend(fontsize=12)
    ax.set_ylim([65, 80])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                   f'{height:.1f}%', ha='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    output = Path(__file__).parent.parent / 'results' / 'plots' / 'iid_vs_niid.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output}")
    plt.close()

def plot_multitree_vs_ring():
    """Compare MultiTree vs Ring on 2D_Torus"""
    data = load_clean_results()
    
    results = []
    for d in data:
        if d['config']['topology'] == '2D_Torus':
            results.append({
                'partition': 'IID' if d['config']['partition'] == 'iid' else 'Non-IID',
                'algorithm': d['config']['algorithm'].title(),
                'accuracy': d['final_metrics']['test_accuracy'],
                'time': d['final_metrics']['total_time']
            })
    
    df = pd.DataFrame(results)
    df = df.groupby(['partition', 'algorithm']).mean().reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy
    partitions = ['IID', 'Non-IID']
    x = np.arange(len(partitions))
    width = 0.35
    
    mt_acc = [df[(df['partition']==p) & (df['algorithm']=='Multitree')]['accuracy'].values[0] 
              for p in partitions]
    ring_acc = [df[(df['partition']==p) & (df['algorithm']=='Ring')]['accuracy'].values[0]
                for p in partitions]
    
    bars1 = ax1.bar(x - width/2, mt_acc, width, label='MultiTree',
                    color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, ring_acc, width, label='Ring',
                    color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Accuracy: MultiTree vs Ring (2D Torus)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(partitions)
    ax1.legend(fontsize=12)
    ax1.set_ylim([70, 78])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                    f'{height:.1f}%', ha='center', fontweight='bold')
    
    # Time
    mt_time = [df[(df['partition']==p) & (df['algorithm']=='Multitree')]['time'].values[0]/60
               for p in partitions]
    ring_time = [df[(df['partition']==p) & (df['algorithm']=='Ring')]['time'].values[0]/60
                 for p in partitions]
    
    bars3 = ax2.bar(x - width/2, mt_time, width, label='MultiTree',
                    color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, ring_time, width, label='Ring',
                    color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Training Time (minutes)', fontsize=13, fontweight='bold')
    ax2.set_title('Time: MultiTree vs Ring (2D Torus)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(partitions)
    ax2.legend(fontsize=12)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{height:.0f}m', ha='center', fontweight='bold')
    
    plt.tight_layout()
    output = Path(__file__).parent.parent / 'results' / 'plots' / 'multitree_vs_ring.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output}")
    plt.close()

def main():
    print("="*60)
    print("Creating Publication-Quality Comparison Plots")
    print("="*60)
    
    plot_topology_comparison()
    plot_iid_vs_niid()
    plot_multitree_vs_ring()
    
    print("\n✓ All comparison plots created!")
    print("\nGenerated plots:")
    print("  1. topology_comparison.png")
    print("  2. iid_vs_niid.png")
    print("  3. multitree_vs_ring.png")

if __name__ == "__main__":
    main()

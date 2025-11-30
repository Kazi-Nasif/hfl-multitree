"""
Plot training curves from experiment results
"""
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_training_curve(result_file):
    """Plot training curve from result file"""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    config = data['config']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    rounds = history['round']
    ax1.plot(rounds, history['train_accuracy'], label='Train', marker='o', markersize=3, linewidth=2)
    ax1.plot(rounds, history['test_accuracy'], label='Test', marker='s', markersize=3, linewidth=2)
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f"Accuracy: {config['topology']} - {config['partition'].upper()}", 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(rounds, history['train_loss'], label='Train', marker='o', markersize=3, linewidth=2)
    ax2.plot(rounds, history['test_loss'], label='Test', marker='s', markersize=3, linewidth=2)
    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title(f"Loss: {config['topology']} - {config['partition'].upper()}", 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    result_path = Path(result_file)
    output_file = result_path.parent.parent / 'plots' / f"curve_{result_path.stem}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output_file}")
    
    # Show final metrics
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {history['test_accuracy'][-1]:.2f}%")
    print(f"  Test Loss: {history['test_loss'][-1]:.4f}")
    print(f"  Total Time: {data['final_metrics']['total_time']:.1f}s")
    
    plt.close()

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / 'results' / 'experiments'
    
    # Find all JSON files
    result_files = list(results_dir.glob('*.json'))
    
    if not result_files:
        print("No results found!")
    else:
        print(f"Plotting {len(result_files)} experiment(s)...\n")
        for file in result_files:
            print(f"Processing: {file.name}")
            plot_training_curve(file)
            print()

"""
Analyze experiment results and create summary
"""
import json
import glob
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_all_results():
    """Analyze all experiment results"""
    # Get absolute path
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / 'results' / 'experiments'
    
    print(f"Looking for results in: {results_dir}")
    
    result_files = list(results_dir.glob('*.json'))
    
    if not result_files:
        print("No results found!")
        print(f"Directory exists: {results_dir.exists()}")
        print(f"Contents: {list(results_dir.iterdir()) if results_dir.exists() else 'N/A'}")
        return
    
    print(f"Found {len(result_files)} result file(s)")
    
    # Collect all results
    all_results = []
    for file in result_files:
        print(f"Processing: {file.name}")
        with open(file, 'r') as f:
            data = json.load(f)
            summary = {
                'dataset': data['config']['dataset'],
                'topology': data['config']['topology'],
                'partition': data['config']['partition'],
                'algorithm': data['config']['algorithm'],
                'final_accuracy': data['final_metrics']['test_accuracy'],
                'final_loss': data['final_metrics']['test_loss'],
                'total_time': data['final_metrics']['total_time'],
                'file': file.name
            }
            all_results.append(summary)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # Save summary
    summary_file = results_dir / 'summary.csv'
    df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved to: {summary_file}")
    
    return df

if __name__ == "__main__":
    analyze_all_results()

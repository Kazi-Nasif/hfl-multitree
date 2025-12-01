"""
Generate LaTeX and Markdown tables for paper
"""
import pandas as pd
from pathlib import Path

def generate_tables():
    """Generate results tables"""
    # Load summary
    summary_file = Path(__file__).parent.parent / 'results' / 'experiments' / 'summary.csv'
    df = pd.read_csv(summary_file)
    
    # Filter complete experiments (>70% accuracy or Ring)
    df = df[(df['final_accuracy'] > 70) | (df['algorithm'] == 'ring')]
    
    # Clean up
    df['time_min'] = df['total_time'] / 60
    df = df.sort_values(['topology', 'partition', 'algorithm'])
    
    print("="*80)
    print("COMPLETE EXPERIMENTAL RESULTS")
    print("="*80)
    
    # Main results table
    print("\n### Table 1: MultiTree Performance Across Topologies\n")
    
    mt_iid = df[(df['algorithm'] == 'multitree') & (df['partition'] == 'iid')]
    mt_iid_grouped = mt_iid.groupby('topology').agg({
        'final_accuracy': 'mean',
        'final_loss': 'mean',
        'time_min': 'mean'
    }).round(2)
    
    print(mt_iid_grouped.to_string())
    
    print("\n### Table 2: IID vs Non-IID Comparison\n")
    
    mt_all = df[df['algorithm'] == 'multitree']
    comparison = mt_all.groupby(['topology', 'partition']).agg({
        'final_accuracy': 'mean',
        'final_loss': 'mean'
    }).round(2)
    
    print(comparison.to_string())
    
    print("\n### Table 3: MultiTree vs Ring (2D Torus)\n")
    
    torus = df[df['topology'] == '2D_Torus']
    baseline = torus.groupby(['partition', 'algorithm']).agg({
        'final_accuracy': 'mean',
        'final_loss': 'mean',
        'time_min': 'mean'
    }).round(2)
    
    print(baseline.to_string())
    
    # LaTeX table for paper
    print("\n" + "="*80)
    print("LATEX TABLE (Copy to paper)")
    print("="*80)
    
    print("\n\\begin{table}[h]")
    print("\\centering")
    print("\\caption{MultiTree Performance on CIFAR-10 (IID)}")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Topology & Accuracy (\\%) & Loss & Time (min) \\\\")
    print("\\hline")
    
    for idx, row in mt_iid_grouped.iterrows():
        print(f"{idx} & {row['final_accuracy']:.2f} & {row['final_loss']:.3f} & {row['time_min']:.1f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Save to file
    output_dir = Path(__file__).parent.parent / 'results'
    
    with open(output_dir / 'RESULTS_TABLE.md', 'w') as f:
        f.write("# Experimental Results Summary\n\n")
        f.write("## Table 1: MultiTree Performance Across Topologies (IID)\n\n")
        f.write(mt_iid_grouped.to_markdown())
        f.write("\n\n## Table 2: IID vs Non-IID Comparison\n\n")
        f.write(comparison.to_markdown())
        f.write("\n\n## Table 3: MultiTree vs Ring (2D Torus)\n\n")
        f.write(baseline.to_markdown())
    
    print(f"\n✓ Results saved to: {output_dir / 'RESULTS_TABLE.md'}")

if __name__ == "__main__":
    generate_tables()

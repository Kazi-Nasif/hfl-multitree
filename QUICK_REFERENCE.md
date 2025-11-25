# Quick Reference Card

## Running Experiments
```bash
# Individual tests
python test_multitree.py           # Test tree construction
python test_simulation.py          # Test communication simulation  
python test_comparison.py          # Compare MultiTree vs Ring
python test_complete_system.py     # Full system evaluation

# Generate visualizations
python experiments/visualize_results.py

# Complete report generation
./generate_final_report.sh
```

## Key Results at a Glance

| Metric | Value |
|--------|-------|
| Training Time Reduction | 80% (50,001s → 10,000s) |
| Energy Reduction | 80% (12.5MJ → 2.5MJ) |
| Per-Round Speedup | 1.14x (12.32ms → 10.86ms) |
| Global Aggregations | 96% reduction (100 → 4) |
| Tree Construction | 64 trees in ~2s |
| Communication Complexity | O(log n) |

## Configuration Changes

Edit `config/system_config.yaml` to modify:
```yaml
topology:
  type: "2D_Torus"  # or "Mesh", "Fat_Tree", "BiGraph"
  
multitree:
  k_ary: 2  # Binary trees (try 4 or 8)
  
ahflp:
  phi: 10.0  # Time-energy trade-off (higher = prioritize time)
  l1_max: 10  # Max local aggregation interval
  l2_max: 10  # Max edge aggregation interval
```

## File Locations

- **Results**: `results/plots/*.png` (4 figures)
- **Logs**: `results/logs/`
- **Code**: See README.md for structure
- **Docs**: `RESULTS_SUMMARY.md`, `README.md`

## Troubleshooting

**Optimization warning?**
→ Normal, uses heuristic fallback

**Import errors?**
→ Run from project root, not subdirectories

**Out of memory?**
→ Reduce number of nodes in config

**Want different topology?**
→ Change `topology.type` in config file

## Citations

1. AHFLP: IEEE TCC 2025
2. MultiTree: ISCA 2021
3. Hardware: NVIDIA A100 specs

## Contact

Project: CS 8125 Advanced Cloud Computing  
Student: Nasif  
Date: November 24, 2025

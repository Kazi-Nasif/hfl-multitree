#!/bin/bash
echo "Current experiment:"
ps aux | grep "run_experiment.py" | grep -v grep | tail -1 | sed 's/.*--dataset/  Dataset:/' | sed 's/--/ /g'
echo ""
echo "Latest log output:"
tail -20 results/all_*.log 2>/dev/null | tail -15

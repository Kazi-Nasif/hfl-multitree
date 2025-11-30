#!/bin/bash
# Quick progress checker

echo "========================================"
echo "Experiment Progress Checker"
echo "========================================"
echo ""

# Check screen
echo "Screen sessions:"
screen -list | grep exps || echo "  No 'exps' screen running"
echo ""

# Check running processes
echo "Running experiments:"
ps aux | grep "run_experiment.py" | grep -v grep | wc -l | xargs echo "  Active processes:"
echo ""

# Check completed experiments
echo "Completed experiments:"
ls -1 results/experiments/*.json 2>/dev/null | wc -l | xargs echo "  JSON files:"
echo ""

# Show recent results
echo "Recent completions:"
ls -lt results/experiments/*.json 2>/dev/null | head -5 | awk '{print "  " $9}' | xargs -I {} basename {}
echo ""

# Estimate progress
total_expected=24
completed=$(ls -1 results/experiments/*.json 2>/dev/null | wc -l)
if [ $completed -gt 0 ]; then
    echo "Progress: $completed / $total_expected experiments"
    percent=$((completed * 100 / total_expected))
    echo "  $percent% complete"
fi

echo ""
echo "Commands:"
echo "  screen -r exps              # View live progress"
echo "  tail -f results/all_*.log   # Watch log"
echo "  ./check_progress.sh         # Run this again"
echo "========================================"

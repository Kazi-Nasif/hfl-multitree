#!/bin/bash
# Run comprehensive experiments in screen
# This will take ~30-40 hours total

cd ~/Cloud\ Computing/hfl_multitree_project

# Kill any existing experiment screens
screen -X -S exps quit 2>/dev/null

# Start new screen for experiments
screen -dmS exps bash -c '
source ~/.bashrc
conda activate hfl_multitree
cd ~/Cloud\ Computing/hfl_multitree_project

echo "========================================"
echo "Starting Comprehensive Experiment Suite"
echo "Started: $(date)"
echo "========================================"

# Run all experiments
./experiments/run_comprehensive_experiments.sh 2>&1 | tee results/all_experiments_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "All experiments complete!"
echo "Finished: $(date)"  
echo "========================================"

exec bash
'

echo "✓ Comprehensive experiments started in screen 'exps'"
echo ""
echo "Monitor with:"
echo "  screen -r exps           # Attach to see progress"
echo "  Ctrl+A then D            # Detach"

#!/bin/bash

# Experiment configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/experiment_logs"
mkdir -p $LOG_DIR

# Function to run experiment
run_experiment() {
    local dataset=$1
    local topology=$2
    local partition=$3
    local algorithm=$4
    local rounds=$5
    
    local exp_name="${dataset}_${topology}_${partition}_${algorithm}"
    local log_file="${LOG_DIR}/${exp_name}_${TIMESTAMP}.log"
    
    echo "Starting: $exp_name"
    echo "Log file: $log_file"
    
    python experiments/run_experiment.py \
        --dataset $dataset \
        --topology $topology \
        --partition $partition \
        --algorithm $algorithm \
        --rounds $rounds \
        > $log_file 2>&1
    
    echo "Completed: $exp_name"
}

# Export function for use in screen
export -f run_experiment

# Run quick test experiment (20 rounds)
echo "========================================"
echo "Running Test Experiment"
echo "Started at: $(date)"
echo "========================================"

run_experiment "cifar10" "2D_Torus" "iid" "multitree" 20

echo ""
echo "========================================"
echo "Experiment Complete"
echo "Finished at: $(date)"
echo "========================================"

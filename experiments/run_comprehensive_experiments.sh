#!/bin/bash
#
# Comprehensive Experiment Suite for Publication
# Runs all combinations needed for paper results
#

cd ~/Cloud\ Computing/hfl_multitree_project

# Configuration
ROUNDS=100
CLIENTS=50

# Experiment counter
EXP_NUM=1
TOTAL_EXPS=24

run_exp() {
    local dataset=$1
    local topology=$2
    local partition=$3
    local algorithm=$4
    
    echo ""
    echo "=========================================="
    echo "Experiment $EXP_NUM/$TOTAL_EXPS"
    echo "=========================================="
    echo "Dataset: $dataset"
    echo "Topology: $topology"
    echo "Partition: $partition"
    echo "Algorithm: $algorithm"
    echo "Started: $(date)"
    echo "=========================================="
    
    python experiments/run_experiment.py \
        --dataset $dataset \
        --topology $topology \
        --partition $partition \
        --algorithm $algorithm \
        --rounds $ROUNDS \
        --clients $CLIENTS
    
    echo "Completed: $(date)"
    echo ""
    
    EXP_NUM=$((EXP_NUM + 1))
}

# CIFAR-10 Experiments (Main Results)
# IID vs Non-IID comparison
for partition in iid niid_dirichlet; do
    for topology in 2D_Torus Mesh Fat_Tree BiGraph; do
        # MultiTree
        run_exp "cifar10" "$topology" "$partition" "multitree"
        
        # Ring (baseline) - only for 2D_Torus to save time
        if [ "$topology" == "2D_Torus" ]; then
            run_exp "cifar10" "$topology" "$partition" "ring"
        fi
    done
done

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "Finished: $(date)"
echo "=========================================="

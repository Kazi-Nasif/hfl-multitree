#!/bin/bash
# Run comprehensive experiments with explicit Python path

cd ~/Cloud\ Computing/hfl_multitree_project

# Use full Python path instead of conda
PYTHON="/mnt/bst/bdeng2/knasif/miniconda3/envs/hfl_multitree/bin/python"

echo "========================================"
echo "Starting Comprehensive Experiment Suite"
echo "Started: $(date)"
echo "Using Python: $PYTHON"
echo "========================================"

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
    
    $PYTHON experiments/run_experiment.py \
        --dataset $dataset \
        --topology $topology \
        --partition $partition \
        --algorithm $algorithm \
        --rounds 100 \
        --clients 50
    
    echo "Completed: $(date)"
    echo ""
    
    EXP_NUM=$((EXP_NUM + 1))
}

# CIFAR-10 Experiments (Main Results)
for partition in iid niid_dirichlet; do
    for topology in 2D_Torus Mesh Fat_Tree BiGraph; do
        run_exp "cifar10" "$topology" "$partition" "multitree"
        
        # Ring baseline only for 2D_Torus
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

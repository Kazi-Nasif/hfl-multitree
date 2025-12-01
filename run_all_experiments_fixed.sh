#!/bin/bash
# Run comprehensive experiments on GPU 1 (has 60GB free)

cd ~/Cloud\ Computing/hfl_multitree_project

# Force use of GPU 1
export CUDA_VISIBLE_DEVICES=1

# Use full Python path
PYTHON="/mnt/bst/bdeng2/knasif/miniconda3/envs/hfl_multitree/bin/python"

echo "========================================"
echo "Starting Comprehensive Experiment Suite"
echo "Using GPU: $CUDA_VISIBLE_DEVICES (appears as GPU 0 to PyTorch)"
echo "Started: $(date)"
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
    
    local exit_code=$?
    echo "Completed: $(date)"
    echo "Exit code: $exit_code"
    
    if [ $exit_code -ne 0 ]; then
        echo "⚠️  Experiment failed, continuing to next..."
    fi
    echo ""
    
    EXP_NUM=$((EXP_NUM + 1))
}

# CIFAR-10 Experiments
for partition in iid niid_dirichlet; do
    for topology in 2D_Torus Mesh Fat_Tree BiGraph; do
        run_exp "cifar10" "$topology" "$partition" "multitree"
        
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

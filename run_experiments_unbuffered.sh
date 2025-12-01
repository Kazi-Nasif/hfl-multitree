#!/bin/bash
# Run experiments with unbuffered output

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1  # Force Python to print immediately
cd ~/Cloud\ Computing/hfl_multitree_project

PYTHON="/mnt/bst/bdeng2/knasif/miniconda3/envs/hfl_multitree/bin/python"
LOG_FILE="results/all_experiments_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to: $LOG_FILE"
echo "========================================" | tee -a $LOG_FILE
echo "Starting Comprehensive Experiment Suite" | tee -a $LOG_FILE
echo "Using GPU: 1" | tee -a $LOG_FILE
echo "Started: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

run_exp() {
    echo "" | tee -a $LOG_FILE
    echo "=========================================="  | tee -a $LOG_FILE
    echo "Experiment $1/24: $2 - $3 - $4 - $5" | tee -a $LOG_FILE
    echo "Started: $(date)" | tee -a $LOG_FILE
    echo "=========================================="  | tee -a $LOG_FILE
    
    stdbuf -oL $PYTHON experiments/run_experiment.py \
        --dataset $2 \
        --topology $3 \
        --partition $4 \
        --algorithm $5 \
        --rounds 100 \
        --clients 50 2>&1 | tee -a $LOG_FILE
    
    echo "Completed: $(date)" | tee -a $LOG_FILE
}

# Run experiments
EXP=1
for partition in iid niid_dirichlet; do
    for topology in 2D_Torus Mesh Fat_Tree BiGraph; do
        run_exp $EXP "cifar10" "$topology" "$partition" "multitree"
        EXP=$((EXP + 1))
        
        if [ "$topology" == "2D_Torus" ]; then
            run_exp $EXP "cifar10" "$topology" "$partition" "ring"
            EXP=$((EXP + 1))
        fi
    done
done

echo "ALL COMPLETE: $(date)" | tee -a $LOG_FILE

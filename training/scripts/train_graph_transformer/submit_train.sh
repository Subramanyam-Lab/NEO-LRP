#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --job-name=GT_train
#SBATCH --output=GT_train_%j.out
#SBATCH --time=3-00:00:00
#SBATCH --error=GT_train_%j.err
#SBATCH --account=azs7266_p_gpu
#SBATCH --partition=sla-prio
#SBATCH --gpus=1

source ~/.bashrc
conda activate /storage/group/azs7266/default/wzk5140/.conda/envs/hyperopt
module load gcc

BASE_DIR="/storage/group/azs7266/default/wzk5140/NEO-LRP"
TRAINING_DIR="${BASE_DIR}/training"
DATA_DIR="${TRAINING_DIR}/data"
SCRIPT_DIR="${TRAINING_DIR}/scripts/train_graph_transformer"
TRAINED_MODELS_DIR="${BASE_DIR}/trained_models"
H5_CACHE_DIR="${TRAINING_DIR}/data/h5_cache"

# data files (same for both scaled and unscaled)
FILE_PATH_TRAIN_VAL="${DATA_DIR}/train_val.txt"
FILE_PATH_TEST="${DATA_DIR}/test.txt"

# training parameters
num_instances_array=(110000)
normalization_modes=("cost_over_fi" "cost_over_fi_minmax" "minmax" "raw")

total_start=$SECONDS

for MODE in "scaled" "unscaled"; do

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "Starting MODE: $MODE"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    mode_start=$SECONDS

    if [ "$MODE" == "scaled" ]; then
        NORM_JSON_PATH="${SCRIPT_DIR}/label_range_scaled.json"
    else
        NORM_JSON_PATH="${SCRIPT_DIR}/label_range_unscaled.json"
    fi

    export mode="$MODE"
    export file_path_train_val="$FILE_PATH_TRAIN_VAL"
    export file_path_test="$FILE_PATH_TEST"
    export norm_json_path="$NORM_JSON_PATH"
    export h5_cache_dir="$H5_CACHE_DIR"
    export trained_models_dir="$TRAINED_MODELS_DIR"

    cd "$SCRIPT_DIR"

    for normalization_mode in "${normalization_modes[@]}"; do
        echo ">>> Normalization: $normalization_mode"
        export normalization_mode="$normalization_mode"

        for num_instances in "${num_instances_array[@]}"; do
            echo "Training: n=$num_instances"

            config_start=$SECONDS
            export num_instances="$num_instances"

            /storage/group/azs7266/default/wzk5140/.conda/envs/hyperopt/bin/python train.py

            config_elapsed=$((SECONDS - config_start))
            echo ">>> Finished: mode=$MODE, norm=$normalization_mode, n=$num_instances in ${config_elapsed}s ($(($config_elapsed / 60))m $(($config_elapsed % 60))s)"
        done
    done

    mode_elapsed=$((SECONDS - mode_start))
    echo "Completed MODE: $MODE in ${mode_elapsed}s ($(($mode_elapsed / 3600))h $(($mode_elapsed % 3600 / 60))m)"

done

total_elapsed=$((SECONDS - total_start))
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "All training completed (scaled and unscaled)"
echo "Total time: ${total_elapsed}s ($(($total_elapsed / 3600))h $(($total_elapsed % 3600 / 60))m $(($total_elapsed % 60))s)"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
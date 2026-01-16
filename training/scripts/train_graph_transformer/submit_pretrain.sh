#!/bin/bash
# SLURM settings (remove these lines if not using SLURM)
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --job-name=GT_pretrain
#SBATCH --output=GT_pretrain_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --error=GT_pretrain_%j.err
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --gpus=1

source ~/.bashrc
conda activate <your_conda_env>

# paths (modify these)
BASE_DIR="<path_to_NEO-LRP>"
TRAINING_DIR="${BASE_DIR}/training"
DATA_DIR="${TRAINING_DIR}/data"
SCRIPT_DIR="${TRAINING_DIR}/scripts/train_graph_transformer"
H5_CACHE_DIR="${TRAINING_DIR}/data/h5_cache"
OUTPUT_DIR="${TRAINING_DIR}/results/graph_transformer"

FILE_PATH_TRAIN_VAL="${DATA_DIR}/train_val.txt"
FILE_PATH_TEST="${DATA_DIR}/test.txt"

# hpo parameters
SWEEP_COUNT=50
num_instances_array=(110000)
normalization_modes=("cost_over_fi")

# wandb setup (set your API key or disable)
export WANDB_API_KEY="<your_wandb_api_key>"
# export WANDB_DISABLED="true"  # uncomment to disable wandb

total_start=$SECONDS

for MODE in "scaled"; do

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "Starting HPO MODE: $MODE"
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
    export output_dir="$OUTPUT_DIR"
    export sweep_count="$SWEEP_COUNT"

    cd "$SCRIPT_DIR"

    for normalization_mode in "${normalization_modes[@]}"; do
        echo ">>> HPO Normalization: $normalization_mode"
        export normalization_mode="$normalization_mode"

        for num_instances in "${num_instances_array[@]}"; do
            echo "HPO: n=$num_instances"

            config_start=$SECONDS
            export num_instances="$num_instances"

            python pretrain.py

            config_elapsed=$((SECONDS - config_start))
            echo ">>> Finished HPO: mode=$MODE, norm=$normalization_mode, n=$num_instances in ${config_elapsed}s ($(($config_elapsed / 60))m $(($config_elapsed % 60))s)"
        done
    done

    mode_elapsed=$((SECONDS - mode_start))
    echo "Completed HPO MODE: $MODE in ${mode_elapsed}s ($(($mode_elapsed / 3600))h $(($mode_elapsed % 3600 / 60))m)"

done

total_elapsed=$((SECONDS - total_start))
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "All HPO completed (scaled and unscaled)"
echo "Total time: ${total_elapsed}s ($(($total_elapsed / 3600))h $(($total_elapsed % 3600 / 60))m $(($total_elapsed % 60))s)"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

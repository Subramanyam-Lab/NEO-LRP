#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --job-name=DS_deepsets
#SBATCH --output=DS_deepsets_%j.out
#SBATCH --time=2-00:00:00
#SBATCH --error=DS_deepsets_%j.err
#SBATCH --account=azs7266_p_gpu
#SBATCH --partition=sla-prio
#SBATCH --gpus=1

source ~/.bashrc
conda activate /storage/group/azs7266/default/wzk5140/.conda/envs/hyperopt
module load gcc

BASE_DIR="/storage/group/azs7266/default/wzk5140/NEO-LRP"
TRAINING_DIR="${BASE_DIR}/training"
DATA_DIR="${TRAINING_DIR}/data"
SCRIPT_DIR="${TRAINING_DIR}/scripts/train_deepsets"
TRAINED_MODELS_DIR="${BASE_DIR}/trained_models"

# data files (same for both scaled and unscaled)
FILE_PATH_TRAIN_VAL="${DATA_DIR}/train_val.txt"
FILE_PATH_TEST="${DATA_DIR}/test.txt"

# training parameters
num_instances_array=(110 1100 11000 110000)
max_evals_array=(150 100 50 25)
normalization_modes=("cost_over_fi" "cost_over_fi_minmax" "minmax" "raw")

total_start=$SECONDS

for MODE in "scaled" "unscaled"; do

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "Starting MODE: $MODE"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    mode_start=$SECONDS

    if [ "$MODE" == "scaled" ]; then
        NORM_JSON_PATH="${SCRIPT_DIR}/label_range_scaled.json"
        OUTPUT_DIR="${TRAINING_DIR}/results/deepsets/scaled"
    else
        NORM_JSON_PATH="${SCRIPT_DIR}/label_range_unscaled.json"
        OUTPUT_DIR="${TRAINING_DIR}/results/deepsets/unscaled"
    fi

    mkdir -p "$OUTPUT_DIR"

    export mode="$MODE"
    export file_path_train_val="$FILE_PATH_TRAIN_VAL"
    export file_path_test="$FILE_PATH_TEST"
    export norm_json_path="$NORM_JSON_PATH"
    export output_dir="$OUTPUT_DIR"
    export trained_models_dir="$TRAINED_MODELS_DIR"

    echo "Mode: $MODE"
    echo "Norm JSON: $NORM_JSON_PATH"
    echo "Output: $OUTPUT_DIR"

    cd "$SCRIPT_DIR"

    for normalization_mode in "${normalization_modes[@]}"; do
        echo ">>> Normalization: $normalization_mode"

        for index in "${!num_instances_array[@]}"; do
            num_instances=${num_instances_array[$index]}
            max_evals=${max_evals_array[$index]}

            echo "Training: n=$num_instances, evals=$max_evals"

            export num_instances
            export max_evals
            export normalization_mode

            config_start=$SECONDS

            /storage/group/azs7266/default/wzk5140/.conda/envs/hyperopt/bin/python hpo.py --surrogate_model GP

            config_elapsed=$((SECONDS - config_start))
            echo ">>> Finished: mode=$MODE, norm=$normalization_mode, n=$num_instances in ${config_elapsed}s ($(($config_elapsed / 60))m $(($config_elapsed % 60))s)"

            find . -type d -name "checkpoints_trial_*" -exec rm -rf {} +
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
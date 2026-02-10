#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio

# Usage:
#   sbatch --job-name=prodhon_ds submit.sh P_prodhon cost_over_fi 110000 vroom deepsets 5
#   sbatch --job-name=prodhon_gt submit.sh P_prodhon cost_over_fi 110000 vroom graph_transformer 5
#   sbatch --job-name=schneider_ds --array=1-203 submit.sh S_schneider cost_over_fi 110000 vroom deepsets 1 configs/schneider_instances.txt
#   sbatch --job-name=tuzun submit.sh T_tuzun cost_over_fi 110000 vroom deepsets 5
#   sbatch --job-name=barreto submit.sh B_barreto cost_over_fi 110000 vroom deepsets 5
#
# Bash usage (remove/comment SBATCH lines first):
#   bash submit.sh P_prodhon cost_over_fi 110000 vroom deepsets 5

DATASET="$1"
NORMALIZATION="$2"
N="$3"
SOLVER="$4"
MODEL_TYPE="$5"
NUM_RUNS="$6"
INSTANCES_FILE="${7:-}"  # optional

# Validate required arguments
if [[ -z "$DATASET" ]]; then
    echo "ERROR: DATASET (arg 1) is required"
    echo "Usage: submit.sh DATASET NORMALIZATION N SOLVER MODEL_TYPE NUM_RUNS [INSTANCES_FILE]"
    exit 1
fi

if [[ -z "$NORMALIZATION" ]]; then
    echo "ERROR: NORMALIZATION (arg 2) is required"
    echo "Usage: submit.sh DATASET NORMALIZATION N SOLVER MODEL_TYPE NUM_RUNS [INSTANCES_FILE]"
    exit 1
fi

if [[ -z "$N" ]]; then
    echo "ERROR: N (arg 3) is required"
    echo "Usage: submit.sh DATASET NORMALIZATION N SOLVER MODEL_TYPE NUM_RUNS [INSTANCES_FILE]"
    exit 1
fi

if [[ -z "$SOLVER" ]]; then
    echo "ERROR: SOLVER (arg 4) is required"
    echo "Usage: submit.sh DATASET NORMALIZATION N SOLVER MODEL_TYPE NUM_RUNS [INSTANCES_FILE]"
    exit 1
fi

if [[ -z "$MODEL_TYPE" ]]; then
    echo "ERROR: MODEL_TYPE (arg 5) is required"
    echo "Usage: submit.sh DATASET NORMALIZATION N SOLVER MODEL_TYPE NUM_RUNS [INSTANCES_FILE]"
    exit 1
fi

if [[ -z "$NUM_RUNS" ]]; then
    echo "ERROR: NUM_RUNS (arg 6) is required"
    echo "Usage: submit.sh DATASET NORMALIZATION N SOLVER MODEL_TYPE NUM_RUNS [INSTANCES_FILE]"
    exit 1
fi

# Validate argument values
if [[ "$DATASET" != "P_prodhon" && "$DATASET" != "S_schneider" && "$DATASET" != "T_tuzun" && "$DATASET" != "B_barreto" ]]; then
    echo "ERROR: Invalid DATASET: $DATASET"
    echo "Must be one of: P_prodhon, S_schneider, T_tuzun, B_barreto"
    exit 1
fi

if [[ "$NORMALIZATION" != "raw" && "$NORMALIZATION" != "minmax" && "$NORMALIZATION" != "cost_over_fi" && "$NORMALIZATION" != "cost_over_fi_minmax" ]]; then
    echo "ERROR: Invalid NORMALIZATION: $NORMALIZATION"
    echo "Must be one of: raw, minmax, cost_over_fi, cost_over_fi_minmax"
    exit 1
fi

if [[ "$N" != "110" && "$N" != "1100" && "$N" != "11000" && "$N" != "110000" ]]; then
    echo "ERROR: Invalid N: $N"
    echo "Must be one of: 110, 1100, 11000, 110000"
    exit 1
fi

if [[ "$SOLVER" != "vroom" && "$SOLVER" != "ortools" && "$SOLVER" != "vrpeasy" ]]; then
    echo "ERROR: Invalid SOLVER: $SOLVER"
    echo "Must be one of: vroom, ortools, vrpeasy"
    exit 1
fi

if [[ "$MODEL_TYPE" != "deepsets" && "$MODEL_TYPE" != "graph_transformer" ]]; then
    echo "ERROR: Invalid MODEL_TYPE: $MODEL_TYPE"
    echo "Must be one of: deepsets, graph_transformer"
    exit 1
fi

if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [[ "$NUM_RUNS" -lt 1 ]]; then
    echo "ERROR: Invalid NUM_RUNS: $NUM_RUNS"
    echo "Must be a positive integer"
    exit 1
fi

# Base paths
BASE_DIR="/storage/group/azs7266/default/wzk5140/NEO-LRP"
NEOLRP_DIR="${BASE_DIR}/neo-lrp"
CONFIGS_DIR="${NEOLRP_DIR}/configs"
BKS_DIR="${CONFIGS_DIR}/BKS"
INSTANCES_DIR="${BASE_DIR}/benchmark_instances"
MODELS_DIR="${BASE_DIR}/trained_models"

# Determine scaled vs unscaled based on dataset
if [[ "$DATASET" == "P_prodhon" || "$DATASET" == "S_schneider" ]]; then
    SCALE_TYPE="scaled"
elif [[ "$DATASET" == "T_tuzun" || "$DATASET" == "B_barreto" ]]; then
    SCALE_TYPE="unscaled"
else
    echo "ERROR: Unknown dataset for scale type: $DATASET"
    exit 1
fi

# Model paths (for validation only - run.py auto-discovers these)
if [[ "$MODEL_TYPE" == "deepsets" ]]; then
    PHI_LOC="${MODELS_DIR}/deepsets/${SCALE_TYPE}/phi/${NORMALIZATION}/${N}.onnx"
    RHO_LOC="${MODELS_DIR}/deepsets/${SCALE_TYPE}/rho/${NORMALIZATION}/${N}.onnx"
else
    PHI_LOC="${MODELS_DIR}/graph_transformer/${SCALE_TYPE}/${NORMALIZATION}/${N}.pth"
    RHO_LOC=""
fi

# Config and BKS paths (for validation only)
CONFIG_PATH="${CONFIGS_DIR}/${DATASET}.json"
BKS_PATH="${BKS_DIR}/${DATASET}.json"
INSTANCE_DIR="${INSTANCES_DIR}/${DATASET}"

# Output directory
OUTPUT_DIR="${NEOLRP_DIR}/output"

# Environment setup
source ~/.bashrc
module load anaconda
module load gurobi
module load gcc
conda activate /storage/group/azs7266/default/wzk5140/.conda/envs/neolrp

PYTHON="/storage/group/azs7266/default/wzk5140/.conda/envs/neolrp/bin/python"

# Change to neo-lrp directory
cd "${NEOLRP_DIR}"

echo " ~~~~~ SLURM Job Info  ~~~~~"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Task ID:       ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Dataset:       ${DATASET}"
echo "Normalization: ${NORMALIZATION}"
echo "N:             ${N}"
echo "Solver:        ${SOLVER}"
echo "Num runs:      ${NUM_RUNS}"
echo "Model type:    ${MODEL_TYPE}"
echo "Scale type:    ${SCALE_TYPE}"
echo "Phi model:     ${PHI_LOC}"
echo "Rho model:     ${RHO_LOC:-N/A}"
echo "Config:        ${CONFIG_PATH}"
echo "BKS:           ${BKS_PATH}"
echo "Instance dir:  ${INSTANCE_DIR}"
echo "Output dir:    ${OUTPUT_DIR}"
echo " ~~~~~ ~~~~~ ~~~~~ ~~~~~ ~~~~~"

# Validate files exist
if [[ ! -f "$PHI_LOC" ]]; then
    echo "ERROR: Missing phi model: $PHI_LOC"
    exit 1
fi

if [[ "$MODEL_TYPE" == "deepsets" && ! -f "$RHO_LOC" ]]; then
    echo "ERROR: Missing rho model: $RHO_LOC"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Missing config: $CONFIG_PATH"
    exit 1
fi

if [[ ! -d "$INSTANCE_DIR" ]]; then
    echo "ERROR: Missing instance directory: $INSTANCE_DIR"
    exit 1
fi

# Build command - use run.py instead of core.neos_runner
CMD="${PYTHON} run.py \
    --dataset ${DATASET} \
    --normalization ${NORMALIZATION} \
    --N ${N} \
    --solver ${SOLVER} \
    --model_type ${MODEL_TYPE} \
    --num_runs ${NUM_RUNS} \
    --output_dir ${OUTPUT_DIR}"

# Handle array jobs with instances file
if [[ -n "$INSTANCES_FILE" ]]; then
    if [[ ! -f "$INSTANCES_FILE" ]]; then
        echo "ERROR: Instances file not found: $INSTANCES_FILE"
        exit 1
    fi
    
    if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
        echo "ERROR: INSTANCES_FILE provided but not running as array job"
        echo "Use: sbatch --array=1-N submit.sh ..."
        exit 1
    fi
    
    mapfile -t instances < "$INSTANCES_FILE"
    idx=$((SLURM_ARRAY_TASK_ID - 1))
    
    if (( idx < 0 || idx >= ${#instances[@]} )); then
        echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (1-${#instances[@]})"
        exit 1
    fi
    
    INSTANCE_NAME="${instances[$idx]}"
    echo "Instance:      ${INSTANCE_NAME}"
    CMD="${CMD} --instance ${INSTANCE_NAME}"
fi

echo ""
echo "Running command:"
echo "$CMD"
echo ""

$CMD
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: run.py exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "~~~~~Job completed successfully~~~~~"
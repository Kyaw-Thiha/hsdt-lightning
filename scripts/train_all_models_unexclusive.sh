#!/usr/bin/env bash
#
# train_all_models_slurm.sh
# -------------------------
# Submit with `sbatch scripts/train_all_models_slurm.sh`.
# The job sequentially trains each registered model (tdsat, hsdt, hdst, ssrt)
# under the Slurm allocation. Additional CLI arguments supplied to this script
# are forwarded to every `python main.py ...` invocation.
#
# Require prior environment setup via scripts/setup.sh so the virtualenv lives
# at ${PROJECT_ROOT}/.env.
#
# Slurm directives (override on sbatch command line if desired):
#   sbatch --gres=gpu:1 --cpus-per-task=8 scripts/train_all_models_slurm.sh

#SBATCH --job-name=hsdt-suite
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.log
#SBATCH --error=logs/slurm/%x_%j.log

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.env"
LOG_DIR="${PROJECT_ROOT}/logs/slurm"

mkdir -p "${LOG_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "❌ Virtualenv not found at ${VENV_DIR}. Run scripts/setup.sh first." >&2
  exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "❌ This script is meant to run under Slurm via 'sbatch'. Aborting." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"

declare -a MODELS=("tdsat" "hsdt" "hdst" "ssrt")

echo "🟢 Slurm job ${SLURM_JOB_ID:-N/A}: starting sequential trainings"
echo "    Project root: ${PROJECT_ROOT}"
echo "    Models: ${MODELS[*]}"

cd "${PROJECT_ROOT}"

for model in "${MODELS[@]}"; do
  echo ""
  echo "=============================================="
  echo "🚀 Launching training for model: ${model}"
  echo "=============================================="

  srun --ntasks=1 python main.py fit --config "config/models/${model}.yaml" "$@"
done

echo ""
echo "✅ All model trainings completed."

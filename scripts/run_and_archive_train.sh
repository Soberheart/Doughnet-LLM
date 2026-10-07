#!/usr/bin/env bash
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}" || exit 1

MODEL="${1:-ae}"
export WANDB_MODE="${WANDB_MODE:-offline}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_DIR}/records/train_${MODEL}_${STAMP}.log"
mkdir -p "${PROJECT_DIR}/records"

export DOUGHNET_COMMAND="bash train_doughnet_with_archive.sh ${MODEL}"
echo "Command: ${DOUGHNET_COMMAND}" | tee "${LOG_FILE}"
bash train_doughnet_with_archive.sh "${MODEL}" 2>&1 | tee -a "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"

bash "${PROJECT_DIR}/scripts/archive_doughnet_run.sh" "training_${MODEL}" "${LOG_FILE}"
exit "${STATUS}"

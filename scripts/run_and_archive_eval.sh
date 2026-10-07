#!/usr/bin/env bash
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}" || exit 1

export WANDB_MODE="${WANDB_MODE:-offline}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_DIR}/records/evaluation_${STAMP}.log"
mkdir -p "${PROJECT_DIR}/records"

export DOUGHNET_COMMAND="python net/prediction.py --config-name dyn settings.test_only=True"
echo "Command: ${DOUGHNET_COMMAND}" | tee "${LOG_FILE}"
python net/prediction.py --config-name dyn settings.test_only=True 2>&1 | tee -a "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"

bash "${PROJECT_DIR}/scripts/archive_doughnet_run.sh" official_evaluation "${LOG_FILE}"
exit "${STATUS}"

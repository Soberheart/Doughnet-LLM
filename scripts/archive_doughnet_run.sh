#!/usr/bin/env bash
set -u

# Archive a completed DoughNet evaluation or training run.
# Usage: bash scripts/archive_doughnet_run.sh <kind> <log_file> [output_dir]

KIND="${1:-run}"
LOG_FILE="${2:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="${3:-${PROJECT_DIR}/records/${KIND}_${STAMP}}"

mkdir -p "${ARCHIVE_DIR}"
cd "${PROJECT_DIR}" || exit 1

printf '%s\n' "DoughNet run archive" > "${ARCHIVE_DIR}/README.txt"
printf 'Archive time: %s\n' "$(date -Is)" >> "${ARCHIVE_DIR}/README.txt"
printf 'Run kind: %s\n' "${KIND}" >> "${ARCHIVE_DIR}/README.txt"
printf 'Project directory: %s\n' "${PROJECT_DIR}" >> "${ARCHIVE_DIR}/README.txt"

if [ -n "${LOG_FILE}" ] && [ -f "${LOG_FILE}" ]; then
    cp -p "${LOG_FILE}" "${ARCHIVE_DIR}/evaluation_or_training.log"
else
    printf 'No log supplied or log not found: %s\n' "${LOG_FILE}" >> "${ARCHIVE_DIR}/README.txt"
fi

{
    printf 'date: %s\n' "$(date -Is)"
    printf 'command_line: %s\n' "${DOUGHNET_COMMAND:-unknown}"
    printf 'git_commit: '
    git rev-parse HEAD 2>/dev/null || printf 'unavailable\n'
    printf 'git_branch: '
    git branch --show-current 2>/dev/null || printf 'unavailable\n'
    printf 'git_status:\n'
    git status --short 2>/dev/null || true
} > "${ARCHIVE_DIR}/run_metadata.txt"

{
    printf 'python: '
    python --version 2>&1 || true
    python - <<'PY'
import sys
print("python_executable:", sys.executable)
try:
    import torch
    print("pytorch:", torch.__version__)
    print("pytorch_cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print("gpu_%d:" % i, torch.cuda.get_device_name(i))
except Exception as exc:
    print("torch_error:", repr(exc))
PY
    python -m pip freeze 2>/dev/null || true
} > "${ARCHIVE_DIR}/environment.txt"

for file in data/dataset.h5 weights/ae.pth weights/dyn.pth; do
    if [ -f "${file}" ]; then
        stat -c '%s %n' "${file}" 2>/dev/null || wc -c "${file}"
        sha256sum "${file}"
    fi
done > "${ARCHIVE_DIR}/input_checksums.sha256"

if [ -d wandb ]; then
    find wandb -maxdepth 1 -type d -name 'offline-run-*' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -10 > "${ARCHIVE_DIR}/wandb_offline_runs.txt"
fi

find . -type f \( -name 'best.pth' -o -name 'last.pth' -o -name '*.pth' \) \
    -not -path './weights/*' -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null \
    | sort -r > "${ARCHIVE_DIR}/model_outputs.txt"

printf 'Created archive: %s\n' "${ARCHIVE_DIR}"

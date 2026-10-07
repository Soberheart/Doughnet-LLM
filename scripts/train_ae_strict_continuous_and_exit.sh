#!/usr/bin/env bash

# Source this file from an active PBS interactive job after activating doughnet.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run with: source scripts/train_ae_strict_continuous_and_exit.sh"
    exit 2
fi

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "PBS_JOBID is not set; refusing to start training outside a PBS job."
    return 2
fi

if ! command -v torchrun >/dev/null 2>&1; then
    echo "torchrun was not found. Activate the doughnet environment first."
    return 2
fi

REPO="$HOME/projects/doughnet"
TRAIN_DIR="$REPO/records/training_ae_strict_continuous"
LOG="$TRAIN_DIR/train_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -d "$REPO/net" ]]; then
    echo "Project directory not found: $REPO"
    return 2
fi

mkdir -p "$TRAIN_DIR" || return 2

echo "Starting 30-epoch AE training. Log: $LOG"
(
    cd "$REPO" || exit 1
    set -o pipefail
    torchrun --standalone --nproc_per_node=4 net/prediction.py \
        --config-name ae \
        settings.ddp=True \
        settings.test_only=False \
        settings.resume=False \
        settings.resume_freeze=False \
        training.epochs=30 \
        training.bs=14 \
        2>&1 | tee "$LOG"
)
train_status=$?

if [[ "$train_status" -ne 0 ]]; then
    echo "Training failed with exit code $train_status. PBS session remains open."
    echo "Log: $LOG"
    return "$train_status"
fi

if ! grep -Eq 'end of epoch 30:' "$LOG"; then
    echo "Training exited without a confirmed epoch 30. PBS session remains open."
    echo "Log: $LOG"
    return 1
fi

echo "Confirmed epoch 30 completed. Exiting the PBS interactive session."
exit 0

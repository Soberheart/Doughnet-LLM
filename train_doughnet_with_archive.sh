#!/usr/bin/env bash
set -Eeuo pipefail

# Run on an allocated NSCC GPU compute node from the DoughNet repository root.
# Usage: bash train_doughnet_with_archive.sh ae|dyn

stage="${1:-}"
case "$stage" in
  ae|dyn) ;;
  *) printf 'Usage: bash %s ae|dyn\n' "$0" >&2; exit 2 ;;
esac

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

if [[ ! -f net/prediction.py || ! -f "net/config/${stage}.yaml" ]]; then
  printf 'Run this script from the DoughNet repository root.\n' >&2
  exit 1
fi

if ! python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  printf 'CUDA is unavailable. Run this on an allocated GPU compute node with the DoughNet environment active.\n' >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
results_root="${DOUGHNET_RESULTS_DIR:-$repo_root/results}"
run_dir="$results_root/$stage/$stamp"
mkdir -p "$run_dir/checkpoints" "$run_dir/config" "$run_dir/logs" "$run_dir/metadata"

cp "net/config/${stage}.yaml" "$run_dir/config/${stage}.yaml"
cp net/config/common.yaml "$run_dir/config/common.yaml"
cp .gitmodules "$run_dir/config/.gitmodules" 2>/dev/null || true

{
  printf 'date=%s\n' "$(date --iso-8601=seconds)"
  printf 'stage=%s\n' "$stage"
  printf 'host=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$repo_root"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'git_status:\n'
  git status --short 2>/dev/null || true
  printf '\npython:\n'
  python --version 2>&1
  printf 'torch/cuda/gpu:\n'
  python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))'
  printf '\nPBS_JOBID=%s\n' "${PBS_JOBID:-not-set}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-not-set}"
  printf '\nGPU status:\n'
  nvidia-smi 2>&1 || true
  printf '\nInput files:\n'
  ls -lh data/dataset.h5 weights/ae.pth weights/dyn.pth 2>&1 || true
  printf '\nSubmodule/source revisions:\n'
  git -C net/nvdiffrast rev-parse HEAD 2>/dev/null || true
  git -C sim/sdftoolbox rev-parse HEAD 2>/dev/null || true
} > "$run_dir/metadata/run_info.txt"

python -m pip freeze > "$run_dir/metadata/pip_freeze.txt"

hydra_dir="$run_dir/hydra"
log_file="$run_dir/logs/train.log"

archive_checkpoints() {
  local rc=$?
  for name in last best; do
    if [[ -f "$hydra_dir/$name.pth" ]]; then
      cp -f "$hydra_dir/$name.pth" "$run_dir/checkpoints/${stage}_${name}.pth"
    fi
  done
  printf 'exit_code=%s\nfinished_at=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$run_dir/metadata/exit_status.txt"
  printf '\nExperiment directory: %s\n' "$run_dir"
  return 0
}
trap archive_checkpoints EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export WANDB_MODE="${WANDB_MODE:-offline}"
set +e
if [[ "$stage" == ae ]]; then
  python net/prediction.py --config-name ae \
    "settings.test_only=False" "settings.resume=False" "hydra.run.dir=$hydra_dir" \
    2>&1 | tee "$log_file"
else
  python net/prediction.py --config-name dyn \
    "settings.test_only=False" "hydra.run.dir=$hydra_dir" \
    2>&1 | tee "$log_file"
fi
train_status=${PIPESTATUS[0]}
set -e
exit "$train_status"

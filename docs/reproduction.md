# DoughNet 复现与后续工作指南

## 1. 当前进度与实验范围

- Level 1：完成 AE 从随机初始化连续训练 30 epoch，并完成完整测试。
- Level 2：加载已验证 AE，冻结非 `condition` 参数，完成 Dyn 连续训练 30 epoch、单步及序列测试和归档。
- Level 3：待复现以目标点云为条件的 CEM 动作规划。
- 语言约束与 LLM 集成：尚未实现，应在点云规划基线验证后开展。

这些结果来自一次训练运行，尚无多随机种子统计。连续完成 30 epoch 不等于完整复现论文所有实验，也不构成会议录用标准。此前五个 AE 可视化案例来自旧 checkpoint，不能作为新 30 epoch AE 的可视化验证。

## 2. 安装与数据准备

在 Linux GPU 环境执行：

```bash
git clone https://github.com/Soberheart/Doughnet-LLM.git
cd Doughnet-LLM
conda env create -f environment.yml
conda activate doughnet
git clone https://github.com/NVlabs/nvdiffrast.git net/nvdiffrast
git clone https://github.com/cheind/sdftoolbox.git sim/sdftoolbox
pip install -e net/nvdiffrast
pip install -e sim/sdftoolbox
```

依赖目录已存在时不要重复克隆。当前未记录第三方依赖的精确 commit；上述命令获取当前版本，不保证与已完成实验一致。后续应记录两个依赖的 `git rev-parse HEAD` 并锁定版本。

原始 `environment.yml` 是安装起点；此前服务器记录为 Python 3.9.18、PyTorch 2.0.0+cu118、CUDA 11.8，AE 使用四张 A100-SXM4-40GB。不要把环境文件中的版本与实际运行环境视为完全一致。

从 [官方数据下载页](https://real.stanford.edu/doughnet/) 准备数据与官方权重：

```text
data/dataset.h5
weights/ae.pth
weights/dyn.pth
```

Git 忽略数据、权重、outputs、records、results、wandb 和第三方依赖目录。克隆本仓库不会下载自训模型。NSCC 上训练与评估应在已分配的 GPU 计算节点执行，不要在登录节点启动。

## 3. Level 1：AE 训练与测试

进入项目目录并激活环境，确认 `nvidia-smi` 能看到四张 GPU：

```bash
mkdir -p records/training_ae_strict_continuous
export WANDB_MODE=offline
set -o pipefail
torchrun --standalone --nproc_per_node=4 net/prediction.py \
  --config-name ae settings.ddp=True settings.test_only=False \
  settings.resume=False settings.resume_freeze=False \
  training.epochs=30 training.bs=14 \
  2>&1 | tee records/training_ae_strict_continuous/train.log
```

每卡 batch 14，全局 batch 56。建议用 PBS 批处理包装命令，避免交互连接中断。脚本 `train_ae_strict_continuous_and_exit.sh` 则用于通过 `source` 在交互 PBS shell 中运行，成功后退出该 shell，不会关闭集群机器。

在一张 GPU 上测试本次已完成实验：

```bash
set -o pipefail
python net/prediction.py --config-name ae \
  settings.ddp=False settings.test_only=True \
  settings.test_path=outputs/2026-10-01/18-29-03/best.pth \
  2>&1 | tee records/training_ae_strict_continuous/test.log
```

新训练会生成新目录，必须相应替换 checkpoint 路径。

| 指标（%） | 官方 AE 对照（此前记录的四舍五入值） | 自训连续 30 epoch |
| --- | ---: | ---: |
| VIoU | 94.8 | 95.045 |
| CIoU | 94.3 | 94.521 |
| AccC | 99.7 | 99.918 |
| AccG | 99.1 | 99.231 |

四项均达到与该对照相差 1 个百分点以内的目标。这里是官方预训练模型的评估对照，不能泛化为所有论文设置的复现结论。

VIoU 衡量整体占据体积的交并比；CIoU 衡量组件级重建重叠；AccC 是组件数量预测准确率；AccG 是 genus 预测准确率。它们不是机器人规划任务成功率。

## 4. Level 2：Dyn 训练与测试

先检查 `scripts/train_dyn_30.pbs`，修改项目目录、虚拟环境路径、AE checkpoint，以及集群队列、项目和资源字段。当前脚本含原实验的 NSCC 绝对路径，不能直接用于其他机器。

```bash
grep '^#PBS' scripts/train_dyn_30.pbs
grep -nE 'AE_CHECKPOINT|resume_path|resume_freeze|training\.epochs|training\.bs' scripts/train_dyn_30.pbs
qsub scripts/train_dyn_30.pbs
qstat -u "$USER"
```

当前脚本设置四卡、每卡 batch 12（全局 48）、30 epoch、从自训 AE 初始化并冻结 AE。`settings.resume=True` 在这里表示加载预训练权重，不是恢复已中断 Dyn 训练。

已完成作业 `25678817.pbs101` 的退出码为 0，用时 03:23:59。训练日志为 `records/training_dyn/train_30epoch_25678817.pbs101.log`，模型目录为 `outputs/2026-10-04/14-59-24/`。

分配一张 GPU 后测试验证集选出的最佳 checkpoint：

```bash
mkdir -p records/training_dyn
export WANDB_MODE=offline
set -o pipefail
python net/prediction.py --config-name dyn \
  settings.ddp=False settings.test_only=True \
  settings.test_single=True settings.test_multi=True \
  settings.test_path=outputs/2026-10-04/14-59-24/best.pth \
  2>&1 | tee records/training_dyn/test_30epoch_best.log
```

| 测试模式（%） | VIoU | CIoU | AccC | AccG |
| --- | ---: | ---: | ---: | ---: |
| 单步预测 | 94.7 | 93.6 | 98.3 | 99.0 |
| 完整序列平均 | 92.3 | 91.2 | 97.8 | 98.8 |
| 最后一帧 | 85.5 | 79.1 | 90.3 | 91.5 |

序列末帧质量低于单步，说明误差随 rollout 累积，后续规划必须检查多步预测与实际执行结果。不要根据测试集反复选取 checkpoint；当前 `best.pth` 使用验证指标选择。

## 5. checkpoint 与实验归档

当前 `net/prediction.py` 的保存逻辑仅保存 `model_state_dict`，缺少 optimizer、scheduler、epoch 和随机数状态。中断后加载权重再训练 8 epoch 不等价于原 30 epoch 连续训练。严格续训功能仍待实现与验证。

已有归档：

```text
records/training_ae_strict_continuous_20261002_133250/
records/training_dyn_30epoch_20261004_184359/
```

Dyn 归档与校验示例：

```bash
bash scripts/archive_doughnet_run.sh training_dyn_30epoch \
  records/training_dyn/train_30epoch_25678817.pbs101.log
# 用上条命令实际返回的目录替换 ARCHIVE。
ARCHIVE=records/training_dyn_30epoch_20261004_184359
cp records/training_dyn/test_30epoch_best.log "$ARCHIVE/"
sha256sum outputs/2026-10-04/14-59-24/best.pth \
  outputs/2026-10-04/14-59-24/last.pth > "$ARCHIVE/checkpoint_checksums.sha256"
sha256sum -c "$ARCHIVE/checkpoint_checksums.sha256"
```

校验命令必须从项目根目录运行。哈希清单不包含 checkpoint 文件本身，归档也不能仅凭日志认定为完整可迁移备份。另行保存权重、实际 `.hydra/config.yaml`、`.hydra/overrides.yaml`、训练脚本和源代码 commit。

已记录的最佳 checkpoint SHA256：

```text
AE:  43e388893e404081d4db953791d8976a94dc38ceec5a5111131a544ce1778a0e
Dyn: a05452f681f89ee26763959ec77db234dc0bdcf2cbd2996f7361beebb563acae
```

## 6. Level 3：目标点云规划的具体流程

目的：建立不依赖语言的动作规划基线，验证模型能否根据当前观测和明确目标搜索有效动作。当前仓库未完成 CEM planner；动作生成脚本不能直接当作规划器。

1. 核对论文与代码接口。定位观测编码、动作参数、Dyn rollout、解码及模拟器执行入口；确认坐标单位、四元数顺序、工具类型、闭合宽度和时间间隔。
2. 固定任务与目标观测。先选一个可复现初始场景，保存当前观测、目标点云、随机种子和 checkpoint 哈希，核实目标点云与训练输入的预处理一致。
3. 验证记录动作 rollout。先用一个已知动作运行模型，检查输出形状、坐标及拓扑，避免把接口错误带入搜索。
4. 实现候选动作评分。以目标 latent 与预测 latent 的余弦相似度为起点，最小化 `J_point(tau) = -cos(z_goal, z_pred(tau))`。核实原始实现对多个 latent token 的聚合方式后再固定目标函数。
5. 实现 CEM。核实并记录官方候选数、elite 数、迭代次数、采样范围、分布更新方式和规划时域；在这些信息确定前，不把经验参数宣称为官方设置。
6. 在模拟器中执行所选动作。记录执行后的真实状态，与预测及目标比较。数据集中原记录动作的未来帧不能充当新规划动作的执行结果。
7. 加入相同候选评估预算的随机搜索对照，在固定任务集上统计目标形状匹配、拓扑正确率、任务成功率和规划耗时。
8. 归档每轮候选动作与分数、elite、最优动作、预测状态、执行状态、配置和随机种子。完成这条基线后再加入语言目标或约束。

首个可交付目标：一个固定场景完成“目标点云 → 搜索动作 → 模拟执行 → 评估与归档”的闭环。机器人实机与语言实验不属于当前已完成范围。

## 7. 日常 Git 同步

服务器现有 `origin` 指向官方仓库，`personal` 指向本仓库；`main` 已跟踪 `personal/main`。

```bash
git status --short
git pull --ff-only personal main
# 修改代码或文档后，明确指定要提交的文件。
git add README.md docs/reproduction.md
git diff --cached --check
git commit -m "Document reproduction results and planning workflow"
git push personal main
```

存在未提交修改时先检查再拉取；不要强制覆盖服务器上的实验代码。发布时保留原项目的作者、许可证及引用，并区分原始实现与本仓库新增内容。

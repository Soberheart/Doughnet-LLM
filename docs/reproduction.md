# DoughNet 复现与后续工作指南

## 1. 当前进度与实验范围

- Level 1：完成 AE 从随机初始化连续训练 30 epoch，并完成完整测试。
- Level 2：加载已验证 AE，冻结非 `condition` 参数，完成 Dyn 连续训练 30 epoch、单步及序列测试和归档。
- Level 3：已完成固定场景模拟、预处理、记录动作 Dyn/AE 对照及表面拓扑审计；目标评分核对正在进行，CEM 搜索和所选动作的模拟执行尚未完成。
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

### 模拟器缺少 board 网格时

官方仓库未包含 `sim/mpm/assets/meshes/processed/board-board.obj`，但包含碰撞用的 `board-128.sdf`。如果 OBJ 缺失，可先在登录节点恢复网格；这一步只需要 NumPy，不需要 GPU：

```bash
cd ~/projects/doughnet
source ~/venvs/doughnet/bin/activate
python scripts/restore_board_mesh.py
ls -lh sim/mpm/assets/meshes/processed/board-board.obj \
  sim/mpm/assets/meshes/processed/board-128.sdf
```

脚本从 SDF 零等值面提取网格，并使用 `T_mesh_to_voxels` 的逆变换恢复坐标；检查网格闭合及面朝向，生成 `board-board.provenance.json` 记录来源、哈希及方法。它不修改 SDF，也不覆盖已有 OBJ 或来源记录。

该网格是从已有 SDF 恢复的资源，并非作者原始 OBJ；模拟实验应归档来源记录并注明这一差异。成功恢复资源后，继续在分配到的 GPU 节点上验证固定动作模拟，不能仅凭网格检查通过就宣称规划复现完成。

### GPU 节点无窗口模拟

`render=False` 需要连同底层窗口创建一起关闭。当前实现通过 `enable_visualization=config.render` 控制 Scene；关闭时不创建 Visualizer，也不调用依赖 Vulkan 的 `ti.ui.Window`。MPM 初始化明确使用 CUDA，并禁止静默回退到 CPU。开启渲染时仍创建可视化器，此模式需要另行验证图形环境。

服务器尚未更新这三处代码时，可上传 `scripts/level3_headless_cuda.patch`，在项目根目录先运行 `git apply --check scripts/level3_headless_cuda.patch`，检查通过后运行 `git apply scripts/level3_headless_cuda.patch`。该补丁只包含 `sim/generate/builder.py`、`sim/mpm/__init__.py` 和 `sim/mpm/engine/scene.py`。

如果计算节点没有 Git，可上传 `scripts/apply_level3_headless_cuda.py`，先运行 `python scripts/apply_level3_headless_cuda.py --check`，看到 `CHECK_PASS` 后运行 `python scripts/apply_level3_headless_cuda.py`。脚本只依赖 Python 标准库，预先核对所有修改并检查语法，再备份原文件到 `records/level3_patch_backups/` 后应用相同修改；重复运行不会重新修改已应用的内容。此脚本不要求 `.patch` 文件存在。

在已分配的 GPU 节点验证后端：

```bash
export TI_ARCH=cuda
python -u -c "import taichi as ti; ti.init(arch=ti.cuda, enable_fallback=False); print('CUDA_INIT_PASS')"
```

看到 `Starting on arch=cuda` 和 `CUDA_INIT_PASS` 后，在新建的独立输出目录运行固定动作模拟，保持 `render=False`。使用 `python -u` 将初始化与执行日志及时写入文件；CUDA 后端检查通过不等于完整模拟已通过。

### 固定场景模拟、预处理与记录动作预测检查

2026-10-07 的固定场景 `0000` 已完成 CUDA 无窗口模拟，得到 63 帧（步数 0 至 992，间隔 16），4163 个粒子；位置和速度均为有限值，工具产生接触。末帧早于 `max_horizon=3200`，结合生成器的动作完成退出逻辑与正常结束日志，表明动作流程结束。模拟日志的拓扑标签由一个 genus=1 的组件变为两个 genus=1 的组件。

官方预处理耗时 443 秒，产生 `scenes/data.h5`。服务器检查输出 `H5_CHECK_PASS`：一个场景、63 帧，物体网格每帧 9216 个顶点/3072 个三角形槽位，工具网格 192 个顶点/64 个三角形，工具点云为 1024×3；物体有效三角形数为 2522 至 3050。退化三角形用于固定形状的填充。`genus` 是模拟注释的转存，尚未通过网格几何独立验证。预处理结构通过不等于模型预测或规划通过。

如缺少预处理依赖，已使用的安装命令为 `python -m pip install "pymeshfix==0.16.2" "numpy==1.24.4"`。安装后使用原始预处理入口：

```bash
python -u sim/process.py --config-name common \
  base_dir="$LEVEL3_RUN/scenes" num_processes=1 num_frames=63
```

下一步在已分配的 CUDA 节点运行 `scripts/check_level3_dyn.py`。`LEVEL3_RUN` 应指向上述已有模拟及预处理结果的实验目录；换终端时需要重新设置它。脚本直接读取单场景 HDF5、训练时的 Hydra 配置及完整 Dyn 权重，使用原始 Renderer 生成观测，严格检查所有模型权重键，包括 `condition`。它不需要 Git、W&B 或 Taichi 图形窗口。

```bash
cd ~/projects/doughnet
source ~/venvs/doughnet/bin/activate
: "${LEVEL3_RUN:?请先设置为已有的 Level 3 实验目录}"
DYN_CHECK_DIR=$(mktemp -d "$LEVEL3_RUN/dyn_check_XXXXXX")
set -o pipefail
python -u scripts/check_level3_dyn.py \
  --scene-dir "$LEVEL3_RUN/scenes/0000" \
  --processed "$LEVEL3_RUN/scenes/data.h5" \
  --checkpoint outputs/2026-10-04/14-59-24/best.pth \
  --config outputs/2026-10-04/14-59-24/.hydra/config.yaml \
  --expected-sha256 a05452f681f89ee26763959ec77db234dc0bdcf2cbd2996f7361beebb563acae \
  --output "$DYN_CHECK_DIR/artifacts" \
  2>&1 | tee "$DYN_CHECK_DIR/run.log"
DYN_EXIT=${PIPESTATUS[0]}
echo "dyn_check_exit_code=$DYN_EXIT"
```

训练配置的 `next_frame_offset=5` 对应 0.160 秒。脚本逐步预测 0→5→…→60 的 12 个等间隔区间：单步模式每次重新编码当前观测；连续模式只由第 0 帧初始化并传递预测 latent，后续仅使用记录的工具点云。第 62 帧另行渲染并保存为目标观测，不将 60→62 作为一个训练步长预测。

输出目录包含 `observations.npz`、各帧观测 PLY、预测 NPZ/PLY、逐步 `metrics.csv`、汇总 `summary.json`、配置、源数据和权重哈希。`observed_062.ply` 是后续目标点云。分数由仓库 Evaluater 在原查询网格上计算；工具占据标签被显式转换为布尔掩码，避免整数标签被用于索引。这与现有 benchmark 入口直接传递整数工具标签不同，因此这些单场景诊断指标不能直接视为与此前 benchmark 完全同协议的成绩。latent 余弦值仅用于诊断，其 token 聚合尚未核实为官方规划目标函数。

看到无缺失键的 `Checkpoint load`、`OBSERVATION_PASS`、三个指标汇总、`FIXED_ACTION_INFERENCE_PASS` 且退出码为 0，表示有限数值推理及导出流程完成；仍需检查预测形状与拓扑质量。CEM 搜索和新选择动作的模拟执行仍未实现。

脚本的 CPU 边界检查可运行 `python -m unittest discover -s tests -p test_level3_dyn_helpers.py -v`；该检查覆盖数据损坏、checkpoint 映射、预测时间间隔及导出，不验证 CUDA 渲染或模型预测质量。

### 新场景的预测退化与 AE 对照

上述场景的服务器诊断结果（2026-10-07，12 个等间隔预测区间）为：单步平均 VIoU/CIoU 为 91.907/88.279，连续预测平均为 81.951/81.499；第 60 帧连续预测为 57.336/51.905。它们是单场景诊断成绩，不是规划成功率或完整测试集成绩。

| 目标帧 | 单步 VIoU | 单步 CIoU | 连续 VIoU | 连续 CIoU |
| --- | ---: | ---: | ---: | ---: |
| 30 | 93.720 | 93.720 | 92.159 | 92.159 |
| 35 | 86.639 | 86.639 | 80.804 | 80.804 |
| 50 | 90.653 | 90.653 | 67.257 | 67.257 |
| 55 | 87.074 | 46.423 | 64.569 | 64.569 |
| 60 | 85.359 | 82.479 | 57.336 | 51.905 |

第 50 帧两种模式的 genus 准确率均为 0；第 55 帧组件数量准确率均为 0。第 60 帧两种模式的拓扑头均预测两个组件，与标注组件数一致，但按 CIoU 匹配后的 genus 为 `[0, 1]`，标注为 `[1, 1]`。组件数量正确不等于拓扑全部正确；拓扑头输出也不等于对预测网格进行独立拓扑验证。

连续预测存在明显退化，单步模式在后段也出现误差。尚不能确定 AE 重建、观测质量、数据分布或 Dyn 转移各自的影响，不据此直接修改权重或重训。下一项对照是同一观测的 AE 直接重建。

上传 `scripts/check_level3_ae.py` 到服务器的 `scripts/`，保留已有的 `check_level3_dyn.py`。在重新分配到的单 GPU 节点执行：

```bash
cd ~/projects/doughnet
source ~/venvs/doughnet/bin/activate
# DYN_CHECK_DIR 指向已完成检查的 dyn_check_XXXXXX 目录。
: "${DYN_CHECK_DIR:?请先设置为已有的 Dyn 诊断目录}"
AE_CHECK_DIR=$(mktemp -d "$(dirname "$DYN_CHECK_DIR")/ae_check_XXXXXX")
set -o pipefail
python -u scripts/check_level3_ae.py \
  --dyn-artifacts "$DYN_CHECK_DIR/artifacts" \
  --output "$AE_CHECK_DIR/artifacts" \
  2>&1 | tee "$AE_CHECK_DIR/run.log"
AE_EXIT=${PIPESTATUS[0]}
echo "ae_check_exit_code=$AE_EXIT"
cat "$AE_CHECK_DIR/artifacts/summary.json"
cat "$AE_CHECK_DIR/artifacts/ae_metrics.csv"
```

脚本使用同一 Dyn checkpoint 内冻结的 AE 权重，不调用 condition 模块；复用保存的观测和 reference latent，并重新编码核对 latent 是否对应当前模型。它不重新模拟或渲染，对源网格、权重和相关评估源码进行哈希核对，使用原诊断的查询网格、布尔工具掩码和组件匹配。对已预测的各帧还核对其真实标签与先前导出一致。

输出包含各帧 AE NPZ/PLY、`ae_metrics.csv`、三种模式的 `comparison.csv`、`summary.json` 和来源记录。AE 在第 0 帧及目标第 62 帧也进行直接重建；三种模式均值仅比较共同的 5、10、…、60 帧，不能混入第 62 帧。`AE_RECONSTRUCTION_DIAGNOSTIC_PASS` 表示检查及导出完成，不代表质量合格。

如果后段 AE 直接重建也较差，优先检查观测、源几何与 AE 在该场景的表现；如果 AE 较好而单步较差，检查 Dyn 的动作/时间接口及转移预测；如果 AE 和单步都较好而连续预测较差，重点检查长期误差累积。根据对照结果再决定后续处理，继续完成动作搜索和模拟执行闭环。

2026-10-07 服务器完成了 AE 对照：14 个观测帧，12 个共同目标帧，耗时 24.77 秒；重新编码与保存 latent 的最大绝对差为 0。共同目标帧的 AE 平均 VIoU/CIoU 为 94.713/91.689，AccC/AccG 为 83.333/75.000。不能将 14 帧 AE 总平均与 12 帧 Dyn 平均混合比较。

帧 0 至 45 的已检查观测标注均为 `[1]`，AE 拓扑头均匹配正确；帧 50、55 标注变为一个 genus=2 的组件，AE 拓扑头预测两个组件、匹配 genus=1，但几何 CIoU 仍超过 94%。帧 60、62 标注为两个 genus=1 的组件；AE 拓扑头数量为 2，但匹配 genus 为 `[0, 0]`，CIoU 分别仅为 50.214、39.750。几何分割、拓扑头和模拟标注是不同证据来源，不能互相替代。

帧 35 的 AE VIoU 为 94.709，单步 Dyn 为 86.639、连续 Dyn 为 80.804，说明 Dyn 的退化在 AE 直接几何重建仍较好时已发生。后段同时存在 AE/目标表示问题，不能将全部误差归为连续 Dyn 累积，也不能据此证明训练轮数不足。

### 处理后网格的独立拓扑核对

`sim/process.py` 的 genus 由模拟日志转存；源模拟标签来自 `SceneGraph` 的图结构。该处理入口通过表面重建、平滑、简化和 `pymeshfix.clean_from_arrays` 生成网格，并未重新计算其几何 genus。下一步核对处理后网格与图标注是否一致，保留当前权重和场景。

上传 `scripts/audit_level3_topology.py`，保留已有 `check_level3_dyn.py`；以下 CPU 检查可在登录节点运行，不要求 GPU。将路径设置为本次已完成的诊断目录：

```bash
cd ~/projects/doughnet
source ~/venvs/doughnet/bin/activate
DYN_ARTIFACTS="records/level3_pointcloud/sim_check_FhY2DG/dyn_check_ml9xhm/artifacts"
TOPO_CHECK_DIR=$(mktemp -d records/level3_topology_check_XXXXXX)
set -o pipefail
python -u scripts/audit_level3_topology.py \
  --dyn-artifacts "$DYN_ARTIFACTS" \
  --output "$TOPO_CHECK_DIR/artifacts" \
  2>&1 | tee "$TOPO_CHECK_DIR/run.log"
TOPO_EXIT=${PIPESTATUS[0]}
echo "topology_check_exit_code=$TOPO_EXIT"
cat "$TOPO_CHECK_DIR/artifacts/mesh_topology.csv"
```

脚本默认核对 0、30、35、45、50、55、60、62 帧，去除零面积填充三角形，仅合并坐标完全相同的重复顶点；它不修补、平滑或重建网格。Open3D 检查闭合边流形、顶点流形、可定向性及自交；检查合格后，按每个连通表面的 `g=(2-(V-E+F))/2` 计算 genus。

`matches` 表示某一标签下单个合格连通表面的 genus 与该标签的模拟标注一致；`differs` 表示这两者不同；`unresolved` 表示网格条件不支持该比较。多个表面壳层不能直接视为多个实体组件。本检查审计的是处理后表面，不能单独判定源模拟图或粒子体积哪一个正确。

输出包括逐标签 `mesh_topology.csv`、完整检查结果 `summary.json`、物体网格 `gt_XXX.ply` 和工具网格 `tool_XXX.ply`。`SURFACE_TOPOLOGY_AUDIT_COMPLETE` 表示审计及导出完成，不能视为标注全部验证通过。结合这些结果和已保存的 AE/Dyn 预测，再决定是否检查粒子表面处理或模型的困难场景表现，继续完成规划闭环。

2026-10-07 服务器审计正常完成（退出码 0）。所有已审计标签表面均为单个闭合、边与顶点流形、可定向、无自交的表面，且无重复三角形。结果如下：

| 帧 | 模拟标注 genus | 处理后表面 genus | 审计结果 |
| --- | --- | --- | --- |
| 0、30、35 | `[1]` | `[1]` | 一致 |
| 45 | `[1]` | `[2]` | 不一致 |
| 50、55 | `[2]` | `[2]` | 一致 |
| 60、62 | `[1, 1]` | 两个标签表面各为 genus=1 | 各标签一致 |

第 45 帧证明模拟图标注与处理后网格存在差异；这不能单独确定模拟图变化检测或网格处理哪一个偏离原粒子几何。AE 在该帧输出的 genus=1 与图标注一致，但与处理后表面 genus=2 不一致，因此该帧原 AccG=100% 不能直接解释为处理后几何拓扑正确。

第 50、55、60、62 帧的图标注得到各标签网格审计支持；这些帧已有 AE 拓扑头预测错误，不能由第 45 帧的差异解释。第 35 帧表面与标注也一致，而 Dyn 已有明显几何预测误差。保留原始数据和权重，用独立审计记录注明差异；不根据这一场景直接推断需要增加训练轮数，也不根据解码失败断定目标 latent 不包含拓扑信息。

下一项补查仅覆盖第 35 至 62 帧的每一帧，定位网格和标注变化的时间，无需重新模拟或申请 GPU。现有脚本支持 `--frames`；在 Bash 中使用范围展开：

```bash
cd ~/projects/doughnet
source ~/venvs/doughnet/bin/activate
DYN_ARTIFACTS="records/level3_pointcloud/sim_check_FhY2DG/dyn_check_ml9xhm/artifacts"
TOPO_TRANSITION_DIR=$(mktemp -d records/level3_topology_transition_XXXXXX)
set -o pipefail
python -u scripts/audit_level3_topology.py \
  --dyn-artifacts "$DYN_ARTIFACTS" \
  --frames {35..62} \
  --output "$TOPO_TRANSITION_DIR/artifacts" \
  2>&1 | tee "$TOPO_TRANSITION_DIR/run.log"
TRANSITION_EXIT=${PIPESTATUS[0]}
echo "transition_check_exit_code=$TRANSITION_EXIT"
cat "$TOPO_TRANSITION_DIR/artifacts/mesh_topology.csv"
```

补查若出现 `unresolved`，保留对应帧的检查详情及网格供查看；不将无法验证的帧计为标注正确。对比逐帧转变后，继续验证目标评分和动作搜索；当前仓库未找到原作者发布的 CEM 入口，仍需从论文及补充材料核实参数、动作参数化和 latent 评分聚合。接口检查及拓扑审计不替代 CEM 搜索和所选动作的模拟执行。

### 逐帧审计结果与目标评分核对

2026-10-07 的补查正常结束（退出码 0），覆盖第 35 至 62 帧，共 28 帧、34 个标签表面。所有检查的标签表面均闭合、边与顶点流形、可定向、无自交，且无重复三角形。存在差异的 9 个帧如下，其余帧各标签均一致：

| 帧 | 图标注 genus | 处理后表面 genus |
| --- | --- | --- |
| 40、41、45、47、49 | 1 | 2 |
| 46 | 1 | 3 |
| 48 | 1 | 4 |
| 52 | 2 | 1 |
| 61（仅标签 4） | 1 | 2 |

差异出现后又消失，并非单一固定延迟。该结果仍不能定位差异源自粒子几何、图判定还是表面处理。

核对 [原论文 §4.2](https://arxiv.org/html/2404.12524v1) 后，应进一步区分：论文的模拟标注通过反事实分离扰动判断物理连接，以区分暂时接触与真正合并；处理后静态表面的连通性和 genus 并不自动等同于这种物理拓扑。不能用本次 Euler 审计直接覆盖图标注，也不能仅凭静态表面差异判定标注有误。保留两套结果并注明定义。第 50、55、60、62 帧的既有表面审计与各标签图标注一致，已有 AE 失败仍需单独记录。

下一步使用 `scripts/check_level3_goal_score.py`，在 CPU 上复用缓存，比较第 62 帧目标 latent 与实际观测状态、AE 重建、单步和连续预测的分数。无需重新申请 GPU 或模拟，上传这一个脚本即可：

```bash
cd ~/projects/doughnet
source ~/venvs/doughnet/bin/activate
DYN_ARTIFACTS="records/level3_pointcloud/sim_check_FhY2DG/dyn_check_ml9xhm/artifacts"
GOAL_CHECK_DIR=$(mktemp -d records/level3_goal_score_XXXXXX)
set -o pipefail
python -u scripts/check_level3_goal_score.py \
  --dyn-artifacts "$DYN_ARTIFACTS" \
  --output "$GOAL_CHECK_DIR/artifacts" \
  2>&1 | tee "$GOAL_CHECK_DIR/run.log"
GOAL_EXIT=${PIPESTATUS[0]}
echo "goal_score_exit_code=$GOAL_EXIT"
if [ "$GOAL_EXIT" -eq 0 ]; then
  cat "$GOAL_CHECK_DIR/artifacts/summary.json"
  cat "$GOAL_CHECK_DIR/artifacts/goal_scores.csv"
fi
```

脚本在 `records/level3_ae_check_*/artifacts` 和本次模拟目录的 `ae_check_*/artifacts` 下寻找与当前 Dyn 路径及四个缓存文件哈希匹配的成功 AE 结果。只有一个匹配项时自动选用；多项或无匹配项时列出情况，需用 `--ae-artifacts` 指定既有成功目录。它不接受空路径作为项目目录，不混用其他场景的缓存。逐帧继续核对查询网格、目标标签、参考 latent、工具掩码和预测时间间隔，所有输入文件记录 SHA-256。

输出 `goal_scores.csv`、`summary.json`、`metadata.json`。评分分别使用对应 token 的 cosine 均值、全部 latent 展平后的 cosine、最后一个 token 的 cosine。原论文 §5.3 给出 latent cosine 评分，但未明确 257×512 数组的聚合方式，仓库也未找到原始规划器入口；这三种是诊断选项，尚不能称为原作者实现。最后一个 token 在 completer 自注意力之前由聚合结果池化生成，不能当作最终所有 token 的算术均值。

几何列 `goal_occupancy_iou_pct` 比较当前状态/解码预测与第 62 帧真实目标的占据重叠，忽略部件编号，排除两帧工具及桌面掩码的并集。它使用缓存网格，不是完整表面距离、拓扑成功率或官方 benchmark 指标；不同帧的有效比较区域可能不同，因此还输出有效网格点数和目标占据点数。第 60 帧预测比第 62 帧目标早 0.064 秒，应保留该时间差。

汇总按模式分别报告评分与目标几何重叠度的秩相关、最高分帧及最高几何重叠帧；排除目标自比较帧，避免把 cosine=1 的平凡情况计入排序。同时保留目标自比较和目标 AE 重建，以区分 latent 自相似和解码质量。实际轨迹不一定单调接近目标，评分不单调本身不证明评分错误；一条已记录轨迹的相关性也不能证明不同候选动作的排序能力。

`GOAL_SCORE_DIAGNOSTIC_COMPLETE` 仅表示缓存核对和分数导出完成。评分高不能解释为成功概率；几何 IoU 高也不保证 genus 或组件数正确。收到结果后固定并记录规划评分约定，再实现动作参数化、CEM、同预算随机搜索对照和新动作的模拟执行。

### 已核实的论文规划参数及待补事项

[原论文 §5.3](https://arxiv.org/html/2404.12524v1) 写明初始候选数 64、精英保留数 8、迭代 10 轮；工具类别为窄/常规/宽，初始概率各 1/3，保留精英并重估分布。平移搜索范围为两个平面方向各 ±40 mm，平面旋转范围为 ±10°。文中分布写为 `N_t(0, I·8)` 与 `N_theta(0, 10)`，尚未核实其尺度是方差还是标准差，不能直接将 8、10 写成采样标准差。

还需核实平移与旋转相对哪个工具初始姿态、三种工具几何与数据的对应关系、最终闭合宽度的搜索范围、停止规则和模型预测时域。当前固定动作的目标宽度 0.005 不等于论文规划搜索范围；63 帧记录也不等于 CEM 的官方规划时域。若无法从公开材料取得某些细节，明确记录重实现的选择，不宣称参数完全等同原实现。

本阶段尚缺三个实际产物：候选动作搜索及每轮精英/分数日志；从相同初始状态执行最优新动作及同预算随机基线；真实执行结果与目标的几何、组件和 genus 评估及归档。上述接口检查、轨迹评分和审计均不能替代这些规划闭环实验。

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

# 天翼(TienYi)真机推理部署方案（pi05_tienyi_eggplant_delta）

> 2026-07-06（v1）。基于天工方案（`docs/推理部署方案.md`）适配天翼茄子放盘任务训练权重。
> 交付脚本：`scripts/deploy/tienyi/serve_http.py`、`scripts/deploy/tienyi/robot_client.py`、`scripts/deploy/tienyi/verify_serve_http.py`。
> 与天工共用 xRocs 双臂/夹爪/相机栈，脚本以天工三件套为基线，仅改任务绑定项。

## 一、与天工方案的关系

天工和天翼共用同一套 xRocs 双臂（7+7）+ 软夹爪 + 3 相机栈，关节限位、控制安全机制、全链路统一训练布局 `[L7, R7, Lg, Rg]` 完全一致。因此本方案只列**与天工不同的任务绑定项**，其余（数据流、布局、delta 还原、norm stats 自动加载、HTTP 协议、执行安全机制）见 `docs/推理部署方案.md`。

## 二、任务绑定项（与天工的差异）

| 项 | 天工 lint-roller | 天翼 eggplant | 依据 |
| --- | --- | --- | --- |
| config | `pi05_tienkung_lint_roller_delta` | `pi05_tienyi_eggplant_delta` | `src/openpi/training/config.py` |
| 数据集 repo_id | `tienkung_lint_roller_lerobot` | `tienyi_eggplant_lerobot` | norm stats 自动从 `<ckpt>/assets/tienyi_eggplant_lerobot/` 读取 |
| 最优 checkpoint | 29999 | **99999** | 离线评测单调下降，见 agent.md |
| prompt（逐字） | Use an electrostatic roller... | `Place the eggplant on the plate` | `meta/tasks.jsonl` |
| 双臂运动 | 仅右臂（左臂静止） | **双臂都动** | 数据实测；verify 断言相应放宽 |
| `COLLECTION_INIT_POSE` | lint-roller 首帧均值 | eggplant 首帧均值（见下） | 554 条 episode 首帧均值 |
| 相机键 | top/left/right | top/left/right（俯视相机数据里叫 `camera_top`，由 `camera_head` 重命名而来） | client 兼容 top/head |
| `CONTROL_DT` | 1/29 | 1/29（相同，fps=29） | 数据集 fps |

训练 prompt（逐字，来自 `datasets/tienyi_eggplant_lerobot/meta/tasks.jsonl`）：

```
Place the eggplant on the plate
```

`COLLECTION_INIT_POSE`（**训练布局 [L7, R7, Lg, Rg]**，554 条 episode 首帧均值实算，夹爪恒 0.011765）：

```python
COLLECTION_INIT_POSE = np.asarray([
    # 左臂 0:7
    0.122871, 0.116640, -0.129596, -1.853367, -1.278654, -0.054483, -0.000901,
    # 右臂 7:14
    0.145657, -0.131415,  0.346642, -1.964025,  1.511000,  0.265127,  0.115378,
    # 左夹爪 14, 右夹爪 15
    0.011765, 0.011765,
], dtype=np.float64)
```

## 三、相机命名兼容（top/head）

训练数据集把平台俯视相机 `camera_head` 重命名为 `camera_top`（见 `scripts/convert_tienyi_v3_to_v21.py`），因此模型侧只认 `camera_top/left/right`。真机上俯视相机可能仍叫 `head`，client 的 `_pick_camera` 已对 `top` 增加 `head` 别名（先精确匹配、再模糊匹配 `top`/`head`），无论 robot 暴露为 top 还是 head 都能取到。若真机相机键完全不同，需在真机 dry-run 时核对 `相机: [...]` 打印并调整。

## 四、本地验证（无机器人，GPU 机上完成）

`scripts/deploy/tienyi/verify_serve_http.py`（硬断言，失败非零退出；内置 `HF_LEROBOT_HOME` 默认值）：

```bash
# server 起在 18000 后：
JAX_PLATFORMS=cpu uv run python scripts/deploy/tienyi/verify_serve_http.py
# 可选参数：--server / --episode / --config-name / --exp-name / --step
```

断言项（相对天工的差异）：**左右臂均用 `ARM_MAE_MAX=0.05`**（茄子双臂都动，不再假设左臂钉死）；运动错位护栏改为对“该窗口 GT 幅度更大的臂”做 pred/gt 幅度比检查；其余（chunk `[32,16]`、首步贴近 state、夹爪 [-0.1,1.1]、与 `eval_offline` 同帧交叉校验）保留。交叉校验默认指向 `checkpoints/pi05_tienyi_eggplant_delta/eggplant_delta_v1_8xa800/eval_offline/trajectories/step99999_ep553.npz`。

## 五、部署命令

```bash
# GPU 机
cd /mnt/cpk/magic/openpi
.venv/bin/python scripts/deploy/tienyi/serve_http.py \
    --config-name pi05_tienyi_eggplant_delta \
    --checkpoint-dir checkpoints/pi05_tienyi_eggplant_delta/eggplant_delta_v1_8xa800/99999 \
    --port 18000

# 机器人本体（旧项目环境）
python scripts/deploy/tienyi/robot_client.py \
    --infer-url http://<GPU机IP>:18000/inference \
    --task "Place the eggplant on the plate"

# 真机前 dry-run（不向机器人下发任何指令，仅打印）
python scripts/deploy/tienyi/robot_client.py --infer-url http://<GPU机IP>:18000/inference --dry-run
```

## 六、风险与备注

- 与天工方案共通的风险（prompt 逐字、控制频率、JPEG 压缩、blocking 延迟、`.venv` 迁移）见 `docs/推理部署方案.md` 第六节。
- 双臂都动：真机首跑仍用 `CHUNK_CONFIRM=True` 逐块确认，重点观察左右臂是否都跟随预测、无错位。
- 相机键：真机 dry-run 时核对 `_pick_camera` 是否正确取到俯视相机（top 或 head）。
- 关节顺序：dry-run 核对 `_get_state_for_server()` 读数与 `COLLECTION_INIT_POSE` 最大偏差（prepare 已自动打印）。

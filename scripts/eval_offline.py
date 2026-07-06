"""Offline open-loop evaluation for TienKung fine-tuned checkpoints.

Mimics the reference eval (eval_pertask_actioncond_fix): slide a window over dataset
episodes with a fixed stride, run policy inference on the observation at each window
start, and compare the predicted action chunk against the ground-truth chunk.

Outputs (under <exp_dir>/eval_offline/ by default):
- metrics_sweep.json: overall / per-checkpoint / per-episode MSE & MAE, plus per-group
  (left arm / right arm / grippers) breakdowns in raw physical units (rad).
- plots/step<N>_ep<E>.png: per-dim predicted vs ground-truth trajectories.
- trajectories/step<N>_ep<E>.npz: raw predicted/GT chunks and window indices.

Example:
    HF_LEROBOT_HOME=/mnt/cpk/magic/openpi/datasets python scripts/eval_offline.py \
        --config-name pi05_tienkung_lint_roller_delta \
        --exp-name lint_roller_delta_v1_8xa800
"""

import dataclasses
import gc
import json
import logging
import pathlib

import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.transforms as _transforms

DIM_LABELS = (
    [f"L_arm{i}" for i in range(7)] + [f"R_arm{i}" for i in range(7)] + ["L_grip", "R_grip"]
)
DIM_GROUPS = {"left_arm": slice(0, 7), "right_arm": slice(7, 14), "grippers": slice(14, 16)}


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_tienkung_lint_roller_delta"
    exp_name: str = "lint_roller_delta_v1_8xa800"
    # Checkpoint steps to evaluate; empty means all steps found in the experiment directory.
    steps: tuple[int, ...] = ()
    # Episode indices to evaluate; empty means the last `n_episodes` episodes.
    episodes: tuple[int, ...] = ()
    n_episodes: int = 5
    # Frames between consecutive window starts (matches the reference eval).
    stride: int = 16
    output_dir: str | None = None
    seed: int = 0


def build_eval_dataset(config: _config.TrainConfig):
    """Dataset yielding repacked samples: obs keys + absolute GT action chunk [horizon, 16]."""
    data_config = config.data.create(config.assets_dirs, config.model)
    horizon = config.model.action_horizon

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(horizon)] for key in data_config.action_sequence_keys
        },
    )
    transforms = []
    if data_config.prompt_from_task:
        transforms.append(_transforms.PromptFromLeRobotTask(dataset_meta.tasks))
    transforms.extend(data_config.repack_transforms.inputs)
    transform = _transforms.compose(transforms)

    episode_bounds = {
        ep: (int(dataset.episode_data_index["from"][ep]), int(dataset.episode_data_index["to"][ep]))
        for ep in range(dataset_meta.total_episodes)
    }
    return dataset, transform, episode_bounds, horizon


def eval_checkpoint(
    config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    dataset,
    transform,
    episode_bounds: dict[int, tuple[int, int]],
    episodes: list[int],
    horizon: int,
    stride: int,
    seed: int,
    plots_dir: pathlib.Path,
    traj_dir: pathlib.Path,
    step: int,
) -> dict:
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    # Fixed rng so different checkpoints see identical flow-matching noise.
    policy._rng = jax.random.key(seed)  # noqa: SLF001

    per_episode = {}
    all_sq_err = []  # per-window squared error arrays [horizon, 16]
    for ep in episodes:
        ep_from, ep_to = episode_bounds[ep]
        # Windows whose action chunk stays inside the episode.
        starts = list(range(ep_from, ep_to - horizon + 1, stride))
        preds, gts = [], []
        for start in starts:
            sample = transform(dataset[start])
            gt = np.asarray(sample.pop("actions"), dtype=np.float64)  # [horizon, 16] absolute
            pred = np.asarray(policy.infer(sample)["actions"], dtype=np.float64)  # [horizon, 16]
            preds.append(pred)
            gts.append(gt)

        preds, gts = np.stack(preds), np.stack(gts)  # [n_windows, horizon, 16]
        sq_err = (preds - gts) ** 2
        abs_err = np.abs(preds - gts)
        all_sq_err.append(sq_err)

        per_episode[ep] = {
            "n_windows": len(starts),
            "avg_mse": float(sq_err.mean()),
            "avg_mae": float(abs_err.mean()),
            "per_group": {
                name: {"avg_mse": float(sq_err[..., sl].mean()), "avg_mae": float(abs_err[..., sl].mean())}
                for name, sl in DIM_GROUPS.items()
            },
        }

        np.savez_compressed(
            traj_dir / f"step{step}_ep{ep}.npz",
            pred_chunks=preds.astype(np.float32),
            gt_chunks=gts.astype(np.float32),
            window_starts=np.asarray(starts) - ep_from,  # frame index within the episode
            stride=stride,
        )
        plot_episode(preds, gts, np.asarray(starts) - ep_from, stride, plots_dir / f"step{step}_ep{ep}.png", step, ep)
        logging.info(
            f"step {step} ep {ep}: {len(starts)} windows, "
            f"mse={per_episode[ep]['avg_mse']:.6f}, right_arm_mse={per_episode[ep]['per_group']['right_arm']['avg_mse']:.6f}"
        )

    del policy
    gc.collect()

    all_sq = np.concatenate(all_sq_err)
    return {
        "n_total_windows": int(all_sq.shape[0]),
        "avg_per_window_mse": float(all_sq.mean()),
        "avg_per_window_mae": float(np.sqrt(all_sq).mean()),
        "per_group": {
            name: {"avg_mse": float(all_sq[..., sl].mean()), "avg_mae": float(np.sqrt(all_sq[..., sl]).mean())}
            for name, sl in DIM_GROUPS.items()
        },
        "per_episode": per_episode,
    }


def plot_episode(
    preds: np.ndarray,
    gts: np.ndarray,
    window_starts: np.ndarray,
    stride: int,
    out_path: pathlib.Path,
    step: int,
    ep: int,
) -> None:
    """Per-dim GT trajectory vs receding-horizon prediction (first `stride` steps of each chunk)."""
    # Stitch executed trajectory: first `stride` steps of each window's prediction.
    exec_frames, exec_pred, exec_gt = [], [], []
    for w, start in enumerate(window_starts):
        n_exec = min(stride, preds.shape[1])
        exec_frames.append(np.arange(start, start + n_exec))
        exec_pred.append(preds[w, :n_exec])
        exec_gt.append(gts[w, :n_exec])
    frames = np.concatenate(exec_frames)
    pred_traj = np.concatenate(exec_pred)
    gt_traj = np.concatenate(exec_gt)

    fig, axes = plt.subplots(4, 4, figsize=(20, 12), sharex=True)
    for dim in range(16):
        ax = axes[dim // 4][dim % 4]
        ax.plot(frames, gt_traj[:, dim], label="GT", color="black", lw=1.2)
        ax.plot(frames, pred_traj[:, dim], label="Pred", color="tab:red", lw=1.0, alpha=0.8)
        ax.set_title(DIM_LABELS[dim], fontsize=10)
        if dim == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"step {step} | episode {ep} | open-loop (stride={stride})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = _config.get_config(args.config_name)
    exp_dir = pathlib.Path("checkpoints") / args.config_name / args.exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    steps = list(args.steps) or sorted(int(p.name) for p in exp_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"No checkpoint steps found in {exp_dir}")

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else exp_dir / "eval_offline"
    plots_dir, traj_dir = out_dir / "plots", out_dir / "trajectories"
    plots_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    dataset, transform, episode_bounds, horizon = build_eval_dataset(config)
    total_episodes = len(episode_bounds)
    episodes = list(args.episodes) or list(range(total_episodes - args.n_episodes, total_episodes))
    logging.info(f"Evaluating steps {steps} on episodes {episodes} (stride={args.stride}, horizon={horizon})")

    per_checkpoint = {}
    for step in steps:
        logging.info(f"===== checkpoint step {step} =====")
        per_checkpoint[str(step)] = eval_checkpoint(
            config,
            exp_dir / str(step),
            dataset,
            transform,
            episode_bounds,
            episodes,
            horizon,
            args.stride,
            args.seed,
            plots_dir,
            traj_dir,
            step,
        )

    metrics = {
        "model": f"{args.config_name}/{args.exp_name}",
        "mode": "offline_open_loop_sweep",
        "action_space": "absolute joint positions (rad) + gripper opening",
        "stride": args.stride,
        "action_horizon": horizon,
        "episodes": episodes,
        "note": (
            "Per-window MSE/MAE computed in raw physical units over all horizon steps and 16 dims. "
            "All episodes were part of the training set (train-set open-loop error, not generalization). "
            "per_group splits: left_arm dims 0:7 (idle in this task), right_arm dims 7:14, grippers dims 14:16."
        ),
        "per_checkpoint": per_checkpoint,
    }
    metrics_path = out_dir / "metrics_sweep.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logging.info(f"Saved metrics to {metrics_path}")

    # Summary table on stdout.
    print(f"\n{'step':>8} {'mse':>12} {'mae':>12} {'right_arm_mse':>14} {'gripper_mse':>12}")
    for step in steps:
        m = per_checkpoint[str(step)]
        print(
            f"{step:>8} {m['avg_per_window_mse']:>12.6f} {m['avg_per_window_mae']:>12.6f} "
            f"{m['per_group']['right_arm']['avg_mse']:>14.6f} {m['per_group']['grippers']['avg_mse']:>12.6f}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))

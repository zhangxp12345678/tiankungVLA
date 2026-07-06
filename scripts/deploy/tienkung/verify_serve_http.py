"""Local verification for serve_tienkung_http.py (no robot needed).

Takes a real dataset frame, sends it through the HTTP channel exactly like the robot
client would (JPEG b64 images + training-layout state), then hard-asserts:
1. response shape [horizon, 16]
2. layout correctness: right-arm motion lands in dims 7:14, left arm 0:7 stays pinned,
   grippers 14:16 in range
3. first prediction step stays near the current state (delta-model property)
4. consistency with the offline eval trajectory for the same frame (if available)

Exits non-zero on any failed check.

Usage:
    uv run python scripts/verify_serve_http.py            # server on :18000 (docs default)
    uv run python scripts/verify_serve_http.py --server http://127.0.0.1:18001/inference
"""

import argparse
import base64
import json
import os
import pathlib
import sys
import time
import urllib.request

# Default to the local dataset copy so the script works offline (must be set before lerobot import).
os.environ.setdefault("HF_LEROBOT_HOME", "/mnt/cpk/magic/openpi/datasets")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")

import cv2
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np

import openpi.training.config as _config
import openpi.transforms as _transforms

PROMPT = "Use an electrostatic roller to pick up debris on the desktop"

# Assertion thresholds (rad). Measured spread across flow-matching noise: right-arm
# MAE 0.007~0.038, first-step deviation ~0.006.
RIGHT_ARM_MAE_MAX = 0.05
LEFT_ARM_MAE_MAX = 0.01
FIRST_STEP_DEV_MAX = 0.05

_failures: list[str] = []


def check(name: str, *, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        _failures.append(name)


def to_b64_jpg(img) -> str:
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:  # CHW float [0,1] -> HWC uint8
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the TienKung HTTP inference server.")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:18000/inference",
        help="Inference endpoint (default matches docs deploy command).",
    )
    parser.add_argument("--config-name", default="pi05_tienkung_lint_roller_delta")
    parser.add_argument("--exp-name", default="lint_roller_delta_v1_8xa800")
    parser.add_argument("--step", type=int, default=29999, help="Checkpoint step for offline cross-check.")
    parser.add_argument("--episode", type=int, default=400)
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    horizon = config.model.action_horizon

    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={k: [t / meta.fps for t in range(horizon)] for k in data_config.action_sequence_keys},
    )
    transform = _transforms.compose(
        [_transforms.PromptFromLeRobotTask(meta.tasks), *data_config.repack_transforms.inputs]
    )

    ep_from = int(dataset.episode_data_index["from"][args.episode])
    sample = transform(dataset[ep_from])
    state = np.asarray(sample["observation/state"], dtype=np.float32)  # training layout
    gt = np.asarray(sample["actions"], dtype=np.float64)  # [horizon, 16] absolute

    payload = {
        "images": {
            "top": to_b64_jpg(sample["observation/images/camera_top"]),
            "left": to_b64_jpg(sample["observation/images/camera_left"]),
            "right": to_b64_jpg(sample["observation/images/camera_right"]),
        },
        "state": state.tolist(),
        "task": PROMPT,
        "request_id": "verify-001",
    }

    print(f"POST {args.server} (episode {args.episode}, frame {ep_from})")
    t0 = time.monotonic()
    req = urllib.request.Request(
        args.server, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    except Exception as exc:
        print(f"FAIL: cannot reach server at {args.server}: {exc}")
        return 1
    rtt = (time.monotonic() - t0) * 1000

    if resp.get("status") != "success":
        print(f"FAIL: server returned error: {resp}")
        return 1

    pred = np.asarray(resp["action_pred"], dtype=np.float64)
    print(f"HTTP OK, rtt={rtt:.0f} ms, policy infer={resp['policy_timing'].get('infer_ms', 0):.0f} ms\n")

    # --- hard checks ---
    check("chunk_shape", ok=pred.shape == (horizon, 16), detail=f"{pred.shape} (expect ({horizon}, 16))")

    left_mae = float(np.abs(pred[:, 0:7] - gt[:, 0:7]).mean())
    right_mae = float(np.abs(pred[:, 7:14] - gt[:, 7:14]).mean())
    check("right_arm_mae", ok=right_mae < RIGHT_ARM_MAE_MAX, detail=f"{right_mae:.5f} rad (< {RIGHT_ARM_MAE_MAX})")
    check("left_arm_mae", ok=left_mae < LEFT_ARM_MAE_MAX, detail=f"{left_mae:.5f} rad (< {LEFT_ARM_MAE_MAX})")

    # Layout misplacement guard: if the right arm moves in GT, the prediction must move
    # comparably in the SAME dims (a swapped layout leaves them near-constant).
    pred_r_range = float(np.ptp(pred[:, 7:14]))
    gt_r_range = float(np.ptp(gt[:, 7:14]))
    if gt_r_range > 0.05:
        ratio = pred_r_range / gt_r_range
        check("right_arm_motion_ratio", ok=0.5 < ratio < 2.0, detail=f"pred/gt range ratio {ratio:.2f} (0.5~2.0)")
    else:
        print(f"  [SKIP] right_arm_motion_ratio: gt range too small in this window ({gt_r_range:.4f})")

    first_dev = float(np.abs(pred[0] - state).max())
    check("first_step_near_state", ok=first_dev < FIRST_STEP_DEV_MAX, detail=f"{first_dev:.4f} rad (< {FIRST_STEP_DEV_MAX})")

    grip_ok = bool(np.all(pred[:, 14:16] > -0.1) and np.all(pred[:, 14:16] < 1.1))
    check(
        "gripper_range",
        ok=grip_ok,
        detail=f"min={pred[:, 14:16].min():.3f} max={pred[:, 14:16].max():.3f} ([-0.1, 1.1])",
    )

    # --- offline eval cross-check (informational + loose assertion) ---
    traj_file = (
        pathlib.Path("checkpoints")
        / args.config_name
        / args.exp_name
        / "eval_offline/trajectories"
        / f"step{args.step}_ep{args.episode}.npz"
    )
    if traj_file.exists():
        offline_pred = np.asarray(np.load(traj_file)["pred_chunks"][0], dtype=np.float64)
        mae_offline = float(np.abs(offline_pred - gt).mean())
        mae_http = float(np.abs(pred - gt).mean())
        # Same model + same frame: HTTP channel should not be dramatically worse than the
        # offline path (allow generous headroom for sampling noise).
        check(
            "consistent_with_offline_eval",
            ok=mae_http < max(5 * mae_offline, 0.05),
            detail=f"HTTP MAE {mae_http:.5f} vs offline {mae_offline:.5f}",
        )
    else:
        print(f"  [SKIP] consistent_with_offline_eval: {traj_file} not found")

    if _failures:
        print(f"\nFAILED checks: {_failures}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

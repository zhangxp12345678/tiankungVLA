"""HTTP inference server for the TienYi dual-arm robot (pi05_tienyi_* configs).

Structurally identical to the TienKung server (scripts/deploy/tienkung/serve_http.py):
the server is fully config-driven and has no robot-specific constants. The whole chain
uses the TRAINING layout [left arm 7, right arm 7, left gripper 1, right gripper 1] --
the state from the client is passed through to `observation/state` unchanged, and the
returned action chunk keeps the same layout. See docs/推理部署方案_tienyi.md.

Request (POST /inference, JSON):
    images: {"top": <b64 jpeg>, "left": <b64 jpeg>, "right": <b64 jpeg>}
    state: [16] float, training layout
    task: prompt string (must match the training prompt verbatim)
    request_id: optional passthrough

Response:
    action_pred: [horizon, 16] absolute joint positions (rad) + gripper opening,
                 training layout
"""

import argparse
import base64
import io
import logging
import pathlib
import time

from fastapi import FastAPI
from fastapi import Request
import numpy as np
from PIL import Image
import uvicorn

from openpi.policies import policy_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as config_loader

STATE_DIM = 16

app = FastAPI()
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Injected from __main__ before startup.
CONFIG_NAME: str | None = None
CHECKPOINT_DIR: pathlib.Path | None = None
ASSETS_DIR: pathlib.Path | None = None


@app.on_event("startup")
async def startup_event():
    if CONFIG_NAME is None or CHECKPOINT_DIR is None:
        raise RuntimeError("Server not initialized. Launch with --config-name and --checkpoint-dir.")

    logger.info("Loading config: %s", CONFIG_NAME)
    cfg = config_loader.get_config(CONFIG_NAME)
    logger.info("Loading checkpoint: %s", CHECKPOINT_DIR)

    norm_stats = None
    if ASSETS_DIR is not None:
        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
        norm_stats = _checkpoints.load_norm_stats(ASSETS_DIR, data_config.asset_id)
        logger.info("Loaded norm stats override from: %s", ASSETS_DIR)

    app.state.policy = policy_config.create_trained_policy(cfg, CHECKPOINT_DIR, norm_stats=norm_stats)
    logger.info("Policy loaded, ready for inference.")


def decode_image(b64_str: str) -> np.ndarray:
    if not b64_str:
        raise ValueError("Empty image payload")
    img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    return np.asarray(img)  # uint8 HWC RGB


def _pick_image_key(images_b64: dict, short_key: str) -> str:
    if short_key in images_b64:
        return short_key
    for key in images_b64:
        if short_key in str(key).lower():
            return key
    return ""


@app.post("/inference")
async def inference(request: Request):
    current_policy = getattr(app.state, "policy", None)
    if current_policy is None:
        return {"status": "error", "message": "Policy not loaded."}

    try:
        request_start = time.monotonic()
        server_received_ns = time.time_ns()
        data = await request.json()
        images_b64 = data.get("images", {})
        task = data.get("task", "")
        request_id = data.get("request_id")
        state = np.asarray(data.get("state", []), dtype=np.float32)

        if not isinstance(images_b64, dict):
            return {"status": "error", "message": "images must be a dict"}
        if state.shape != (STATE_DIM,):
            return {"status": "error", "message": f"Expected state shape ({STATE_DIM},), got {state.shape}"}

        logger.info(
            "inference request: task=%r image_keys=%s state_head=%s",
            task,
            list(images_b64.keys()),
            np.round(state[:4], 4).tolist(),
        )

        top_key = _pick_image_key(images_b64, "top")
        left_key = _pick_image_key(images_b64, "left")
        right_key = _pick_image_key(images_b64, "right")
        if not (top_key and left_key and right_key):
            return {
                "status": "error",
                "message": f"Missing camera image; need top/left/right, got {list(images_b64.keys())}",
            }

        # State is already in the training layout [L7, R7, Lg, Rg]: pass through unchanged.
        obs = {
            "observation/state": state,
            "observation/images/camera_top": decode_image(images_b64[top_key]),
            "observation/images/camera_left": decode_image(images_b64[left_key]),
            "observation/images/camera_right": decode_image(images_b64[right_key]),
            "prompt": task,
        }

        result = current_policy.infer(obs)
        action_chunk = np.asarray(result["actions"])  # [horizon, 16] training layout
        server_sent_ns = time.time_ns()

        return {
            "status": "success",
            "request_id": request_id,
            "action_pred": action_chunk.tolist(),
            "chunk_len": len(action_chunk),
            "policy_timing": result.get("policy_timing", {}),
            "server_timing": {
                "received_unix_ns": server_received_ns,
                "sent_unix_ns": server_sent_ns,
                "total_ms": (time.monotonic() - request_start) * 1000,
            },
        }
    except Exception as exc:
        logger.exception("Inference error")
        return {"status": "error", "message": str(exc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TienYi dual-arm HTTP inference server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=18000, help="Bind port.")
    parser.add_argument("--config-name", required=True, help="TrainConfig name, e.g. pi05_tienyi_eggplant_delta.")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=pathlib.Path,
        help="Checkpoint step directory, e.g. checkpoints/<config>/<exp>/99999.",
    )
    parser.add_argument(
        "--assets-dir",
        type=pathlib.Path,
        default=None,
        help="Optional norm stats assets dir override (default: <checkpoint-dir>/assets).",
    )
    args = parser.parse_args()

    CONFIG_NAME = args.config_name
    CHECKPOINT_DIR = args.checkpoint_dir
    ASSETS_DIR = args.assets_dir

    # Standard asyncio loop avoids uvloop conflicts with Orbax metadata loading.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", loop="asyncio")

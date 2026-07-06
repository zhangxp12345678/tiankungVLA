"""TienKung dual-arm (station, soft gripper, 3 cameras) policy transforms.

State/action layout (16 dims, all from puppet fields, no head):
    0:7   left arm joints (rad)
    7:14  right arm joints (rad)
    14    left gripper opening
    15    right gripper opening
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_tienkung_example() -> dict:
    """Creates a random input example for the TienKung policy."""
    return {
        "observation/state": np.random.rand(16),
        "observation/images/camera_top": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/images/camera_left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/images/camera_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "Use an electrostatic roller to pick up debris on the desktop",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class TienkungInputs(transforms.DataTransformFn):
    """Converts TienKung inputs to the model's expected format.

    Used for both training and inference. Expects the unified keys produced by
    `TienkungRepack` during training; the robot client must send the same keys
    at inference time:
    - observation/state: [16]
    - observation/images/camera_top | camera_left | camera_right
    - actions: [action_horizon, 16] (training only)
    - prompt
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # During training LeRobot decodes videos as float32 (C,H,W) in [0,1];
        # at inference the robot sends uint8 (H,W,C) and parsing is a no-op.
        base_image = _parse_image(data["observation/images/camera_top"])
        left_image = _parse_image(data["observation/images/camera_left"])
        right_image = _parse_image(data["observation/images/camera_right"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TienkungOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the TienKung 16-dim action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        # The model outputs are padded to action_dim=32; only the first 16 dims are real.
        return {"actions": np.asarray(data["actions"][..., :16])}

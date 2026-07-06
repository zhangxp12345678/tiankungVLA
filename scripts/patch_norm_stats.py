"""Apply a minimum-range floor to quantile norm stats.

For near-constant action/state dimensions (e.g. an unused arm), q99 - q01 can be
tiny (~1e-5). Quantile normalization divides by this range, so a rare small movement
gets amplified by orders of magnitude and produces huge loss spikes during training.

This script widens any dimension whose (q99 - q01) is below `min_range`, expanding
symmetrically around the midpoint. Stats are modified in place with a `.orig` backup.

Usage:
    python scripts/patch_norm_stats.py --config-name pi05_tienkung_lint_roller_delta [--min-range 0.01]

Run AFTER compute_norm_stats.py and BEFORE train.py. The patched stats are then
baked into training and saved into checkpoints automatically.
"""

import pathlib
import shutil

import numpy as np
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def _floor_range(norm_stats: dict[str, normalize.NormStats], min_range: float) -> dict[str, normalize.NormStats]:
    patched = {}
    for key, stats in norm_stats.items():
        q01, q99 = np.asarray(stats.q01, dtype=np.float64), np.asarray(stats.q99, dtype=np.float64)
        rng = q99 - q01
        narrow = rng < min_range
        if narrow.any():
            mid = (q01 + q99) / 2
            q01 = np.where(narrow, mid - min_range / 2, q01)
            q99 = np.where(narrow, mid + min_range / 2, q99)
            print(f"[{key}] widened dims {np.flatnonzero(narrow).tolist()} (range < {min_range})")
        patched[key] = normalize.NormStats(mean=stats.mean, std=stats.std, q01=q01, q99=q99)
    return patched


def main(config_name: str, min_range: float = 0.01):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    stats_dir = pathlib.Path(config.assets_dirs) / data_config.repo_id

    stats_file = stats_dir / "norm_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"{stats_file} not found -- run compute_norm_stats.py first.")

    backup = stats_file.with_suffix(".json.orig")
    if not backup.exists():
        shutil.copy2(stats_file, backup)
        print(f"Backup saved to: {backup}")

    norm_stats = normalize.load(stats_dir)
    patched = _floor_range(norm_stats, min_range)
    normalize.save(stats_dir, patched)
    print(f"Patched stats written to: {stats_file}")


if __name__ == "__main__":
    tyro.cli(main)

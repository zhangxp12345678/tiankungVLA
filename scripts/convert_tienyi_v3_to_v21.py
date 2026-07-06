"""Merge three TienYi v3.0-style LeRobot datasets into one local v2.1 dataset.

Sources (read-only, never modified):
    /media/unify/Special_Project/tienyi_prod2_dualArm-gripper-3cameras_20/
        tienyi_prod2_dualArm-gripper-3cameras_20_chufang_20260624/success/lerobot_RoboMIND
        tienyi_prod2_dualArm-gripper-3cameras_20_chufang_20260625/success/lerobot_RoboMIND
        tienyi_prod2_dualArm-gripper-3cameras_20_chufang_20260626/success/lerobot_RoboMIND

Output:
    <openpi>/datasets/tienyi_eggplant_lerobot   (repo_id="tienyi_eggplant_lerobot")

The sources use v3.0 metadata (flattened "stats/<feature>/<stat>" keys, file-based path
templates in info.json) but store per-episode parquet/mp4 files, so conversion is:
  1. renumber episode_index globally (source indices are non-contiguous: episodes deleted),
  2. rewrite the global "index" column and set task_index=0,
  3. rebuild v2.1 meta files (info.json / tasks.jsonl / episodes.jsonl / episodes_stats.jsonl),
  4. copy videos with renamed episode indices,
  5. rename the "camera_head" video key to "camera_top" so the existing TienkungRepack /
     TienkungInputs code (written for the lint-roller dataset) works unchanged.

Usage:
    python scripts/convert_tienyi_v3_to_v21.py [--dry-run] [--skip-videos]
"""

import argparse
import json
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SOURCE_BASE = pathlib.Path("/media/unify/Special_Project/tienyi_prod2_dualArm-gripper-3cameras_20")
SOURCES = [
    SOURCE_BASE / f"tienyi_prod2_dualArm-gripper-3cameras_20_chufang_{d}/success/lerobot_RoboMIND"
    for d in ("20260624", "20260625", "20260626")
]
OUTPUT = REPO_ROOT / "datasets" / "tienyi_eggplant_lerobot"

# Natural-language prompt replacing the internal task id "ur2_place_eggplant_to_plate".
PROMPT = "Place the eggplant on the plate"

# Source video key -> output video key. "camera_head" is the overhead camera on this platform;
# it is renamed to "camera_top" to match the key layout used by TienkungRepack/TienkungInputs.
VIDEO_KEY_MAP = {
    "camera_observations.color_images.camera_head": "camera_observations.color_images.camera_top",
    "camera_observations.color_images.camera_left": "camera_observations.color_images.camera_left",
    "camera_observations.color_images.camera_right": "camera_observations.color_images.camera_right",
}
# Stats kept in v2.1 episodes_stats.jsonl (source also has q01/q10/q50/q90/q99).
STAT_KEYS = ("min", "max", "mean", "std", "count")


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def scalar_stats(values: np.ndarray) -> dict:
    return {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(len(values))],
    }


def convert_episode_stats(flat: dict, new_index: int, global_start: int, length: int) -> dict:
    """v3.0 flattened 'stats/<feature>/<stat>' -> v2.1 nested stats dict."""
    stats: dict[str, dict] = {}
    for key, value in flat.items():
        if not key.startswith("stats/"):
            continue
        _, feature, stat = key.split("/", 2) if key.count("/") == 2 else (None, None, None)
        if feature is None or stat not in STAT_KEYS:
            continue
        feature = VIDEO_KEY_MAP.get(feature, feature)
        stats.setdefault(feature, {})[stat] = value
    # Index columns are renumbered, so their stats must be recomputed.
    stats["episode_index"] = scalar_stats(np.full(length, new_index))
    stats["index"] = scalar_stats(np.arange(global_start, global_start + length))
    stats["task_index"] = scalar_stats(np.zeros(length, dtype=np.int64))
    return stats


def rewrite_parquet(src: pathlib.Path, dst: pathlib.Path, new_index: int, global_start: int) -> int:
    table = pq.read_table(src)
    n = table.num_rows
    replacements = {
        "episode_index": pa.array(np.full(n, new_index, dtype=np.int64)),
        "index": pa.array(np.arange(global_start, global_start + n, dtype=np.int64)),
        "task_index": pa.array(np.zeros(n, dtype=np.int64)),
    }
    for name, arr in replacements.items():
        i = table.schema.get_field_index(name)
        table = table.set_column(i, table.schema.field(i).name, arr)
    pq.write_table(table, dst)
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Enumerate and validate only, write nothing.")
    parser.add_argument("--skip-videos", action="store_true", help="Skip video copying (for quick re-runs).")
    args = parser.parse_args()

    # 1. Validate sources are compatible.
    infos = [json.load(open(s / "meta" / "info.json")) for s in SOURCES]
    features = infos[0]["features"]
    for info in infos[1:]:
        assert info["features"] == features, "Source datasets have different features."
        assert info["fps"] == infos[0]["fps"], "Source datasets have different fps."
    fps = infos[0]["fps"]

    # 2. Enumerate episodes that actually exist on disk (some were deleted -> gaps).
    plan = []  # (source_root, src_index, length)
    for src, info in zip(SOURCES, infos):
        episodes = load_jsonl(src / "meta" / "episodes.jsonl")
        stats_lines = {line["episode_index"]: line for line in load_jsonl(src / "meta" / "episodes_stats.jsonl")}
        for ep in episodes:
            idx = ep["episode_index"]
            parquet = src / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
            videos = [src / "videos" / "chunk-000" / vk / f"episode_{idx:06d}.mp4" for vk in VIDEO_KEY_MAP]
            if not parquet.exists() or not all(v.exists() for v in videos):
                print(f"WARN: skipping {src.parent.parent.name} episode {idx} (missing files)")
                continue
            assert idx in stats_lines, f"missing stats for {src} ep {idx}"
            plan.append((src, idx, ep["length"], stats_lines[idx]))

    total_frames = sum(p[2] for p in plan)
    print(f"Merging {len(plan)} episodes, {total_frames} frames from {len(SOURCES)} sources -> {OUTPUT}")
    if args.dry_run:
        return

    # 3. Prepare output layout (single chunk: 554 < 1000).
    assert len(plan) <= 1000, "More than one chunk not implemented."
    if OUTPUT.exists():
        raise SystemExit(f"Output {OUTPUT} already exists; remove it first to re-convert.")
    (OUTPUT / "meta").mkdir(parents=True)
    (OUTPUT / "data" / "chunk-000").mkdir(parents=True)
    for vk_out in VIDEO_KEY_MAP.values():
        (OUTPUT / "videos" / "chunk-000" / vk_out).mkdir(parents=True)

    episodes_out, stats_out = [], []
    global_start = 0
    for new_index, (src, src_index, length, stats_line) in enumerate(tqdm.tqdm(plan, desc="episodes")):
        n = rewrite_parquet(
            src / "data" / "chunk-000" / f"episode_{src_index:06d}.parquet",
            OUTPUT / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet",
            new_index,
            global_start,
        )
        assert n == length, f"length mismatch: parquet={n} meta={length} ({src} ep {src_index})"
        if not args.skip_videos:
            for vk_src, vk_out in VIDEO_KEY_MAP.items():
                shutil.copyfile(
                    src / "videos" / "chunk-000" / vk_src / f"episode_{src_index:06d}.mp4",
                    OUTPUT / "videos" / "chunk-000" / vk_out / f"episode_{new_index:06d}.mp4",
                )
        episodes_out.append({"episode_index": new_index, "tasks": [PROMPT], "length": length})
        stats_out.append(
            {"episode_index": new_index, "stats": convert_episode_stats(stats_line, new_index, global_start, length)}
        )
        global_start += length

    # 4. Write v2.1 meta files.
    with open(OUTPUT / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": PROMPT}) + "\n")
    with open(OUTPUT / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes_out:
            f.write(json.dumps(ep) + "\n")
    with open(OUTPUT / "meta" / "episodes_stats.jsonl", "w") as f:
        for line in stats_out:
            f.write(json.dumps(line) + "\n")

    features_out = {VIDEO_KEY_MAP.get(k, k): v for k, v in features.items()}
    info_out = {
        "codebase_version": "v2.1",
        "robot_type": infos[0].get("robot_type"),
        "total_episodes": len(plan),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(plan) * len(VIDEO_KEY_MAP),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(plan)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features_out,
    }
    with open(OUTPUT / "meta" / "info.json", "w") as f:
        json.dump(info_out, f, indent=4)

    print(f"Done: {len(plan)} episodes, {total_frames} frames -> {OUTPUT}")


if __name__ == "__main__":
    main()

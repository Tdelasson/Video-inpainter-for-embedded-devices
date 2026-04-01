from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, ".")

from Metrics.official_eval import run_official_synthetic_eval

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class VideoSample:
    name: str
    frames: list[np.ndarray]
    masks: list[np.ndarray]
    dataset: str
    mask_type: str


def get_image_paths(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_davis_synthetic_subset(data_root: Path, limit: int) -> list[VideoSample]:
    frames_dir = data_root / "DAVIS" / "JPEGImages"
    masks_dir = data_root / "DAVIS" / "SyntheticMasks"

    with open(data_root / "DAVIS" / "test.json", encoding="utf-8") as f:
        video_names = sorted(json.load(f).keys())[:limit]

    videos = []
    for video_name in video_names:
        frame_paths = get_image_paths(frames_dir / video_name)
        mask_paths = get_image_paths(masks_dir / video_name)

        frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]

        masks = []
        for p in mask_paths:
            mask = np.array(Image.open(p))
            if mask.ndim == 3:
                mask = mask[..., 0]
            masks.append((mask > 127).astype(np.uint8))

        videos.append(
            VideoSample(
                name=video_name,
                frames=frames,
                masks=masks,
                dataset="DAVIS",
                mask_type="synthetic",
            )
        )

    return videos


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    official_eval_repo = Path.home() / "projects" / "video-inpainting-evaluation-public"

    videos = load_davis_synthetic_subset(
        data_root=repo_root / "Test_Data",
        limit=15,
    )

    metrics = run_official_synthetic_eval(
        videos=videos,
        pred_root=repo_root / "Results2" / "FuseFormer_OM" / "DAVIS" / "synthetic" / "_official_eval_pred",
        repo_root=official_eval_repo,
        output_size=(432, 240),
        metrics=("vfid", "warp_error_mask"),
    )

    print(metrics)


if __name__ == "__main__":
    main()

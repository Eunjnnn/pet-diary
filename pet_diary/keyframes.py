"""Extract representative keyframes from a video clip using ffmpeg.

Strategy: scene-change detection first; if the clip has fewer scene changes
than requested, fall back to uniform temporal sampling.
"""

import json
import subprocess
from pathlib import Path


def video_duration(video: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(video),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def extract_keyframes(
    video: Path,
    out_dir: Path,
    max_frames: int = 6,
    scene_threshold: float = 0.3,
    max_width: int = 1024,
) -> list[Path]:
    """Extract up to max_frames JPEG keyframes. Returns paths in time order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem

    # Pass 1: scene-change frames
    pattern = out_dir / f"{stem}_scene_%03d.jpg"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video),
            "-vf", f"select='gt(scene,{scene_threshold})',scale='min({max_width},iw)':-2",
            "-vsync", "vfr", "-frames:v", str(max_frames),
            str(pattern),
        ],
        check=True,
    )
    frames = sorted(out_dir.glob(f"{stem}_scene_*.jpg"))

    # Pass 2 fallback: uniform sampling when the clip is mostly one scene
    if len(frames) < max_frames:
        for f in frames:
            f.unlink()
        duration = video_duration(video)
        interval = duration / (max_frames + 1)
        for i in range(1, max_frames + 1):
            out_path = out_dir / f"{stem}_uniform_{i:03d}.jpg"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-ss", f"{interval * i:.3f}", "-i", str(video),
                    "-vf", f"scale='min({max_width},iw)':-2",
                    "-frames:v", "1", str(out_path),
                ],
                check=True,
            )
        frames = sorted(out_dir.glob(f"{stem}_uniform_*.jpg"))

    return frames

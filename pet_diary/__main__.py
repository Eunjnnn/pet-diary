"""Pet diary prototype pipeline.

Usage:
    python -m pet_diary <video_dir> [--out outputs] [--date 2026-07-21] [--max-frames 6]

For each video in <video_dir>: extract keyframes -> caption with Claude vision,
then compose all captions into one first-person diary entry.
"""

import argparse
import sys
from pathlib import Path

import anthropic

from .caption import caption_clip
from .diary import write_diary
from .keyframes import extract_keyframes

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a pet diary from video clips.")
    parser.add_argument("video_dir", type=Path, help="Directory containing video clips")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--date", default="오늘", help="Date label for the diary entry")
    parser.add_argument("--max-frames", type=int, default=6, help="Keyframes per clip")
    args = parser.parse_args()

    videos = sorted(p for p in args.video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"No videos found in {args.video_dir}", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()
    frames_dir = args.out / "keyframes"
    args.out.mkdir(parents=True, exist_ok=True)

    observations = []
    for video in videos:
        print(f"[keyframes] {video.name}")
        frames = extract_keyframes(video, frames_dir, max_frames=args.max_frames)
        print(f"[caption  ] {video.name} ({len(frames)} frames)")
        obs = caption_clip(client, video.stem, frames)
        observations.append(obs)
        (args.out / f"{video.stem}.caption.txt").write_text(obs, encoding="utf-8")

    print("[diary    ] composing entry")
    entry = write_diary(client, args.date, observations)
    diary_path = args.out / "diary.md"
    diary_path.write_text(entry, encoding="utf-8")
    print(f"\n=== {diary_path} ===\n")
    print(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())

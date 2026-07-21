"""Pet diary prototype pipeline.

Usage:
    python -m pet_diary <video_dir> [--backend local|claude] [--out outputs]
                        [--date 2026-07-21] [--max-frames 6]

For each video in <video_dir>: extract keyframes -> caption with a VLM,
then compose all captions into one first-person diary entry.

Backends:
    local  — Qwen2.5-VL on a local GPU (default; no API key needed)
    claude — Claude API (requires ANTHROPIC_API_KEY)
"""

import argparse
import sys
from pathlib import Path

from .keyframes import extract_keyframes
from .prompts import NO_PET_SENTINEL

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


def make_backend(name: str):
    if name == "claude":
        from .claude_backend import ClaudeBackend
        return ClaudeBackend()
    from .local_backend import LocalVLMBackend
    return LocalVLMBackend()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a pet diary from video clips.")
    parser.add_argument("video_dir", type=Path, help="Directory containing video clips")
    parser.add_argument("--backend", choices=["local", "claude"], default="local")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--date", default="오늘", help="Date label for the diary entry")
    parser.add_argument("--lang", choices=["ko", "en"], default="ko", help="Diary language")
    parser.add_argument("--max-frames", type=int, default=6, help="Keyframes per clip")
    args = parser.parse_args()

    videos = sorted(p for p in args.video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"No videos found in {args.video_dir}", file=sys.stderr)
        return 1

    print(f"[backend  ] loading '{args.backend}'")
    backend = make_backend(args.backend)

    frames_dir = args.out / "keyframes"
    args.out.mkdir(parents=True, exist_ok=True)

    observations = []
    for video in videos:
        print(f"[keyframes] {video.name}")
        frames = extract_keyframes(video, frames_dir, max_frames=args.max_frames)
        print(f"[caption  ] {video.name} ({len(frames)} frames)")
        obs = backend.caption_clip(video.stem, frames)
        (args.out / f"{video.stem}.caption.txt").write_text(obs, encoding="utf-8")
        if obs.strip().startswith(NO_PET_SENTINEL):
            print(f"[skip     ] {video.name}: no pet visible")
            continue
        observations.append(obs)

    if not observations:
        print("No clips with a pet visible — nothing to write about.", file=sys.stderr)
        return 1

    print("[diary    ] composing entry")
    entry = backend.write_diary(args.date, observations, lang=args.lang)
    diary_path = args.out / "diary.md"
    diary_path.write_text(entry, encoding="utf-8")
    print(f"\n=== {diary_path} ===\n")
    print(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())

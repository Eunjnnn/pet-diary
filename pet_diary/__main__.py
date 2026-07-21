"""Pet diary prototype pipeline.

Usage:
    python -m pet_diary <video_dir_or_file> [--backend local|claude] [--out outputs]
                        [--date 2026-07-21] [--lang ko|en] [--max-frames 6]

Input can be a directory of short clips, or a single long recording — in the
latter case motion events are detected first and each event becomes a clip.
For each clip: extract keyframes -> caption with a VLM, then compose all
captions into one first-person diary entry.

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
    parser.add_argument("video_path", type=Path,
                        help="Directory of video clips, or a single long recording")
    parser.add_argument("--backend", choices=["local", "claude"], default="local")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--date", default="오늘", help="Date label for the diary entry")
    parser.add_argument("--lang", choices=["ko", "en"], default="ko", help="Diary language")
    parser.add_argument("--max-frames", type=int, default=6, help="Keyframes per clip")
    args = parser.parse_args()

    time_labels: dict[str, str] = {}
    if args.video_path.is_file():
        from .events import detect_events, extract_event_clips

        print("[events   ] detecting motion events")
        events = detect_events(args.video_path)
        if not events:
            print("No motion events found in the recording.", file=sys.stderr)
            return 1
        videos = extract_event_clips(args.video_path, events, args.out / "events")
        time_labels = {c.stem: e.label() for c, e in zip(videos, events)}
        print(f"[events   ] {len(events)} event(s): "
              + ", ".join(e.label() for e in events))
    else:
        videos = sorted(p for p in args.video_path.iterdir()
                        if p.suffix.lower() in VIDEO_EXTS)
        if not videos:
            print(f"No videos found in {args.video_path}", file=sys.stderr)
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
        label = time_labels.get(video.stem)
        observations.append(f"[Time {label}] {obs}" if label else obs)

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

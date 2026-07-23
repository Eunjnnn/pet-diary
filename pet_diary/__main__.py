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

from .pipeline import generate_diary


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
    parser.add_argument("--name", default=None, help="Pet name (diary is written as this pet)")
    parser.add_argument("--max-frames", type=int, default=8, help="Keyframes per clip")
    args = parser.parse_args()

    print(f"[backend  ] loading '{args.backend}'")
    backend = make_backend(args.backend)

    try:
        result = generate_diary(
            args.video_path, backend, args.out,
            date_label=args.date, lang=args.lang, pet_name=args.name, max_frames=args.max_frames,
            log=lambda msg: print(f"[pipeline ] {msg}"),
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"\n=== {args.out / 'diary.md'} ===\n")
    print(result["diary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

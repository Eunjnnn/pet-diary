"""Reusable pipeline core shared by the CLI and the web app."""

from pathlib import Path
from typing import Callable

from .events import detect_events, extract_event_clips
from .keyframes import extract_keyframes, video_duration
from .prompts import NO_PET_SENTINEL

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
SHORT_CLIP_SEC = 15.0  # below this, skip motion analysis and use the clip as-is


def generate_diary(
    video_path: Path,
    backend,
    out_dir: Path,
    date_label: str = "오늘",
    lang: str = "ko",
    pet_name: str | None = None,
    max_frames: int = 8,
    log: Callable[[str], None] = lambda msg: None,
) -> dict:
    """Run the full pipeline for one recording file or a directory of clips.

    Returns {"diary", "captions", "keyframes", "time_labels", "skipped"}.
    Raises ValueError when there is nothing to write about.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    time_labels: dict[str, str] = {}
    if video_path.is_file():
        duration = video_duration(video_path)
        if duration <= SHORT_CLIP_SEC:
            # Too short for meaningful motion analysis (MOG2 warmup alone
            # eats the first second) — just treat it as one clip.
            log(f"Short recording ({duration:.0f}s) — using it as a single clip")
            videos = [video_path]
        else:
            log("Detecting motion events...")
            events = detect_events(video_path)
            if events:
                videos = extract_event_clips(video_path, events, out_dir / "events")
                time_labels = {c.stem: e.label() for c, e in zip(videos, events)}
                log(f"{len(events)} event(s): " + ", ".join(e.label() for e in events))
            else:
                # A still pet (sleeping, sitting) produces no motion — that is
                # still a valid, diary-worthy day. Fall back to uniform
                # snapshots over the whole recording.
                log("No motion detected — quiet recording; sampling snapshots instead")
                videos = [video_path]
    else:
        videos = sorted(
            p for p in video_path.iterdir() if p.suffix.lower() in VIDEO_EXTS
        )
        if not videos:
            raise ValueError(f"No videos found in {video_path}")

    observations: list[str] = []
    captions: dict[str, str] = {}
    keyframes: dict[str, list[Path]] = {}
    skipped: list[str] = []

    for video in videos:
        log(f"Extracting keyframes: {video.name}")
        frames = extract_keyframes(video, out_dir / "keyframes", max_frames=max_frames)
        keyframes[video.stem] = frames

        log(f"Watching clip: {video.name} ({len(frames)} frames)")
        obs = backend.caption_clip(video.stem, frames)
        captions[video.stem] = obs
        (out_dir / f"{video.stem}.caption.txt").write_text(obs, encoding="utf-8")

        if obs.strip().startswith(NO_PET_SENTINEL):
            log(f"Skipping {video.name}: no pet visible")
            skipped.append(video.stem)
            continue
        label = time_labels.get(video.stem)
        observations.append(f"[Time {label}] {obs}" if label else obs)

    if not observations:
        raise ValueError("No clips with a pet visible — nothing to write about.")

    log("Writing today's diary...")
    entry = backend.write_diary(date_label, observations, lang=lang, pet_name=pet_name)
    (out_dir / "diary.md").write_text(entry, encoding="utf-8")

    return {
        "diary": entry,
        "captions": captions,
        "keyframes": keyframes,
        "time_labels": time_labels,
        "skipped": skipped,
    }

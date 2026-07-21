"""Motion-triggered event segmentation for long home-camera recordings.

Detects motion with MOG2 background subtraction on downscaled, subsampled
frames, groups active moments into events, and cuts each event into its own
clip with ffmpeg. No deep-learning model involved — fast enough for
hours-long recordings.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ANALYSIS_WIDTH = 320  # frames are downscaled to this width before detection


@dataclass
class Event:
    start: float  # seconds, padding already applied
    end: float

    def label(self) -> str:
        def hms(t: float) -> str:
            t = int(t)
            return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"
        return f"{hms(self.start)}-{hms(self.end)}"


def detect_events(
    video: Path,
    sample_fps: float = 5.0,
    motion_threshold: float = 0.01,
    min_event_sec: float = 1.0,
    merge_gap_sec: float = 3.0,
    pad_sec: float = 1.0,
    warmup_sec: float = 1.0,
) -> list[Event]:
    """Return motion events (in seconds) found in the video.

    motion_threshold is the fraction of foreground pixels above which a
    sampled frame counts as "moving". Events closer than merge_gap_sec are
    merged; events shorter than min_event_sec are dropped; pad_sec is added
    on both sides of each event.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    step = max(1, round(fps / sample_fps))

    mog = cv2.createBackgroundSubtractorMOG2(
        history=int(sample_fps * 20), varThreshold=32, detectShadows=False
    )

    active_times: list[float] = []
    frame_idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = frame_idx / fps
            h, w = frame.shape[:2]
            scale = ANALYSIS_WIDTH / w
            small = cv2.resize(frame, (ANALYSIS_WIDTH, int(h * scale)))
            mask = mog.apply(small)
            ratio = float(np.count_nonzero(mask)) / mask.size
            # MOG2 needs a warmup period; everything is "foreground" at first.
            if t >= warmup_sec and ratio >= motion_threshold:
                active_times.append(t)
        frame_idx += 1
    cap.release()

    # Group active samples into events.
    events: list[Event] = []
    for t in active_times:
        if events and t - events[-1].end <= merge_gap_sec:
            events[-1].end = t
        else:
            events.append(Event(start=t, end=t))

    events = [e for e in events if e.end - e.start >= min_event_sec]
    for e in events:
        e.start = max(0.0, e.start - pad_sec)
        e.end = min(duration, e.end + pad_sec)
    return events


def extract_event_clips(video: Path, events: list[Event], out_dir: Path) -> list[Path]:
    """Cut each event into its own mp4 clip. Returns paths in time order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, e in enumerate(events, 1):
        clip = out_dir / f"event_{i:03d}_{e.label().replace(':', '')}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{e.start:.2f}", "-i", str(video),
                "-t", f"{e.end - e.start:.2f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an",
                str(clip),
            ],
            check=True,
        )
        clips.append(clip)
    return clips

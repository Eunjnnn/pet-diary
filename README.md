# pet-diary

An AI pet diary prototype: watches home-camera / phone video clips of your pet and writes a first-person diary entry about their day.

## Pipeline

```
video clips ──ffmpeg──▶ keyframes ──Claude vision──▶ per-clip observations ──Claude──▶ diary.md
```

1. **keyframes** — extracts up to 6 keyframes per clip via scene-change detection (falls back to uniform temporal sampling when the clip is mostly one scene)
2. **caption** — sends each clip's keyframes in a single vision request and produces a factual observation log for the clip
3. **diary** — composes all of the day's observations into one first-person diary entry (markdown)

Steps 2 and 3 call the Claude API (`claude-opus-4-8`), so an Anthropic API key is required. Step 1 runs fully locally.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

Requires `ffmpeg` / `ffprobe` on PATH.

## Run

```bash
.venv/bin/python -m pet_diary data/samples --out outputs --date 2026-07-21
```

Outputs:
- `outputs/keyframes/` — extracted keyframes
- `outputs/<clip>.caption.txt` — per-clip observation logs
- `outputs/diary.md` — the diary entry

## Sample data

Sample videos under `data/samples/` are from Wikimedia Commons (CC BY / CC BY-SA) — see [ATTRIBUTION.md](data/samples/ATTRIBUTION.md). The video files themselves are gitignored; re-download them from the linked source pages.

## Notes

- Diary output language is currently Korean (see the prompts in `pet_diary/caption.py` and `pet_diary/diary.py`).
- Planned: motion-triggered event clipping for long recordings, best-shot album selection, local VLM backend option.

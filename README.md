# 🐾 pet-diary

> *"Dear diary, today I chased the red dot again. I almost got it this time."* — 🐱

Ever wondered what your cat did all day while you were at work?
**pet-diary** watches your home-camera clips and writes your pet's day as a diary —
in your pet's own voice. 🐱🐶🐰

```
 /\_/\      ┌─────────────┐      ┌──────────────┐      📔
( o.o ) ──▶ │ 🎬 keyframes │ ──▶ │ 👀 VLM looks  │ ──▶  diary.md
 > ^ <      └─────────────┘      └──────────────┘
home cam        ffmpeg            Qwen2.5-VL / Claude
```

## ✨ What it does

1. 🎬 **keyframes** — picks up to 6 snapshots per clip (scene-change detection, with uniform sampling as fallback)
2. 👀 **caption** — a vision-language model looks at each clip and writes an observation log *("a grey tabby is hunting the laser dot...")*
3. 📔 **diary** — all observations become one first-person diary entry, written by your pet

Clips where no furry friend shows up are politely skipped. 🙈

## 🤖 Backends

| Backend | Model | Needs |
|---|---|---|
| `local` *(default)* | Qwen2.5-VL-7B on your GPU | ~16GB VRAM, no API key 🎉 |
| `claude` | Claude API (`claude-opus-4-8`) | `ANTHROPIC_API_KEY` |

## 🛠️ Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # + torch/transformers for local backend
```

Requires `ffmpeg` / `ffprobe` on PATH. 🍿

## 🚀 Run

```bash
.venv/bin/python -m pet_diary data/samples --date 2026-07-21 --lang ko
```

- `--lang ko|en` — diary language 🇰🇷/🇺🇸
- `--backend local|claude` — pick your brain 🧠
- `--max-frames 6` — snapshots per clip 📸

Then open `outputs/diary.md` and meet your pet's inner monologue:

> **고양이의 하루일기** 🐱
> 오늘은 정말 재미있었어! 아침부터 레이저 포인트를 쫓아 다녔어.
> 그 작은 빨간 점이 벽을 따라 올라가는 걸 볼 때마다 나도 모르게 몸이 움직였어. …
> **오늘의 기분:** 오늘은 참 행복한 하루였어! 💛

## 📁 Outputs

```
outputs/
├── keyframes/            📸 the snapshots
├── <clip>.caption.txt    👀 what the VLM saw
└── diary.md              📔 today's masterpiece
```

## 🐈 Sample data

Cat videos under `data/samples/` are from Wikimedia Commons (CC BY / CC BY-SA) —
see [ATTRIBUTION.md](data/samples/ATTRIBUTION.md). Video files are gitignored;
grab them from the linked source pages.

## 🗺️ Roadmap

- [ ] 🎥 Motion-triggered event clipping for long recordings (feed it a whole day!)
- [ ] 🏆 Best-shot album — the day's cutest moments, embedded in the diary
- [ ] 🐱🐱 Multi-pet identity ("was it Nabi or Coco who ate the churu?")
- [ ] 📬 Daily diary delivery via messenger bot

---

Made with 💖 (and a lot of cat videos)

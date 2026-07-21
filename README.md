# pet-diary

홈캠/폰 영상에서 반려동물의 하루를 관찰해 1인칭 일기로 써주는 AI 펫 다이어리 프로토타입.

## Pipeline

```
video clips ──ffmpeg──▶ keyframes ──Claude vision──▶ per-clip observations ──Claude──▶ diary.md
```

1. **keyframes** — 클립마다 장면 전환 감지(부족하면 균등 샘플링)로 키프레임 최대 6장 추출
2. **caption** — 키프레임들을 한 번의 vision 요청으로 보내 클립별 관찰 기록 생성
3. **diary** — 하루치 관찰 기록을 모아 반려동물 1인칭 일기(markdown) 작성

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

ffmpeg/ffprobe가 설치되어 있어야 합니다.

## Run

```bash
.venv/bin/python -m pet_diary data/samples --out outputs --date 2026-07-21
```

출력:
- `outputs/keyframes/` — 추출된 키프레임
- `outputs/<clip>.caption.txt` — 클립별 관찰 기록
- `outputs/diary.md` — 오늘의 일기

## Sample data

`data/samples/`의 영상 출처와 라이선스는 [ATTRIBUTION.md](data/samples/ATTRIBUTION.md) 참고 (Wikimedia Commons, CC BY / CC BY-SA).

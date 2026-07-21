"""Caption a video clip by sending its keyframes to Claude in one vision request."""

import base64
from pathlib import Path

import anthropic

MODEL = "claude-opus-4-8"

CAPTION_SYSTEM = (
    "You are an observer describing home-camera footage of pets. "
    "You receive keyframes sampled in time order from ONE video clip. "
    "Describe factually in Korean: which animals appear (species, coat color/pattern), "
    "what they are doing, where they are, and how the activity changes across frames. "
    "Do not invent details that are not visible."
)


def _image_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }


def caption_clip(client: anthropic.Anthropic, clip_name: str, frames: list[Path]) -> str:
    """Return a Korean description of what happens in the clip."""
    content: list[dict] = []
    for i, frame in enumerate(frames, 1):
        content.append({"type": "text", "text": f"[프레임 {i}/{len(frames)}]"})
        content.append(_image_block(frame))
    content.append({
        "type": "text",
        "text": (
            f"클립 이름: {clip_name}\n"
            "위 프레임들은 이 클립에서 시간 순서대로 뽑은 것입니다. "
            "이 클립에서 일어난 일을 3~5문장으로 관찰 기록처럼 요약해주세요."
        ),
    })

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=CAPTION_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return next(b.text for b in response.content if b.type == "text")

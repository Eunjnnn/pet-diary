"""Claude API backend (requires ANTHROPIC_API_KEY)."""

import base64
from pathlib import Path

import anthropic

from .prompts import CAPTION_SYSTEM, caption_user_text, diary_system, diary_user_text

MODEL = "claude-opus-4-8"


def _image_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }


class ClaudeBackend:
    def __init__(self, model: str = MODEL):
        self.client = anthropic.Anthropic()
        self.model = model

    def _text_of(self, response) -> str:
        return next(b.text for b in response.content if b.type == "text")

    def caption_clip(self, clip_name: str, frames: list[Path]) -> str:
        content: list[dict] = []
        for i, frame in enumerate(frames, 1):
            content.append({"type": "text", "text": f"[Frame {i}/{len(frames)}]"})
            content.append(_image_block(frame))
        content.append({"type": "text", "text": caption_user_text(clip_name)})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=CAPTION_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        return self._text_of(response)

    def write_diary(self, date_label: str, observations: list[str], lang: str = "ko") -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=diary_system(lang),
            messages=[{
                "role": "user",
                "content": diary_user_text(date_label, observations, lang),
            }],
        )
        return self._text_of(response)

"""Turn per-clip observations into a first-person pet diary entry."""

import anthropic

MODEL = "claude-opus-4-8"

DIARY_SYSTEM = (
    "You write a pet's diary in Korean, in the pet's first-person voice — "
    "playful, warm, a little self-important, like a cat narrating its own day. "
    "Base every event strictly on the provided observations; do not invent events. "
    "Output markdown: a title line, then 2-4 short paragraphs, "
    "and finish with a one-line '오늘의 기분' summary."
)


def write_diary(client: anthropic.Anthropic, date_label: str, observations: list[str]) -> str:
    """Compose one diary entry from a day's clip observations."""
    obs_text = "\n\n".join(
        f"[관찰 {i}]\n{obs}" for i, obs in enumerate(observations, 1)
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=DIARY_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"날짜: {date_label}\n\n"
                f"오늘 홈캠에 기록된 관찰 내용입니다:\n\n{obs_text}\n\n"
                "이 관찰들을 바탕으로 오늘의 일기를 써주세요."
            ),
        }],
    )
    return next(b.text for b in response.content if b.type == "text")

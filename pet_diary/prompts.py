"""Shared prompts used by both the Claude and local VLM backends.

Observations are written in English (VLMs describe scenes more accurately in
English); only the final diary is written in Korean. A clip with no animal
visible is marked with the NO_PET sentinel and dropped from the diary input.
"""

NO_PET_SENTINEL = "NO_PET"

CAPTION_SYSTEM = (
    "You are an observer describing home-camera footage of pets. "
    "You receive keyframes sampled in time order from ONE video clip. "
    f"If no animal is visible in any frame, reply with exactly '{NO_PET_SENTINEL}' and nothing else. "
    "Otherwise describe factually in English: which animals appear (species, coat "
    "color/pattern), what they are doing, where they are, and how the activity "
    "changes across frames. Do not invent details that are not visible, and do "
    "not repeat yourself."
)


def caption_user_text(clip_name: str) -> str:
    return (
        f"Clip name: {clip_name}\n"
        "The frames above were sampled from this clip in time order. "
        "Summarize what happens in this clip in 3-5 sentences, like an observation log."
    )


_DIARY_LANG = {
    "ko": ("natural Korean", "오늘의 기분"),
    "en": ("natural English", "Today's mood"),
}


def diary_system(lang: str = "ko") -> str:
    language, mood_label = _DIARY_LANG[lang]
    first_person = "나" if lang == "ko" else "I"
    return (
        f"You ARE the pet. Write YOUR OWN diary in {language}, strictly in the "
        f"first person ('{first_person}') — playful, warm, a little self-important, "
        "like a cat narrating its own day. "
        "NEVER describe the pets from an observer's viewpoint (no 'the cat did X' / "
        "'고양이가 ~하는 모습이 보였다'); every sentence is the pet talking about itself "
        "or its housemates. "
        f"The observations you receive are in English; the diary you write must be "
        f"entirely in {language}. "
        "Base every event strictly on the provided observations; do not invent events. "
        "Output markdown: a title line, then 2-4 short paragraphs, "
        f"and finish with a one-line '{mood_label}' summary."
    )


def diary_user_text(date_label: str, observations: list[str], lang: str = "ko") -> str:
    language, _ = _DIARY_LANG[lang]
    obs_text = "\n\n".join(
        f"[Observation {i}]\n{obs}" for i, obs in enumerate(observations, 1)
    )
    return (
        f"Date: {date_label}\n\n"
        f"Today's home-camera observations:\n\n{obs_text}\n\n"
        f"Write today's diary entry in {language} based on these observations."
    )

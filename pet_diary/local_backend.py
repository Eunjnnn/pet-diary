"""Local VLM backend using Qwen2.5-VL (runs fully offline on GPU)."""

from pathlib import Path

from .prompts import CAPTION_SYSTEM, DIARY_SYSTEM, caption_user_text, diary_user_text

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class LocalVLMBackend:
    def __init__(self, model_id: str = MODEL_ID):
        # Heavy imports kept local so `--backend claude` never needs torch.
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _chat(self, system: str, user_content: list[dict], max_new_tokens: int) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def caption_clip(self, clip_name: str, frames: list[Path]) -> str:
        content: list[dict] = []
        for i, frame in enumerate(frames, 1):
            content.append({"type": "text", "text": f"[프레임 {i}/{len(frames)}]"})
            content.append({"type": "image", "image": f"file://{frame.resolve()}"})
        content.append({"type": "text", "text": caption_user_text(clip_name)})
        return self._chat(CAPTION_SYSTEM, content, max_new_tokens=512)

    def write_diary(self, date_label: str, observations: list[str]) -> str:
        content = [{"type": "text", "text": diary_user_text(date_label, observations)}]
        return self._chat(DIARY_SYSTEM, content, max_new_tokens=1024)

"""VLM critic: compares a rendered figure against its text description and
produces the feedback observation for the next policy turn."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from PIL import Image

from .env import Critique

CRITIC_PROMPT = """You are reviewing a scientific figure rendered from TikZ code.

Target description:
{description}

Compare the rendered image against the description. Reply with JSON only:
{{"matches": true/false, "issues": ["issue 1", "issue 2", ...]}}

"matches" is true only if every object, label, color, and spatial relation in
the description appears correctly in the image. List concrete, actionable
issues otherwise (e.g. "node B should be below node A, it is to the right")."""


def parse_critique(text: str) -> Critique:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return Critique(matches=False, feedback=text.strip() or "Critique unavailable.")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Critique(matches=False, feedback=text.strip())
    issues = data.get("issues") or []
    feedback = (
        "The figure matches the description."
        if data.get("matches")
        else "The rendered figure does not match the description:\n- "
        + "\n- ".join(str(i) for i in issues)
    )
    return Critique(matches=bool(data.get("matches")), feedback=feedback)


class VLMCritic:
    """Qwen2.5-VL (or any mlx-vlm model) as judge of render vs. description."""

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
                 max_tokens: int = 500):
        from . import _mlx_compat  # noqa: F401  (must precede mlx imports)
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)
        self.max_tokens = max_tokens

    def critique(self, image: Image.Image, description: str) -> Critique:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(
            self.processor, self.config,
            CRITIC_PROMPT.format(description=description),
            num_images=1,
        )
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "render.png"
            image.save(img_path)
            result = generate(
                self.model, self.processor, prompt, image=[str(img_path)],
                max_tokens=self.max_tokens, verbose=False,
            )
        text = result.text if hasattr(result, "text") else str(result)
        return parse_critique(text)

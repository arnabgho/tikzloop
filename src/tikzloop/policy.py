"""Policy abstraction: description (+ feedback history) -> TikZ code.

MLXPolicy runs any HF chat model converted for MLX (mlx-community/* or a
local conversion of TikZilla). MockPolicy keeps tests model-free.
"""

from __future__ import annotations

import re
from typing import Protocol

SYSTEM_PROMPT = (
    "You are an expert in TikZ. Given a description of a scientific figure, "
    "output only a complete TikZ picture that faithfully renders it. "
    "Output a single ```latex code block containing either a full standalone "
    "document or a tikzpicture environment, and nothing else."
)

CODE_FENCE = re.compile(r"```(?:latex|tex|tikz)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull TikZ source out of a model response (fenced block, else raw)."""
    if m := CODE_FENCE.search(text):
        return m.group(1).strip()
    if "\\begin{tikzpicture}" in text:
        start = text.index("\\documentclass") if "\\documentclass" in text else \
            text.index("\\begin{tikzpicture}")
        return text[start:].strip()
    return text.strip()


def build_messages(description: str, history: list[tuple[str, str]]) -> list[dict]:
    """history: [(code, feedback), ...] from earlier turns of this episode."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Draw the following figure in TikZ:\n\n{description}"},
    ]
    for code, feedback in history:
        messages.append({"role": "assistant", "content": f"```latex\n{code}\n```"})
        messages.append(
            {"role": "user",
             "content": f"{feedback}\n\nProvide the corrected TikZ code."}
        )
    return messages


class Policy(Protocol):
    def generate(self, messages: list[dict]) -> str: ...


class MockPolicy:
    """Replays canned responses; for tests and dry runs."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages: list[dict]) -> str:
        resp = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return resp


class MLXPolicy:
    def __init__(self, model_id: str, max_tokens: int = 1500, temperature: float = 0.7):
        from . import _mlx_compat  # noqa: F401  (must precede mlx imports)
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, messages: list[dict]) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temperature),
            verbose=False,
        )

"""Gym-style multi-turn environment: TikZ code in, feedback + reward out.

The critic drives the loop (its feedback is the next-turn observation for
the policy). The reward function only scores against a ground-truth image
when one exists — it is never shown to the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from PIL import Image

from .sandbox import CompileResult, compile_tikz


class Critic(Protocol):
    def critique(self, image: Image.Image, description: str) -> "Critique": ...


@dataclass
class Critique:
    matches: bool
    feedback: str


class CompileOnlyCritic:
    """Cheapest critic: any compiled figure is accepted. Useful for tests
    and for pure compiler-feedback repair loops."""

    def critique(self, image: Image.Image, description: str) -> Critique:
        return Critique(matches=True, feedback="Figure compiled.")


@dataclass
class StepResult:
    turn: int
    code: str
    compiled: bool
    feedback: str
    done: bool
    image: Image.Image | None = field(default=None, repr=False)
    reward: float | None = None
    compile_seconds: float = 0.0


class TikZEnv:
    def __init__(
        self,
        critic: Critic | None = None,
        reward_fn: Callable[[Image.Image, Image.Image], float] | None = None,
        max_turns: int = 4,
        compile_timeout: float = 25.0,
    ):
        self.critic = critic or CompileOnlyCritic()
        self.reward_fn = reward_fn
        self.max_turns = max_turns
        self.compile_timeout = compile_timeout
        self.description: str = ""
        self.gt_image: Image.Image | None = None
        self.history: list[StepResult] = []

    def reset(self, description: str, gt_image: Image.Image | None = None) -> str:
        self.description = description
        self.gt_image = gt_image
        self.history = []
        return description

    def step(self, code: str) -> StepResult:
        turn = len(self.history) + 1
        result: CompileResult = compile_tikz(code, timeout=self.compile_timeout)
        out_of_turns = turn >= self.max_turns

        if not result.ok:
            step = StepResult(
                turn=turn,
                code=code,
                compiled=False,
                feedback=f"Compilation failed.\n{result.log}\nFix the TikZ code.",
                done=out_of_turns,
                reward=0.0 if self.reward_fn else None,
                compile_seconds=result.seconds,
            )
            self.history.append(step)
            return step

        crit = self.critic.critique(result.image, self.description)
        reward = None
        if self.reward_fn and self.gt_image is not None:
            reward = self.reward_fn(result.image, self.gt_image)

        step = StepResult(
            turn=turn,
            code=code,
            compiled=True,
            feedback=crit.feedback,
            done=crit.matches or out_of_turns,
            image=result.image,
            reward=reward,
            compile_seconds=result.seconds,
        )
        self.history.append(step)
        return step

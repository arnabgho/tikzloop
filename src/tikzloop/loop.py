"""Inference loops: multi-turn generate-compile-critique-repair, and best-of-N."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .env import StepResult, TikZEnv
from .policy import Policy, build_messages, extract_code
from .sandbox import compile_tikz


@dataclass
class Episode:
    description: str
    steps: list[StepResult] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return bool(self.steps) and self.steps[-1].compiled and self.steps[-1].done

    @property
    def final_code(self) -> str | None:
        for step in reversed(self.steps):
            if step.compiled:
                return step.code
        return None

    def to_jsonl_record(self) -> str:
        steps = [
            {k: v for k, v in asdict(s).items() if k != "image"}
            for s in self.steps
        ]
        return json.dumps({"description": self.description, "solved": self.solved,
                           "steps": steps})


def run_episode(env: TikZEnv, policy: Policy, description: str,
                gt_image=None, verbose: bool = False) -> Episode:
    """Multi-turn loop: the policy sees compiler errors and critic feedback
    from prior turns and revises until the critic accepts or turns run out."""
    env.reset(description, gt_image=gt_image)
    episode = Episode(description=description)
    feedback_history: list[tuple[str, str]] = []

    while True:
        messages = build_messages(description, feedback_history)
        code = extract_code(policy.generate(messages))
        step = env.step(code)
        episode.steps.append(step)
        if verbose:
            status = "compiled" if step.compiled else "compile error"
            print(f"[turn {step.turn}] {status} "
                  f"({step.compile_seconds:.1f}s) done={step.done} reward={step.reward}")
        if step.done:
            return episode
        feedback_history.append((code, step.feedback))


def best_of_n(policy: Policy, description: str, n: int, scorer=None) -> list[dict]:
    """Sample n independent programs, compile all, score compiled ones.

    scorer: (image) -> float, e.g. lambda im: reward(im, gt_image). Without a
    scorer, compiled candidates are simply preferred over failed ones.
    Returns candidates sorted best-first.
    """
    candidates = []
    messages = build_messages(description, [])
    for i in range(n):
        code = extract_code(policy.generate(messages))
        result = compile_tikz(code)
        score = None
        if result.ok and scorer is not None:
            score = scorer(result.image)
        candidates.append({"code": code, "compiled": result.ok, "score": score,
                           "image": result.image, "log": result.log})
    return sorted(
        candidates,
        key=lambda c: (c["compiled"], c["score"] if c["score"] is not None else 0.0),
        reverse=True,
    )


def append_jsonl(episode: Episode, path: Path | str) -> None:
    with open(path, "a") as f:
        f.write(episode.to_jsonl_record() + "\n")

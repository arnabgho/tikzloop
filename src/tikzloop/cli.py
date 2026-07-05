"""CLI entry points.

  tikzloop compile fig.tex          # sandbox smoke test
  tikzloop demo "a red circle..."   # full multi-turn loop with a real model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_compile(args) -> int:
    from .sandbox import compile_tikz

    code = Path(args.file).read_text()
    outdir = Path(args.outdir)
    result = compile_tikz(code, workdir=outdir)
    if result.ok:
        print(f"OK ({result.seconds:.1f}s) -> {result.png_path}")
        return 0
    print(f"FAILED ({result.seconds:.1f}s)\n{result.log}", file=sys.stderr)
    return 1


def cmd_demo(args) -> int:
    from .env import TikZEnv
    from .loop import run_episode
    from .policy import MLXPolicy

    print(f"Loading policy {args.model} ...")
    policy = MLXPolicy(args.model, temperature=args.temperature)

    if args.critic == "vlm":
        from .critic import VLMCritic

        print(f"Loading critic {args.critic_model} ...")
        critic = VLMCritic(args.critic_model)
    else:
        critic = None  # CompileOnlyCritic: loop ends on first successful compile

    env = TikZEnv(critic=critic, max_turns=args.max_turns)
    episode = run_episode(env, policy, args.description, verbose=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if episode.final_code:
        (outdir / "final.tex").write_text(episode.final_code)
        final = episode.steps[-1]
        if final.image is not None:
            final.image.save(outdir / "final.png")
        print(f"\nsolved={episode.solved} turns={len(episode.steps)} -> {outdir}/")
        return 0
    print(f"\nNo compiling program after {len(episode.steps)} turns.", file=sys.stderr)
    (outdir / "last_attempt.tex").write_text(episode.steps[-1].code)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="tikzloop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("compile", help="compile a .tex/.tikz file in the sandbox")
    p.add_argument("file")
    p.add_argument("--outdir", default="out")
    p.set_defaults(fn=cmd_compile)

    p = sub.add_parser("demo", help="run the multi-turn loop on one description")
    p.add_argument("description")
    p.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    p.add_argument("--critic", choices=["none", "vlm"], default="none")
    p.add_argument("--critic-model", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--outdir", default="out")
    p.set_defaults(fn=cmd_demo)

    args = parser.parse_args()
    sys.exit(args.fn(args))

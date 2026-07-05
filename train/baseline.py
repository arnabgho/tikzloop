"""Zero-shot baseline: the number the RL run has to beat.

Samples one completion per validation prompt from the untouched base model,
scores with the same reward as training, writes runs/baseline.json (the
dashboard draws it as a rule on the reward and compile-rate panels).

    uv run python train/baseline.py --model Qwen/Qwen2.5-3B-Instruct --n 16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    os.environ.setdefault("TIKZLOOP_RUN_DIR", str(ROOT / "runs" / "_baseline_scratch"))
    import tikzloop._mlx_compat  # noqa: F401
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rewards_tikz", ROOT / "train" / "rewards_tikz.py")
    rewards_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rewards_mod)

    rows = [json.loads(l) for l in
            (ROOT / "data" / "tikz" / "valid.jsonl").read_text().splitlines()]
    rows = rows[: args.n]

    print(f"loading {args.model} ...")
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=args.temperature)

    completions, answers = [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": row["system"]},
             {"role": "user", "content": row["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        out = generate(model, tokenizer, prompt=prompt,
                       max_tokens=args.max_tokens, sampler=sampler, verbose=False)
        completions.append(out)
        answers.append(row["answer"])
        print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)")

    rewards = rewards_mod.tikz_render_reward(
        prompts=[r["prompt"] for r in rows], completions=completions,
        answer=answers, types=["tikz"] * len(rows))

    compiled = sum(1 for r in rewards if r > 0)
    result = {
        "model": args.model,
        "n": len(rows),
        "compile_rate": round(compiled / len(rows), 4),
        "reward_mean": round(sum(rewards) / len(rewards), 4),
        "temperature": args.temperature,
    }
    out_path = ROOT / "runs" / "baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()

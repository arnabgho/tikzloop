"""GRPO reward function for text-to-TikZ + per-rollout telemetry.

Reward: 0.0 if the completion doesn't compile, else 0.1 + 0.9 * SigLIP-EMD
similarity between the rendered figure and the ground-truth image (whose path
arrives via the dataset's `answer` field).

Telemetry (the whole point of this run) goes to $TIKZLOOP_RUN_DIR:
  rollouts.jsonl   one row per rollout: compiled, sim, reward, ink fraction,
                   lengths, timing, error class + group-level stats
  samples/         contact sheets every SAMPLE_EVERY calls: GT + the group's
                   renders side by side, reward stamped on each

mlx-lm-lora calls this batched with batch_size*group_size items;
group size arrives via $TIKZLOOP_GROUP_SIZE (set by run_grpo.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function
from tikzloop.policy import extract_code
from tikzloop.sandbox import compile_tikz

RUN_DIR = Path(os.environ.get("TIKZLOOP_RUN_DIR", "runs/adhoc"))
GROUP_SIZE = int(os.environ.get("TIKZLOOP_GROUP_SIZE", "4"))
SAMPLE_EVERY = int(os.environ.get("TIKZLOOP_SAMPLE_EVERY", "25"))
FORMAT_FLOOR = 0.1

_pool = ThreadPoolExecutor(max_workers=6)
_gt_cache: dict[str, Image.Image] = {}
_sig = None
_call_idx = 0


def _scorer():
    """Lazy SigLIP; falls back to CPU if Metal/MPS memory is contended."""
    global _sig
    if _sig is None:
        from tikzloop.reward import SigLIPReward

        try:
            _sig = SigLIPReward(device="mps")
            _sig(Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64)))  # probe
        except Exception:
            _sig = SigLIPReward(device="cpu")
    return _sig


def _gt(path: str) -> Image.Image:
    if path not in _gt_cache:
        _gt_cache[path] = Image.open(path).convert("RGB")
        if len(_gt_cache) > 600:
            _gt_cache.pop(next(iter(_gt_cache)))
    return _gt_cache[path]


def _ink_fraction(img: Image.Image) -> float:
    """Share of non-near-white pixels — the blank-figure reward-hack tripwire."""
    a = np.asarray(img.convert("L").resize((128, 128)))
    return float((a < 245).mean())


def _error_class(result) -> str:
    if result is None:
        return "no-code"
    log = result.log or ""
    if "TIMEOUT" in log:
        return "timeout"
    if "REJECTED" in log:
        return "rejected"
    return "latex-error"


def _contact_sheet(gt: Image.Image, renders: list, rewards: list, path: Path,
                   tile: int = 220):
    """GT + G rollout renders in a row, reward stamped under each."""
    n = 1 + len(renders)
    sheet = Image.new("RGB", (tile * n, tile + 26), "white")
    draw = ImageDraw.Draw(sheet)

    def paste(img, i, label):
        cell = Image.new("RGB", (tile, tile), "white")
        if img is not None:
            im = img.copy()
            im.thumbnail((tile - 8, tile - 8))
            cell.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
        else:
            draw.text((i * tile + tile // 2 - 14, tile // 2), "FAIL", fill="#a4243b")
        sheet.paste(cell, (i * tile, 0))
        draw.text((i * tile + 6, tile + 6), label, fill="black")
        draw.rectangle([i * tile, 0, (i + 1) * tile - 1, tile + 25],
                       outline="#cccccc")

    paste(gt, 0, "ground truth")
    for j, (img, r) in enumerate(zip(renders, rewards)):
        paste(img, j + 1, f"rollout {j + 1}  r={r:.3f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _score_one(completion: str, gt_path: str):
    """Compile + render one rollout. Returns everything the log needs."""
    t0 = time.monotonic()
    code = extract_code(completion)
    # Guard against harness-level reward hacking: prose wrapped by
    # ensure_standalone compiles as a text document and scores well.
    is_tikz = "\\begin{tikzpicture}" in code or "\\begin{circuitikz}" in code \
        or "\\begin{tikzcd}" in code
    result = compile_tikz(code, timeout=20.0) if is_tikz else None
    ok = bool(result and result.ok and result.image is not None)
    return {
        "code": code,
        "result": result,
        "ok": ok,
        "compile_s": round(time.monotonic() - t0, 2),
    }


@register_reward_function()
def tikz_render_reward(prompts, completions, answer, types=None, **kwargs):
    global _call_idx
    _call_idx += 1
    t_call = time.monotonic()

    scored = list(_pool.map(_score_one, completions, answer))

    rewards, rows = [], []
    for i, (s, gt_path, completion) in enumerate(zip(scored, answer, completions)):
        sim = ink = None
        if s["ok"]:
            img = s["result"].image
            ink = _ink_fraction(img)
            sim = float(_scorer()(img, _gt(gt_path)))
            reward = FORMAT_FLOOR + (1 - FORMAT_FLOOR) * max(0.0, min(1.0, sim))
        else:
            reward = 0.0
        rewards.append(reward)
        rows.append({
            "call": _call_idx,
            "idx": i,
            "compiled": s["ok"],
            "sim": None if sim is None else round(sim, 4),
            "reward": round(reward, 4),
            "ink_frac": None if ink is None else round(ink, 4),
            "completion_chars": len(completion),
            "code_chars": len(s["code"]),
            "compile_s": s["compile_s"],
            "error": None if s["ok"] else _error_class(s["result"]),
        })

    # group-level stats (one group = GROUP_SIZE consecutive rollouts)
    groups = [rewards[g:g + GROUP_SIZE] for g in range(0, len(rewards), GROUP_SIZE)]
    all_fail = sum(1 for g in groups if max(g) == 0.0) / max(1, len(groups))
    zero_var = sum(1 for g in groups if len(set(g)) == 1) / max(1, len(groups))
    group_std = float(np.mean([np.std(g) for g in groups]))
    distinct = len({hashlib.md5(s["code"].encode()).hexdigest() for s in scored})

    summary = {
        "call": _call_idx,
        "kind": "group_summary",
        "n": len(rewards),
        "compile_rate": round(sum(r["compiled"] for r in rows) / len(rows), 4),
        "reward_mean": round(float(np.mean(rewards)), 4),
        "all_fail_group_frac": round(all_fail, 4),
        "zero_variance_group_frac": round(zero_var, 4),
        "within_group_reward_std": round(group_std, 4),
        "distinct_completion_ratio": round(distinct / len(scored), 4),
        "reward_phase_s": round(time.monotonic() - t_call, 2),
        "t": time.time(),
    }

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / "rollouts.jsonl").open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.write(json.dumps(summary) + "\n")

    if (_call_idx - 1) % SAMPLE_EVERY == 0:  # includes the very first call
        g0 = scored[:GROUP_SIZE]
        _contact_sheet(
            _gt(answer[0]),
            [s["result"].image if s["ok"] else None for s in g0],
            rewards[:GROUP_SIZE],
            RUN_DIR / "samples" / f"call_{_call_idx:05d}.png",
        )

    return rewards

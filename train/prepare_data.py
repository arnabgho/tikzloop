"""Build a GRPO training subset from DaTikZ-v3.

Streams nllg/datikz-v3, filters for usable caption+image pairs, saves ground
truth images to disk, and writes data/tikz/{train,valid}.jsonl in the format
mlx-lm-lora's GRPODataset expects: prompt / answer / system / type, where
`answer` carries the absolute path of the ground-truth PNG (the reward
function resolves it back to an image).

Usage:
    uv run python train/prepare_data.py --train 500 --valid 48
    uv run python train/prepare_data.py --redescribe   # VLM descriptions (slow)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "tikz"

# One clear instruction; per-record `system` is mandatory — without it,
# mlx-lm-lora injects an R1-style <think>/<answer> system prompt.
SYSTEM_PROMPT = (
    "You are an expert TikZ programmer. Given a description of a scientific "
    "figure, reply with a single ```latex code block containing a complete "
    "standalone TikZ picture that faithfully matches the description. "
    "Output only the code block."
)

MIN_CAPTION, MAX_CAPTION = 30, 600
MAX_CODE = 4000


def usable(row) -> bool:
    cap, code = (row.get("caption") or "").strip(), row.get("code") or ""
    return (
        MIN_CAPTION <= len(cap) <= MAX_CAPTION
        and len(code) <= MAX_CODE
        and row.get("image") is not None
    )


def redescribe(image, caption: str) -> str:
    """Optionally replace a raw caption with a VLM-generated description."""
    from tikzloop.critic import VLMCritic  # lazy: heavy import

    global _vlm
    if "_vlm" not in globals():
        _vlm = VLMCritic()
    prompt = (
        "Describe this scientific figure precisely so someone could redraw it "
        "in TikZ: every shape, label, color, and spatial relation. Reply with "
        "the description only."
    )
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    import tempfile

    tpl = apply_chat_template(_vlm.processor, _vlm.config, prompt, num_images=1)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "img.png"
        image.save(p)
        out = generate(_vlm.model, _vlm.processor, tpl, image=[str(p)],
                       max_tokens=400, verbose=False)
    return (out.text if hasattr(out, "text") else str(out)).strip() or caption


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--valid", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--redescribe", action="store_true",
                    help="replace captions with VLM descriptions (hours)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    from datasets import load_dataset  # after dotenv so HF_TOKEN applies

    n_needed = args.train + args.valid
    # Stream ~6x the needed rows, filter, then sample for code-length spread.
    ds = load_dataset("nllg/datikz-v3", split="train", streaming=True)
    pool = []
    for row in ds:
        if usable(row):
            pool.append(row)
        if len(pool) >= n_needed * 6:
            break
    print(f"pooled {len(pool)} usable rows")

    # Curriculum-friendly: sort by code length, sample evenly across strata so
    # simple figures (higher early compile odds) are well represented.
    pool.sort(key=lambda r: len(r["code"]))
    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(len(pool)), n_needed)) if len(pool) > n_needed \
        else list(range(len(pool)))
    rows = [pool[i] for i in idx]
    rng.shuffle(rows)

    gt_dir = DATA_DIR / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": rows[: args.train], "valid": rows[args.train:]}

    for split, split_rows in splits.items():
        out = DATA_DIR / f"{split}.jsonl"
        with out.open("w") as f:
            for i, row in enumerate(split_rows):
                img = row["image"].convert("RGB")
                gt_path = gt_dir / f"{split}_{i:05d}.png"
                img.save(gt_path)
                prompt = row["caption"].strip()
                if args.redescribe:
                    prompt = redescribe(img, prompt)
                f.write(json.dumps({
                    "prompt": prompt,
                    "answer": str(gt_path.resolve()),
                    "system": SYSTEM_PROMPT,
                    "type": "tikz",
                }) + "\n")
        print(f"{split}: {len(split_rows)} rows -> {out}")


if __name__ == "__main__":
    main()

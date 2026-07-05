"""GRPO training driver for text-to-TikZ on Apple Silicon (mlx-lm-lora).

Drives mlx_lm_lora.train.run() from Python rather than the CLI so we can
(a) force importance_sampling_level="token" — the library default of None
    produces an exactly-zero policy gradient (upstream issue #55);
(b) inject a JSONL metrics callback (chains under WandBCallback when --wandb
    is set and WANDB_API_KEY resolves);
(c) log a LoRA-B-norm canary each report, proving gradients are flowing.

Usage:
  uv run python train/run_grpo.py --iters 12 --run-name smoke-0.5b
  uv run python train/run_grpo.py --model Qwen/Qwen2.5-3B-Instruct \
      --load-in-4bits --iters 400 --run-name night-3b [--wandb tikzloop]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import types
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--data", default=str(ROOT / "data" / "tikz"))
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-completion", type=int, default=768)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--load-in-4bits", action="store_true")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--report-every", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=2)
    ap.add_argument("--sample-every", type=int, default=25,
                    help="reward calls between contact sheets")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true",
                    help="resume from <run>/adapters/adapters.safetensors")
    ap.add_argument("--wandb", default=None, help="W&B project name")
    return ap.parse_args()


def lora_b_norm(adapter_file: Path) -> float | None:
    """Canary for upstream issue #55: if LoRA-B stays 0.0, nothing is training."""
    if not adapter_file.exists():
        return None
    try:
        import numpy as np
        from safetensors.numpy import load_file

        weights = load_file(str(adapter_file))
        norms = [float(np.linalg.norm(v)) for k, v in weights.items()
                 if "lora_b" in k.lower()]
        return round(sum(norms), 6) if norms else None
    except Exception:
        return None


def main():
    cli = parse_args()
    load_dotenv(ROOT / ".env")

    run_dir = ROOT / "runs" / cli.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = run_dir / "adapters"

    # Reward-module knobs travel via env (the module is loaded by run()).
    os.environ["TIKZLOOP_RUN_DIR"] = str(run_dir)
    os.environ["TIKZLOOP_GROUP_SIZE"] = str(cli.group_size)
    os.environ["TIKZLOOP_SAMPLE_EVERY"] = str(cli.sample_every)

    import tikzloop._mlx_compat  # noqa: F401  (must precede mlx-lm imports)
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm_lora.train import CONFIG_DEFAULTS, run

    wandb_project = None
    if cli.wandb:
        if os.environ.get("WANDB_API_KEY"):
            wandb_project = cli.wandb
        else:
            print("[tikzloop] --wandb set but no WANDB_API_KEY in env/.env; "
                  "continuing local-only")

    class JsonlCallback(TrainingCallback):
        def __init__(self):
            self.t0 = time.time()
            self.metrics = run_dir / "metrics.jsonl"

        def _write(self, info: dict, kind: str):
            info = {k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in info.items()}
            info.update(kind=kind, wall_s=round(time.time() - self.t0, 1),
                        lora_b_norm=lora_b_norm(adapter_path / "adapters.safetensors"))
            with self.metrics.open("a") as f:
                f.write(json.dumps(info) + "\n")

        def on_train_loss_report(self, train_info: dict):
            self._write(train_info, "train")

        def on_val_loss_report(self, val_info: dict):
            self._write(val_info, "val")

    args = types.SimpleNamespace(**{
        **CONFIG_DEFAULTS,
        "model": cli.model,
        "train": True,
        "test": False,
        "train_type": "lora",
        "train_mode": "grpo",
        "data": cli.data,
        "seed": cli.seed,
        "num_layers": cli.num_layers,
        "batch_size": cli.batch_size,
        "iters": cli.iters,
        "learning_rate": cli.lr,
        "steps_per_report": cli.report_every,
        "steps_per_eval": cli.eval_every,
        "val_batches": cli.val_batches,
        "save_every": cli.save_every,
        "adapter_path": str(adapter_path),
        "resume_adapter_file": (
            str(adapter_path / "adapters.safetensors") if cli.resume else None),
        "max_seq_length": cli.max_seq_length,
        "max_completion_length": cli.max_completion,
        "grad_checkpoint": True,
        "load_in_4bits": cli.load_in_4bits,
        "lora_parameters": {"rank": cli.lora_rank, "dropout": 0.0, "scale": 10.0},
        # GRPO specifics — paper-faithful, and the issue-#55 fix:
        "beta": cli.beta,
        "group_size": cli.group_size,
        "temperature": cli.temperature,
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "grpo_loss_type": "dr_grpo",
        "importance_sampling_level": "token",
        "reward_functions_file": str(ROOT / "train" / "rewards_tikz.py"),
        "reward_functions": "tikz_render_reward",
        "wandb": wandb_project,
    })

    (run_dir / "config.json").write_text(json.dumps(
        {k: v for k, v in vars(args).items() if not k.startswith("_")},
        indent=2, default=str))
    print(f"[tikzloop] run dir: {run_dir}")
    print(f"[tikzloop] dashboard: uv run python train/dashboard.py --run {run_dir}")

    run(args, training_callback=JsonlCallback())


if __name__ == "__main__":
    main()

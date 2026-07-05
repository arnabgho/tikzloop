# Local GRPO training (M3 Pro)

GRPO-LoRA training of Qwen models on text-to-TikZ, with the `tikzloop`
sandbox as the reward and heavy instrumentation for building training
intuitions. Trainer: [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora).

## Setup (once)

```bash
uv sync --extra mlx --extra train --group dev
cp .env.example .env                     # add WANDB_API_KEY if you want W&B
uv run python train/prepare_data.py --train 500 --valid 48
```

## Run

```bash
# baseline: the number RL has to beat (drawn as a rule on the dashboard)
uv run python train/baseline.py --model Qwen/Qwen2.5-3B-Instruct --n 16

# smoke test (~15 min): verifies the whole loop end to end
uv run python train/run_grpo.py --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --iters 6 --run-name smoke-0.5b --report-every 1 --sample-every 2

# the overnight run (~3-4 min/iter, plug the laptop in)
uv run python train/run_grpo.py --model Qwen/Qwen2.5-3B-Instruct --load-in-4bits \
    --iters 400 --run-name night-3b [--wandb tikzloop]

# resume after an interruption
uv run python train/run_grpo.py ... --run-name night-3b --resume

# dashboard (separate terminal)
uv run python train/dashboard.py --run runs/night-3b   # -> localhost:8787
```

## What to watch (the intuitions this rig exists to teach)

| Panel | Healthy | Sick |
|---|---|---|
| Reward mean ± std | jumps to the 0.1 floor early, then grinds up | flat at 0 for hours (all-fail groups — prompts too hard) |
| Compile rate | steep early climb (RL prunes syntax errors first) | stuck low → check samples for truncated code, raise `--max-completion` |
| Degenerate groups | all-fail & zero-variance fractions falling | persistently high = zero gradient, wasted compute |
| Similarity vs ink | both stable or rising together | **sim ↑ while ink ↓ = blank-figure reward hack** — the model draws less and less |
| Diversity | distinct ratio near 1.0 | falling toward 1/G = entropy collapse; groups stop exploring |
| KL / clip | KL drifts slowly; some high-side clipping | KL spike = about to go degenerate; constant clipping = lr too high |
| LoRA-B norm | grows off zero within ~10 iters | **flat 0.0 = upstream zero-gradient bug — stop, check importance_sampling_level** |
| Samples gallery | figures resemble ground truth more over time | reward climbing while figures get worse — trust the pictures, not the curve |

Two failure modes were already caught during construction (both would have
silently corrupted training):
- prose completions ("I cannot draw that") compiled as text documents and
  scored 0.82 — the reward now requires a real `tikzpicture`;
- the trainer's default `importance_sampling_level=None` yields exactly-zero
  gradients (upstream issue #55) — `run_grpo.py` forces `"token"` and the
  dashboard's LoRA-B canary proves gradients flow.

## Files

- `prepare_data.py` — DaTikZ-v3 subset → `data/tikz/{train,valid}.jsonl` + GT pngs
- `rewards_tikz.py` — compile→render→SigLIP-EMD reward + rollout telemetry
- `run_grpo.py` — training driver (paper hyperparams: dr_grpo, ε=0.2/0.28, β=0)
- `baseline.py` — zero-shot eval → `runs/baseline.json`
- `dashboard.py` / `dashboard.html` — live dashboard

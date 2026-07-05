# tikzloop

Prototype of a **sandbox + multi-turn environment + inference loop** for
text-to-TikZ experiments, extending ideas from
[TikZilla (arXiv:2603.03072)](https://arxiv.org/abs/2603.03072).
Designed to run on Apple Silicon (MLX) and port cleanly to Colab
(swap MLX policy/critic for CUDA; sandbox and env are pure CPU).

## Pieces

- `sandbox.py` — safe compile+render: tectonic `--untrusted`, pre-compile
  denylist (`\write18`, `\input`, ...), timeout, Linux memory rlimit,
  capped render DPI. `TikZ source -> CompileResult{ok, log, image}`.
- `env.py` — gym-style `TikZEnv`: `reset(description, gt_image)` /
  `step(code) -> StepResult{feedback, reward, done}`. Critic feedback drives
  the loop; reward (vs. ground truth) is for training/eval only.
- `policy.py` — `MLXPolicy` (any mlx-lm chat model; point it at a TikZilla
  conversion), `MockPolicy` for tests.
- `critic.py` — `VLMCritic` (Qwen2.5-VL via mlx-vlm) judges render vs.
  description, returns JSON verdict + actionable issues.
- `reward.py` — SigLIP patch embeddings + relaxed EMD similarity, a
  prototype stand-in for TikZilla's DeTikZify-based reward encoder.
- `loop.py` — `run_episode` (multi-turn repair loop), `best_of_n`
  (reranking), JSONL episode logging for hindsight-repair training data.

## Usage

```bash
uv sync --extra mlx --group dev
uv run pytest                      # sandbox + env tests (no models needed)

# sandbox smoke test
uv run tikzloop compile examples/flowchart.tex

# multi-turn loop, compiler feedback only
uv run tikzloop demo "a red circle connected to a blue square by an arrow"

# with the VLM critic in the loop
uv run tikzloop demo "..." --critic vlm

# semantic reward (needs: uv sync --extra reward)
# from tikzloop.reward import SigLIPReward; env = TikZEnv(reward_fn=SigLIPReward())
```

## Next steps

- Swap `MockPolicy`-grade baselines for the released TikZilla-3B weights
  (convert to MLX via `mlx_lm.convert`).
- Best-of-N reranking eval with `SigLIPReward` against DaTikZ-V4 test pairs.
- Log episodes (`append_jsonl`) to build a hindsight-repair dataset, then
  GRPO-LoRA on Colab with the same sandbox as the reward function.

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

VALID = (
    "```latex\n\\begin{tikzpicture}\\node[draw, circle, fill=red!30]{A};"
    "\\end{tikzpicture}\n```"
)
BROKEN = "```latex\n\\begin{tikzpicture}\\node[draw{oops;\\end{tikzpicture}\n```"
EMPTY = "I cannot draw that."


@pytest.fixture()
def rewards_module(tmp_path, monkeypatch):
    pytest.importorskip("mlx_lm_lora")
    pytest.importorskip("torch")
    monkeypatch.setenv("TIKZLOOP_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("TIKZLOOP_GROUP_SIZE", "3")
    monkeypatch.setenv("TIKZLOOP_SAMPLE_EVERY", "1")
    import tikzloop._mlx_compat  # noqa: F401
    spec = importlib.util.spec_from_file_location(
        "rewards_tikz_test", Path(__file__).parent.parent / "train" / "rewards_tikz.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rewards_tikz_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_reward_batch(rewards_module, tmp_path):
    gt = tmp_path / "gt.png"
    Image.new("RGB", (200, 200), "white").save(gt)

    completions = [VALID, BROKEN, EMPTY]
    rewards = rewards_module.tikz_render_reward(
        prompts=["a red circle"] * 3,
        completions=completions,
        answer=[str(gt)] * 3,
        types=["tikz"] * 3,
    )
    assert len(rewards) == 3
    assert 0.1 <= rewards[0] <= 1.0          # compiled -> format floor + sim
    assert rewards[1] == 0.0 and rewards[2] == 0.0
    assert all(isinstance(r, float) for r in rewards)

    run_dir = Path(rewards_module.RUN_DIR)
    lines = [json.loads(l) for l in (run_dir / "rollouts.jsonl").read_text().splitlines()]
    per_sample = [l for l in lines if l.get("kind") != "group_summary"]
    summary = [l for l in lines if l.get("kind") == "group_summary"]
    assert len(per_sample) == 3 and len(summary) == 1
    assert summary[0]["compile_rate"] == pytest.approx(1 / 3, abs=0.01)
    assert per_sample[0]["ink_frac"] is not None
    assert per_sample[1]["error"] == "latex-error"
    assert per_sample[2]["error"] == "no-code"  # prose must never score

    sheets = list((run_dir / "samples").glob("*.png"))
    assert sheets, "contact sheet should be written on first call"

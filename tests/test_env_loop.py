from tikzloop.env import Critique, TikZEnv
from tikzloop.loop import run_episode
from tikzloop.policy import MockPolicy, build_messages, extract_code

GOOD = "```latex\n\\begin{tikzpicture}\\node[draw]{ok};\\end{tikzpicture}\n```"
BAD = "```latex\n\\begin{tikzpicture}\\node[draw{broken;\\end{tikzpicture}\n```"


class RejectOnceCritic:
    def __init__(self):
        self.calls = 0

    def critique(self, image, description):
        self.calls += 1
        if self.calls == 1:
            return Critique(matches=False, feedback="Node label is wrong.")
        return Critique(matches=True, feedback="Matches.")


def test_extract_code_variants():
    assert extract_code(GOOD).startswith("\\begin{tikzpicture}")
    raw = "Here you go: \\begin{tikzpicture}\\node{x};\\end{tikzpicture}"
    assert extract_code(raw).startswith("\\begin{tikzpicture}")


def test_repair_after_compile_error():
    env = TikZEnv(max_turns=4)  # CompileOnlyCritic
    policy = MockPolicy([BAD, GOOD])
    episode = run_episode(env, policy, "a node labeled ok")
    assert [s.compiled for s in episode.steps] == [False, True]
    assert episode.solved
    assert "tikzpicture" in episode.final_code


def test_critic_feedback_reaches_next_prompt():
    env = TikZEnv(critic=RejectOnceCritic(), max_turns=4)
    policy = MockPolicy([GOOD, GOOD])
    episode = run_episode(env, policy, "a node")
    assert len(episode.steps) == 2
    assert not episode.steps[0].done
    assert episode.steps[1].done and episode.solved
    # the repair prompt must contain the critic's feedback
    messages = build_messages("a node", [(extract_code(GOOD), episode.steps[0].feedback)])
    assert any("Node label is wrong." in m["content"] for m in messages)


def test_max_turns_terminates():
    env = TikZEnv(max_turns=2)
    policy = MockPolicy([BAD])  # never fixes it
    episode = run_episode(env, policy, "anything")
    assert len(episode.steps) == 2
    assert not episode.solved
    assert episode.final_code is None

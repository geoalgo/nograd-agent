import re

import pytest
import yaml

from autoresearch import AgentResponse, CodeAgent, load_turns
from autoresearch import cli

TRAIN = """\
VALUE = 1
print(f"score: {VALUE}")
"""


def write_src(tmp_path, config: dict | None = None):
    src_dir = tmp_path / "src"
    (src_dir / "lib").mkdir(parents=True)
    (src_dir / "train.py").write_text(TRAIN)
    (src_dir / "lib" / "train.py").write_text(TRAIN)
    (src_dir / "eval.py").write_text("")
    (src_dir / "PROMPT.md").write_text("Maximize the score.")
    if config is not None:
        (src_dir / cli.CONFIG_FILE).write_text(yaml.safe_dump(config))
    return src_dir


def test_settings_precedence(tmp_path):
    src_dir = write_src(
        tmp_path, {"metric": "score", "prompt_file": "PROMPT.md", "n_workers": 2, "max_trials": 5}
    )
    settings = cli.load_settings(["--src_dir", str(src_dir), "--max_trials", "3"])
    assert settings["max_trials"] == 3  # command line
    assert settings["n_workers"] == 2  # configuration file
    assert settings["trial_timeout"] == 120.0  # default
    assert settings["prompt"] == "Maximize the score."
    assert settings["editable_files"] == ["train.py"]
    assert settings["frozen_files"] == []

    # Paths are relative to the source folder, the eval script is frozen.
    settings = cli.load_settings(
        ["--src_dir", str(src_dir), "--train_script", "lib/train.py", "--eval_script", "eval.py"]
    )
    assert settings["editable_files"] == ["lib/train.py"]
    assert settings["frozen_files"] == ["eval.py"]

    # A prompt on the command line replaces the prompt file of the configuration.
    settings = cli.load_settings(["--src_dir", str(src_dir), "--prompt", "Be quick."])
    assert settings["prompt"] == "Be quick."


@pytest.mark.parametrize(
    "config, args, error",
    [
        ({"metric": "score", "prompt": "x", "max_trial": 3}, [], "unknown settings in .*: max_trial"),
        ({"prompt": "x"}, [], "the metric is required"),
        ({"metric": "score"}, [], "describe the task"),
        ({"metric": "score", "prompt": "x"}, ["--train_script", "missing.py"], "train_script missing.py not found"),
        ({"metric": "score", "prompt": "x"}, ["--train_script", "../train.py"], "must be inside the source folder"),
        ({"metric": "score", "prompt": "x"}, ["--frozen_files", "train.py"], "both editable and frozen"),
    ],
)
def test_invalid_settings(tmp_path, capsys, config, args, error):
    src_dir = write_src(tmp_path, config)
    (tmp_path / "train.py").write_text(TRAIN)
    with pytest.raises(SystemExit):
        cli.load_settings(["--src_dir", str(src_dir), *args])
    assert re.search(error, capsys.readouterr().err)


class IncrementAgent(CodeAgent):
    def __init__(self, **kwargs):
        self.calls = 0

    def send(self, message, cwd, system_prompt=None):
        self.calls += 1
        (cwd / "train.py").write_text(TRAIN.replace("VALUE = 1", f"VALUE = {self.calls + 1}"))
        return AgentResponse(f"SUMMARY: set VALUE to {self.calls + 1}", [], "session", 0.01, 0.0)


@pytest.mark.timeout(120)
def test_cli_single_script(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNETUNE_FOLDER", str(tmp_path / "experiments"))
    monkeypatch.setattr(cli, "ClaudeCodeAgent", IncrementAgent)
    src_dir = write_src(tmp_path, {"metric": "score", "prompt_file": "PROMPT.md", "trial_timeout": 30})
    cli.main(["--src_dir", str(src_dir), "--max_trials", "3"])

    (tuner_path,) = (tmp_path / "experiments").iterdir()
    turns = load_turns(tuner_path)
    # Parsed from the output of train.py, there is no evaluation script.
    assert [turn.metric for turn in turns][:3] == [1.0, 2.0, 3.0]
    saved = yaml.safe_load((tuner_path / cli.CONFIG_FILE).read_text())
    assert saved["max_trials"] == 3 and saved["src_dir"] == str(src_dir.resolve())

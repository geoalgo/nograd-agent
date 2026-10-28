import subprocess
import time
from pathlib import Path

import pytest

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import LocalBackend
from syne_tune.results_callback import StoreResultsCallback

from autoresearch import (
    ENTRY_POINT,
    AgentResponse,
    AutoSearch,
    AutoSearchCallback,
    AutoSearchLogger,
    CodeAgent,
    LLMSearcher,
    load_turns,
)
from autoresearch.run_trial import build_trial_dir, check_frozen_files, parse_metric, run

TRAIN = """\
from pathlib import Path

VALUE = 1
Path("value.txt").write_text(str(VALUE))
"""

EVALUATE = """\
import json
from pathlib import Path

print(json.dumps({"score": float(Path("value.txt").read_text())}))
"""


class FakeAgent(CodeAgent):
    """Edits train.py like an agent would, without calling any model.

    The first proposal also tries to change the frozen evaluate.py, the second
    one is broken, the next ones increase VALUE.
    """

    def __init__(self):
        self.prompts = []
        self.frozen_write_refused = False

    def send(self, message, cwd, system_prompt=None):
        self.prompts.append(message)
        call = len(self.prompts)
        train = cwd / "train.py"
        if call == 1:
            train.write_text(TRAIN.replace("VALUE = 1", "VALUE = 2"))
            try:
                (cwd / "evaluate.py").write_text('print(\'{"score": 100.0}\')\n')
            except PermissionError:
                self.frozen_write_refused = True
        elif call == 2:
            train.write_text("def broken(:\n")
        else:
            train.write_text(TRAIN.replace("VALUE = 1", f"VALUE = {call}"))
        tool_use = {
            "type": "tool_use",
            "name": "Write",
            "input": {"file_path": str(train)},
        }
        messages = [
            {"type": "assistant", "message": {"content": [tool_use]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
        ]
        return AgentResponse(
            text=f"Changed VALUE.\nSUMMARY: proposal {call}",
            messages=messages,
            session_id="fake-session",
            cost_usd=0.01,
            duration_s=0.0,
        )


def run_tuning(tmp_path, monkeypatch, n_workers: int, max_trials: int, metric: str = "score"):
    monkeypatch.setenv("SYNETUNE_FOLDER", str(tmp_path / "experiments"))
    base_dir = tmp_path / "script"
    base_dir.mkdir()
    (base_dir / "train.py").write_text(TRAIN)
    (base_dir / "evaluate.py").write_text(EVALUATE)
    agent = FakeAgent()
    searcher = LLMSearcher(
        task_prompt="Maximize the score.",
        base_dir=base_dir,
        editable_files=["train.py"],
        frozen_files=["evaluate.py"],
        evaluate_command=["python", "evaluate.py"],
        metric=metric,
        mode="max",
        agent=agent,
        trial_timeout=30,
    )
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=ENTRY_POINT, pass_args_as_json=True),
        scheduler=AutoSearch(searcher),
        callbacks=[StoreResultsCallback(), AutoSearchCallback()],
        stop_criterion=StoppingCriterion(max_num_trials_finished=max_trials),
        n_workers=n_workers,
        max_failures=max_trials,
        wait_trial_completion_when_stopping=True,
        sleep_time=0.1,
        output_logger=AutoSearchLogger(searcher),
    )
    tuner.run()
    return tuner, searcher, agent


@pytest.mark.timeout(120)
def test_autoresearch_sequential(tmp_path, monkeypatch, capsys):
    tuner, searcher, agent = run_tuning(tmp_path, monkeypatch, n_workers=1, max_trials=5)
    history = searcher.history()

    output = capsys.readouterr().out
    assert "AutoSearch - Syne Tune" in output
    assert "Trial 1 started: proposal 1" in output
    # FakeAgent costs 0.01$ per proposal, the baseline is free.
    assert f"{0.01 * (len(history) - 1):.2f}$" in output

    assert history.loc[0, "summary"] == "Unmodified baseline."
    assert history.loc[0, "metric"] == 1.0
    # The frozen evaluate.py is read-only for the agent.
    assert agent.frozen_write_refused
    assert history.loc[1, "metric"] == 2.0
    assert history.loc[2, "status"] == "error"
    assert "SyntaxError" in history.loc[2, "error"]
    # The error is shown to the agent in the next prompt.
    assert "SyntaxError" in agent.prompts[2]
    # The task is only described in the first message.
    assert "Maximize the score." in agent.prompts[0]
    assert "Maximize the score." not in agent.prompts[1]
    assert searcher.best().metric == history["metric"].max() > 2.0
    assert history.loc[3, "parent_id"] == 1
    assert history.loc[1, "summary"] == "proposal 1"
    assert "+VALUE = 2" in history.loc[1, "diff"]
    assert len(searcher.tool_calls()) == len(history) - 1

    # Everything is stored in the tuning folder, conversations in trial folders.
    assert searcher.output_dir == tuner.tuner_path / "autoresearch"
    reloaded = load_turns(tuner.tuner_path)
    assert [t.status for t in reloaded] == list(history["status"])
    trial_dir = Path(tuner.trial_backend.trial_path(history.loc[1, "trial_id"]))
    assert "VALUE = 2" in (trial_dir / "train.py").read_text()
    assert "Changed VALUE." not in (trial_dir / "conversation.md").read_text()
    assert "proposal 1" in (trial_dir / "conversation.md").read_text()
    assert (trial_dir / "conversation.json").exists()


@pytest.mark.timeout(120)
def test_autoresearch_parallel(tmp_path, monkeypatch):
    _, searcher, agent = run_tuning(tmp_path, monkeypatch, n_workers=3, max_trials=6)
    history = searcher.history()

    assert set(history["status"]) <= {"completed", "error"}
    assert history["trial_id"].notna().all()
    # The first proposal is made while the baseline is still running.
    assert "running" in agent.prompts[0]
    assert searcher.best().metric == history["metric"].max()


@pytest.mark.timeout(30)
def test_timeout_kills_child_processes(tmp_path):
    # The grandchild keeps stdout open, the trial must not wait for it to exit.
    script = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    start = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        run(["python", "-c", script], tmp_path, timeout=2)
    assert time.time() - start < 10


def test_frozen_files_in_trial(tmp_path):
    base_dir, candidate_dir, trial_dir = (tmp_path / n for n in ("base", "candidate", "trial"))
    for folder in (base_dir, candidate_dir):
        folder.mkdir()
        (folder / "train.py").write_text(TRAIN)
        (folder / "evaluate.py").write_text(EVALUATE)
    (candidate_dir / "evaluate.py").write_text("tampered")
    build_trial_dir(base_dir, candidate_dir, trial_dir, ["evaluate.py"])

    # Restored from the base folder and read-only.
    assert (trial_dir / "evaluate.py").read_text() == EVALUATE
    with pytest.raises(PermissionError):
        (trial_dir / "evaluate.py").write_text("tampered")
    check_frozen_files(base_dir, trial_dir, ["evaluate.py"])

    # Training can still change it deliberately, which fails the trial.
    (trial_dir / "evaluate.py").chmod(0o644)
    (trial_dir / "evaluate.py").write_text("tampered")
    with pytest.raises(RuntimeError, match="evaluate.py is frozen"):
        check_frozen_files(base_dir, trial_dir, ["evaluate.py"])

    # The trial folder can be built again, as when tuning resumes and reuses a
    # trial id, although it contains a read-only copy.
    (trial_dir / "evaluate.py").chmod(0o444)
    build_trial_dir(base_dir, candidate_dir, trial_dir, ["evaluate.py"])
    assert (trial_dir / "evaluate.py").read_text() == EVALUATE


def test_parse_metric():
    assert parse_metric('training\n{"score": 1.5, "other": 2}\n', "score") == 1.5
    # The last report is used.
    assert parse_metric("epoch 1 score: 1\nepoch 2 score=2.5e-1\n", "score") == 0.25
    # Other metrics containing the name are not matched.
    assert parse_metric("test_score: 9\nscore: 1", "score") == 1.0
    assert parse_metric("acc (val) 0.9", "acc", r"acc \(val\) ([0-9.]+)") == 0.9
    # A missing or non-finite metric fails the trial, the agent sees why.
    for output in ('{"other": 1.0}', "score: nan", '{"score": NaN}', "score: high", "test_score: 1"):
        with pytest.raises(RuntimeError, match="Missing metric:.*score"):
            parse_metric(output, "score")


@pytest.mark.timeout(60)
def test_baseline_without_metric_stops_tuning(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="baseline did not report the metric, check metric=.typo."):
        run_tuning(tmp_path, monkeypatch, n_workers=1, max_trials=3, metric="typo")


def test_editable_files_in_subfolder(tmp_path):
    base_dir = tmp_path / "script"
    (base_dir / "src").mkdir(parents=True)
    (base_dir / "src" / "model.py").write_text("VALUE = 1\n")
    (base_dir / "evaluate.py").write_text(EVALUATE)

    class SubfolderAgent(CodeAgent):
        def send(self, message, cwd, system_prompt=None):
            (cwd / "src" / "model.py").write_text("VALUE = 2\n")
            return AgentResponse("SUMMARY: set VALUE to 2", [], "session", 0.0, 0.0)

    searcher = LLMSearcher(
        task_prompt="Maximize the score.",
        base_dir=base_dir,
        editable_files=["src/model.py"],
        frozen_files=["evaluate.py"],
        metric="score",
        agent=SubfolderAgent(),
    )
    searcher.set_output_dir(tmp_path / "output")
    for _ in range(2):
        searcher.suggest()
    assert (searcher.candidate_dir(0) / "src" / "model.py").read_text() == "VALUE = 1\n"
    assert (searcher.candidate_dir(1) / "src" / "model.py").read_text() == "VALUE = 2\n"


def test_resume_from_checkpoint(tmp_path):
    import copy

    base_dir = tmp_path / "script"
    base_dir.mkdir()
    (base_dir / "train.py").write_text(TRAIN)
    (base_dir / "evaluate.py").write_text(EVALUATE)
    searcher = LLMSearcher(
        task_prompt="Maximize the score.",
        base_dir=base_dir,
        editable_files=["train.py"],
        frozen_files=["evaluate.py"],
        metric="score",
        agent=FakeAgent(),
    )
    searcher.set_output_dir(tmp_path / "output")
    searcher.suggest()
    searcher.suggest()
    searcher.on_trial_complete(0, {"score": 1.0})
    # Tuner.load restores the searcher of the last checkpoint, here taken while
    # candidate 1 was running and before candidate 2 was proposed.
    checkpoint = copy.deepcopy(searcher)
    searcher.suggest()

    # Tuner.run then calls set_output_dir again on the same folder.
    checkpoint.set_output_dir(tmp_path / "output")
    history = checkpoint.history()
    assert list(history["status"]) == ["completed", "interrupted"]
    checkpoint.suggest()
    prompt = checkpoint.agent.prompts[-1]
    assert "| 1 | 0 | INTERRUPTED, not evaluated |" in prompt
    assert "Error of candidate" not in prompt
    assert "proposals for candidate 2 were lost" in prompt
    assert "running" not in prompt
    # Only reported once.
    checkpoint.suggest()
    assert "lost" not in checkpoint.agent.prompts[-1]

    # Reusing the searcher for another run is refused.
    with pytest.raises(ValueError, match="create a new LLMSearcher"):
        checkpoint.set_output_dir(tmp_path / "other")


def test_claude_code_agent_sends_system_prompt_on_every_call(tmp_path, monkeypatch):
    import json
    import shutil

    import autoresearch.searchers.agent as agent_module

    commands = []

    messages = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        messages.append(kwargs["stdin"].read())
        result = {"type": "result", "session_id": "s1", "result": "SUMMARY: x", "total_cost_usd": 0.1 * len(commands)}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(result) + "\n", stderr="")

    monkeypatch.setattr(shutil, "which", lambda binary: binary)
    monkeypatch.setattr(agent_module.subprocess, "run", fake_run)
    agent = agent_module.ClaudeCodeAgent()
    long_message = "x" * 1_000_000
    responses = [agent.send(long_message, tmp_path, system_prompt="RULES") for _ in range(2)]

    # The message is sent as input, not as an argument.
    assert messages == [long_message, long_message]
    assert all(long_message not in cmd for cmd in commands)

    for cmd in commands:
        assert cmd[cmd.index("--append-system-prompt") + 1] == "RULES"
    assert "--resume" not in commands[0]
    assert commands[1][commands[1].index("--resume") + 1] == "s1"
    # Claude Code reports the cumulative cost of the session.
    assert [round(r.cost_usd, 6) for r in responses] == [0.1, 0.1]


def test_error_of_trial_killed_without_message(tmp_path):
    base_dir = tmp_path / "script"
    base_dir.mkdir()
    (base_dir / "train.py").write_text(TRAIN)
    searcher = LLMSearcher(
        task_prompt="Maximize the score.",
        base_dir=base_dir,
        editable_files=["train.py"],
        metric="score",
        agent=FakeAgent(),
    )
    searcher.set_output_dir(tmp_path / "tuner" / "autoresearch")
    searcher.suggest()
    searcher.on_trial_add(0, trial_id=0)
    # Written by the local backend, error.txt is missing as for a killed trial.
    (tmp_path / "tuner" / "0").mkdir()
    (tmp_path / "tuner" / "0" / "std.err").write_text("loading data\nKilled\n")
    searcher.on_trial_error(0)
    error = searcher.history().loc[0, "error"]
    assert "stopped without an error message" in error
    assert error.endswith("loading data\nKilled")


def test_summaries_fit_in_tables():
    from autoresearch.searchers.llm_searcher import Turn
    from autoresearch.searchers.prompts import format_candidate, parse_summary

    assert parse_summary("Done.\nSUMMARY:  use a | b \n") == "use a | b"
    # Without a summary line, the end of the reply is kept on a single line.
    assert parse_summary("I changed\n\nthe model.\n") == "I changed the model."
    row = format_candidate(Turn(candidate_id=1, parent_id=0, summary="use a | b"), "score")
    assert row == "| 1 | 0 | running | use a \\| b |"


@pytest.mark.parametrize("failures", [2, 10])
def test_agent_retries(tmp_path, failures):
    base_dir = tmp_path / "script"
    base_dir.mkdir()
    (base_dir / "train.py").write_text(TRAIN)

    class FlakyAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def send(self, message, cwd, system_prompt=None):
            self.attempts += 1
            if self.attempts <= failures:
                # Partial edit before failing, discarded before the retry.
                (cwd / "train.py").write_text("partial")
                raise RuntimeError("rate limited")
            assert (cwd / "train.py").read_text() == TRAIN
            return super().send(message, cwd, system_prompt)

    agent = FlakyAgent()
    searcher = LLMSearcher(
        task_prompt="Maximize the score.",
        base_dir=base_dir,
        editable_files=["train.py"],
        metric="score",
        agent=agent,
        agent_retries=3,
        agent_retry_wait=0.0,
    )
    searcher.set_output_dir(tmp_path / "output")
    searcher.suggest()
    if failures <= 3:
        searcher.suggest()
        assert agent.attempts == failures + 1
        assert "VALUE = 2" in (searcher.candidate_dir(1) / "train.py").read_text()
    else:
        with pytest.raises(RuntimeError, match="failed 4 times in a row"):
            searcher.suggest()
        assert agent.attempts == 4

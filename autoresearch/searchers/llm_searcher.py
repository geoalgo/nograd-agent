import difflib
import json
import logging
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from autoresearch.run_trial import (
    ERROR_FILE,
    ERROR_TAIL_LINES,
    MISSING_METRIC_ERROR,
    SPEC_FILE,
)
from autoresearch.searchers.agent import ClaudeCodeAgent, CodeAgent
from autoresearch.searchers.prompts import SYSTEM_PROMPT, build_prompt, parse_summary

logger = logging.getLogger(__name__)

# Written to each candidate folder, and so copied to its trial folder.
CONVERSATION_JSON = "conversation.json"
CONVERSATION_MD = "conversation.md"


@dataclass
class Turn:
    """One suggestion: the agent call that proposed a candidate, and its outcome."""

    candidate_id: int
    parent_id: int | None
    trial_id: int | None = None
    prompt: str = ""
    messages: list[dict] = field(default_factory=list)  # raw agent messages
    summary: str = ""
    diff: str = ""
    session_id: str | None = None
    cost_usd: float = 0.0
    duration_s: float = 0.0
    status: str = "pending"  # pending | completed | error | interrupted
    metric: float | None = None
    result: dict[str, Any] = field(default_factory=dict)  # all reported metrics
    error: str | None = None


class LLMSearcher:
    """Proposes new versions of a script with a coding agent, given the results
    of the previous versions.

    Every suggestion is a code candidate: the agent edits the editable files in a
    fixed workspace folder, and the edited files are copied to
    ``candidates/<id>/``. Only editable files are copied, so changes the agent
    makes to other files are discarded. The trial backend then runs the
    candidate with ``run_trial.py``. The agent never runs code itself.

    Suggestions are serialized because the tuner blocks on ``suggest``. Results
    of trials running in parallel are given to the agent in the next prompt,
    together with the table of all candidates and the ones still running. The
    next candidate is always edited from the best one so far.

    The workspace, candidates and the log of all turns (``turns.jsonl``) are
    stored in the folder set by :meth:`set_output_dir`, which
    :class:`AutoSearch` sets to ``<tuner_path>/autoresearch``.

    :param task_prompt: Description of the problem, sent in the first message
    :param base_dir: Folder with the initial version of the script. All its files
        are copied to every trial folder.
    :param editable_files: Files the agent may change, relative to ``base_dir``
    :param frozen_files: Files the agent can read but not change, for instance the
        evaluation code. They are read-only in the workspace of the agent, and
        restored read-only from ``base_dir`` in the trial folder, where the trial
        fails if training changed them.
    :param train_command: Command run first in the trial folder, ``python`` is
        replaced by the interpreter running the trial
    :param evaluate_command: Optional command run after training in the trial
        folder. The metric is parsed from the output of the last command run.
    :param metric: Name of the metric to optimize
    :param mode: "max" or "min"
    :param metric_regex: Regular expression whose first group is the value of
        the metric, the last match in the output is used. Defaults to a line
        containing ``<metric>: <value>`` or ``<metric>=<value>``.
    :param agent: Coding agent, defaults to :class:`ClaudeCodeAgent` with Sonnet
    :param trial_timeout: Maximum seconds for training plus evaluation of a trial
    :param evaluate_baseline: Whether the first candidate is the unmodified script
    :param agent_retries: Number of times a failed agent call is retried, for
        instance after a rate limit or a timeout, before tuning is stopped
    :param agent_retry_wait: Seconds before the first retry, doubled for each
        following one
    """

    def __init__(
        self,
        task_prompt: str,
        base_dir: str | Path,
        editable_files: list[str],
        metric: str,
        mode: str = "max",
        frozen_files: list[str] = (),
        train_command: list[str] = ("python", "train.py"),
        evaluate_command: list[str] | None = None,
        metric_regex: str | None = None,
        agent: CodeAgent | None = None,
        trial_timeout: float = 120.0,
        evaluate_baseline: bool = True,
        agent_retries: int = 3,
        agent_retry_wait: float = 10.0,
    ):
        assert mode in ("max", "min"), f"mode must be 'max' or 'min', got {mode}"
        self.task_prompt = task_prompt
        self.base_dir = Path(base_dir).resolve()
        self.editable_files = list(editable_files)
        self.frozen_files = list(frozen_files)
        for name in self.editable_files + self.frozen_files:
            assert (self.base_dir / name).is_file(), f"{self.base_dir / name} not found"
        self.train_command = list(train_command)
        self.evaluate_command = None if evaluate_command is None else list(evaluate_command)
        if metric_regex is not None:
            assert re.compile(metric_regex).groups >= 1, (
                f"metric_regex must capture the value in a group, got {metric_regex!r}"
            )
        self.metric_regex = metric_regex
        self.metric = metric
        self.mode = mode
        self.agent = agent if agent is not None else ClaudeCodeAgent()
        self.trial_timeout = trial_timeout
        self.evaluate_baseline = evaluate_baseline
        self.agent_retries = agent_retries
        self.agent_retry_wait = agent_retry_wait
        self.turns: list[Turn] = []
        # Candidates finished since the last prompt, reported in the next one.
        self._unreported: list[int] = []
        # Candidates proposed after the checkpoint tuning resumed from, reported
        # in the next prompt since the agent conversation still contains them.
        self._lost: list[int] = []
        self.output_dir: Path | None = None

    def set_output_dir(self, output_dir: str | Path):
        """Set the folder for the workspace, candidates and log, must be called
        before the first :meth:`suggest`. It can only be changed before the first
        suggestion, since candidates and the agent session belong to one run."""
        output_dir = Path(output_dir).resolve()
        if self.turns and output_dir != self.output_dir:
            raise ValueError(
                f"This searcher already ran in {self.output_dir}, create a new "
                "LLMSearcher and agent for every tuning run."
            )
        self.output_dir = output_dir
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "base_dir": str(self.base_dir),
            "frozen_files": self.frozen_files,
            "train_command": self.train_command,
            "evaluate_command": self.evaluate_command,
            "timeout": self.trial_timeout,
            "metric": self.metric,
            "metric_regex": self.metric_regex,
        }
        self.spec_path.write_text(json.dumps(spec, indent=2))
        # When tuning resumes with Tuner.load, the searcher is the one of the
        # last checkpoint: later candidates are only in the log, and trials
        # running at the checkpoint are never reported as finished.
        # TODO: fix in Syne Tune instead: on resume, Tuner.run should report trials
        # that were running when it was saved to the scheduler, e.g. with on_trial_error.
        if self.log_path.exists():
            logged = {turn.candidate_id for turn in load_turns(self.log_path)}
            self._lost = sorted(i for i in logged if i >= len(self.turns))
        for turn in self.turns:
            if turn.status == "pending":
                turn.status = "interrupted"
                self._finish(turn)

    @property
    def workspace(self) -> Path:
        # Must stay fixed: Claude Code stores sessions per working directory.
        return self.output_dir / "workspace"

    @property
    def candidates_dir(self) -> Path:
        return self.output_dir / "candidates"

    @property
    def log_path(self) -> Path:
        return self.output_dir / "turns.jsonl"

    @property
    def spec_path(self) -> Path:
        return self.output_dir / SPEC_FILE

    def candidate_dir(self, candidate_id: int) -> Path:
        return self.candidates_dir / str(candidate_id)

    # ---- Called by AutoSearch ----

    def suggest(self) -> dict[str, Any]:
        """Propose a new candidate and return the config of the trial running it."""
        assert self.output_dir is not None, "call set_output_dir before suggest"
        candidate_id = len(self.turns)
        if candidate_id == 0 and self.evaluate_baseline:
            turn = Turn(candidate_id=0, parent_id=None, summary="Unmodified baseline.")
            self._write_candidate(turn, self._read_files(self.base_dir))
        else:
            turn = self._propose(candidate_id)
        self.turns.append(turn)
        self._save_turn(turn)
        self._write_conversation(turn)
        return {
            "candidate_dir": str(self.candidate_dir(candidate_id)),
            "trial_spec": str(self.spec_path),
        }

    def on_trial_add(self, candidate_id: int, trial_id: int):
        self.turns[candidate_id].trial_id = trial_id

    def on_trial_complete(self, candidate_id: int, result: dict[str, Any]):
        turn = self.turns[candidate_id]
        turn.status = "completed"
        turn.metric = result.get(self.metric)
        turn.result = {k: v for k, v in result.items() if not k.startswith("st_")}
        self._finish(turn)

    def on_trial_error(self, candidate_id: int):
        turn = self.turns[candidate_id]
        turn.status = "error"
        error_path = self.candidate_dir(candidate_id) / ERROR_FILE
        turn.error = (
            error_path.read_text() if error_path.exists() else self._trial_stderr(turn)
        )
        self._finish(turn)
        # Most likely a wrong metric name or regex, every candidate would fail
        # the same way.
        if candidate_id == 0 and self.evaluate_baseline and MISSING_METRIC_ERROR in turn.error:
            raise ValueError(
                f"The unmodified baseline did not report the metric, check "
                f"metric={self.metric!r}, metric_regex and the output of the "
                f"script:\n{turn.error}"
            )

    # ---- Queryable history ----

    def history(self) -> pd.DataFrame:
        """One row per candidate, without the prompts and raw agent messages."""
        return turns_to_dataframe(self.turns)

    def tool_calls(self) -> pd.DataFrame:
        """One row per tool call made by the agent."""
        rows = []
        for turn in self.turns:
            for message in turn.messages:
                if message.get("type") != "assistant":
                    continue
                for block in message["message"]["content"]:
                    if block.get("type") == "tool_use":
                        rows.append(
                            {
                                "candidate_id": turn.candidate_id,
                                "tool": block["name"],
                                "file": block["input"].get("file_path"),
                                "input": block["input"],
                            }
                        )
        return pd.DataFrame(rows)

    def best(self) -> Turn | None:
        completed = [t for t in self.turns if t.metric is not None]
        if not completed:
            return None
        sign = 1 if self.mode == "max" else -1
        return max(completed, key=lambda t: sign * t.metric)

    # ---- Internals ----

    def _finish(self, turn: Turn):
        self._unreported.append(turn.candidate_id)
        self._save_turn(turn)

    def _trial_stderr(self, turn: Turn) -> str:
        """Error message of a trial that stopped without writing one, for
        instance killed for using too much memory: the end of its error output,
        in the trial folder of the local backend."""
        message = "The trial stopped without an error message, it may have been killed."
        stderr = self.output_dir.parent / str(turn.trial_id) / "std.err"
        if turn.trial_id is not None and stderr.exists():
            tail = stderr.read_text().strip().splitlines()[-ERROR_TAIL_LINES:]
            if tail:
                message += " End of its error output:\n" + "\n".join(tail)
        return message

    def _save_turn(self, turn: Turn):
        # Appends a record per state change, load_turns keeps the latest one.
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(turn), default=str) + "\n")

    def _read_files(self, folder: Path) -> dict[str, str]:
        return {name: (folder / name).read_text() for name in self.editable_files}

    def _write_candidate(self, turn: Turn, files: dict[str, str]):
        folder = self.candidate_dir(turn.candidate_id)
        folder.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text(content)

    def _write_conversation(self, turn: Turn):
        folder = self.candidate_dir(turn.candidate_id)
        record = {
            k: v
            for k, v in asdict(turn).items()
            if k not in ("trial_id", "status", "metric", "result", "error")
        }
        (folder / CONVERSATION_JSON).write_text(json.dumps(record, indent=2))
        (folder / CONVERSATION_MD).write_text(conversation_markdown(turn))

    def _reset_workspace(self, parent_files: dict[str, str]):
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)
        for name in self.frozen_files:
            (self.workspace / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.base_dir / name, self.workspace / name)
            # The agent only has file tools and cannot make it writable again.
            (self.workspace / name).chmod(0o444)
        for name, content in parent_files.items():
            (self.workspace / name).parent.mkdir(parents=True, exist_ok=True)
            (self.workspace / name).write_text(content)

    def _propose(self, candidate_id: int) -> Turn:
        parent = self.best()
        if parent is None and self.turns:
            parent = self.turns[0]
        if parent is None:
            parent_files = self._read_files(self.base_dir)
        else:
            parent_files = self._read_files(self.candidate_dir(parent.candidate_id))
        turn = Turn(
            candidate_id=candidate_id,
            parent_id=None if parent is None else parent.candidate_id,
        )
        first_message = all(t.session_id is None for t in self.turns)
        turn.prompt = build_prompt(
            candidate_id=candidate_id,
            turns=self.turns,
            new_results=[self.turns[i] for i in self._unreported],
            parent=parent,
            parent_files=parent_files,
            metric=self.metric,
            mode=self.mode,
            task_prompt=self.task_prompt if first_message else None,
            read_only_files=self.frozen_files,
            lost_candidates=self._lost,
        )
        self._unreported = []
        self._lost = []
        self._reset_workspace(parent_files)

        message = turn.prompt
        new_files = parent_files
        for _ in range(2):
            response = self._send(message, parent_files)
            turn.messages += response.messages
            turn.session_id = response.session_id
            turn.cost_usd += response.cost_usd
            turn.duration_s += response.duration_s
            turn.summary = parse_summary(response.text)
            new_files = {
                name: (self.workspace / name).read_text()
                for name in self.editable_files
                if (self.workspace / name).exists()
            }
            turn.diff = self._diff(parent_files, new_files)
            if turn.diff:
                break
            message = (
                "You did not change any editable file. Edit "
                f"{', '.join(self.editable_files)} in place to make your proposal."
            )
        if not turn.diff:
            logger.warning(f"candidate {candidate_id} is identical to its parent")
        self._write_candidate(turn, {**parent_files, **new_files})
        return turn

    def _send(self, message: str, parent_files: dict[str, str]):
        """Send a message to the agent, retrying failed calls with exponential
        backoff. If every attempt fails, the error is raised and stops tuning."""
        # TODO: record the turn as failed instead of raising after the last
        # attempt, so that running trials are not killed.
        for attempt in range(self.agent_retries + 1):
            try:
                return self.agent.send(
                    message, cwd=self.workspace, system_prompt=SYSTEM_PROMPT
                )
            except Exception as e:
                if attempt == self.agent_retries:
                    raise RuntimeError(
                        f"The agent failed {attempt + 1} times in a row, stopping."
                    ) from e
                wait = self.agent_retry_wait * 2**attempt
                logger.warning(
                    f"agent call failed ({e!r}), retrying in {wait:.0f}s "
                    f"({attempt + 1}/{self.agent_retries})"
                )
                time.sleep(wait)
                # Discards partial edits of the failed attempt.
                self._reset_workspace(parent_files)

    def _diff(self, old: dict[str, str], new: dict[str, str]) -> str:
        return "".join(
            "".join(
                difflib.unified_diff(
                    old.get(name, "").splitlines(keepends=True),
                    new.get(name, "").splitlines(keepends=True),
                    fromfile=f"a/{name}",
                    tofile=f"b/{name}",
                )
            )
            for name in self.editable_files
        )


def load_turns(path: str | Path) -> list[Turn]:
    """Read a ``turns.jsonl`` log, keeping the latest record of each candidate.

    :param path: Path of the log, or of the tuning folder containing it
    """
    path = Path(path)
    if path.is_dir():
        path = path / "autoresearch" / "turns.jsonl"
    turns = {}
    for line in path.read_text().splitlines():
        turn = Turn(**json.loads(line))
        turns[turn.candidate_id] = turn
    return [turns[k] for k in sorted(turns)]


def turns_to_dataframe(turns: list[Turn]) -> pd.DataFrame:
    """One row per candidate, without the prompts and raw agent messages."""
    rows = [
        {k: v for k, v in asdict(t).items() if k not in ("messages", "prompt")}
        for t in turns
    ]
    return pd.DataFrame(rows).set_index("candidate_id") if rows else pd.DataFrame()


def conversation_markdown(turn: Turn, max_tool_output: int = 2000) -> str:
    """Readable rendering of the agent conversation that proposed a candidate."""
    parent = "none" if turn.parent_id is None else turn.parent_id
    lines = [
        f"# Candidate {turn.candidate_id} (parent: {parent})",
        "",
        f"**Summary:** {turn.summary}",
        "",
        f"Session `{turn.session_id}`, cost ${turn.cost_usd:.4f}, "
        f"{turn.duration_s:.1f}s",
    ]
    if turn.prompt:
        lines += ["", "## Prompt", "", turn.prompt]
    if turn.messages:
        lines += ["", "## Agent"]
    for message in turn.messages:
        if message.get("type") not in ("assistant", "user"):
            continue
        content = message["message"]["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for block in content:
            kind = block.get("type")
            if kind == "text":
                lines += ["", block["text"]]
            elif kind == "tool_use":
                arguments = json.dumps(block["input"], indent=2)
                lines += ["", f"**Tool call** `{block['name']}`", "```json", arguments, "```"]
            elif kind == "tool_result":
                output = block.get("content")
                if isinstance(output, list):
                    output = "\n".join(c.get("text", "") for c in output)
                output = str(output)
                if len(output) > max_tool_output:
                    output = output[:max_tool_output] + "\n[...]"
                lines += ["", "**Tool result**", "```", output, "```"]
    if turn.diff:
        lines += ["", "## Diff", "", "```diff", turn.diff.rstrip("\n"), "```"]
    return "\n".join(lines) + "\n"

"""Entry point run by the trial backend for every candidate of the LLMSearcher.

Builds the trial folder from the base folder overlaid with the candidate files,
restores the frozen files from the base folder as read-only files, then runs
the train command, and the evaluate command if any, in the trial folder. The
trial fails if training changed a frozen file. This guards against accidental
changes only: candidates run as the same user and are trusted not to cheat.

The metric is parsed from the output of the last command run, by default from
the last line containing ``<metric>: <value>`` or ``<metric>=<value>``, and is
reported together with the training time.

On failure, the end of the error output is written to <candidate_dir>/error.txt
so that the searcher can show it to the agent, and the trial exits with an error.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
from syne_tune import Reporter
from syne_tune.constants import ST_CHECKPOINT_DIR
from syne_tune.utils import add_config_json_to_argparse, load_config_json

ERROR_FILE = "error.txt"
SPEC_FILE = "trial_spec.json"
ERROR_TAIL_LINES = 60
# Start of the error of a trial whose evaluation did not report the metric.
MISSING_METRIC_ERROR = "Missing metric:"
IGNORED = shutil.ignore_patterns("__pycache__", ERROR_FILE)
NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?i:nan|inf)"


def default_metric_regex(metric: str) -> str:
    """Matches ``metric: 0.87``, ``metric=0.87`` and ``"metric": 0.87`` in JSON."""
    return rf"(?<![\w.]){re.escape(metric)}\"?\s*[:=]\s*({NUMBER})"


def build_trial_dir(
    base_dir: Path, candidate_dir: Path, trial_dir: Path, frozen_files: list[str]
):
    # Read-only copies left by an earlier run of this trial id, when tuning is
    # resumed from a checkpoint, cannot be overwritten.
    for name in frozen_files:
        (trial_dir / name).unlink(missing_ok=True)
    shutil.copytree(base_dir, trial_dir, dirs_exist_ok=True, ignore=IGNORED)
    shutil.copytree(candidate_dir, trial_dir, dirs_exist_ok=True, ignore=IGNORED)
    for name in frozen_files:
        shutil.copy2(base_dir / name, trial_dir / name)
        # Read-only so that accidental writes by the candidate fail early.
        (trial_dir / name).chmod(0o444)


def check_frozen_files(base_dir: Path, trial_dir: Path, frozen_files: list[str]):
    """Raise if the candidate changed or deleted a frozen file while training.
    This catches accidental changes, not deliberate ones: candidates run as the
    same user and are trusted."""
    for name in frozen_files:
        path = trial_dir / name
        if not path.exists() or path.read_bytes() != (base_dir / name).read_bytes():
            raise RuntimeError(
                f"{name} is frozen but was changed or deleted during training."
            )


def parse_metric(output: str, metric: str, metric_regex: str | None = None) -> float:
    """Parse the value of the metric from the last match of metric_regex in the
    output, raise if there is none or if it is not finite.

    :param metric_regex: Regular expression whose first group is the value,
        defaults to :func:`default_metric_regex`
    """
    pattern = metric_regex or default_metric_regex(metric)
    matches = list(re.finditer(pattern, output))
    if not matches:
        tail = "\n".join(output.strip().splitlines()[-5:])
        raise RuntimeError(
            f"{MISSING_METRIC_ERROR} no line of the output reports {metric} "
            f"(regex {pattern!r}), the output ends with:\n{tail}"
        )
    text = matches[-1].group(1)
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise RuntimeError(
            f"{MISSING_METRIC_ERROR} {metric} must be a finite number, the "
            f"output reported {text!r}"
        )
    return value


def kill_process_tree(pid: int):
    """Kill a process and all its descendants, as syne-tune's LocalBackend does
    when stopping a trial. Descendants would otherwise keep the output pipes
    open and block the trial past its timeout."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for process in [*parent.children(recursive=True), parent]:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _forward(stream, out, lines: list[str]):
    """Write every line of stream to out as soon as it is read and keep it."""
    for line in stream:
        out.write(line)
        out.flush()
        lines.append(line)


def run(command: list[str], trial_dir: Path, timeout: float) -> str:
    """Run a command in trial_dir and return its stdout, raise on failure.

    The output of the command is forwarded line by line to the output of the
    trial while it runs, so that it can be followed live.
    """
    # "python" refers to the interpreter running the trial.
    command = [sys.executable if c == "python" else c for c in command]
    # Unbuffered so that the output of Python commands arrives line by line.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        command,
        cwd=trial_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    stdout, stderr = [], []
    threads = [
        threading.Thread(target=_forward, args=(process.stdout, sys.stdout, stdout)),
        threading.Thread(target=_forward, args=(process.stderr, sys.stderr, stderr)),
    ]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(process.pid)
        raise
    finally:
        for thread in threads:
            thread.join()
    if process.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(command)}` exited with code {process.returncode}:\n"
            f"{''.join(stderr)}"
        )
    return "".join(stdout)


def write_error(candidate_dir: Path, message: str):
    tail = "\n".join(message.strip().splitlines()[-ERROR_TAIL_LINES:])
    (candidate_dir / ERROR_FILE).write_text(tail)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate_dir", type=str)
    parser.add_argument("--trial_spec", type=str)
    parser.add_argument(f"--{ST_CHECKPOINT_DIR}", type=str, required=True)
    add_config_json_to_argparse(parser)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    config = load_config_json(args) if "st_config_json_filename" in args else args
    candidate_dir = Path(config["candidate_dir"])
    spec = json.loads(Path(config["trial_spec"]).read_text())
    trial_dir = Path(config[ST_CHECKPOINT_DIR]).parent
    timeout = spec["timeout"]

    start = time.time()
    try:
        build_trial_dir(
            Path(spec["base_dir"]), candidate_dir, trial_dir, spec["frozen_files"]
        )
        output = run(spec["train_command"], trial_dir, timeout)
        train_time = time.time() - start
        check_frozen_files(Path(spec["base_dir"]), trial_dir, spec["frozen_files"])
        if spec["evaluate_command"]:
            output = run(
                spec["evaluate_command"], trial_dir, max(timeout - train_time, 1.0)
            )
        value = parse_metric(output, spec["metric"], spec["metric_regex"])
        Reporter()(**{"train_time": train_time, spec["metric"]: value})
    except subprocess.TimeoutExpired:
        write_error(candidate_dir, f"Timeout: exceeded {timeout:.0f}s.")
        raise
    except Exception as e:
        write_error(candidate_dir, str(e))
        raise


if __name__ == "__main__":
    main()

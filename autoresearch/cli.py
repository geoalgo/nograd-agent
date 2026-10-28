"""Improve a training script with a coding agent, every version runs as a trial.

The source folder contains the training script, by default train.py, and an
optional configuration file, by default nograd.yaml, whose keys are the long
options below without dashes. Options passed on the command line override the
configuration file. Paths of scripts and files are relative to the source
folder, which is copied to every trial.

Usage:
    nograd-agent --src_dir examples/tabular/task --max_trials 3
"""

import argparse
import logging
import re
from pathlib import Path

import yaml
from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import LocalBackend
from syne_tune.results_callback import StoreResultsCallback

from autoresearch import (
    ENTRY_POINT,
    AutoSearch,
    AutoSearchCallback,
    AutoSearchLogger,
    ClaudeCodeAgent,
    LLMSearcher,
)

CONFIG_FILE = "nograd.yaml"

EPILOG = """\
The metric is parsed from the output of the training script, or of the
evaluation script if one is given: the last line containing "<metric>: <value>"
or "<metric>=<value>", or the last match of --metric_regex, whose first group
is the value. The training script must not change the evaluation script and
the frozen files, which the agent can only read.
"""

# Settings that can be set in the configuration file or on the command line,
# with their default.
DEFAULTS = {
    "metric": None,
    "mode": "max",
    "metric_regex": None,
    "prompt": None,
    "prompt_file": None,
    "train_script": "train.py",
    "eval_script": None,
    "editable_files": None,
    "frozen_files": [],
    "n_workers": 1,
    "max_trials": 10,
    "trial_timeout": 120.0,
    "evaluate_baseline": True,
    "model": "sonnet",
    "effort": None,
    "name": None,
}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nograd-agent",
        description=__doc__.split("\n\n")[0],
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src_dir", required=True, help="Folder with the scripts, copied to every trial"
    )
    parser.add_argument(
        "--config", help=f"Configuration file, defaults to <src_dir>/{CONFIG_FILE}"
    )
    # Every default is None so that only options passed override the
    # configuration file, the actual defaults are in DEFAULTS.
    task = parser.add_argument_group("task")
    task.add_argument("--metric", help="Name of the metric to optimize, required")
    task.add_argument("--mode", choices=["max", "min"], help="Default: max")
    task.add_argument("--metric_regex", help="Regex whose first group is the metric")
    task.add_argument("--prompt", help="Description of the task given to the agent")
    task.add_argument("--prompt_file", help="File containing the description of the task")
    task.add_argument("--train_script", help="Default: train.py")
    task.add_argument("--eval_script", help="Run after training if given, frozen")
    task.add_argument(
        "--editable_files", nargs="+", help="Files the agent may change, default: the train script"
    )
    task.add_argument(
        "--frozen_files", nargs="+", help="Files the agent can only read, besides the eval script"
    )
    budget = parser.add_argument_group("budget")
    budget.add_argument("--n_workers", type=int, help="Trials run in parallel, default: 1")
    budget.add_argument("--max_trials", type=int, help="Default: 10")
    budget.add_argument("--trial_timeout", type=float, help="Seconds, default: 120")
    budget.add_argument(
        "--evaluate_baseline",
        action=argparse.BooleanOptionalAction,
        help="Whether the first trial runs the unmodified script, default: true",
    )
    agent = parser.add_argument_group("agent")
    agent.add_argument("--model", help="Claude model, default: sonnet")
    agent.add_argument("--effort", help="Reasoning effort of the model")
    parser.add_argument("--name", help="Name of the experiment, default: the source folder name")
    return parser


def load_settings(argv: list[str] | None = None) -> dict:
    """Settings from the defaults, overridden by the configuration file, then by
    the command line, with paths checked and resolved in the source folder."""
    parser = make_parser()
    args = vars(parser.parse_args(argv))
    src_dir = Path(args.pop("src_dir")).resolve()
    if not src_dir.is_dir():
        parser.error(f"--src_dir {src_dir} is not a folder")
    config_path = args.pop("config")
    config = {}
    if config_path is not None or (src_dir / CONFIG_FILE).exists():
        config_path = Path(config_path or src_dir / CONFIG_FILE)
        if not config_path.exists():
            parser.error(f"configuration file {config_path} not found")
        config = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(config, dict):
            parser.error(f"{config_path} must contain a mapping of settings")
        unknown = sorted(set(config) - set(DEFAULTS))
        if unknown:
            parser.error(
                f"unknown settings in {config_path}: {', '.join(unknown)}, "
                f"expected some of {', '.join(DEFAULTS)}"
            )
    passed = {k: v for k, v in args.items() if v is not None}
    # A prompt passed on the command line replaces the one of the configuration
    # file, whichever way either is given.
    if "prompt" in passed or "prompt_file" in passed:
        config = {k: v for k, v in config.items() if k not in ("prompt", "prompt_file")}
    settings = {**DEFAULTS, **config, **passed, "src_dir": str(src_dir)}

    if settings["metric"] is None:
        parser.error("the metric is required, set it with --metric or in the configuration file")
    if settings["mode"] not in ("max", "min"):
        parser.error(f"mode must be max or min, got {settings['mode']}")

    def relative(name: str, setting: str) -> str:
        path = (src_dir / name).resolve()
        if not path.is_relative_to(src_dir):
            parser.error(f"{setting} {name} must be inside the source folder {src_dir}")
        if not path.is_file():
            parser.error(f"{setting} {name} not found in {src_dir}")
        return path.relative_to(src_dir).as_posix()

    if settings["prompt"] is not None and settings["prompt_file"] is not None:
        parser.error("set either prompt or prompt_file, not both")
    if settings["prompt_file"] is not None:
        prompt_file = relative(settings["prompt_file"], "prompt_file")
        settings["prompt"] = (src_dir / prompt_file).read_text()
    if not settings["prompt"]:
        parser.error("describe the task with --prompt or --prompt_file, or in the configuration file")

    settings["train_script"] = relative(settings["train_script"], "train_script")
    if settings["eval_script"] is not None:
        settings["eval_script"] = relative(settings["eval_script"], "eval_script")
    editable = settings["editable_files"] or [settings["train_script"]]
    settings["editable_files"] = [relative(name, "editable file") for name in editable]
    frozen = [relative(name, "frozen file") for name in settings["frozen_files"]]
    if settings["eval_script"] is not None and settings["eval_script"] not in frozen:
        frozen.append(settings["eval_script"])
    settings["frozen_files"] = frozen
    both = set(settings["editable_files"]) & set(frozen)
    if both:
        parser.error(f"files cannot be both editable and frozen: {', '.join(sorted(both))}")
    if settings["name"] is None:
        settings["name"] = src_dir.name
    return settings


def make_tuner(settings: dict) -> Tuner:
    eval_script = settings["eval_script"]
    searcher = LLMSearcher(
        task_prompt=settings["prompt"],
        base_dir=settings["src_dir"],
        editable_files=settings["editable_files"],
        frozen_files=settings["frozen_files"],
        train_command=["python", settings["train_script"]],
        evaluate_command=None if eval_script is None else ["python", eval_script],
        metric=settings["metric"],
        mode=settings["mode"],
        metric_regex=settings["metric_regex"],
        agent=ClaudeCodeAgent(model=settings["model"], effort=settings["effort"]),
        trial_timeout=settings["trial_timeout"],
        evaluate_baseline=settings["evaluate_baseline"],
    )
    return Tuner(
        trial_backend=LocalBackend(entry_point=ENTRY_POINT, pass_args_as_json=True),
        scheduler=AutoSearch(searcher),
        callbacks=[StoreResultsCallback(), AutoSearchCallback()],
        stop_criterion=StoppingCriterion(max_num_trials_finished=settings["max_trials"]),
        n_workers=settings["n_workers"],
        # Broken candidates are expected and reported back to the agent.
        max_failures=settings["max_trials"],
        # Lets the last running trials finish so that every proposal is scored.
        wait_trial_completion_when_stopping=True,
        sleep_time=1.0,
        tuner_name=re.sub(r"[^a-zA-Z0-9-]", "-", settings["name"]),
        output_logger=AutoSearchLogger(searcher),
    )


def main(argv: list[str] | None = None):
    settings = load_settings(argv)
    logging.basicConfig(level=logging.INFO)
    tuner = make_tuner(settings)
    # Records the settings actually used.
    tuner.tuner_path.mkdir(parents=True, exist_ok=True)
    (tuner.tuner_path / CONFIG_FILE).write_text(yaml.safe_dump(settings, sort_keys=False))
    tuner.run()


if __name__ == "__main__":
    main()

"""Example of a coding agent, here Claude Code, improving a small tabular model.

Every trial runs a new version of task/train.py proposed by the agent from the
results of the previous versions. Requires the Claude Code CLI
(https://docs.claude.com/en/docs/claude-code).

This is the Python API, the same search runs from the command line with:
    nograd-agent --src_dir examples/tabular/task

Usage:
    python examples/tabular/launch.py --n_workers 2 --max_trials 10
"""

import argparse
import logging
from pathlib import Path

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

TASK_DIR = Path(__file__).parent / "task"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n_workers", type=int, default=2)
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--trial_timeout", type=float, default=120.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    searcher = LLMSearcher(
        task_prompt=(TASK_DIR / "PROMPT.md").read_text(),
        base_dir=TASK_DIR,
        editable_files=["train.py"],
        frozen_files=["eval.py"],
        train_command=["python", "train.py"],
        # Parsed from the line "valid_accuracy: 0.85" printed by train.py.
        metric="valid_accuracy",
        mode="max",
        agent=ClaudeCodeAgent(model=args.model),
        trial_timeout=args.trial_timeout,
    )
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=ENTRY_POINT, pass_args_as_json=True),
        scheduler=AutoSearch(searcher),
        callbacks=[StoreResultsCallback(), AutoSearchCallback()],
        stop_criterion=StoppingCriterion(max_num_trials_finished=args.max_trials),
        n_workers=args.n_workers,
        # Broken candidates are expected and reported back to the agent.
        max_failures=args.max_trials,
        # Lets the last running trials finish so that every proposal is scored.
        wait_trial_completion_when_stopping=True,
        sleep_time=1.0,
        tuner_name="autoresearch-tabular",
        output_logger=AutoSearchLogger(searcher),
    )
    tuner.run()

    best = searcher.best()
    if best is not None:
        print(f"best candidate {best.candidate_id}: {best.result}")
    print(f"agent conversations and candidates: {searcher.output_dir}")


if __name__ == "__main__":
    main()

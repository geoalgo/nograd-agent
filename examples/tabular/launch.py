"""Example of a coding agent, here Claude Code, improving a small tabular model.

Every trial runs a new version of task/train.py proposed by the agent from the
results of the previous versions. Requires the Claude Code CLI
(https://docs.claude.com/en/docs/claude-code).

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
    AutoResearch,
    AutoResearchCallback,
    AutoResearchLogger,
    ClaudeCodeAgent,
    LLMSearcher,
)

TASK_PROMPT = """\
Improve the accuracy of a classifier on the Adult (census income) tabular \
dataset: predict whether income is ">50K" or "<=50K" from 14 numeric and \
categorical columns, some with missing values.

train.py fits a scikit-learn model on the training split and saves it with \
joblib; eval.py scores it on a validation split, which is the metric you \
optimize. Constraints:
- train.py must keep a main() function that fits and saves the model to \
MODEL_PATH, and the saved model must accept the raw feature DataFrame in predict().
- Only use the training data from load_train_data(); never load the validation \
or test data or call functions from eval.py other than load_train_data.
- Training and evaluation together must finish in under 60 seconds on a laptop \
CPU. Only scikit-learn, numpy, pandas and joblib are installed.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n_workers", type=int, default=2)
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--trial_timeout", type=float, default=120.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    searcher = LLMSearcher(
        task_prompt=TASK_PROMPT,
        base_dir=Path(__file__).parent / "task",
        editable_files=["train.py"],
        frozen_files=["eval.py"],
        # Imports train.py as a module rather than running it as __main__, so
        # that classes it defines can be unpickled when evaluating.
        train_command=["python", "-c", "import train; train.main()"],
        evaluate_command=["python", "eval.py"],
        metric="valid_accuracy",
        mode="max",
        agent=ClaudeCodeAgent(model=args.model),
        trial_timeout=args.trial_timeout,
    )
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=ENTRY_POINT, pass_args_as_json=True),
        scheduler=AutoResearch(searcher),
        callbacks=[StoreResultsCallback(), AutoResearchCallback()],
        stop_criterion=StoppingCriterion(max_num_trials_finished=args.max_trials),
        n_workers=args.n_workers,
        # Broken candidates are expected and reported back to the agent.
        max_failures=args.max_trials,
        # Lets the last running trials finish so that every proposal is scored.
        wait_trial_completion_when_stopping=True,
        sleep_time=1.0,
        tuner_name="autoresearch-tabular",
        output_logger=AutoResearchLogger(searcher),
    )
    tuner.run()

    best = searcher.best()
    if best is not None:
        print(f"best candidate {best.candidate_id}: {best.result}")
    print(f"agent conversations and candidates: {searcher.output_dir}")


if __name__ == "__main__":
    main()

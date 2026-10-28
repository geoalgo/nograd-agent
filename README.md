# Nograd-agent: No-grad Agent Descent

This repository gives code to run Autoresearch with [Syne Tune](https://github.com/syne-tune/syne-tune). 
In case you know what grad-student descent is, this repo is doing no-grad agent descent as agents are *not yet*
differentiable.

A coding agent (Claude Code) proposes new versions of a training script, and Syne Tune
runs and scores them as trials. The example task improves a small scikit-learn
classifier on the Adult census income dataset.

This is a standalone package that depends on the released `syne-tune` from
PyPI, it plugs into Syne Tune through its public scheduler, backend and
callback interfaces and does not modify or vendor any Syne Tune code.

The key benefits are to get:
- trial abstractions and execution: multiprocessing / multiGPU supported out of the box
- easy to report scores without modifying train.py functions by just setting regexp
- early stopping / successive halving available out of the box

The repo is very much WIP, any feedback welcome. 

**Related repos.**

I was looking for a repository to run autoresearch and did not find any except the [original](https://github.com/karpathy/autoresearch) from Karpathy, which
is minimal (mostly a markdown file). That is great, but it does not offer many features either.
On the HPO / autoresearch space I am only aware of [autotune](https://github.com/RobertTLange/autotune), which derives a search space from your program, whereas
nograd-agent uses autoresearch as a *searcher* in a standard hyperparameter optimization library.
This [repo](https://github.com/ferreirafabio/autoresearch-automl) is also nice and benchmarks autoresearch against standard HPO methods, whereas nograd-agent can combine autoresearch with them.

## Layout

- `autoresearch/searchers/`: the searcher and the coding agent
  - `llm_searcher.py`: `LLMSearcher`, asks the agent for a new candidate given
    the results so far, stores candidates, conversations and the log of turns
  - `agent.py`: `CodeAgent` interface and `ClaudeCodeAgent`, which calls the
    Claude Code CLI in print mode
  - `prompts.py`: system prompt and prompt asking for the next candidate
- `autoresearch/scheduler.py`: `AutoResearch`, the Syne Tune scheduler that
  forwards trial events to the searcher, and `AutoResearchCallback`, which
  sets the output folder of the searcher to `<tuner_path>/autoresearch`
- `autoresearch/logger.py`: `AutoResearchLogger`, output of the tuner showing
  the summary of each candidate and the total LLM cost, passed with
  `Tuner(..., output_logger=AutoResearchLogger(searcher))`
- `autoresearch/run_trial.py`: entry point of every trial (`ENTRY_POINT`),
  builds the trial folder from the candidate, runs training and evaluation
  and reports the metrics
- `examples/tabular/launch.py`: launches the search on the example task
- `examples/tabular/task/train.py`: the training script the agent modifies
- `examples/tabular/task/eval.py`: the fixed evaluation, the agent cannot
  modify it
- `tests/`: tests of the whole loop with a fake agent

## Running

Requires [uv](https://docs.astral.sh/uv/) and the
[Claude Code CLI](https://docs.claude.com/en/docs/claude-code) (logged in).

```bash
uv sync --extra examples
uv run python examples/tabular/launch.py --n_workers 2 --max_trials 10
```

Options of `launch.py`: `--model` (default `sonnet`) and `--trial_timeout` in
seconds (default 120). Agent conversations and candidate scripts are written
to the folder printed at the end of the run.

To train and evaluate the current script by hand:

```bash
cd examples/tabular/task && uv run python train.py && uv run python eval.py
```

To run the tests:

```bash
uv run pytest tests
```

## Using it on another task

Put the initial script in a folder with a fixed evaluation script whose last
line of output is a JSON dictionary of metrics, then give `LLMSearcher` the
files the agent may edit (`editable_files`), the ones it may only read
(`frozen_files`, read-only for the agent and restored in every trial, which
fails if training changed them; this catches mistakes, candidates are trusted
not to cheat on purpose), the train and evaluate commands and the metric to optimize.
See `examples/tabular/launch.py`.

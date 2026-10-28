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
- `autoresearch/scheduler.py`: `AutoSearch`, the Syne Tune scheduler that
  forwards trial events to the searcher, and `AutoSearchCallback`, which
  sets the output folder of the searcher to `<tuner_path>/autoresearch`
- `autoresearch/logger.py`: `AutoSearchLogger`, output of the tuner showing
  the summary of each candidate and the total LLM cost, passed with
  `Tuner(..., output_logger=AutoSearchLogger(searcher))`
- `autoresearch/run_trial.py`: entry point of every trial (`ENTRY_POINT`),
  builds the trial folder from the candidate, runs the script and reports
  the metric parsed from its output
- `autoresearch/cli.py`: the `nograd-agent` command
- `examples/tabular/task/`: the example task
  - `train.py`: the training script the agent modifies, prints the scores
  - `eval.py`: the fixed data splits and evaluation, the agent cannot modify it
  - `PROMPT.md`: description of the task given to the agent
  - `nograd.yaml`: settings of the search
- `examples/tabular/launch.py`: the same search with the Python API
- `tests/`: tests of the whole loop with a fake agent

## Running

Requires [uv](https://docs.astral.sh/uv/) and the
[Claude Code CLI](https://docs.claude.com/en/docs/claude-code) (logged in).

```bash
uv sync --extra examples
uv run nograd-agent --src_dir examples/tabular/task --max_trials 5
```

Settings are read from `nograd.yaml` in the source folder, options passed on
the command line override them, see `nograd-agent --help`. Agent
conversations and candidate scripts are written to the folder printed at the
end of the run, together with the settings used.

To train and evaluate the current script by hand:

```bash
cd examples/tabular/task && uv run python train.py
```

To run the tests:

```bash
uv run pytest tests
```

## Using it on another task

Put the training script in a folder, by default `train.py`, which prints the
metric to optimize as `<metric>: <value>` or `<metric>=<value>`; the last
such line is used. Describe the task in a prompt and set the metric in
`nograd.yaml`:

```yaml
metric: valid_accuracy
mode: max                  # or min
prompt_file: PROMPT.md     # or `prompt: ...`
frozen_files: [eval.py]    # files the agent can read but not change
```

Then run `nograd-agent --src_dir <folder>`. The whole folder is copied to
every trial and paths are relative to it. Other settings:

- `train_script`: the script run in every trial, default `train.py`
- `editable_files`: files the agent may change, default the train script
- `eval_script`: a script run after training, from whose output the metric is
  then parsed; it is frozen
- `metric_regex`: regular expression whose first group is the value of the
  metric, for other output formats
- `n_workers`, `max_trials`, `trial_timeout`, `model`, `effort`, `name`

Frozen files are read-only for the agent and restored in every trial, which
fails if training changed them. This catches mistakes, candidates are trusted
not to cheat on purpose: the training script prints its own score.

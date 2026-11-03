"""Summarize a search: statistics in text and a bar plot of the metric per candidate.

Every bar is labelled with a 2 to 4 word description of the change of the
candidate (for instance "tuned learning rate"), written by a Claude model from
the summary of the agent and cached in <run>/autoresearch/trial_labels.json.
Without the Claude Code CLI or with --no_llm_labels, the first words of the
summary are used instead. Bars are colored with viridis by candidate index.

Usage:
    nograd-analyse                                  # latest search
    nograd-analyse ~/syne-tune/<search> --output plot.png
"""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from syne_tune.constants import ST_METADATA_FILENAME

from autoresearch.run_trial import SPEC_FILE
from autoresearch.searchers.llm_searcher import load_turns, turns_to_dataframe

LABELS_FILE = "trial_labels.json"
# Width of the plot per candidate, in inches.
BAR_INCHES = 0.25


def experiments_folder() -> Path:
    return Path(os.environ.get("SYNETUNE_FOLDER", Path.home() / "syne-tune")).expanduser()


def latest_run() -> Path:
    """The latest search with at least one evaluated candidate."""
    logs = sorted(
        experiments_folder().glob("*/autoresearch/turns.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for log in logs:
        if any(turn.metric is not None for turn in load_turns(log)):
            return log.parent.parent
    raise FileNotFoundError(f"no search with an evaluated candidate in {experiments_folder()}")


def load_candidates(run: Path) -> pd.DataFrame:
    return turns_to_dataframe(load_turns(run))


def metric_and_mode(run: Path, candidates: pd.DataFrame, metric=None, mode=None) -> tuple[str, str]:
    """The metric and mode given, else the ones saved with the search: in the
    Syne Tune metadata, the trial spec or the settings of the command line.
    Searches saved before the metric was recorded get the result matching the
    metric of the candidates."""
    saved = {}
    metadata_path = run / ST_METADATA_FILENAME
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("metric_names"):
            saved = {"metric": metadata["metric_names"][0], "mode": metadata.get("metric_mode")}
    for path in (run / "nograd.yaml", run / "autoresearch" / SPEC_FILE):
        if path.exists():
            saved = {**yaml.safe_load(path.read_text()), **saved}
    metric = metric or saved.get("metric")
    if metric is None:
        completed = candidates[candidates["metric"].notna()]
        if completed.empty:
            raise ValueError(
                f"no candidate of {run} was evaluated, so its metric is unknown: "
                "pass --metric, or another search folder"
            )
        names = {
            name
            for _, row in completed.iterrows()
            for name, value in row["result"].items()
            if value == row["metric"]
        }
        if len(names) != 1:
            raise ValueError(f"cannot tell the metric of {run}, pass --metric")
        metric = names.pop()
    return metric, mode or saved.get("mode", "max")


def first_words(summary: str, n: int = 4) -> str:
    words = summary.rstrip(".").split()
    return " ".join(words[:n]) + ("..." if len(words) > n else "")


def short_labels(run: Path, candidates: pd.DataFrame, model: str | None = "haiku") -> dict[int, str]:
    """A label of 2 to 4 words per candidate, asking the model once for the
    candidates not in the cache of the run, the first words of the summary if
    model is None or the call fails."""
    cache_path = run / "autoresearch" / LABELS_FILE
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    labels = cache.get(model, {}) if model else {}
    missing = [c for c in candidates.index if str(c) not in labels]
    if model and missing:
        prompt = (
            "Each line below describes the change one trial made to a training "
            "script. Give each one a label of 2 to 4 words, like 'tuned LR', "
            "'deeper trees' or 'one-hot categoricals'. Answer only with a JSON "
            "object mapping each id to its label.\n\n"
            + "\n".join(f"{c}: {candidates.loc[c, 'summary']}" for c in missing)
        )
        # Allows the script to run from within a Claude Code session.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        command = ["claude", "-p", "--model", model, "--restricted", "--strict-mcp-config", "--tools", ""]
        try:
            output = subprocess.run(
                command, input=prompt, capture_output=True, text=True, timeout=120, env=env, check=True
            ).stdout
            answer = json.loads(output[output.index("{") : output.rindex("}") + 1])
            labels.update({str(k): str(v) for k, v in answer.items()})
            cache[model] = labels
            cache_path.write_text(json.dumps(cache, indent=2))
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            print(f"could not get labels from {model}, using the first words of the summaries: {e}")
    return {c: labels.get(str(c)) or first_words(candidates.loc[c, "summary"]) for c in candidates.index}


def statistics(run: Path, candidates: pd.DataFrame, metric: str, mode: str) -> str:
    sign = 1 if mode == "max" else -1
    counts = candidates["status"].value_counts()
    completed = candidates[candidates["metric"].notna()]
    lines = [
        f"Search: {run}",
        f"Metric: {metric}, {'higher' if mode == 'max' else 'lower'} is better",
        f"Candidates: {len(candidates)} proposed, {counts.get('completed', 0)} evaluated, "
        f"{counts.get('error', 0)} failed, {counts.get('interrupted', 0)} interrupted, "
        f"{counts.get('pending', 0)} running",
    ]
    baseline = None
    if 0 in completed.index and pd.isna(candidates.loc[0, "parent_id"]) and not candidates.loc[0, "diff"]:
        baseline = completed.loc[0, "metric"]
        lines.append(f"Baseline (candidate 0): {baseline:.4f}")
    if not completed.empty:
        best_id = (sign * completed["metric"]).idxmax()
        best = completed.loc[best_id, "metric"]
        line = f"Best (candidate {best_id}): {best:.4f}"
        if baseline is not None:
            gain = best - baseline
            line += f", {gain:+.4f} over the baseline"
            if baseline != 0:
                line += f" ({100 * gain / abs(baseline):+.1f}%)"
        lines.append(line)
        lines.append(f"  {completed.loc[best_id, 'summary']}")
        others = {
            k: v for k, v in completed.loc[best_id, "result"].items()
            if k != metric and isinstance(v, (int, float))
        }
        if others:
            lines.append("  other results: " + ", ".join(f"{k}={v:.4g}" for k, v in others.items()))
        lines.append(f"  files: {run / 'autoresearch' / 'candidates' / str(best_id)}")
        # Candidates that improved on all the previous ones.
        record, improvements = -math.inf, []
        for c, value in completed["metric"].items():
            if sign * value > record:
                record = sign * value
                improvements.append(str(c))
        lines.append(f"New best found by candidates: {', '.join(improvements)}")
    train_time = sum(r.get("train_time", 0.0) for r in completed["result"])
    lines.append(
        f"LLM cost: {candidates['cost_usd'].sum():.2f}$, agent time "
        f"{candidates['duration_s'].sum() / 60:.1f} min, training time {train_time / 60:.1f} min"
    )
    return "\n".join(lines)


def candidates_table(candidates: pd.DataFrame, metric: str) -> str:
    table = pd.DataFrame(
        {
            "parent": candidates["parent_id"].map(lambda p: "-" if p is None or pd.isna(p) else int(p)),
            metric: candidates["metric"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}"),
            "status": candidates["status"],
            "change": candidates["label"],
        }
    )
    return table.to_string()


def plot(run: Path, candidates: pd.DataFrame, metric: str, mode: str, output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(candidates)
    ids = list(candidates.index)
    cmap = plt.get_cmap("viridis")
    color = {c: cmap(i / max(n - 1, 1)) for i, c in enumerate(ids)}
    fontsize = 6 if n > 30 else 8
    fig, ax = plt.subplots(figsize=(max(6.0, BAR_INCHES * n + 2.5), 6.0), constrained_layout=True)

    done = candidates[candidates["metric"].notna()]
    values = done["metric"]
    if values.empty:
        low, high = 0.0, 1.0
    else:
        span = (values.max() - values.min()) or abs(values.max()) or 1.0
        low, high = values.min() - 0.15 * span, values.max() + 0.2 * span
    ax.bar(
        done.index,
        values - low,
        bottom=low,
        color=[color[c] for c in done.index],
        width=0.75,
        edgecolor="none",
        linewidth=0,
    )
    for c, value in values.items():
        ax.annotate(
            f"{value:.4f}",
            (c, value),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=fontsize,
            color="0.25",
            # Readable over the line of the best metric.
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.8},
            zorder=4,
        )
    # Candidates without a metric have no bar, their status is in their label.
    ticks = []
    for c in ids:
        status = candidates.loc[c, "status"]
        prefix = {"error": "failed: ", "interrupted": "interrupted: ", "pending": "running: "}
        ticks.append(f"{prefix.get(status, '')}{candidates.loc[c, 'label']} ({c})")

    # Best metric found so far, as a step line over the candidates.
    if not values.empty:
        best = values.cummax() if mode == "max" else values.cummin()
        ax.step(best.index, best.values, where="mid", color="red", lw=1.5, label="best so far", zorder=3)
        if 0 in values.index:
            ax.axhline(values.loc[0], color="black", ls="--", lw=1, label="baseline", zorder=3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax.set_ylim(low, high)
    ax.set_xlim(min(ids) - 0.6, max(ids) + 0.6)
    ax.set_xticks(ids, ticks, rotation=90, fontsize=fontsize)
    for tick, c in zip(ax.get_xticklabels(), ids):
        if c not in values.index:
            tick.set_color("0.5")
    ax.set_ylabel(f"{metric} ({'higher' if mode == 'max' else 'lower'} is better)")
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(run.name, fontsize=10, color="0.4")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="nograd-analyse", description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, nargs="?", help="Search folder, default: the latest one")
    parser.add_argument("--output", type=Path, help="Plot file, default: <run>/analysis.png")
    parser.add_argument("--metric", help="Default: the metric of the search")
    parser.add_argument("--mode", choices=["max", "min"], help="Default: the mode of the search")
    parser.add_argument("--label_model", default="haiku", help="Claude model writing the labels, default: haiku")
    parser.add_argument(
        "--no_llm_labels", action="store_true", help="Label bars with the first words of the summaries"
    )
    args = parser.parse_args(argv)
    try:
        run = (args.run or latest_run()).expanduser().resolve()
        candidates = load_candidates(run)
        if candidates.empty:
            parser.error(f"no candidate in {run}")
        metric, mode = metric_and_mode(run, candidates, args.metric, args.mode)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
    model = None if args.no_llm_labels else args.label_model
    analyse(run, candidates, metric, mode, args.output, model)


def analyse(
    run: Path,
    candidates: pd.DataFrame | None = None,
    metric: str | None = None,
    mode: str | None = None,
    output: Path | None = None,
    label_model: str | None = "haiku",
):
    """Print the statistics and the table of candidates of a search and save its
    plot, by default to <run>/analysis.png.

    :param label_model: Claude model writing the labels of the candidates, None
        to use the first words of their summaries
    """
    if candidates is None:
        candidates = load_candidates(run)
    metric, mode = metric_and_mode(run, candidates, metric, mode)
    output = output or run / "analysis.png"
    candidates["label"] = pd.Series(short_labels(run, candidates, label_model))

    print(statistics(run, candidates, metric, mode))
    print()
    print(candidates_table(candidates, metric))
    try:
        plot(run, candidates, metric, mode, output)
    except ImportError:
        print("\nInstall matplotlib to plot the results, for instance with `uv sync --extra plot`.")
    else:
        print(f"\nPlot saved to {output}")


if __name__ == "__main__":
    main()

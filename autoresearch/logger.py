from pathlib import Path

import pandas as pd
from syne_tune.tuner_logger import Colors, TunerLogger
from syne_tune.tuning_status import TuningStatus

from autoresearch.searchers.llm_searcher import LLMSearcher


class AutoResearchLogger(TunerLogger):
    """Output of the tuner adapted to an :class:`LLMSearcher`: titles name
    AutoResearch, trials are shown with the summary of their change, and the
    total cost of the agent is printed at the end. Pass it to the tuner with
    ``Tuner(..., output_logger=AutoResearchLogger(searcher))``.

    :param searcher: Searcher of the :class:`AutoResearch` scheduler
    :param max_summary_width: Summaries are cut to this number of characters in
        the status table
    """

    def __init__(
        self,
        searcher: LLMSearcher,
        max_summary_width: int = 100,
        use_colors: bool = True,
        use_emojis: bool = True,
    ):
        super().__init__(use_colors=use_colors, use_emojis=use_emojis)
        self.searcher = searcher
        self.max_summary_width = max_summary_width

    def print_experiment_header(
        self,
        name: str,
        backend_name: str,
        n_workers: int,
        scheduler_name: str,
        results_path: Path,
        log_path: str,
        **kwargs,
    ):
        separator = "━" * 80
        title = self._format_message("🔬", "AutoResearch - Syne Tune", Colors.CYAN + Colors.BOLD)
        direction = "higher" if self.searcher.mode == "max" else "lower"
        print(f"\n{title}")
        print(separator)
        print(f"\n{self._format_message('📋', 'Experiment Configuration', Colors.BOLD)}")
        print(f"├─ Name: {self._color(name, Colors.GREEN)}")
        print(f"├─ Agent: {self._color(self._agent_name(), Colors.MAGENTA)}")
        print(f"├─ Workers: {self._color(str(n_workers), Colors.YELLOW)}")
        print(f"├─ Editable files: {self._color(', '.join(self.searcher.editable_files), Colors.CYAN)}")
        print(f"├─ Metric: {self._color(f'{self.searcher.metric} ({direction} is better)', Colors.CYAN)}")
        print(f"└─ Results Path: {self._color(str(results_path), Colors.BLUE)}")
        print(f"\n{separator}\n")

    def print_tuning_start(self):
        print(self._format_message("🏁", "Starting AutoResearch...", Colors.GREEN))

    def print_tuning_status(self, tuning_status: TuningStatus):
        separator = "━" * 80
        print(f"\n{separator}")
        print(self._format_message("📊", "AutoResearch Status", Colors.BOLD))
        print(self._history_table())
        print(f"{separator}\n")

    def print_trial_started(self, trial_id: int, config: dict):
        turn = self._turn(trial_id)
        summary = f": {turn.summary}" if turn is not None else ""
        timestamp = self._get_timestamp()
        print(f"{timestamp} {self._format_message('🚀', f'Trial {trial_id} started{summary}', Colors.GREEN)}")

    def print_searcher_out_of_candidates(self):
        print(self._format_message("🔍", "The agent made no new proposal, stopping.", Colors.CYAN))

    def print_tuning_complete(self):
        separator = "━" * 80
        print(f"\n{separator}")
        print(self._format_message("🎉🎉🎉", "AUTORESEARCH COMPLETE!", Colors.GREEN + Colors.BOLD))
        print(f"{separator}\n")
        print(self._history_table())

    def print_tuning_finished(self, results_path: Path):
        print(f"\n💾 {self._color('Results saved to:', Colors.BOLD)} {self._color(str(results_path), Colors.BLUE)}")
        cost = sum(turn.cost_usd for turn in self.searcher.turns)
        print(f"💰 {self._color('LLM cost:', Colors.BOLD)} {self._color(f'{cost:.2f}$', Colors.YELLOW)}")
        print(f"✨ {self._color('Happy training with your optimized function!', Colors.CYAN)}\n")

    def print_best_config_instructions(self, trial_id: int, config: dict):
        # Retraining is done from the candidate files rather than a config.
        turn = self._turn(trial_id)
        if turn is not None and self.searcher.output_dir is not None:
            folder = self.searcher.candidate_dir(turn.candidate_id)
            print(f"\n💡 {self._color('Best candidate files:', Colors.CYAN + Colors.BOLD)} {folder}\n")

    def _agent_name(self) -> str:
        agent = self.searcher.agent
        model = getattr(agent, "model", None)
        return type(agent).__name__ + (f" ({model})" if model else "")

    def _turn(self, trial_id: int):
        return next((t for t in self.searcher.turns if t.trial_id == trial_id), None)

    def _history_table(self) -> str:
        history = self.searcher.history()
        if history.empty:
            return "No trial yet."
        table = history.reset_index()[["trial_id", "metric", "summary", "status"]]
        table = table.rename(columns={"metric": self.searcher.metric})
        width = self.max_summary_width
        summaries = table["summary"].map(
            lambda s: s if len(s) <= width else s[: width - 3] + "..."
        )
        # Left-aligned, pandas right-aligns text columns by default.
        table["summary"] = summaries.str.ljust(summaries.str.len().max())
        table["trial_id"] = table["trial_id"].astype("Int64")
        with pd.option_context("display.max_colwidth", None, "display.width", None):
            return table.to_string(index=False)

import logging
from pathlib import Path
from typing import Any

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import TrialScheduler, TrialSuggestion
from syne_tune.tuner_callback import TunerCallback

from autoresearch.searchers.llm_searcher import LLMSearcher

logger = logging.getLogger(__name__)


class AutoResearch(TrialScheduler):
    """Tuning method where a coding agent proposes new versions of a script.

    Trials are not stopped early: every candidate is trained and evaluated fully.

    Forwards trial events to an :class:`LLMSearcher` and stores its workspace,
    candidates and log in ``<tuner_path>/autoresearch``, which requires passing
    :class:`AutoResearchCallback` to the tuner. Trials must be run with
    :const:`autoresearch.ENTRY_POINT` as entry point, for instance:

    .. code-block:: python

        from autoresearch import (
            ENTRY_POINT,
            AutoResearch,
            AutoResearchCallback,
            LLMSearcher,
        )
        from syne_tune.results_callback import StoreResultsCallback

        tuner = Tuner(
            trial_backend=LocalBackend(entry_point=ENTRY_POINT, pass_args_as_json=True),
            scheduler=AutoResearch(LLMSearcher(...)),
            callbacks=[StoreResultsCallback(), AutoResearchCallback()],
            ...
        )

    ``pass_args_as_json=True`` is needed when paths contain spaces.

    :param searcher: Searcher proposing the code candidates
    """

    def __init__(self, searcher: LLMSearcher):
        super().__init__(random_seed=0)
        self.searcher = searcher

    def suggest(self) -> TrialSuggestion | None:
        return TrialSuggestion.start_suggestion(config=self.searcher.suggest())

    @staticmethod
    def _candidate_id(trial: Trial) -> int:
        return int(Path(trial.config["candidate_dir"]).name)

    def on_trial_add(self, trial: Trial):
        self.searcher.on_trial_add(self._candidate_id(trial), trial.trial_id)

    def on_trial_complete(self, trial: Trial, result: dict[str, Any]):
        self.searcher.on_trial_complete(self._candidate_id(trial), result)

    def on_trial_error(self, trial: Trial):
        self.searcher.on_trial_error(self._candidate_id(trial))

    def metric_names(self) -> list[str]:
        return [self.searcher.metric]

    def metric_mode(self) -> str:
        return self.searcher.mode

    def metadata(self) -> dict[str, Any]:
        return {
            "scheduler_name": type(self).__name__,
            "agent": type(self.searcher.agent).__name__,
            "model": getattr(self.searcher.agent, "model", None),
            "editable_files": self.searcher.editable_files,
            "frozen_files": self.searcher.frozen_files,
        }


class AutoResearchCallback(TunerCallback):
    """Sets the output folder of the :class:`AutoResearch` searcher to
    ``<tuner_path>/autoresearch`` when tuning starts."""

    # Syne Tune does not give schedulers the tuner path, hence this callback.
    def on_tuning_start(self, tuner):
        assert isinstance(
            tuner.scheduler, AutoResearch
        ), "AutoResearchCallback requires an AutoResearch scheduler"
        tuner.scheduler.searcher.set_output_dir(tuner.tuner_path / "autoresearch")

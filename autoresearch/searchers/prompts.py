SYSTEM_PROMPT = """\
You are an expert machine learning engineer improving a training script through \
a sequence of experiments. Each of your proposals is run as a trial by a tuning \
system and you are told its score afterwards.
Rules:
- Only edit the files you are told are editable, in the current folder. Edit them \
in place with the Edit or Write tools; never create other files.
- You cannot run code. Make sure the script is correct by reading it carefully.
- Make one focused change per proposal so that its effect can be measured, unless \
told otherwise.
- Finish every reply with a single line starting with "SUMMARY:" that describes \
the change in at most 25 words.
"""

SUMMARY_PREFIX = "SUMMARY:"


def format_candidate(turn, metric: str) -> str:
    if turn.status == "completed":
        outcome = f"{metric}={turn.metric:.4f}"
    elif turn.status == "error":
        outcome = "FAILED"
    elif turn.status == "interrupted":
        outcome = "INTERRUPTED, not evaluated"
    else:
        outcome = "running"
    parent = "-" if turn.parent_id is None else turn.parent_id
    # Escaped so that the summary stays in its cell of the Markdown table.
    summary = turn.summary.replace("|", "\\|")
    return f"| {turn.candidate_id} | {parent} | {outcome} | {summary} |"


def build_prompt(
    candidate_id: int,
    turns: list,
    new_results: list,
    parent,
    parent_files: dict[str, str],
    metric: str,
    mode: str,
    task_prompt: str | None,
    read_only_files: list[str],
    lost_candidates: list[int] = (),
) -> str:
    """Prompt asking for the next candidate.

    :param task_prompt: Description of the problem, only given in the first
        message of a conversation
    :param lost_candidates: Candidates the agent proposed after the checkpoint
        tuning was resumed from, their ids are reused
    """
    sections = []
    if task_prompt is not None:
        sections.append(f"# Task\n\n{task_prompt}")
        if read_only_files:
            sections.append(
                "Read-only files for context, in the current folder: "
                + ", ".join(read_only_files)
            )
    if lost_candidates:
        ids = ", ".join(str(i) for i in lost_candidates)
        candidates = "candidates" if len(lost_candidates) > 1 else "candidate"
        sections.append(
            "# Tuning restarted\n\n"
            "The tuning was restarted from a checkpoint. Your proposals for "
            f"{candidates} {ids} were lost before being evaluated, ignore them: "
            "their ids are reused, and the results below are the current state."
        )
    if new_results:
        lines = []
        for turn in new_results:
            lines.append(format_candidate(turn, metric))
            if turn.status == "error":
                lines.append(
                    f"Error of candidate {turn.candidate_id}:\n```\n{turn.error}\n```"
                )
        sections.append(
            "# New results since your last proposal\n\n" + "\n".join(lines)
        )
    if turns:
        direction = "higher" if mode == "max" else "lower"
        table = "\n".join(format_candidate(turn, metric) for turn in turns)
        sections.append(
            f"# All candidates ({metric}, {direction} is better)\n\n"
            f"| id | parent | result | change |\n|---|---|---|---|\n{table}"
        )
    if any(turn.status == "pending" for turn in turns):
        sections.append(
            "Candidates marked running are being evaluated in parallel right now. "
            "Propose something different from them."
        )
    if parent is None:
        origin = "the initial version"
    else:
        score = (
            f" ({metric}={parent.metric:.4f})"
            if parent.metric is not None
            else " (not evaluated yet)"
        )
        origin = f"candidate {parent.candidate_id}{score}, the best so far"
    code = "\n\n".join(
        f"`{name}`:\n```\n{content}```" for name, content in parent_files.items()
    )
    # The current code is always included because the agent may otherwise rely
    # on outdated file contents from earlier in the conversation.
    sections.append(
        f"# Your next proposal: candidate {candidate_id}\n\n"
        f"The editable files in the current folder were reset to {origin}. "
        f"Their current content, which replaces any earlier version you saw:\n\n"
        f"{code}\n\n"
        f"Edit {', '.join(parent_files)} in place to improve {metric}."
    )
    return "\n\n".join(sections)


def parse_summary(text: str) -> str:
    """The summary line of the reply, or its end if there is none, on a single
    line so that it fits in a row of the tables of candidates."""
    summaries = [line for line in text.splitlines() if line.startswith(SUMMARY_PREFIX)]
    if summaries:
        summary = summaries[-1].removeprefix(SUMMARY_PREFIX)
    else:
        summary = text[-300:]
    return " ".join(summary.split())

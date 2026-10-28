import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentResponse:
    """Result of one message sent to a :class:`CodeAgent`.

    :param text: Final text reply of the agent
    :param messages: Raw messages of the exchange (text, tool calls, results)
    :param session_id: Identifier of the conversation
    :param cost_usd: Cost of this call only
    :param duration_s: Wall-clock duration of this call
    """

    text: str
    messages: list[dict]
    session_id: str | None
    cost_usd: float
    duration_s: float


class CodeAgent:
    """Coding agent that edits files in a working directory and keeps one
    conversation across calls of :meth:`send`.
    """

    def send(
        self, message: str, cwd: Path, system_prompt: str | None = None
    ) -> AgentResponse:
        """Send a message and return once the agent has finished editing.

        :param message: User message
        :param cwd: Working directory, the agent may only edit files in it. It must
            be the same folder for all calls of a conversation.
        :param system_prompt: Instructions appended to the agent's system prompt,
            the same on every call of a conversation
        """
        raise NotImplementedError


class ClaudeCodeAgent(CodeAgent):
    """:class:`CodeAgent` calling the Claude Code CLI in print mode.

    The conversation is continued with ``claude -p --resume <session_id>``, so it
    survives restarts of this process. Claude can only use file tools, which are
    confined to the working directory; tools that run code are not available, and
    user/project settings, CLAUDE.md files, hooks and MCP servers are ignored.

    :param model: Model alias or name passed to ``claude --model``
    :param effort: Reasoning effort passed to ``claude --effort`` (low, medium,
        high, xhigh or max), the default of Claude Code if None
    :param timeout: Maximum seconds for one call
    :param tools: Built-in tools available to Claude
    :param binary: Name or path of the Claude Code executable
    """

    def __init__(
        self,
        model: str = "sonnet",
        effort: str | None = None,
        timeout: float = 600.0,
        tools: tuple[str, ...] = ("Read", "Edit", "Write", "Glob", "Grep"),
        binary: str = "claude",
    ):
        if shutil.which(binary) is None:
            raise FileNotFoundError(
                f"Claude Code executable `{binary}` not found, see "
                "https://docs.claude.com/en/docs/claude-code for installation."
            )
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.tools = tools
        self.binary = binary
        self.session_id: str | None = None
        # Claude Code reports the cumulative cost of a session, not per call.
        self._session_cost = 0.0

    def send(
        self, message: str, cwd: Path, system_prompt: str | None = None
    ) -> AgentResponse:
        cmd = [
            self.binary,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model,
            "--restricted",
            "--strict-mcp-config",
            "--tools",
            ",".join(self.tools),
            "--permission-mode",
            "acceptEdits",
        ]
        if self.effort is not None:
            cmd += ["--effort", self.effort]
        # Passed on every call: Claude Code reuses the prompt recorded on the
        # first call, until the conversation is compacted where it is rebuilt
        # from this flag, without which the rules would be lost.
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        if self.session_id is not None:
            cmd += ["--resume", self.session_id]
        # Allows tuning to be launched from within a Claude Code session.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        start = time.time()
        # The message is read from a file given as input rather than passed as
        # an argument, whose length is limited by the operating system.
        with tempfile.TemporaryFile("w+") as message_file:
            message_file.write(message)
            message_file.seek(0)
            process = subprocess.run(
                cmd,
                cwd=cwd,
                stdin=message_file,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        duration = time.time() - start
        messages = [json.loads(line) for line in process.stdout.splitlines() if line]
        result = next((m for m in messages if m.get("type") == "result"), None)
        if process.returncode != 0 or result is None or result.get("is_error"):
            raise RuntimeError(
                f"{self.binary} failed with code {process.returncode}: "
                f"{process.stderr[-2000:] or (result or {}).get('result')}"
            )
        self.session_id = result["session_id"]
        session_cost = result.get("total_cost_usd") or 0.0
        cost = max(session_cost - self._session_cost, 0.0)
        self._session_cost = session_cost
        return AgentResponse(
            text=result.get("result") or "",
            messages=messages,
            session_id=self.session_id,
            cost_usd=cost,
            duration_s=duration,
        )

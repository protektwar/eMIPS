"""Compatibility: the session moved to :mod:`emips.session` (shared by
the GUI and the CLI debugger)."""

from ..session import InputNeeded, Session, StepResult  # noqa: F401

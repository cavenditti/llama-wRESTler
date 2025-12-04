"""
Logging configuration module for llama_wrestler.

This module provides centralized logging setup with rich console output
and custom log levels for structured phase and step tracking.
"""

import logging
import sys
from typing import Optional, Any
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from devtools import debug, pformat


# Custom log levels
PHASE_LEVEL = 35  # Between WARNING (30) and CRITICAL (50)
STEP_LEVEL = 25  # Between INFO (20) and WARNING (30)
UPDATE_LEVEL = 22  # Between INFO (20) and STEP (25) - for relevant updates

# Add custom level names and colors
logging.addLevelName(PHASE_LEVEL, "PHASE")
logging.addLevelName(STEP_LEVEL, "STEP")
logging.addLevelName(UPDATE_LEVEL, "UPDATE")

# Default level when no verbosity flags are used
DEFAULT_LEVEL = UPDATE_LEVEL


class Logger(logging.Logger):
    """Extended Logger class with custom log levels for phase, step, and update messages."""

    def phase(self, message: str, *args, **kwargs) -> None:
        """Log a phase message with automatic separators."""
        if self.isEnabledFor(PHASE_LEVEL):
            self._log(PHASE_LEVEL, message, args, **kwargs)

    def step(self, message: str, *args, **kwargs) -> None:
        """Log a step message with automatic separator."""
        if self.isEnabledFor(STEP_LEVEL):
            self._log(STEP_LEVEL, message, args, **kwargs)

    def update(self, message: str, *args, **kwargs) -> None:
        """Log a relevant update message (shown by default)."""
        if self.isEnabledFor(UPDATE_LEVEL):
            self._log(UPDATE_LEVEL, message, args, **kwargs)

    def debugf(self, obj: Any) -> None:
        """Log an object using devtools formatting at DEBUG level."""
        if self.isEnabledFor(logging.DEBUG):
            self.debug(debug.format(obj))


# Set our custom Logger class as the default
logging.setLoggerClass(Logger)


class CustomRichHandler(RichHandler):
    """CustomRichHandler that adds separators for PHASE and STEP levels."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record with custom formatting for PHASE and STEP levels."""
        # Add separator for PHASE level
        if record.levelno == PHASE_LEVEL:
            separator = "═" * 120
            self.console.print(separator, style="bold cyan")
            super().emit(record)
        # Add separator for STEP level
        elif record.levelno == STEP_LEVEL:
            separator = "─" * 120
            self.console.print(separator, style="dim cyan")
            super().emit(record)
        else:
            super().emit(record)


def setup_logging(
    level: int = DEFAULT_LEVEL,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> Logger:
    """
    Configure logging for the application using rich console output.

    Args:
        level: Logging level (default: UPDATE_LEVEL). Use logging.INFO for -v, logging.DEBUG for -vv
        log_file: Optional file path to write logs to
        format_string: Optional custom format string (ignored for console, used for file)

    Returns:
        Configured root logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get or create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create custom theme for rich
    theme = Theme(
        {
            "logging.level.debug": "dim cyan",
            "logging.level.info": "green",
            "logging.level.update": "blue",
            "logging.level.warning": "yellow",
            "logging.level.error": "red",
            "logging.level.critical": "bold red",
            "logging.level.phase": "bold cyan",
            "logging.level.step": "dim cyan",
        }
    )

    # Console handler with rich
    console = Console(file=sys.stdout, theme=theme, force_terminal=True)
    rich_handler = CustomRichHandler(
        console=console,
        level=level,
        show_time=True,
        show_path=False,
    )
    root_logger.addHandler(rich_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger  # type: ignore[return-value]


def get_logger(name: str) -> Logger:
    """
    Get a logger with the specified name.

    Convenience function for getting a logger instance for a module.

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)  # type: ignore[return-value]

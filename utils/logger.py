import logging
import os
import sys
import traceback
import json
from typing import Any, Dict
from pathlib import Path
from datetime import datetime

# Import optional dependencies
try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": "\n".join(traceback.format_tb(record.exc_info[2])) if record.exc_info[2] else None
            }

        # Add any extra fields from record
        for key, value in record.__dict__.items():
            if key not in ("name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                         "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                         "created", "msecs", "relativeCreated", "thread", "threadName", "process",
                         "processName", "getMessage"):
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


class TracebackFormatter(logging.Formatter):
    """Custom formatter that always includes tracebacks for errors."""

    def __init__(self, include_traceback: bool = True):
        super().__init__()
        self.include_traceback = include_traceback

    def format(self, record: logging.LogRecord) -> str:
        formatted_message = super().format(record)

        if self.include_traceback and record.exc_info:
            tb_lines = traceback.format_exception(*record.exc_info)
            formatted_message += "\n" + "".join(tb_lines)

        return formatted_message


def setup_logger(
    name: str = "rag_system",
    level: str = "INFO",
    log_file: str = None,
    use_json: bool = False,
    include_sentry: bool = True
) -> logging.Logger:
    """
    Setup logger with console and file handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        use_json: Use JSON formatting for structured logging
        include_sentry: Include Sentry error capturing

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Choose formatter
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = TracebackFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Configure Sentry (if available)
    if include_sentry and SENTRY_AVAILABLE:
        dsns = os.getenv("SENTRY_DSN")
        if dsns:
            sentry_sdk.init(
                dsn=dsns,
                traces_sample_rate=1.0,
                profiles_sample_rate=1.0,
            )
            logger.info("Sentry error tracking enabled")
        else:
            logger.warning("SENTRY_DSN not configured, Sentry disabled")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    Falls back to the root rag_system logger if not found.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Configure with defaults if not configured
        logger = setup_logger(name)
    return logger


# Global logger instance
logger = setup_logger()


def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None):
    """Decorator to log function calls with parameters."""
    def decorator(func):
        def wrapper(*func_args, **func_kwargs):
            logger.debug(f"Calling {func_name}",
                        extra={"function": func_name, "args": args, "kwargs": kwargs})
            try:
                result = func(*func_args, **func_kwargs)
                logger.debug(f"Function {func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func_name} failed: {e}",
                           exc_info=True, extra={"function": func_name})
                raise
        return wrapper
    return decorator


import os  # Import here to avoid circular dependencies

"""
Logging configuration for NautilusTrader integration.

This module provides structured logging with correlation IDs and comprehensive
error tracking, following the patterns established in the knowledge-ingestion system.
"""

import logging
import logging.handlers
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor

from nautilus_integration.core.config import LoggingConfig

# Context variable for correlation ID tracking
correlation_id_context: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_context.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get the correlation ID for the current context."""
    return correlation_id_context.get()


def add_correlation_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add correlation ID to log events."""
    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_component_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add component information to log events."""
    # Extract component from logger name
    logger_name = getattr(logger, "name", "unknown")
    if "nautilus_integration" in logger_name:
        component = logger_name.replace("nautilus_integration.", "").split(".")[0]
        event_dict["component"] = component
    return event_dict


def add_integration_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add integration-specific context to log events."""
    event_dict["integration"] = "nautilus-trader"
    event_dict["system"] = "algorithmic-trading"
    return event_dict


def filter_sensitive_data(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Filter sensitive data from log events."""
    sensitive_keys = {
        "password", "token", "key", "secret", "auth", "credential",
        "api_key", "access_token", "refresh_token"
    }
    
    def _filter_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively filter sensitive data from dictionary."""
        filtered = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                filtered[key] = "***REDACTED***"
            elif isinstance(value, dict):
                filtered[key] = _filter_dict(value)
            elif isinstance(value, list):
                filtered[key] = [
                    _filter_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value
        return filtered
    
    # Filter the entire event dict
    return _filter_dict(event_dict)


class CorrelationIdFilter(logging.Filter):
    """Logging filter to add correlation ID to standard library logs."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        correlation_id = get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        else:
            record.correlation_id = "N/A"
        return True


def setup_standard_logging(config: LoggingConfig) -> None:
    """Set up standard library logging."""
    # Create formatter
    if config.structured_logging:
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "correlation_id": "%(correlation_id)s", '
            '"message": "%(message)s"}'
        )
    else:
        formatter = logging.Formatter(config.format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CorrelationIdFilter())
    root_logger.addHandler(console_handler)
    
    # Add file handler if configured
    if config.log_file_path:
        log_path = Path(config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config.log_rotation_enabled:
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=_parse_size(config.log_max_size),
                backupCount=config.log_backup_count
            )
        else:
            file_handler = logging.FileHandler(log_path)
        
        file_handler.setFormatter(formatter)
        file_handler.addFilter(CorrelationIdFilter())
        root_logger.addHandler(file_handler)


def setup_structlog(config: LoggingConfig) -> None:
    """Set up structured logging with structlog."""
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_correlation_id,
        add_component_info,
        add_integration_context,
        filter_sensitive_data,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    
    if config.structured_logging:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.level)
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _parse_size(size_str: str) -> int:
    """Parse size string (e.g., '100MB') to bytes."""
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


def setup_logging(config: LoggingConfig) -> None:
    """
    Set up comprehensive logging for NautilusTrader integration.
    
    This function configures both standard library logging and structured
    logging with correlation ID tracking and component attribution.
    
    Args:
        config: Logging configuration
    """
    # Set up standard library logging first
    setup_standard_logging(config)
    
    # Set up structured logging
    setup_structlog(config)
    
    # Configure NautilusTrader logging
    nautilus_logger = logging.getLogger("nautilus_trader")
    nautilus_logger.setLevel(getattr(logging, config.level))
    
    # Configure integration logging
    integration_logger = logging.getLogger("nautilus_integration")
    integration_logger.setLevel(getattr(logging, config.level))
    
    # Log setup completion
    logger = structlog.get_logger("nautilus_integration.logging")
    logger.info(
        "Logging setup completed",
        level=config.level,
        structured=config.structured_logging,
        correlation_tracking=config.correlation_id_enabled,
        file_logging=config.log_file_path is not None,
        rotation_enabled=config.log_rotation_enabled,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


class LoggingContextManager:
    """Context manager for correlation ID tracking."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize logging context manager.
        
        Args:
            correlation_id: Optional correlation ID (generates new one if None)
        """
        self.correlation_id = correlation_id or generate_correlation_id()
        self.previous_correlation_id: Optional[str] = None
    
    def __enter__(self) -> str:
        """Enter the context and set correlation ID."""
        self.previous_correlation_id = get_correlation_id()
        set_correlation_id(self.correlation_id)
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and restore previous correlation ID."""
        if self.previous_correlation_id:
            set_correlation_id(self.previous_correlation_id)
        else:
            correlation_id_context.set(None)


def with_correlation_id(correlation_id: Optional[str] = None) -> LoggingContextManager:
    """
    Create a logging context with correlation ID tracking.
    
    Args:
        correlation_id: Optional correlation ID (generates new one if None)
        
    Returns:
        Context manager for correlation ID tracking
        
    Example:
        with with_correlation_id() as cid:
            logger.info("Processing request", request_id="123")
            # All logs in this context will have the same correlation ID
    """
    return LoggingContextManager(correlation_id)


def log_error_with_context(
    logger: structlog.BoundLogger,
    error: Exception,
    context: Dict[str, Any],
    message: str = "Error occurred"
) -> None:
    """
    Log an error with comprehensive context information.
    
    Args:
        logger: Structured logger
        error: Exception that occurred
        context: Additional context information
        message: Error message
    """
    logger.error(
        message,
        error_type=type(error).__name__,
        error_message=str(error),
        **context,
        exc_info=True
    )
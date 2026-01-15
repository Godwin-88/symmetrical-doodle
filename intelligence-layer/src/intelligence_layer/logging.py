"""Logging configuration for intelligence layer."""

import logging
import sys
from typing import Any, Dict
import structlog
from .config import LoggingConfig


def configure_logging(config: LoggingConfig) -> None:
    """Configure structured logging."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.level.upper()),
    )
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if config.format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class RequestLogger:
    """Request logging middleware."""
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    async def log_request(self, request: Any, response: Any, duration: float) -> None:
        """Log request details."""
        self.logger.info(
            "request_completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration_ms=duration * 1000,
        )
    
    async def log_error(self, request: Any, error: Exception) -> None:
        """Log request error."""
        self.logger.error(
            "request_error",
            method=request.method,
            url=str(request.url),
            error=str(error),
            error_type=type(error).__name__,
        )
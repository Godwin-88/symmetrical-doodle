"""
Structured logging infrastructure with correlation IDs for Google Drive Knowledge Base Ingestion.
Provides comprehensive logging capabilities for debugging and monitoring.
"""

import sys
import uuid
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import structlog
from structlog.types import FilteringBoundLogger
from structlog.stdlib import LoggerFactory

from .config import get_settings, LoggingConfig


# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


@dataclass
class LogContext:
    """Log context information"""
    correlation_id: str
    component: str
    operation: Optional[str] = None
    file_id: Optional[str] = None
    document_id: Optional[str] = None
    phase: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CorrelationIDProcessor:
    """Processor to add correlation ID to log records"""
    
    def __call__(self, logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        correlation_id = correlation_id_var.get()
        if correlation_id:
            event_dict["correlation_id"] = correlation_id
        return event_dict


class TimestampProcessor:
    """Processor to add ISO timestamp to log records"""
    
    def __call__(self, logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        return event_dict


class ComponentProcessor:
    """Processor to add component information to log records"""
    
    def __init__(self, component: str):
        self.component = component
    
    def __call__(self, logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        event_dict["component"] = self.component
        return event_dict


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "lineno", "funcName", "created", 
                          "msecs", "relativeCreated", "thread", "threadName", 
                          "processName", "process", "getMessage", "exc_info", "exc_text", "stack_info"]:
                log_data[key] = value
                
        return json.dumps(log_data, default=str)


class LoggingManager:
    """Central logging manager for the knowledge ingestion system"""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or get_settings().logging
        self._configured = False
        self._loggers: Dict[str, FilteringBoundLogger] = {}
        
    def configure_logging(self):
        """Configure structured logging with correlation IDs"""
        if self._configured:
            return
            
        # Configure structlog
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            TimestampProcessor(),
            CorrelationIDProcessor(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if self.config.format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
            
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure standard library logging
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if self.config.format == "json":
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        root_logger.addHandler(console_handler)
        
        # File handler if configured
        if self.config.log_file:
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            
            if self.config.format == "json":
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            root_logger.addHandler(file_handler)
            
        self._configured = True
    
    def get_logger(self, name: str, component: Optional[str] = None) -> FilteringBoundLogger:
        """Get a structured logger for a component"""
        if not self._configured:
            self.configure_logging()
            
        if name not in self._loggers:
            logger = structlog.get_logger(name)
            if component:
                logger = logger.bind(component=component)
            self._loggers[name] = logger
            
        return self._loggers[name]
    
    def set_correlation_id(self, correlation_id: Optional[str] = None) -> str:
        """Set correlation ID for current context"""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
        return correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID"""
        return correlation_id_var.get()
    
    def clear_correlation_id(self):
        """Clear correlation ID from current context"""
        correlation_id_var.set(None)
    
    def create_log_context(
        self, 
        component: str, 
        operation: Optional[str] = None,
        file_id: Optional[str] = None,
        document_id: Optional[str] = None,
        phase: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> LogContext:
        """Create a log context with correlation ID"""
        if correlation_id is None:
            correlation_id = self.get_correlation_id() or self.set_correlation_id()
            
        return LogContext(
            correlation_id=correlation_id,
            component=component,
            operation=operation,
            file_id=file_id,
            document_id=document_id,
            phase=phase,
            metadata=metadata
        )


# Global logging manager instance
logging_manager = LoggingManager()


def configure_logging(config: Optional[LoggingConfig] = None):
    """Configure global logging"""
    global logging_manager
    if config:
        logging_manager = LoggingManager(config)
    logging_manager.configure_logging()


def get_logger(name: str, component: Optional[str] = None) -> FilteringBoundLogger:
    """Get a structured logger"""
    return logging_manager.get_logger(name, component)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for current context"""
    return logging_manager.set_correlation_id(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return logging_manager.get_correlation_id()


def clear_correlation_id():
    """Clear correlation ID from current context"""
    logging_manager.clear_correlation_id()


def create_log_context(
    component: str, 
    operation: Optional[str] = None,
    **kwargs
) -> LogContext:
    """Create a log context"""
    return logging_manager.create_log_context(component, operation, **kwargs)


class LogContextManager:
    """Context manager for logging with correlation ID"""
    
    def __init__(self, context: LogContext):
        self.context = context
        self.previous_correlation_id = None
        
    def __enter__(self) -> LogContext:
        self.previous_correlation_id = get_correlation_id()
        set_correlation_id(self.context.correlation_id)
        return self.context
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_correlation_id:
            set_correlation_id(self.previous_correlation_id)
        else:
            clear_correlation_id()


def log_context(
    component: str, 
    operation: Optional[str] = None,
    **kwargs
) -> LogContextManager:
    """Create a logging context manager"""
    context = create_log_context(component, operation, **kwargs)
    return LogContextManager(context)


# Performance logging utilities
class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger: FilteringBoundLogger):
        self.logger = logger
        
    def log_operation_start(self, operation: str, **kwargs):
        """Log operation start"""
        self.logger.info(
            "Operation started",
            operation=operation,
            **kwargs
        )
        
    def log_operation_complete(self, operation: str, duration_ms: float, **kwargs):
        """Log operation completion"""
        self.logger.info(
            "Operation completed",
            operation=operation,
            duration_ms=duration_ms,
            **kwargs
        )
        
    def log_operation_error(self, operation: str, error: Exception, duration_ms: float, **kwargs):
        """Log operation error"""
        self.logger.error(
            "Operation failed",
            operation=operation,
            error=str(error),
            error_type=type(error).__name__,
            duration_ms=duration_ms,
            **kwargs
        )


def get_performance_logger(name: str) -> PerformanceLogger:
    """Get a performance logger"""
    logger = get_logger(name)
    return PerformanceLogger(logger)
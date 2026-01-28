"""
Logging module for NautilusTrader integration.

This module provides a simple interface to the nautilus_logging module
to maintain compatibility with existing imports.
"""

# Re-export everything from nautilus_logging for compatibility
from .nautilus_logging import (
    get_logger,
    set_correlation_id,
    get_correlation_id,
    generate_correlation_id,
    setup_logging,
    with_correlation_id,
    log_error_with_context,
    LoggingContextManager,
    CorrelationIdFilter,
    add_correlation_id,
    add_component_info,
    add_integration_context,
    filter_sensitive_data
)

__all__ = [
    'get_logger',
    'set_correlation_id', 
    'get_correlation_id',
    'generate_correlation_id',
    'setup_logging',
    'with_correlation_id',
    'log_error_with_context',
    'LoggingContextManager',
    'CorrelationIdFilter',
    'add_correlation_id',
    'add_component_info',
    'add_integration_context',
    'filter_sensitive_data'
]
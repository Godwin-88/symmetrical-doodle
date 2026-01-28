"""
NautilusTrader Integration Package

This package provides integration between NautilusTrader and the existing
algorithmic trading platform, following the patterns established in the
knowledge-ingestion system.
"""

__version__ = "0.1.0"
__author__ = "Trading System"
__email__ = "dev@tradingsystem.com"

from nautilus_integration.core.config import NautilusConfig
from nautilus_integration.core.nautilus_logging import setup_logging
from nautilus_integration.services.integration_service import NautilusIntegrationService

__all__ = [
    "NautilusConfig",
    "setup_logging", 
    "NautilusIntegrationService",
]